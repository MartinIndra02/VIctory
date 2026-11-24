//! Buffer views in heap slots no table names, reached by a translated
//! program that loads its view's slot when it runs: through a title heap
//! object, the device methods that write its descriptors, a recorded
//! dispatch and a submission, on the host GPU. The title memory is
//! synthetic; nothing here brings a host copy up to date but the code under
//! test.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use victory_shader::ResourceType;
use victory_shader::emit::{HostShader, StrideEntry};
use windows::Win32::Graphics::Direct3D12::*;

use super::raw_view_tests::{
    LOADED_SLOT_READ, R32_TYPELESS, R32G32B32A32_FLOAT, RAW, WIDTH, WINDOW, placed, translate,
};
use super::{HEAP_IIDS, Heap, HeapDesc, Placement, View, create_heap, heap};
use crate::host::{REACHED, host};
use crate::object::{S_OK, This};
use crate::pipeline::host::HostPipeline;

/// One test at a time: they count what submissions copy.
static SERIAL: Mutex<()> = Mutex::new(());

/// The first slot of the table the programs are given: the buffer of heap
/// slots, then the view the first program writes (the second reads).
const TABLE: u32 = 2;

/// Whether the host runs translated shaders (shader model 6.6).
fn translated_shaders_run() -> bool {
    let Some(host) = host() else { return false };
    let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_6,
    };
    // SAFETY: a feature query into a struct of its size.
    let queried = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut model).cast(),
            size_of_val(&model) as u32,
        )
    };
    queried.is_ok() && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0
}

/// Write-watched title memory, as the title's allocations are: the first
/// of `windows` windows the host copies apart. Never freed: host copies of
/// it outlive the test. Every page is touched: the first read of a page
/// never touched counts as a write of it (the kernel zeroes it then), and
/// the tests count what is copied.
fn title_memory(windows: u64) -> u64 {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    // SAFETY: a fresh allocation.
    let p = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            ((windows + 1) * WINDOW) as usize,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
        )
    } as u64;
    assert_ne!(p, 0, "title memory");
    for at in (p..p + (windows + 1) * WINDOW).step_by(4096) {
        // SAFETY: inside the allocation.
        unsafe { (at as *mut u8).write(0) };
    }
    p.next_multiple_of(WINDOW)
}

/// `N` dwords of title memory from dword `first` of the buffer at
/// `address`.
fn dwords<const N: usize>(address: u64, first: usize) -> [u32; N] {
    // SAFETY: inside a buffer of the test's title memory.
    unsafe { (address as *const u32).add(first).cast::<[u32; N]>().read() }
}

/// The title's CPU writes `values` from dword `first` of the buffer at
/// `address`.
fn title_writes(address: u64, first: usize, values: impl IntoIterator<Item = u32>) {
    for (i, v) in values.into_iter().enumerate() {
        // SAFETY: inside a buffer of the test's title memory.
        unsafe { (address as *mut u32).add(first + i).write(v) };
    }
}

/// A shader-visible view heap of the title's with `count` slots.
fn title_heap(count: u32) -> (This, &'static Heap) {
    #[repr(C)]
    struct Extended(HeapDesc, Placement);
    let extended = Extended(
        HeapDesc {
            kind: 0,
            count,
            flags: 1,
            nodes: 0,
        },
        Placement {
            address: 0,
            size: 0,
        },
    );
    let mut h: This = std::ptr::null();
    assert_eq!(
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], &mut h),
        S_OK
    );
    (h, heap(h))
}

/// The title makes `view` (of memory it placed) in slot `slot` of its
/// heap, with the device methods titles call.
fn title_makes(heap: &Heap, slot: u32, view: &View) {
    let View::Placed(unordered, resource, desc) = view else {
        panic!("a view of placed memory");
    };
    let handle = heap.start + u64::from(slot) * 32;
    if *unordered {
        super::create_placed_uav(std::ptr::null(), resource, desc, handle);
    } else {
        super::create_placed_srv(std::ptr::null(), resource, desc, handle);
    }
}

/// The host pipeline of the translated compute program `words`, whose
/// used-resource entries are the table's first two slots (of `kinds`).
fn pipeline(words: &[u32], kinds: [ResourceType; 2], run_time_buffers: u8) -> HostPipeline {
    let (cs, used) = translate(words, kinds);
    assert_eq!(cs.run_time_buffers, run_time_buffers);
    let shader = crate::pipeline::translated_shader("compute", cs, used);
    HostPipeline::new(&[shader], None).expect("a host pipeline")
}

/// The title records a dispatch of two thread groups of `pipeline` with
/// its heap `h` set and the table at [`TABLE`], and submits the list.
fn title_dispatches(h: This, pipeline: &HostPipeline) {
    let host = host().expect("the host");
    let list = host.new_list().expect("a list");
    let mut state = crate::draw::State::default();
    state.set_heaps(&[h]);
    state.set_table(true, 1, heap(h).start + u64::from(TABLE) * 32);
    state.dispatch_with(&list, pipeline, 2, 1, 1);
    list.close();
    host.execute(&[&list]);
}

/// Heaps brought up to date, host copies' ranges checked and bytes copied
/// for shaders that pick their view at run time, so far.
fn reached() -> [u64; 3] {
    [0, 1, 2].map(|i| REACHED[i].load(Ordering::Relaxed))
}

fn structured(stride: u32) -> (u32, u32, u32) {
    (0, stride, 0)
}

const RAW_VIEW: (u32, u32, u32) = (R32_TYPELESS, 0, RAW);
const PAGE: u64 = 64 << 10;

/// A program that loads its view's heap slot from a buffer when it runs
/// reads the title memory of whatever buffer view the title has in that
/// slot, though no table of the dispatch names the slot: a structured view
/// and a typed one, written before and after the host heap existed; again
/// after the title rewrites a slot to another buffer, after it changes a
/// buffer's bytes, and nothing once it puts a view of no buffer there.
/// Without `HostHeap::reach` and the submission's refresh every value read
/// through a picked slot is 0.
#[test]
fn views_in_slots_picked_at_run_time_read_title_memory() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !translated_shaders_run() {
        return; // no host GPU, or no shader model 6.6
    }
    // Title memory in three windows the host copies apart: the buffers the
    // program reads through slots it picks, the buffer of slots, and what
    // it writes. (The host makes a structured view only where its elements
    // fall on its copy's: the data starts a multiple of 12, 16 and 20 bytes
    // in.) A is in the first 64 KiB page of its window, B in the second.
    let memory = title_memory(3);
    let (a, b) = (memory + 0xF000, memory + 0x1_E000);
    let (slots, written) = (memory + WINDOW, memory + 2 * WINDOW);
    title_writes(a, 0, (0..WIDTH / 4).map(|i| 100 + i));
    title_writes(b, 0, (0..WIDTH / 4).map(|i| 200 + i));

    // The table: a raw view of the slots, then what the program writes,
    // elements of 4 bytes from element 5 (20 bytes in).
    let (h, heap) = title_heap(16);
    title_makes(heap, TABLE, &placed(false, slots, RAW_VIEW, (0, 16)));
    title_makes(heap, TABLE + 1, &placed(true, written, structured(4), (5, 8)));
    // Five slots past the table: three elements of 12 bytes of A from
    // element 3 (36 bytes in), written before the host heap exists.
    title_makes(heap, TABLE + 5, &placed(false, a, structured(12), (3, 3)));
    let host_heap = heap.host().expect("a host heap");
    // Nine past it: four RGBA32F elements of B from element 2 (32 bytes
    // in), written once it does.
    let typed = (R32G32B32A32_FLOAT, 0, 0);
    title_makes(heap, TABLE + 9, &placed(false, b, typed, (2, 4)));
    // No draw names them: neither is made.
    assert_eq!(host_heap.entry_words(TABLE + 5), [0, 0]);
    assert_eq!(host_heap.entry_words(TABLE + 9), [0, 0]);
    assert_eq!(host_heap.reached_counts(), (0, 0));

    // victory-shader's `unknown_stride_views_chosen_by_a_loaded_heap_index`
    // program: each of two thread groups loads a slot (counted from the
    // table) from the buffer in the table's first slot, and its four lanes
    // read `[lane, 4]` of the view there and store it at
    // `[lane, 16 * group]` of the view in the table's second slot.
    let pipeline = pipeline(
        LOADED_SLOT_READ,
        [ResourceType::Srv, ResourceType::Uav],
        HostShader::RUN_TIME_READS,
    );
    let run = |chosen: [u32; 2]| -> [u32; 8] {
        title_writes(slots, 0, chosen);
        title_writes(written, 5, [0; 8]);
        title_dispatches(h, &pipeline);
        // What the shaders wrote, back in title memory.
        dwords(written, 5)
    };

    // Lane l reads [l, 4]: of the structured view A's dwords 10, 13, 16 and
    // (past its three elements) nothing; of the typed one B's dwords 9,
    // 13, 17, 21.
    let before = reached();
    let (twelve, sixteen) = ([110, 113, 116, 0], [209, 213, 217, 221]);
    assert_eq!(run([5, 9]), [twelve, sixteen].concat()[..]);
    // The views that waited were made, the table's two among them: the
    // heap's four buffer views. The structured and typed ones have a raw
    // view in the pool (after the heap's two null ones and the written
    // view's) that their table entries name.
    assert_eq!(host_heap.reached_counts(), (4, 4));
    let pool = host_heap.alias_base() + 2;
    let entry = |stride, raw_slot, bias| {
        StrideEntry {
            stride,
            raw_slot,
            bias,
        }
        .words()
    };
    assert_eq!(host_heap.entry_words(TABLE + 5), entry(12, Some(pool + 1), 4));
    assert_eq!(host_heap.entry_words(TABLE + 9), entry(16, Some(pool + 2), 0));
    // The submission brought one heap's views up to date: three host
    // copies (a window each), whole pages: A's and B's two, the slots'
    // one, the written view's one.
    let first = reached();
    assert_eq!(
        [first[0] - before[0], first[1] - before[1]],
        [1, 3],
        "heaps and host copies"
    );
    assert_eq!(first[2] - before[2], 4 * PAGE, "bytes copied");
    assert_eq!(run([9, 5]), [sixteen, twelve].concat()[..]);
    // Nothing of A or B is copied again: only the pages this test wrote
    // (the slots, the zeroed output) since.
    let second = reached();
    assert_eq!(second[2] - first[2], 2 * PAGE, "bytes copied again");

    // The title rewrites the first picked slot to another buffer: three
    // elements of 12 bytes of B from element 3. Made as it is written, in
    // the raw view slot the slot had.
    title_makes(heap, TABLE + 5, &placed(false, b, structured(12), (3, 3)));
    assert_eq!(host_heap.entry_words(TABLE + 5), entry(12, Some(pool + 1), 4));
    assert_eq!(host_heap.reached_counts(), (5, 4));
    assert_eq!(run([5, 9]), [[210, 213, 216, 0], sixteen].concat()[..]);

    // The title's CPU changes B's bytes between two dispatches.
    let third = reached();
    title_writes(b, 0, (0..WIDTH / 4).map(|i| 300 + i));
    assert_eq!(
        run([5, 9]),
        [[310, 313, 316, 0], [309, 313, 317, 321]].concat()[..]
    );
    // B's page, with the two this test writes each run.
    assert_eq!(reached()[2] - third[2], 3 * PAGE, "bytes copied for B");

    // The second picked slot becomes a raw view of A from dword 8 (its
    // own slot holds it; the raw view slot goes back to the pool).
    title_makes(heap, TABLE + 9, &placed(false, a, RAW_VIEW, (8, 64)));
    assert_eq!(host_heap.entry_words(TABLE + 9), entry(4, None, 0));
    assert_eq!(
        run([5, 9]),
        [[310, 313, 316, 0], [109, 110, 111, 112]].concat()[..]
    );

    // A view of no buffer in the first picked slot: nothing to read, no
    // entry, and one buffer view fewer in the heap.
    let handle = heap.start + u64::from(TABLE + 5) * 32;
    super::create_srv(std::ptr::null(), std::ptr::null(), std::ptr::null(), handle);
    assert_eq!(host_heap.entry_words(TABLE + 5), [0, 0]);
    assert_eq!(host_heap.reached_counts().1, 3);
    assert_eq!(run([5, 9]), [[0; 4], [109, 110, 111, 112]].concat()[..]);
}

/// A view of a buffer object in a slot picked at run time reads the
/// buffer's memory; once the title releases the buffer the slot reads
/// nothing (its memory may be another buffer's by then) and its table entry
/// is gone, and a view of a new buffer written there is read in turn.
#[test]
fn views_of_released_buffers_in_slots_picked_at_run_time_read_nothing() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !translated_shaders_run() {
        return; // no host GPU, or no shader model 6.6
    }
    let memory = title_memory(2);
    let (slots, written) = (memory, memory + WINDOW);
    let (h, heap) = title_heap(16);
    title_makes(heap, TABLE, &placed(false, slots, RAW_VIEW, (0, 16)));
    title_makes(heap, TABLE + 1, &placed(true, written, structured(4), (5, 8)));
    let host_heap = heap.host().expect("a host heap");
    let pipeline = pipeline(
        LOADED_SLOT_READ,
        [ResourceType::Srv, ResourceType::Uav],
        HostShader::RUN_TIME_READS,
    );
    let run = || -> [u32; 8] {
        title_writes(slots, 0, [5, 5]);
        title_writes(written, 5, [0; 8]);
        title_dispatches(h, &pipeline);
        dwords(written, 5)
    };
    // Five slots past the table: three elements of 12 bytes from element 3
    // of a buffer object whose dword i holds `first + i`.
    let handle = heap.start + u64::from(TABLE + 5) * 32;
    let title_buffer = |first: u32| -> This {
        let buffer = crate::resource::test_buffer(0x1_0000);
        let (data, _) = crate::resource::buffer_bytes(buffer).expect("a buffer");
        title_writes(data as u64, 0, (0..64).map(|i| first + i));
        // Format, the buffer dimension, the default mapping, then the
        // first element (64 bits), the count, the stride and the flags.
        let desc = [0, 1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 0, 3, 0, 3, 12, 0, 0];
        super::create_srv(std::ptr::null(), buffer, &desc, handle);
        buffer
    };
    let release = |buffer: This| {
        crate::object::slot::<extern "system" fn(This) -> u32>(buffer, 0x10)(buffer)
    };

    let buffer = title_buffer(500);
    assert_eq!(run(), [[510, 513, 516, 0]; 2].concat()[..]);
    assert_eq!(host_heap.entry_words(TABLE + 5)[0], 12);
    assert_eq!(host_heap.reached_counts().1, 3);

    assert_eq!(release(buffer), 0);
    assert_eq!(run(), [0; 8]);
    assert_eq!(host_heap.entry_words(TABLE + 5), [0, 0]);
    assert_eq!(host_heap.reached_counts().1, 2);

    let buffer = title_buffer(600);
    assert_eq!(run(), [[610, 613, 616, 0]; 2].concat()[..]);
    assert_eq!(host_heap.entry_words(TABLE + 5)[0], 12);
    assert_eq!(release(buffer), 0);
}

/// [`LOADED_SLOT_READ`] with the two views' roles swapped: each of two
/// thread groups loads a slot from the buffer in the table's first slot,
/// and its four lanes read `[lane, 4]` of the view in the table's second
/// slot and store it at `[lane, 16 * group]` of the view in the slot the
/// group loaded.
const LOADED_SLOT_WRITE: &[u32] = &[
    0xBE94_0304, // s_mov_b32 s20, s4
    0xBE95_0380, // s_mov_b32 s21, 0
    0xF408_040A,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
    0x8F07_8205, // s_lshl_b32 s7, s5, 2
    0xF420_0188,
    0x0E00_0000, // s_buffer_load_dword s6, s[16:19], s7
    0x8F06_8506, // s_lshl_b32 s6, s6, 5
    0xF408_020A,
    0x0C00_0000, // s_load_dwordx4 s[8:11], s[20:21], s6
    0xF408_030A,
    0xFA00_0020, // s_load_dwordx4 s[12:15], s[20:21], 0x20
    0x7E02_0284, // v_mov_b32 v1, 4
    0xE030_3000,
    0x8003_0200, // buffer_load_dword v2, v[0:1], s[12:15], 0 idxen offen
    0x8F07_8405, // s_lshl_b32 s7, s5, 4
    0x7E02_0207, // v_mov_b32 v1, s7
    0xE070_3000,
    0x8002_0200, // buffer_store_dword v2, v[0:1], s[8:11], 0 idxen offen
    0xBF81_0000, // s_endpgm
];

/// What a program writes through an unordered access view in a slot it
/// picks when it runs comes back to title memory, as through a view a
/// table names, and lands in the buffer the title has in the slot at the
/// time. Without `HostHeap::reach` the writes are dropped (a null view).
#[test]
fn views_in_slots_picked_at_run_time_are_written_back() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !translated_shaders_run() {
        return; // no host GPU, or no shader model 6.6
    }
    // Three windows: what the program reads, the buffer of slots, and the
    // two buffers it writes through slots it picks (a page each).
    let memory = title_memory(3);
    let (a, slots) = (memory + 0xF000, memory + WINDOW);
    let (c, d) = (memory + 2 * WINDOW, memory + 2 * WINDOW + PAGE);
    title_writes(a, 0, (0..WIDTH / 4).map(|i| 100 + i));
    title_writes(c, 0, [7; 64]);
    title_writes(d, 0, [7; 64]);

    // The table: a raw view of the slots, then three elements of 12 bytes
    // of A from element 3. Six and eight slots past it, for unordered
    // access: elements of 4 bytes of C and of D, from element 5 each.
    let (h, heap) = title_heap(16);
    title_makes(heap, TABLE, &placed(false, slots, RAW_VIEW, (0, 16)));
    title_makes(heap, TABLE + 1, &placed(false, a, structured(12), (3, 3)));
    title_makes(heap, TABLE + 6, &placed(true, c, structured(4), (5, 8)));
    title_makes(heap, TABLE + 8, &placed(true, d, structured(4), (5, 8)));
    let pipeline = pipeline(
        LOADED_SLOT_WRITE,
        [ResourceType::Srv, ResourceType::Srv],
        HostShader::RUN_TIME_READS | HostShader::RUN_TIME_WRITES,
    );

    // Each group's lanes read A's dwords 10, 13, 16 and (past the view's
    // three elements) nothing. The first group stores them from the start
    // of the view it picked (C's dwords 5 to 8), the second 16 bytes into
    // its view (D's dwords 9 to 12). The title's bytes around them stay.
    title_writes(slots, 0, [6, 8]);
    title_dispatches(h, &pipeline);
    assert_eq!(dwords(c, 4), [7, 110, 113, 116, 0, 7, 7, 7, 7, 7]);
    assert_eq!(dwords(d, 4), [7, 7, 7, 7, 7, 110, 113, 116, 0, 7]);

    // The title rewrites the first picked slot to another part of D
    // (elements of 4 bytes from element 32): the next dispatch writes
    // there, and C keeps what the title's CPU puts in it meanwhile.
    title_makes(heap, TABLE + 6, &placed(true, d, structured(4), (32, 8)));
    title_writes(c, 0, [9; 64]);
    title_writes(d, 0, [9; 64]);
    title_dispatches(h, &pipeline);
    assert_eq!(dwords(c, 4), [9; 10]);
    assert_eq!(dwords(d, 4), [9, 9, 9, 9, 9, 110, 113, 116, 0, 9]);
    assert_eq!(dwords(d, 31), [9, 110, 113, 116, 0, 9]);
}
