//! End-to-end integration tests for the round-4 Golomb-Rice
//! `sample_difference` decode path (RFC 9043 §3.8.2 + §4.8).
//!
//! These tests drive [`oxideav_ffv1::decode_line`] over synthetic
//! bit-stream inputs with hand-crafted quantization-table sets. The
//! goal is to exercise the full chain — bit reader → Golomb-Rice
//! prefix/suffix → adaptive VLC state → §3.5 context → §3.5 sign-flip
//! — without depending on the §4.1 quant-table cascade parser
//! (still queued for round 5).

use oxideav_ffv1::{
    decode_line, BitReader, LineDecoderState, LineNeighborBuffers, QuantTableSet, BORDER_WIDTH,
    NUM_QUANT_SUBTABLES, VLC_STATE_INITIAL,
};

/// Build a `(prev_prev, prev, current)` triple sized for a row of
/// `width` real samples plus the §3.1 border padding.
fn make_buffers(width: u32) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let total = BORDER_WIDTH + width as usize + BORDER_WIDTH;
    (vec![0i32; total], vec![0i32; total], vec![0i32; total])
}

/// Quant-table set with Q0[0]=`c` and all other entries zero. Forces
/// every neighbour stencil into context `c` (assuming the other 4
/// neighbour differences round to 0 through their zero subtables).
fn single_context_qtable(c: i32) -> QuantTableSet {
    let mut q = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    q[0][0] = c;
    q
}

#[test]
fn full_zero_row_through_scalar_path() {
    // Force context==7 for every pixel (nonzero → scalar path).
    //
    // The adaptive `k` advances as `count` rises against the static
    // error_sum=4 (every decoded value is zero, so error_sum doesn't
    // grow). Walking the §3.8.2.4 pseudocode:
    //
    //   sample 0: count=1, error_sum=4 → loop k:0(i=1) → 1(i=2) →
    //             2(i=4) exit. k=2. Read "1 00" (3 bits). v=0.
    //             post: count=2.
    //   sample 1: count=2, k=0(i=2)→1(i=4) exit. k=1. Read "1 0"
    //             (2 bits). v=0. post: count=3.
    //   sample 2: count=3, k=0(i=3)→1(i=6) exit. k=1. Read "1 0"
    //             (2 bits). v=0. post: count=4.
    //   sample 3: count=4, k=0(i=4) exit. k=0. Read "1" (1 bit).
    //             v=0. post: count=5.
    //
    // Total: 3+2+2+1 = 8 bits → 0b1001_0101 = 0x95.
    let qtable = single_context_qtable(7);
    let mut state = LineDecoderState::new(16);
    let (prev_prev, prev, mut current) = make_buffers(4);
    let buf = [0x95u8];
    let mut br = BitReader::new(&buf);

    let mut nb = LineNeighborBuffers {
        prev_row: &prev,
        prev_prev_row: &prev_prev,
        current_row: &mut current,
        plane_pixel_width: 4,
    };
    let row = decode_line(&mut br, &mut state, &qtable, &mut nb, 8);
    assert_eq!(row, vec![0, 0, 0, 0]);

    // VLC state for context 7 should have absorbed 4 decodes:
    // count 1→2→3→4→5.
    assert_eq!(state.vlc[7].count, 5);
    // Run-mode untouched on the scalar path.
    assert_eq!(state.run_mode, 0);
    assert_eq!(state.run_index, 0);
}

#[test]
fn negative_context_flips_decoded_sign() {
    // Set up a quant table that produces a negative context (so the
    // sign-flip path is exercised). We use Q3 (which indexes by
    // (L - l) & 0xFF) and rely on the per-pixel writeback: after the
    // first decoded sample writes into current_row[BORDER+0]=v, the
    // second pixel sees L = current_row[idx-2] = current_row[BORDER+1-2]
    // = current_row[BORDER-1] = 0 (border) — so we can drive the
    // negative context through a single pixel only, using the initial
    // border state.
    //
    // For x=0 the stencil is all zeros, so we need Q_i[0] entries to
    // make a negative context. Let Q0[0] = -5; raw context = -5;
    // abs_ctx index = 5, sign_flip = true.
    let mut q = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    q[0][0] = -5;
    let mut state = LineDecoderState::new(16);
    let (prev_prev, prev, mut current) = make_buffers(1);
    // Need a bit pattern that decodes to a nonzero value so the
    // sign-flip is visible. With initial state at k=2: bit pattern
    // "1 10" sr-golomb → unsigned 2 → even → 1, bias=0 → 1; sign-flip
    // negates → -1.
    let buf = [0b1100_0000u8];
    let mut br = BitReader::new(&buf);

    let mut nb = LineNeighborBuffers {
        prev_row: &prev,
        prev_prev_row: &prev_prev,
        current_row: &mut current,
        plane_pixel_width: 1,
    };
    let row = decode_line(&mut br, &mut state, &q, &mut nb, 8);
    assert_eq!(row, vec![-1]);
    assert_eq!(current[BORDER_WIDTH], -1);
}

#[test]
fn run_mode_engages_when_context_is_zero_and_neighbours_match() {
    // All-zero qtable → every pixel sees absolute context 0. The
    // run-mode predicate also needs l == t == tl; since the row is
    // fresh and all border samples are zero, the predicate holds at
    // every step until decoded_diff writes a nonzero value into
    // current_row.
    //
    // Drive the first sample through the "short run, get_bits(0) bits"
    // path: bit 0 → short-run branch, log2_run[0]=0 so rc=0,
    // run_mode=2, returns 0.
    // Second sample: run_count==0 && run_mode==2 → level-coded; read
    // "1" at k=2: bit "1 00" → unsigned 0 → level adjusts to +1.
    // Third sample: state was reset on level transition? Let me check.
    //
    // Actually in our impl, after the run breaks (run_mode==2 → 0 on
    // level read) the next pixel's neighbours are: l = current_row[idx-1]
    // = 1 (because the level-coded sample was 1), t/tl = 0 (still on
    // prev_row). Predicate "l == t == tl" needs 1 == 0 → false. So
    // third sample takes the scalar path.
    let qtable = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    let mut state = LineDecoderState::new(16);
    let (prev_prev, prev, mut current) = make_buffers(2);

    // Bit pattern:
    //   sample 0 run-mode: bit 0 (short run, rc=0)
    //   sample 1 level-coded: "1 00" (k=2, returns 0 → +1 via level)
    //   = 0 1 0 0 = 0b0100 → padded 0b0100_0000 = 0x40.
    let buf = [0x40u8];
    let mut br = BitReader::new(&buf);

    let mut nb = LineNeighborBuffers {
        prev_row: &prev,
        prev_prev_row: &prev_prev,
        current_row: &mut current,
        plane_pixel_width: 2,
    };
    let row = decode_line(&mut br, &mut state, &qtable, &mut nb, 8);
    assert_eq!(row, vec![0, 1]);
    // current_row should reflect the decoded values.
    assert_eq!(current[BORDER_WIDTH], 0);
    assert_eq!(current[BORDER_WIDTH + 1], 1);
}

#[test]
fn row_buffer_writes_persist_across_pixels() {
    // After decode_line, the row buffer carries every decoded
    // sample_difference in the real-sample slots. This is the
    // property a future round will rely on: decoded sample
    // differences in a row form the basis for predicting the next
    // row's context.
    let qtable = single_context_qtable(3);
    let mut state = LineDecoderState::new(8);
    let (prev_prev, prev, mut current) = make_buffers(3);
    // k progression at context 3 (fresh state, error_sum stays at 4
    // because every decoded value is 0):
    //   sample 0: k=2 → "1 00" (3 bits)
    //   sample 1: k=1 → "1 0" (2 bits)
    //   sample 2: k=1 → "1 0" (2 bits)
    // Total 7 bits = 0b1001_0100 = 0x94.
    let buf = [0x94u8];
    let mut br = BitReader::new(&buf);

    {
        let mut nb = LineNeighborBuffers {
            prev_row: &prev,
            prev_prev_row: &prev_prev,
            current_row: &mut current,
            plane_pixel_width: 3,
        };
        let _ = decode_line(&mut br, &mut state, &qtable, &mut nb, 8);
    }
    // Borders zero, real-sample slots also zero (the decoded values).
    assert_eq!(&current[..BORDER_WIDTH], &[0, 0]);
    assert_eq!(current[BORDER_WIDTH], 0);
    assert_eq!(current[BORDER_WIDTH + 1], 0);
    assert_eq!(current[BORDER_WIDTH + 2], 0);
}

#[test]
fn fresh_line_decoder_state_carries_initial_vlc_table() {
    // Sanity: confirm the round-4 API constructs LineDecoderState
    // with all slots at the §3.8.2.5 initial value.
    let s = LineDecoderState::new(5);
    for slot in &s.vlc {
        assert_eq!(*slot, VLC_STATE_INITIAL);
    }
}
