//! Application of the RFC 9043 §4.2.15 explicit initial states
//! (Figures 29/30) to the per-context range-coder windows that
//! §3.8.1.3 otherwise initialises to 128.
//!
//! The states-coded-1 fixture can only pin the PARSE of the
//! `states_coded == 1` path (its deltas all reconstruct to 128, so
//! applying them is indistinguishable from the default). These tests
//! pin the APPLICATION: non-zero reconstructed states must seed the
//! keyframe-initialised windows on both the encode and decode sides,
//! symmetrically enough that a round-trip is bit-exact and visibly
//! enough that ignoring the seeds breaks the decode.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, encode_frame, parse_quantization_table_sets,
    reconstruct_initial_states, ColorspaceType, DecodedFrame, DecodedFramePlane,
    Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

#[path = "data/states_coded_1.rs"]
#[allow(dead_code)]
mod fx;

use fx::{STATES_CODED_1_CONFIG_RECORD, STATES_CODED_1_EXPECTED};

const CONTEXT_SIZE: usize = 32;

// ---------------------------------------------------------------------
// Figure 29/30 reconstruction semantics
// ---------------------------------------------------------------------

/// Single-context QTS (synthetic): every neighbour configuration maps
/// to context `c`; `len_counts` sees no §4.1 shape so
/// `initial_state_row_count` falls back to `context_count`.
fn constant_context_qts(c: u32) -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    tables[0] = [c as i32; 256];
    QuantizationTableSet {
        tables,
        context_count: c + 1,
    }
}

fn gray_cr_with_deltas(
    deltas: Option<Vec<Option<Vec<[i32; CONTEXT_SIZE]>>>>,
) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: 8,
        chroma_planes: false,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra: Some(false),
        initial_state_delta: deltas,
    }
}

/// Figure 29 predictor chain + Figure 30 modular fold, including the
/// 8-bit wrap in both directions and the running (j-1) prediction.
#[test]
fn reconstruction_follows_figures_29_and_30() {
    let qts = vec![constant_context_qts(2)]; // context_count == 3
    let mut rows = vec![[0i32; CONTEXT_SIZE]; 3];
    rows[0][0] = 2; //   (128 + 2) & 255          = 130
    rows[1][0] = -1; //  (130 - 1) & 255          = 129   (chained, not 127)
    rows[0][1] = -200; // (128 - 200) & 255       = 184   (wrap down)
    rows[1][1] = 100; // (184 + 100) & 255        = 28    (wrap up, chained)
    rows[2][31] = 5; //  (128 + 5) & 255          = 133   (k slots independent)
    let cr = gray_cr_with_deltas(Some(vec![Some(rows)]));

    let states = reconstruct_initial_states(&cr, &qts);
    assert_eq!(states.len(), 1);
    let s = states[0].as_ref().expect("set 0 coded");
    assert_eq!(s.len(), 3 * CONTEXT_SIZE, "context_count windows");
    assert_eq!(s[0], 130, "Figure 30: (128 + 2) & 255");
    assert_eq!(s[CONTEXT_SIZE], 129, "Figure 29 chains row j-1, not 128");
    assert_eq!(s[1], 184, "Figure 30 wraps below 0");
    assert_eq!(s[CONTEXT_SIZE + 1], 28, "chained wrap above 255");
    assert_eq!(s[2 * CONTEXT_SIZE + 31], 133, "k slots are independent");
    // A column never touched by any delta stays at 128 all the way
    // down the chain...
    assert_eq!(s[5], 128);
    // ...but a zero delta does NOT reset a touched column: row 2's
    // k = 0 inherits row 1's reconstructed 129 through Figure 29
    // (`pred = initial_states[i][j-1][k]`), not the constant 128.
    assert_eq!(s[2 * CONTEXT_SIZE], 129, "zero delta propagates the chain");
}

/// `states_coded == 0` (record default) reconstructs to all-`None`.
#[test]
fn no_deltas_reconstruct_to_none() {
    let qts = vec![constant_context_qts(2), constant_context_qts(4)];
    let cr = gray_cr_with_deltas(None);
    assert_eq!(reconstruct_initial_states(&cr, &qts), vec![None, None]);
}

/// FFmpeg-interop padding rows (`j >= context_count`) are chained on
/// the wire but never materialised as live windows: the fixture's 942
/// transmitted rows reconstruct to exactly 666 windows, and its lone
/// non-zero delta ([941][31], a padding row) leaves every live state
/// at 128.
#[test]
fn fixture_padding_rows_are_not_materialised() {
    let parsed = parse_quantization_table_sets(STATES_CODED_1_CONFIG_RECORD).expect("parse");
    let states = reconstruct_initial_states(&parsed.record, &parsed.quant_table_sets);
    assert_eq!(states.len(), 2);
    let s0 = states[0].as_ref().expect("set 0 coded");
    assert_eq!(s0.len(), 666 * CONTEXT_SIZE, "live §4.1 contexts only");
    assert!(
        s0.iter().all(|&b| b == 128),
        "all-zero live deltas keep every state at the §3.8.1.3 default"
    );
    assert!(states[1].is_none(), "set 1 wrote states_coded == 0");
}

// ---------------------------------------------------------------------
// End-to-end application: YCbCr / plane-major driver
// ---------------------------------------------------------------------

fn make_header(w: u32, h: u32, quant_index_count: usize) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x: 0,
        slice_y: 0,
        slice_width: w,
        slice_height: h,
        quant_table_set_index_count: quant_index_count,
        quant_table_set_index: [0u32; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

fn gray_frame(samples: Vec<i32>, w: u32, h: u32) -> DecodedFrame {
    DecodedFrame {
        planes: vec![DecodedFramePlane {
            plane_index: 0,
            width: w,
            height: h,
            samples,
        }],
        width: w,
        height: h,
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// Deterministic non-zero delta pattern over the live rows of the
/// fixture's set 0 (`[6,6,6,1,1]`, 666 live + 276 padding rows), zeros
/// on the padding rows.
///
/// The deltas are derived from bounded TARGET states (96..=159) so the
/// Figure 29 running chain never drifts toward the degenerate state 0
/// — the deltas themselves are the state-to-state differences, which
/// keeps every reconstructed state deep inside 1..=255 while still
/// exercising non-trivial magnitudes in both directions.
fn nonzero_live_deltas() -> Vec<[i32; CONTEXT_SIZE]> {
    let target = |j: usize, k: usize| -> i32 { 96 + ((j * 7 + k * 13) % 64) as i32 };
    let mut rows = vec![[0i32; CONTEXT_SIZE]; 942];
    for (j, row) in rows.iter_mut().enumerate().take(666) {
        for (k, slot) in row.iter_mut().enumerate() {
            let pred = if j == 0 { 128 } else { target(j - 1, k) };
            let mut d = target(j, k) - pred;
            if d == 0 {
                // keep every live delta non-zero (target stays in range:
                // consecutive targets differ by 7 mod 64, so d == 0 only
                // via the j == 0 pred; +1 keeps the chain bounded).
                d = 1;
            }
            *slot = d;
        }
    }
    rows
}

/// Non-zero transmitted initial states round-trip bit-exactly through
/// the seeded encoder + seeded decoder — and the SAME coded bytes
/// decoded while ignoring the seeds (the pre-round behaviour) must NOT
/// reproduce the pixels, proving the seeding is load-bearing on both
/// sides.
#[test]
fn seeded_round_trip_is_bit_exact_and_load_bearing() {
    let parsed = parse_quantization_table_sets(STATES_CODED_1_CONFIG_RECORD).expect("parse");
    let mut cr = parsed.record.clone();
    cr.initial_state_delta = Some(vec![Some(nonzero_live_deltas()), None]);

    let pixels: Vec<i32> = STATES_CODED_1_EXPECTED.iter().map(|&b| b as i32).collect();
    let frame = gray_frame(pixels.clone(), 64, 48);
    // Reuse the fixture's own slice geometry (1x1 grid; §4.6.5 slot
    // count from the real reference stream).
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let headers = vec![make_header(1, 1, 2)];

    let bytes = encode_frame(&frame, &cr, &parsed.quant_table_sets, &headers, true)
        .expect("seeded keyframe encode");
    let decoded = decode_frame(&bytes, &cr, &parsed.quant_table_sets, dims, true)
        .expect("seeded decode (slice CRC validated)");
    assert_eq!(
        decoded.planes[0].samples, pixels,
        "seeded encode -> seeded decode is bit-exact"
    );

    // Load-bearing check: strip the seeds from the decode side only.
    let unseeded_cr = parsed.record.clone(); // fixture deltas: all-128 states
    let mut unseeded_cr = unseeded_cr;
    unseeded_cr.initial_state_delta = None;
    // (a desync surfacing as a hard error would be equally conclusive)
    if let Ok(d) = decode_frame(&bytes, &unseeded_cr, &parsed.quant_table_sets, dims, true) {
        assert_ne!(
            d.planes[0].samples, pixels,
            "ignoring the §4.2.15 seeds cannot reproduce the pixels"
        );
    }
}

/// A non-zero delta confined to the FFmpeg-interop padding rows
/// (`j >= context_count`) changes NOTHING: padding rows exist on the
/// wire but never seed a live context, so the coded bytes are
/// identical to the all-zero-delta encode.
#[test]
fn padding_row_deltas_do_not_affect_the_coded_stream() {
    let parsed = parse_quantization_table_sets(STATES_CODED_1_CONFIG_RECORD).expect("parse");
    let pixels: Vec<i32> = STATES_CODED_1_EXPECTED.iter().map(|&b| b as i32).collect();
    let frame = gray_frame(pixels, 64, 48);
    let headers = vec![make_header(1, 1, 2)];

    let mut padded_rows = vec![[0i32; CONTEXT_SIZE]; 942];
    padded_rows[700][5] = 17; // padding row: 666 <= 700 < 942
    let mut cr_padded = parsed.record.clone();
    cr_padded.initial_state_delta = Some(vec![Some(padded_rows), None]);

    let mut cr_zero = parsed.record.clone();
    cr_zero.initial_state_delta = Some(vec![Some(vec![[0i32; CONTEXT_SIZE]; 942]), None]);

    let bytes_padded = encode_frame(&frame, &cr_padded, &parsed.quant_table_sets, &headers, true)
        .expect("encode with padding-row delta");
    let bytes_zero = encode_frame(&frame, &cr_zero, &parsed.quant_table_sets, &headers, true)
        .expect("encode with all-zero deltas");
    assert_eq!(
        bytes_padded, bytes_zero,
        "padding rows never seed a live context"
    );
}

// ---------------------------------------------------------------------
// End-to-end application: RGB / line-major driver
// ---------------------------------------------------------------------

/// The RGB line-major driver seeds its per-slot windows the same way.
#[test]
fn rgb_seeded_round_trip_is_bit_exact_and_load_bearing() {
    let qts = vec![constant_context_qts(3)]; // context_count == 4
    let mut rows = vec![[0i32; CONTEXT_SIZE]; 4];
    for (j, row) in rows.iter_mut().enumerate() {
        for (k, slot) in row.iter_mut().enumerate() {
            let v = ((j * 13 + k * 5) % 7) as i32 - 3; // -3..=3
            *slot = if v == 0 { 2 } else { v };
        }
    }
    let cr = Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::Rgb,
        bits_per_raw_sample: 8,
        chroma_planes: true, // RGB always has three Planes
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra: Some(false),
        initial_state_delta: Some(vec![Some(rows)]),
    };

    let (w, h) = (16u32, 8u32);
    let n = (w * h) as usize;
    let r: Vec<i32> = (0..n).map(|i| ((i * 7) % 256) as i32).collect();
    let g: Vec<i32> = (0..n).map(|i| ((i * 3 + 40) % 256) as i32).collect();
    let b: Vec<i32> = (0..n).map(|i| ((i * 11 + 9) % 256) as i32).collect();
    let frame = DecodedFrame {
        planes: [r, g, b]
            .into_iter()
            .enumerate()
            .map(|(p, samples)| DecodedFramePlane {
                plane_index: p as u8,
                width: w,
                height: h,
                samples,
            })
            .collect(),
        width: w,
        height: h,
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    };
    let headers = vec![make_header(1, 1, 2)];
    let dims = FramePixelDimensions::new(w, h).expect("dims");

    let bytes =
        encode_frame(&frame, &cr, &qts, &headers, true).expect("seeded RGB keyframe encode");
    let decoded = decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("seeded RGB decode");
    for (p, plane) in frame.planes.iter().enumerate() {
        assert_eq!(
            decoded.planes[p].samples, plane.samples,
            "RGB plane {p} bit-exact through the seeded round-trip"
        );
    }

    let mut unseeded_cr = cr.clone();
    unseeded_cr.initial_state_delta = None;
    // (a desync surfacing as a hard error would be equally conclusive)
    if let Ok(d) = decode_frame_rgb(&bytes, &unseeded_cr, &qts, dims, true) {
        assert_ne!(
            d.planes[0].samples, frame.planes[0].samples,
            "ignoring the §4.2.15 seeds cannot reproduce the RGB pixels"
        );
    }
}

/// Degenerate reconstructed states must not hang the encoder. Figure
/// 30's `& 255` can legally reconstruct state 0 — and the §3.8.1.5
/// default transition table's `one_state[1..=8] == 0` entries feed
/// low states INTO 0 — where an unguarded coder would zero `range` on
/// a 1-branch and spin the renormalisation loop forever while growing
/// the output unboundedly (the r390 runaway-allocation incident). The
/// `rangeoff.max(1)` guard in `get_rac`/`put_rac` (a no-op for every
/// valid state) keeps both sides finite AND exact inverses, so the
/// faithfully-seeded degenerate pair still round-trips bit-exactly.
///
/// Every sample of the `constant_context_qts(3)` model maps to
/// context 3, so the degenerate states are planted on context 3's
/// window — slot 0 (the is-zero bit, hammered once per sample) is
/// seeded to state 0 itself, and slot 1 (the first exponent bit) to
/// feeder-band state 1 — guaranteeing the encode path actually codes
/// against them.
#[test]
fn degenerate_zero_state_terminates_and_round_trips() {
    let qts = vec![constant_context_qts(3)]; // context_count == 4
    let mut rows = vec![[0i32; CONTEXT_SIZE]; 4];
    rows[3][0] = -128; // context 3, slot 0: (128 - 128) & 255 == 0
    rows[3][1] = -127; // context 3, slot 1: state 1 (one_state[1] == 0)
    let cr = gray_cr_with_deltas(Some(vec![Some(rows)]));

    let states = reconstruct_initial_states(&cr, &qts);
    let s = states[0].as_ref().expect("set coded");
    assert_eq!(
        s[3 * CONTEXT_SIZE],
        0,
        "Figure 30 faithfully reconstructs the 0 state"
    );
    assert_eq!(s[3 * CONTEXT_SIZE + 1], 1, "feeder-band state 1");

    let pixels: Vec<i32> = (0..64 * 48).map(|i: i32| i % 251).collect();
    let frame = gray_frame(pixels.clone(), 64, 48);
    // §4.6.5: quant_table_set_index_count == 2 for v3 (luma + chroma
    // slot even on gray); both slots select set 0.
    let headers = vec![make_header(1, 1, 2)];
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true)
        .expect("encode terminates despite the degenerate seeds");
    let decoded = decode_frame(&bytes, &cr, &qts, dims, true).expect("decode");
    assert_eq!(
        decoded.planes[0].samples, pixels,
        "guarded coder pair stays bit-exact through degenerate states"
    );
}
