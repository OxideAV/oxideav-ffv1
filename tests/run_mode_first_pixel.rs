//! RFC 9043 §3.8.2.2 / §3.8.2.4.1 run-mode "first Sample" encodability
//! gate on the Golomb-Rice (`coder_type == 0`) encode path.
//!
//! The §3.8.2.2 run state machine begins every run with a `0` Sample
//! Difference: on entering run mode the decoder's first Sample is always
//! `0` (Phase 3 emits a long-run "1" — Sample Difference 0 — or a short
//! run that returns 0 for the current Sample and level-codes the break
//! on the *next* Sample, §3.8.2.4.1). A non-zero `sample_difference` at
//! the **very first** Sample of a run region (absolute context 0 with
//! `l == t == tl`, immediately after a run-state reset) therefore has no
//! Golomb-Rice encoding — there is no preceding zero-run Sample to carry
//! the short-run prefix.
//!
//! A stream a conforming FFV1 decoder produced never exhibits this (every
//! run begins with a 0 Sample Difference), but arbitrary caller pixel
//! data routed into run mode at the first run Sample with a non-zero
//! residual can. Before this round the encoder hit a `debug_assert!` (a
//! no-op in release builds, where it silently emitted a corrupt stream);
//! now [`encode_frame`] surfaces the typed
//! [`oxideav_ffv1::Error::RunModeFirstPixelNonZero`].
//!
//! The range coder (`coder_type ∈ {1, 2}`) has no run mode (§3.8.2.2 is
//! Golomb-Rice-only), so the same pixels encode without restriction —
//! the recommended escape these tests also exercise.

use oxideav_ffv1::{
    decode_frame, encode_frame, ColorspaceType, DecodedFrame, DecodedFramePlane, Error,
    Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

/// Minimal single-plane grayscale v3 config on the chosen entropy coder.
fn grayscale_v3_cr(coder_type: u32, num_h: u32, num_v: u32, bits: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: bits,
        chroma_planes: false,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(1),
        ec: Some(0),
        intra: Some(false),
        initial_state_delta: None,
    }
}

/// Single-context QTS — every neighbour configuration maps to context
/// `c`. `c == 0` makes **every** Sample a run-region Sample (absolute
/// context 0), so the §3.8.2.2 run state machine drives the whole Plane.
fn constant_context_qts(c: u32) -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    tables[0] = [c as i32; 256];
    QuantizationTableSet {
        tables,
        context_count: c + 1,
    }
}

fn make_header(
    slice_x: u32,
    slice_y: u32,
    slice_width: u32,
    slice_height: u32,
    quant_index_count: usize,
) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x,
        slice_y,
        slice_width,
        slice_height,
        quant_table_set_index_count: quant_index_count,
        quant_table_set_index: [0u32; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

fn make_gray_frame(samples: Vec<i32>, w: u32, h: u32, bits: u32) -> DecodedFrame {
    DecodedFrame {
        planes: vec![DecodedFramePlane {
            plane_index: 0,
            width: w,
            height: h,
            samples,
        }],
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// A frame whose very first Sample is non-zero, decoded under a
/// single-context-0 QTS (so pixel (0,0) is a run-region Sample), is
/// unencodable on the Golomb-Rice path. The encoder surfaces
/// `Error::RunModeFirstPixelNonZero { x: 0 }` rather than a corrupt
/// stream.
#[test]
fn golomb_non_zero_first_run_sample_surfaces_typed_error() {
    let cr = grayscale_v3_cr(0, 1, 1, 8);
    let qts = vec![constant_context_qts(0)];
    let headers = vec![make_header(0, 0, 1, 1, cr_slot_count(&cr))];
    // 4x2: first Sample non-zero → run mode at the first run Sample.
    let frame = make_gray_frame(vec![7, 0, 0, 0, 0, 0, 0, 0], 4, 2, 8);

    let err = encode_frame(&frame, &cr, &qts, &headers, false)
        .expect_err("a non-zero first run-region Sample must not encode on the Golomb path");
    assert_eq!(
        err,
        Error::RunModeFirstPixelNonZero { x: 0 },
        "expected the §3.8.2.2 run-mode first-Sample gate, got {err:?}"
    );
}

/// Same first-Sample non-zero, but on a 2x2 slice grid: the unencodable
/// Sample surfaces from whichever Slice covers the top-left raster cell,
/// still as `x == 0` (the first Sample of that Slice's first Line).
#[test]
fn golomb_first_run_sample_gate_fires_on_slice_grid() {
    let cr = grayscale_v3_cr(0, 2, 2, 8);
    let qts = vec![constant_context_qts(0)];
    let slot = cr_slot_count(&cr);
    let headers = vec![
        make_header(0, 0, 1, 1, slot),
        make_header(1, 0, 1, 1, slot),
        make_header(0, 1, 1, 1, slot),
        make_header(1, 1, 1, 1, slot),
    ];
    // 4x4 frame; the (0,0) Sample of the top-left Slice is non-zero.
    let mut samples = vec![0i32; 16];
    samples[0] = 5;
    let frame = make_gray_frame(samples, 4, 4, 8);

    let err = encode_frame(&frame, &cr, &qts, &headers, false)
        .expect_err("first-Slice top-left non-zero run Sample must not encode");
    assert_eq!(err, Error::RunModeFirstPixelNonZero { x: 0 });
}

/// The companion to the gate: a frame whose first run-region Sample is
/// `0` (a zero-run begins normally) followed by a non-zero — Case B in
/// the encoder — round-trips bit-exactly. This pins that the gate is
/// surgical (it rejects only the unrepresentable first-Sample case, not
/// every run-mode non-zero).
#[test]
fn golomb_zero_then_non_zero_run_round_trips() {
    let cr = grayscale_v3_cr(0, 1, 1, 8);
    let qts = vec![constant_context_qts(0)];
    let headers = vec![make_header(0, 0, 1, 1, cr_slot_count(&cr))];
    // First Sample 0 (zero-run starts), then a non-zero level break, then
    // more zeros. The median predictor over the all-zero border keeps
    // pred == 0 so the Sample value equals its Sample Difference here.
    let frame = make_gray_frame(vec![0, 3, 0, 0, 0, 0, 0, 0], 4, 2, 8);

    let bytes = encode_frame(&frame, &cr, &qts, &headers, false)
        .expect("a run that begins with a 0 Sample is encodable (Case B)");
    let decoded = decode_frame(
        &bytes,
        &cr,
        &qts,
        FramePixelDimensions::new(frame.width, frame.height).unwrap(),
        false,
    )
    .expect("the Case B stream must round-trip");
    assert_eq!(
        decoded.planes[0].samples, frame.planes[0].samples,
        "Case B run-mode frame must reconstruct bit-exactly"
    );
}

/// The recommended escape: the exact frame the Golomb path rejects
/// encodes and round-trips bit-exactly on the range coder
/// (`coder_type == 1`), which has no §3.8.2.2 run mode.
#[test]
fn range_coder_encodes_the_golomb_unencodable_frame() {
    let samples = vec![7, 0, 0, 0, 0, 0, 0, 0];

    // Golomb rejects it.
    let cr0 = grayscale_v3_cr(0, 1, 1, 8);
    let qts = vec![constant_context_qts(0)];
    let headers0 = vec![make_header(0, 0, 1, 1, cr_slot_count(&cr0))];
    let frame0 = make_gray_frame(samples.clone(), 4, 2, 8);
    assert_eq!(
        encode_frame(&frame0, &cr0, &qts, &headers0, false),
        Err(Error::RunModeFirstPixelNonZero { x: 0 })
    );

    // Range coder carries the same pixels with no restriction.
    let cr1 = grayscale_v3_cr(1, 1, 1, 8);
    let headers1 = vec![make_header(0, 0, 1, 1, cr_slot_count(&cr1))];
    let frame1 = make_gray_frame(samples, 4, 2, 8);
    let bytes = encode_frame(&frame1, &cr1, &qts, &headers1, false)
        .expect("the range coder has no run mode and must encode the same frame");
    let decoded = decode_frame(
        &bytes,
        &cr1,
        &qts,
        FramePixelDimensions::new(frame1.width, frame1.height).unwrap(),
        false,
    )
    .expect("range-coded frame must round-trip");
    assert_eq!(decoded.planes[0].samples, frame1.planes[0].samples);
}

/// `quant_table_set_index_count` for a single-plane grayscale v3 config:
/// luma slot + a chroma slot (`version <= 3` always reserves one). The
/// header carries that many `quant_table_set_index` entries.
fn cr_slot_count(cr: &Ffv1ConfigurationRecord) -> usize {
    1 + if cr.chroma_planes
        || matches!(
            cr.version,
            Ffv1Version::V0 | Ffv1Version::V1 | Ffv1Version::V3
        ) {
        1
    } else {
        0
    } + if cr.extra_plane { 1 } else { 0 }
}
