//! Round-trip tests for the round-164 range-coded SliceContent frame
//! encoder (`encode_frame_range_coder`).
//!
//! Where `range_reconstruct_plane.rs` validates the per-plane range
//! decoder against synthetic byte streams, this suite drives the
//! **frame-level** range-coded encode → range-coded decode pipeline
//! end-to-end through the public API:
//!
//! ```text
//!   Vec<i32> samples
//!     ── encode_frame_range_coder ──▶ Vec<u8> frame_bytes
//!     ── decode_frame ──────────────▶ DecodedFrame
//!     == samples (bit-exact)
//! ```
//!
//! Every fixture shipped under `docs/video/ffv1/fixtures/` uses
//! `coder_type == 1`, so this is the encode path any future
//! fixture-driven encode test (round 165+) will reach for.

use oxideav_ffv1::{
    decode_frame, encode_frame_range_coder, ColorspaceType, DecodedFrame, DecodedFramePlane,
    Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

/// Minimal grayscale (chroma_planes=false), single-plane v3 config
/// pinned to `coder_type == 1` (range-coded SliceContent).
fn grayscale_v3_range_cr(num_h: u32, num_v: u32, bits: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1, // range-coded SliceContent
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
/// `c`. With `c != 0` the §3.5 sign-flip path stays inactive so the
/// per-context state window's evolution is easy to reason about.
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
    quant_index: u32,
) -> Ffv1SliceHeader {
    let mut idx = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
    for slot in idx.iter_mut().take(quant_index_count) {
        *slot = quant_index;
    }
    Ffv1SliceHeader {
        slice_x,
        slice_y,
        slice_width,
        slice_height,
        quant_table_set_index_count: quant_index_count,
        quant_table_set_index: idx,
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

fn make_gray_decoded_frame(samples: Vec<i32>, w: u32, h: u32, bits: u32) -> DecodedFrame {
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
    }
}

fn assert_round_trip(
    cr: &Ffv1ConfigurationRecord,
    qts: &[QuantizationTableSet],
    headers: &[Ffv1SliceHeader],
    frame: &DecodedFrame,
    ec: bool,
) {
    let bytes = encode_frame_range_coder(frame, cr, qts, headers, ec)
        .expect("range-coded encode must succeed for valid inputs");
    let decoded = decode_frame(
        &bytes,
        cr,
        qts,
        FramePixelDimensions::new(frame.width, frame.height).unwrap(),
        ec,
    )
    .expect("range-coded encoded frame must round-trip through decode_frame");
    assert_eq!(decoded.planes.len(), frame.planes.len());
    for (got, want) in decoded.planes.iter().zip(frame.planes.iter()) {
        assert_eq!(got.samples, want.samples, "Plane samples diverged");
        assert_eq!(got.width, want.width);
        assert_eq!(got.height, want.height);
    }
}

// ----- single-slice grayscale, ec=1 -----------------------------------

#[test]
fn range_encode_round_trips_single_slice_gray_8bit() {
    let cr = grayscale_v3_range_cr(1, 1, 8);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let pixels: Vec<i32> = (0..32).map(|i| (i * 7 + 3) & 0xFF).collect();
    let frame = make_gray_decoded_frame(pixels, 8, 4, 8);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn range_encode_round_trips_single_slice_gray_10bit() {
    let cr = grayscale_v3_range_cr(1, 1, 10);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let pixels: Vec<i32> = (0..20).map(|i| (i * 47) % 1024).collect();
    let frame = make_gray_decoded_frame(pixels, 5, 4, 10);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// ----- slice grid, ec=1 -----------------------------------------------

#[test]
fn range_encode_round_trips_2x2_slice_grid() {
    let cr = grayscale_v3_range_cr(2, 2, 8);
    let qts = vec![constant_context_qts(11)];
    let (fw, fh) = (6u32, 4u32);
    let pixels: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| ((i * 19) ^ 0x5A) & 0xFF)
        .map(|v| v as i32)
        .collect();
    let frame = make_gray_decoded_frame(pixels, fw, fh, 8);
    let headers = vec![
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    assert_round_trip(&cr, &qts, &headers, &frame, true);
}

// ----- ec=0 (3-byte footer) -------------------------------------------

#[test]
fn range_encode_round_trips_ec0_footer() {
    let cr = grayscale_v3_range_cr(1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let pixels: Vec<i32> = vec![0, 1, 254, 255, 128, 64, 32, 16];
    let frame = make_gray_decoded_frame(pixels, 4, 2, 8);
    assert_round_trip(&cr, &qts, &[header], &frame, false);
}

// ----- constant-Plane edge case ---------------------------------------

#[test]
fn range_encode_round_trips_flat_plane() {
    // A flat Plane (every Sample equal) is the lowest-entropy case:
    // every `sample_difference` is zero, and the per-context state
    // window collapses to whatever the arithmetic coder picks for
    // back-to-back zero bits.
    let cr = grayscale_v3_range_cr(1, 1, 8);
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let pixels: Vec<i32> = vec![128; 16];
    let frame = make_gray_decoded_frame(pixels, 4, 4, 8);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// ----- determinism over multiple encode calls -------------------------

#[test]
fn range_encode_is_deterministic_across_calls() {
    let cr = grayscale_v3_range_cr(1, 1, 8);
    let qts = vec![constant_context_qts(4)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let pixels: Vec<i32> = (0..24).map(|i| (i * 13 + 9) & 0xFF).collect();
    let frame = make_gray_decoded_frame(pixels, 6, 4, 8);

    let a =
        encode_frame_range_coder(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    let b =
        encode_frame_range_coder(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(a, b);
}
