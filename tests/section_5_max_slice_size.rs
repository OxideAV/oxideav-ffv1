//! End-to-end tests for the RFC 9043 §5 "Restrictions" max-slice-size
//! gate on the frame-level decode drivers.
//!
//! Round 249 wires
//! [`oxideav_ffv1::validate_slice_max_size_restriction`] into the
//! YCbCr / plane-major [`oxideav_ffv1::decode_frame_with_options`] and
//! the RGB / line-major
//! [`oxideav_ffv1::decode_frame_rgb_with_options`] drivers. The §5
//! restriction is:
//!
//! ```text
//! starting with version 3 and if
//! frame_pixel_width * frame_pixel_height is more than 101376,
//! slice_width * slice_height MUST be less or equal to
//! num_h_slices * num_v_slices / 4.
//! ```
//!
//! (RFC 9043 §5, August 2021.) These tests cover both drivers
//! across:
//!
//! * the below-threshold regime (any raster footprint admissible),
//! * the at-threshold regime (the inequality is strict, so the
//!   threshold itself is admissible), and
//! * the above-threshold regime, in which a Slice that covers more
//!   than `num_h_slices * num_v_slices / 4` cells aborts the frame
//!   decode with [`oxideav_ffv1::Error::SliceMaxSizeExceeded`].
//!
//! Because triggering the above-threshold branch requires a Frame of
//! more than 101376 pixels, the violating-cell tests use a
//! 352 × 290 Frame (102080 pixels, just above the threshold) and a
//! 2×2 raster — the smallest violating combination, where the §5 cap
//! collapses to one cell per Slice and a full-raster (2×2) Slice
//! exceeds the cap by 3.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, decode_frame_rgb_with_options, decode_frame_with_options,
    encode_frame, encode_frame_rgb, ColorspaceType, DecodeOptions, DecodedFrame, DecodedFramePlane,
    Error, Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions,
    PictureStructure, QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS, SECTION_5_MAX_SLICE_AREA_THRESHOLD,
};

// -- shared fixture helpers -------------------------------------------

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
        ec: Some(1),
        intra: Some(false),
        initial_state_delta: None,
    }
}

fn rgb_v3_cr(coder_type: u32, num_h: u32, num_v: u32, bits: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::Rgb,
        bits_per_raw_sample: bits,
        chroma_planes: true,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra: Some(false),
        initial_state_delta: None,
    }
}

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
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

fn make_rgb_decoded_frame(
    r: Vec<i32>,
    g: Vec<i32>,
    b: Vec<i32>,
    w: u32,
    h: u32,
    bits: u32,
) -> DecodedFrame {
    DecodedFrame {
        planes: vec![
            DecodedFramePlane {
                plane_index: 0,
                width: w,
                height: h,
                samples: r,
            },
            DecodedFramePlane {
                plane_index: 1,
                width: w,
                height: h,
                samples: g,
            },
            DecodedFramePlane {
                plane_index: 2,
                width: w,
                height: h,
                samples: b,
            },
        ],
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

// -- below-threshold: §5 is silent on every footprint ----------------

#[test]
fn ycbcr_small_frame_below_threshold_admits_full_raster_slice() {
    // 8 × 4 = 32 pixels, well below the 101376 CIF threshold. On a
    // 1×1 raster a single Slice covers everything (slice_w*slice_h=1,
    // num_h*num_v/4=0); the §5 cap is silent and the encode → decode
    // round-trip works.
    let cr = grayscale_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 11 + 3) as i32 & 0xFF)
        .collect();
    let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes =
        encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame(&bytes, &cr, &qts, dims, true).expect("decode");
    assert_eq!(decoded.planes[0].samples, samples);
}

// -- above-threshold: violating raster footprint aborts ---------------

/// Build a deterministic grayscale `DecodedFrame` of the given
/// dimensions filled with a simple pseudo-random pattern. Kept small
/// (per-Sample work is `O(1)`, total `O(w*h)`) so the test fits in the
/// crate's existing single-thread runtime budget.
fn synth_gray(w: u32, h: u32) -> Vec<i32> {
    (0..(w * h) as usize)
        .map(|i| (i.wrapping_mul(31) + 17) as i32 & 0xFF)
        .collect()
}

fn synth_rgb(w: u32, h: u32, seed: u32) -> Vec<i32> {
    (0..(w * h) as usize)
        .map(|i| (i.wrapping_mul(seed as usize + 19) + 5) as i32 & 0xFF)
        .collect()
}

/// The smallest Frame dimension pair we test above-threshold with:
/// 352 × 290 = 102080 pixels, just past the §5 trigger (101376).
/// Keeping the Frame this small holds encode/decode time well under
/// a second.
const ABOVE_THRESHOLD_W: u32 = 352;
const ABOVE_THRESHOLD_H: u32 = 290;

#[test]
fn frame_area_constants_line_up_with_spec() {
    // Belt-and-braces: the crate's exported threshold matches the
    // RFC text and the test fixture's chosen above-threshold size is
    // strictly above it.
    assert_eq!(SECTION_5_MAX_SLICE_AREA_THRESHOLD, 101_376);
    let area = u64::from(ABOVE_THRESHOLD_W) * u64::from(ABOVE_THRESHOLD_H);
    assert!(
        area > SECTION_5_MAX_SLICE_AREA_THRESHOLD,
        "above-threshold fixture must exceed the §5 trigger: {area} <= {SECTION_5_MAX_SLICE_AREA_THRESHOLD}",
    );
}

#[test]
fn ycbcr_above_threshold_violating_raster_aborts_decode() {
    // 352 × 290 > 101376 — the §5 trigger applies. With a 2×2 raster
    // the cap is 4/4 = 1 cell per Slice. A single Slice that covers
    // the full 2×2 raster (slice_w*slice_h = 4) violates the cap by
    // 3. The frame decode driver must surface SliceMaxSizeExceeded
    // before any per-Plane reconstruction runs on the offending
    // Slice.
    //
    // We still need to *produce* a violating frame to feed the
    // decoder. The encoder does not enforce §5 (round 249 wires the
    // gate on the decoder only), so we can build a bitstream whose
    // single Slice spans the whole 2×2 raster.
    let cr = grayscale_v3_cr(1, 2, 2, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (ABOVE_THRESHOLD_W, ABOVE_THRESHOLD_H);
    let header = make_header(0, 0, 2, 2, 2, 0);
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes =
        encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    match decode_frame(&bytes, &cr, &qts, dims, true) {
        Err(Error::SliceMaxSizeExceeded {
            slice_width,
            slice_height,
            num_h_slices,
            num_v_slices,
            frame_pixel_width,
            frame_pixel_height,
        }) => {
            assert_eq!(slice_width, 2);
            assert_eq!(slice_height, 2);
            assert_eq!(num_h_slices, 2);
            assert_eq!(num_v_slices, 2);
            assert_eq!(frame_pixel_width, fw);
            assert_eq!(frame_pixel_height, fh);
        }
        other => panic!("expected SliceMaxSizeExceeded, got {other:?}"),
    }

    // The options-aware entry point behaves the same — the §5 gate
    // is structural and is not governed by `slice_crc_policy` /
    // `slice_error_status_policy`. Both `strict()` and `lenient()`
    // surface the §5 violation.
    assert!(matches!(
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict()),
        Err(Error::SliceMaxSizeExceeded { .. })
    ));
    assert!(matches!(
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient()),
        Err(Error::SliceMaxSizeExceeded { .. })
    ));
}

#[test]
fn ycbcr_above_threshold_admissible_raster_round_trips() {
    // 352 × 290 > 101376 — the trigger applies. With a 4×4 raster
    // the cap is 16/4 = 4 cells. A 2×2 Slice (4 cells) is at the cap
    // and admissible; the encoder + decoder must round-trip the
    // frame.
    let cr = grayscale_v3_cr(1, 4, 4, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (ABOVE_THRESHOLD_W, ABOVE_THRESHOLD_H);
    // Four 2×2 Slices tile the 4×4 raster, each slice_w*slice_h=4 ==
    // cap=4. All four pass the §5 gate. Per §5 the raster is also
    // covered exactly once (no gaps / no overlaps).
    let headers = [(0u32, 0u32), (2, 0), (0, 2), (2, 2)]
        .iter()
        .map(|&(sx, sy)| make_header(sx, sy, 2, 2, 2, 0))
        .collect::<Vec<_>>();
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame(&bytes, &cr, &qts, dims, true).expect("§5-admissible decode");
    assert_eq!(decoded.planes[0].samples, samples);
}

// -- RGB / line-major mirrors ----------------------------------------

#[test]
fn rgb_above_threshold_violating_raster_aborts_decode() {
    // Mirror of the YCbCr test on the RGB / line-major driver. The
    // gate is wired into the RGB driver too — §5 is colorspace-
    // independent.
    let cr = rgb_v3_cr(1, 2, 2, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (ABOVE_THRESHOLD_W, ABOVE_THRESHOLD_H);
    let header = make_header(0, 0, 2, 2, 2, 0);
    let r = synth_rgb(fw, fh, 13);
    let g = synth_rgb(fw, fh, 17);
    let b = synth_rgb(fw, fh, 19);
    let frame = make_rgb_decoded_frame(r, g, b, fw, fh, 8);
    let bytes = encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true)
        .expect("RGB encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    match decode_frame_rgb(&bytes, &cr, &qts, dims, true) {
        Err(Error::SliceMaxSizeExceeded {
            slice_width,
            slice_height,
            num_h_slices,
            num_v_slices,
            frame_pixel_width,
            frame_pixel_height,
        }) => {
            assert_eq!(slice_width, 2);
            assert_eq!(slice_height, 2);
            assert_eq!(num_h_slices, 2);
            assert_eq!(num_v_slices, 2);
            assert_eq!(frame_pixel_width, fw);
            assert_eq!(frame_pixel_height, fh);
        }
        other => panic!("expected SliceMaxSizeExceeded on RGB driver, got {other:?}"),
    }

    // And under `lenient()` (the §5 gate is independent of the
    // §4.9.2 / §4.9.3 fixity policies).
    assert!(matches!(
        decode_frame_rgb_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient()),
        Err(Error::SliceMaxSizeExceeded { .. })
    ));
}

#[test]
fn rgb_above_threshold_admissible_raster_round_trips() {
    // Mirror of the §5-admissible YCbCr round-trip on the RGB
    // driver: 4×4 raster, four 2×2 Slices, each at the cap.
    let cr = rgb_v3_cr(1, 4, 4, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (ABOVE_THRESHOLD_W, ABOVE_THRESHOLD_H);
    let headers = [(0u32, 0u32), (2, 0), (0, 2), (2, 2)]
        .iter()
        .map(|&(sx, sy)| make_header(sx, sy, 2, 2, 2, 0))
        .collect::<Vec<_>>();
    let r = synth_rgb(fw, fh, 13);
    let g = synth_rgb(fw, fh, 17);
    let b = synth_rgb(fw, fh, 19);
    let frame = make_rgb_decoded_frame(r.clone(), g.clone(), b.clone(), fw, fh, 8);
    let bytes = encode_frame_rgb(&frame, &cr, &qts, &headers, true).expect("RGB encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("§5-admissible decode");
    assert_eq!(decoded.planes[0].samples, r);
    assert_eq!(decoded.planes[1].samples, g);
    assert_eq!(decoded.planes[2].samples, b);
}
