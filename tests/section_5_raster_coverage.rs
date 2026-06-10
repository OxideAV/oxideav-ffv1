//! End-to-end tests for the RFC 9043 §5 "Restrictions" Slice raster-
//! coverage gate on the frame-level decode drivers.
//!
//! Round 260 wires
//! [`oxideav_ffv1::validate_slice_raster_coverage`] into the YCbCr /
//! plane-major [`oxideav_ffv1::decode_frame_with_options`] and the
//! RGB / line-major [`oxideav_ffv1::decode_frame_rgb_with_options`]
//! drivers as a two-pass collect-then-validate preamble. Round 257
//! landed the validator itself; this round folds it into both frame
//! drivers so the §5 partition rule
//!
//! ```text
//! For each Frame, each position in the Slice raster MUST be filled
//! by one and only one Slice of the Frame (no missing Slice position
//! and no Slice overlapping).
//! ```
//!
//! (RFC 9043 §5 second paragraph, August 2021) aborts the Frame
//! decode before any per-Slice pixel reconstruction starts. The §5
//! gate is structural and is not governed by `slice_crc_policy` /
//! `slice_error_status_policy` — both [`DecodeOptions::strict`] and
//! [`DecodeOptions::lenient`] surface the §5 violation, mirroring
//! the round-249 max-slice-size gate.
//!
//! Coverage matrix:
//!
//! * positive — single-Slice 1×1 grid and 4-Slice 2×2 grid encode →
//!   decode round-trips through both drivers (YCbCr and RGB),
//! * overlap — two Slices both addressing cell (0, 0) on a 2×1 grid
//!   surface [`Error::SliceRasterOverlap`] with the lowest forward-
//!   index pair logged,
//! * gap — a single Slice covering only the left half of a 2×1 grid
//!   surfaces [`Error::SliceRasterUncovered`] at the first uncovered
//!   row-major scan-order cell (`x = 1, y = 0`),
//! * lenient passthrough — both error variants still abort under
//!   [`DecodeOptions::lenient`] (the §5 gate is structural).

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, decode_frame_rgb_with_options, decode_frame_with_options,
    encode_frame, encode_frame_rgb, ColorspaceType, DecodeOptions, DecodedFrame, DecodedFramePlane,
    Error, Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions,
    PictureStructure, QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS,
};

// -- shared fixture helpers (mirror section_5_max_slice_size.rs) -------

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
    }
}

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

// -- positive: §5-conforming partitions round-trip -------------------

#[test]
fn ycbcr_single_slice_one_by_one_grid_round_trips() {
    // The most degenerate §5-conforming partition: a 1×1 grid with one
    // Slice covering the whole grid. The validator must accept it; the
    // round-trip must reconstruct the original Samples.
    let cr = grayscale_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("enc");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame(&bytes, &cr, &qts, dims, true).expect("decode");
    assert_eq!(decoded.planes[0].samples, samples);
}

#[test]
fn ycbcr_two_by_two_grid_round_trips() {
    // Four 1×1 Slices tiling a 2×2 grid — the canonical v3-default
    // partition. Every cell painted exactly once: §5-conforming.
    let cr = grayscale_v3_cr(1, 2, 2, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame(&bytes, &cr, &qts, dims, true).expect("decode");
    assert_eq!(decoded.planes[0].samples, samples);
}

#[test]
fn rgb_two_by_two_grid_round_trips() {
    // Same 2×2 grid through the RGB / line-major driver, demonstrating
    // the §5 gate is wired on both drivers symmetrically.
    let cr = rgb_v3_cr(1, 2, 2, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    let r = synth_rgb(fw, fh, 1);
    let g = synth_rgb(fw, fh, 7);
    let b = synth_rgb(fw, fh, 13);
    let frame = make_rgb_decoded_frame(r.clone(), g.clone(), b.clone(), fw, fh, 8);
    let bytes = encode_frame_rgb(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("decode");
    assert_eq!(decoded.planes[0].samples, r);
    assert_eq!(decoded.planes[1].samples, g);
    assert_eq!(decoded.planes[2].samples, b);
}

// -- negative: overlap aborts deterministically ---------------------

#[test]
fn ycbcr_two_slices_overlapping_cell_aborts_with_overlap() {
    // Two Slices on a 2×1 grid; both address cell (0, 0). The
    // canonical partition would be `(0,0)+(1,0)`; this fixture has
    // `(0,0)+(0,0)`. Cell (1, 0) is uncovered AND cell (0, 0) is
    // doubly-painted, so the round-257 validator's deterministic
    // ordering (overlap before gap) must surface
    // `SliceRasterOverlap { x: 0, y: 0, first_slice_index: 0,
    // second_slice_index: 1 }`.
    let cr = grayscale_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [
        make_header(0, 0, 1, 1, 2, 0),
        make_header(0, 0, 1, 1, 2, 0), // overlapping (0, 0)
    ];
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    match decode_frame(&bytes, &cr, &qts, dims, true) {
        Err(Error::SliceRasterOverlap {
            x,
            y,
            first_slice_index,
            second_slice_index,
        }) => {
            assert_eq!(x, 0);
            assert_eq!(y, 0);
            assert_eq!(first_slice_index, 0);
            assert_eq!(second_slice_index, 1);
        }
        other => panic!("expected SliceRasterOverlap, got {other:?}"),
    }
}

#[test]
fn rgb_two_slices_overlapping_cell_aborts_with_overlap() {
    let cr = rgb_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [make_header(0, 0, 1, 1, 2, 0), make_header(0, 0, 1, 1, 2, 0)];
    let r = synth_rgb(fw, fh, 2);
    let g = synth_rgb(fw, fh, 3);
    let b = synth_rgb(fw, fh, 5);
    let frame = make_rgb_decoded_frame(r, g, b, fw, fh, 8);
    let bytes = encode_frame_rgb(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    assert!(matches!(
        decode_frame_rgb(&bytes, &cr, &qts, dims, true),
        Err(Error::SliceRasterOverlap {
            x: 0,
            y: 0,
            first_slice_index: 0,
            second_slice_index: 1,
        })
    ));
}

// -- negative: gap aborts at the canonical first uncovered cell -----

#[test]
fn ycbcr_single_slice_leaves_uncovered_cell_aborts_with_gap() {
    // A 2×1 grid with only one Slice claiming the left cell. Cell
    // (1, 0) is uncovered. The validator's row-major scan-order
    // tie-break must surface `SliceRasterUncovered { x: 1, y: 0 }`.
    let cr = grayscale_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("enc");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    assert!(matches!(
        decode_frame(&bytes, &cr, &qts, dims, true),
        Err(Error::SliceRasterUncovered { x: 1, y: 0 })
    ));
}

#[test]
fn rgb_single_slice_leaves_uncovered_cell_aborts_with_gap() {
    let cr = rgb_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r = synth_rgb(fw, fh, 11);
    let g = synth_rgb(fw, fh, 23);
    let b = synth_rgb(fw, fh, 41);
    let frame = make_rgb_decoded_frame(r, g, b, fw, fh, 8);
    let bytes =
        encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    assert!(matches!(
        decode_frame_rgb(&bytes, &cr, &qts, dims, true),
        Err(Error::SliceRasterUncovered { x: 1, y: 0 })
    ));
}

// -- policy-independence: lenient still aborts on a §5 violation ----

#[test]
fn ycbcr_lenient_still_aborts_on_overlap() {
    // The §5 raster-coverage gate is structural and orthogonal to the
    // §4.9.3 CRC / §4.9.2 error_status policies. Both DecodeOptions
    // policies must surface the violation; lenient cannot swallow a
    // partition error the way it swallows a CRC residue mismatch.
    let cr = grayscale_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [make_header(0, 0, 1, 1, 2, 0), make_header(0, 0, 1, 1, 2, 0)];
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    for opts in [DecodeOptions::strict(), DecodeOptions::lenient()] {
        assert!(matches!(
            decode_frame_with_options(&bytes, &cr, &qts, dims, true, opts),
            Err(Error::SliceRasterOverlap { .. })
        ));
    }
}

#[test]
fn ycbcr_lenient_still_aborts_on_gap() {
    let cr = grayscale_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("enc");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    for opts in [DecodeOptions::strict(), DecodeOptions::lenient()] {
        assert!(matches!(
            decode_frame_with_options(&bytes, &cr, &qts, dims, true, opts),
            Err(Error::SliceRasterUncovered { x: 1, y: 0 })
        ));
    }
}

#[test]
fn rgb_lenient_still_aborts_on_overlap_and_gap() {
    // Symmetric coverage on the RGB / line-major driver.
    let cr = rgb_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);

    // overlap
    let headers_overlap = [make_header(0, 0, 1, 1, 2, 0), make_header(0, 0, 1, 1, 2, 0)];
    let frame_a = make_rgb_decoded_frame(
        synth_rgb(fw, fh, 2),
        synth_rgb(fw, fh, 3),
        synth_rgb(fw, fh, 5),
        fw,
        fh,
        8,
    );
    let bytes_overlap =
        encode_frame_rgb(&frame_a, &cr, &qts, &headers_overlap, true).expect("encode overlap");

    // gap
    let header_single = make_header(0, 0, 1, 1, 2, 0);
    let frame_b = make_rgb_decoded_frame(
        synth_rgb(fw, fh, 11),
        synth_rgb(fw, fh, 13),
        synth_rgb(fw, fh, 17),
        fw,
        fh,
        8,
    );
    let bytes_gap = encode_frame_rgb(
        &frame_b,
        &cr,
        &qts,
        std::slice::from_ref(&header_single),
        true,
    )
    .expect("encode gap");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    for opts in [DecodeOptions::strict(), DecodeOptions::lenient()] {
        assert!(matches!(
            decode_frame_rgb_with_options(&bytes_overlap, &cr, &qts, dims, true, opts),
            Err(Error::SliceRasterOverlap { .. })
        ));
        assert!(matches!(
            decode_frame_rgb_with_options(&bytes_gap, &cr, &qts, dims, true, opts),
            Err(Error::SliceRasterUncovered { x: 1, y: 0 })
        ));
    }
}
