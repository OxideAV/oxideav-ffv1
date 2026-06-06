//! Tests for the unified [`encode_frame`] dispatch helper.
//!
//! `encode_frame` inspects an [`Ffv1ConfigurationRecord`] and forwards
//! to the right specialised encoder so callers no longer replicate the
//! §4.2.3 `coder_type` / §4.2.5 `colorspace_type` switch:
//!
//! ```text
//!   colorspace_type   coder_type   delegate
//!   Rgb (1)           0 | 1 | 2    encode_frame_rgb
//!   YCbCr (0)         0            encode_frame_golomb_rice
//!   YCbCr (0)         1 | 2        encode_frame_range_coder
//! ```
//!
//! Each test pins down the dispatch by asserting the unified helper's
//! output is **byte-identical** to the delegate it should reach, then
//! confirms the bytes round-trip through the matching decoder. A final
//! test pins the out-of-range `coder_type` rejection.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, encode_frame, encode_frame_golomb_rice,
    encode_frame_range_coder, encode_frame_rgb, ColorspaceType, DecodedFrame, DecodedFramePlane,
    Error, Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions,
    PictureStructure, QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS,
};

// ----- shared builders (mirror the per-path encode suites) ------------

fn grayscale_v3_cr(coder_type: u32, bits: u32) -> Ffv1ConfigurationRecord {
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
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: Some(1),
        ec: Some(0),
        intra: Some(false),
        initial_state_delta: None,
    }
}

fn rgb_v3_cr(coder_type: u32, bits: u32) -> Ffv1ConfigurationRecord {
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
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: Some(1),
        ec: Some(0),
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

fn make_header(quant_index_count: usize) -> Ffv1SliceHeader {
    let idx = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
    Ffv1SliceHeader {
        slice_x: 0,
        slice_y: 0,
        slice_width: 1,
        slice_height: 1,
        quant_table_set_index_count: quant_index_count,
        quant_table_set_index: idx,
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
    }
}

fn make_rgb_frame(w: u32, h: u32, bits: u32) -> DecodedFrame {
    let n = (w * h) as usize;
    let mask = (1i32 << bits) - 1;
    let r: Vec<i32> = (0..n as i32).map(|i| (i * 7 + 3) & mask).collect();
    let g: Vec<i32> = (0..n as i32).map(|i| (i * 11 + 5) & mask).collect();
    let b: Vec<i32> = (0..n as i32).map(|i| (i * 13 + 7) & mask).collect();
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
    }
}

// ----- YCbCr coder_type == 0 → encode_frame_golomb_rice ---------------

#[test]
fn dispatch_ycbcr_golomb_matches_specialised_and_round_trips() {
    let cr = grayscale_v3_cr(0, 8);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(2);
    let frame = make_gray_frame((0..32).map(|i| (i * 7 + 3) & 0xFF).collect(), 8, 4, 8);

    let via_helper = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    let via_direct =
        encode_frame_golomb_rice(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(
        via_helper, via_direct,
        "encode_frame must reach encode_frame_golomb_rice for YCbCr coder_type==0"
    );

    let decoded = decode_frame(
        &via_helper,
        &cr,
        &qts,
        FramePixelDimensions::new(frame.width, frame.height).unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(decoded.planes[0].samples, frame.planes[0].samples);
}

// ----- YCbCr coder_type == 1 → encode_frame_range_coder ---------------

#[test]
fn dispatch_ycbcr_range_matches_specialised_and_round_trips() {
    let cr = grayscale_v3_cr(1, 8);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(2);
    let frame = make_gray_frame((0..32).map(|i| (i * 11 + 1) & 0xFF).collect(), 8, 4, 8);

    let via_helper = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    let via_direct =
        encode_frame_range_coder(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(
        via_helper, via_direct,
        "encode_frame must reach encode_frame_range_coder for YCbCr coder_type==1"
    );

    let decoded = decode_frame(
        &via_helper,
        &cr,
        &qts,
        FramePixelDimensions::new(frame.width, frame.height).unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(decoded.planes[0].samples, frame.planes[0].samples);
}

// ----- YCbCr coder_type == 2 → encode_frame_range_coder ---------------

#[test]
fn dispatch_ycbcr_coder_type_2_matches_specialised() {
    let mut cr = grayscale_v3_cr(2, 8);
    for d in cr.state_transition_delta.iter_mut() {
        *d = 1;
    }
    let qts = vec![constant_context_qts(7)];
    let header = make_header(2);
    let frame = make_gray_frame((0..16).map(|i| (i * 5 + 2) & 0xFF).collect(), 4, 4, 8);

    let via_helper = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    let via_direct =
        encode_frame_range_coder(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(
        via_helper, via_direct,
        "encode_frame must reach encode_frame_range_coder for YCbCr coder_type==2"
    );
}

// ----- RGB coder_type == 1 → encode_frame_rgb -------------------------

#[test]
fn dispatch_rgb_range_matches_specialised_and_round_trips() {
    let cr = rgb_v3_cr(1, 8);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(2);
    let frame = make_rgb_frame(8, 4, 8);

    let via_helper = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    let via_direct =
        encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(
        via_helper, via_direct,
        "encode_frame must reach encode_frame_rgb for colorspace Rgb"
    );

    let decoded = decode_frame_rgb(
        &via_helper,
        &cr,
        &qts,
        FramePixelDimensions::new(frame.width, frame.height).unwrap(),
        true,
    )
    .unwrap();
    for (got, want) in decoded.planes.iter().zip(frame.planes.iter()) {
        assert_eq!(
            got.samples, want.samples,
            "RGB plane {} diverged through encode_frame",
            got.plane_index
        );
    }
}

// ----- RGB coder_type == 0 → encode_frame_rgb (Golomb sub-path) -------

#[test]
fn dispatch_rgb_golomb_matches_specialised() {
    let cr = rgb_v3_cr(0, 8);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(2);
    let frame = make_rgb_frame(8, 4, 8);

    let via_helper = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    let via_direct =
        encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(
        via_helper, via_direct,
        "encode_frame must reach encode_frame_rgb for colorspace Rgb even on coder_type==0"
    );
}

// ----- out-of-range coder_type rejection ------------------------------

#[test]
fn dispatch_ycbcr_rejects_out_of_range_coder_type() {
    let cr = grayscale_v3_cr(3, 8);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(2);
    let frame = make_gray_frame(vec![0; 16], 4, 4, 8);

    let err = encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap_err();
    assert!(
        matches!(err, Error::UnsupportedCoderType(3)),
        "coder_type==3 has no §4.2.3 Table 7 entry, got {err:?}"
    );
}
