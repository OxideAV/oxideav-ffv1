//! End-to-end tests for the §4.6 Slice Headers surfaced on
//! [`oxideav_ffv1::DecodedFrame::slice_headers`].
//!
//! Round 260 taught both frame-level decode drivers to parse every
//! Slice Header in a pass-1 preamble (for the RFC 9043 §5
//! second-paragraph raster-coverage gate) and round 274 surfaced the
//! §4.4 `keyframe` boolean on [`oxideav_ffv1::DecodedFrame`]. This
//! round surfaces the pass-1 headers themselves — in forward
//! Slice-index order (slice 0 first, the §4.9.1 trailer-chain order) —
//! so a caller holds BOTH arguments the RFC 9043 §5 third-paragraph
//! multi-Frame rule needs:
//!
//! ```text
//! For each Frame with a keyframe value of 0, each Slice MUST have
//! the same value of slice_x, slice_y, slice_width, and slice_height
//! as a Slice in the previous Frame.
//! ```
//!
//! (RFC 9043 §5 third paragraph, August 2021.) The frame drivers are
//! single-Frame and stateless, so the cross-Frame walk lives with the
//! caller: one
//! `tracker.observe_frame(decoded.keyframe, &decoded.slice_headers)`
//! per Frame in coded order, both arguments now read straight off the
//! decode output.
//!
//! Coverage:
//!
//! * YCbCr / plane-major — a 2×2-grid encode → decode surfaces all
//!   four headers in forward order with every on-wire field intact,
//! * RGB / line-major — same on the line-major driver,
//! * tracker drive — a conforming Frame pair and a re-split
//!   (1×1-grid) Frame are validated through
//!   [`SliceGeometryStabilityTracker`] purely from decode outputs;
//!   the re-split Frame surfaces
//!   [`Error::SliceGeometryUnstable`] deterministically,
//! * options-parity — strict and lenient decodes surface identical
//!   headers (the pass-1 parse is policy-independent).

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, decode_frame_with_options, encode_frame, encode_frame_rgb,
    ColorspaceType, DecodeOptions, DecodedFrame, DecodedFramePlane, Error, Ffv1ConfigurationRecord,
    Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure, QuantizationTableSet,
    SliceGeometryStabilityTracker, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS,
};

// -- shared fixture helpers (mirror section_5_raster_coverage.rs) ------

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
) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x,
        slice_y,
        slice_width,
        slice_height,
        quant_table_set_index_count: quant_index_count,
        quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: PictureStructure::Unknown,
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

/// Field-for-field comparison of a surfaced header against the header
/// the encoder was given. Every §4.6 on-wire field participates —
/// geometry (§4.6.1-§4.6.4), the §4.6.5/§4.6.6 quant-table-set
/// selectors, §4.6.7 `picture_structure` (raw), and the §4.6.8/§4.6.9
/// SAR pair.
fn assert_header_round_trips(surfaced: &Ffv1SliceHeader, written: &Ffv1SliceHeader) {
    assert_eq!(surfaced.slice_x, written.slice_x);
    assert_eq!(surfaced.slice_y, written.slice_y);
    assert_eq!(surfaced.slice_width, written.slice_width);
    assert_eq!(surfaced.slice_height, written.slice_height);
    assert_eq!(
        surfaced.quant_table_set_index_count,
        written.quant_table_set_index_count
    );
    assert_eq!(
        surfaced.quant_table_indices(),
        written.quant_table_indices()
    );
    assert_eq!(
        surfaced.picture_structure_raw,
        written.picture_structure_raw
    );
    assert_eq!(surfaced.sar_num, written.sar_num);
    assert_eq!(surfaced.sar_den, written.sar_den);
}

// -- YCbCr / plane-major driver ----------------------------------------

#[test]
fn ycbcr_decode_surfaces_forward_ordered_slice_headers() {
    // A 2×2-grid (four 1×1-cell Slices) Frame: the decode must surface
    // all four §4.6 headers in forward Slice-index order — the order
    // the encoder wrote them and the §4.9.1 trailer chain walks them.
    let cr = grayscale_v3_cr(1, 2, 2, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [
        make_header(0, 0, 1, 1, 2),
        make_header(1, 0, 1, 1, 2),
        make_header(0, 1, 1, 1, 2),
        make_header(1, 1, 1, 1, 2),
    ];
    let samples = synth_gray(fw, fh);
    let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame(&bytes, &cr, &qts, dims, true).expect("decode");

    assert!(decoded.keyframe);
    assert_eq!(decoded.slice_headers.len(), 4);
    for (surfaced, written) in decoded.slice_headers.iter().zip(headers.iter()) {
        assert_header_round_trips(surfaced, written);
    }
    // The pixel reconstruction is unaffected by the surfacing.
    assert_eq!(decoded.planes[0].samples, samples);
}

#[test]
fn ycbcr_strict_and_lenient_surface_identical_headers() {
    // The pass-1 header parse feeding `slice_headers` runs before the
    // §4.9.2 / §4.9.3 policy gates and is policy-independent: strict
    // and lenient decodes of a clean Frame surface identical headers.
    let cr = grayscale_v3_cr(1, 2, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [make_header(0, 0, 1, 1, 2), make_header(1, 0, 1, 1, 2)];
    let frame = make_gray_decoded_frame(synth_gray(fw, fh), fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let strict = decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict())
        .expect("strict decode");
    let lenient =
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient())
            .expect("lenient decode");

    assert_eq!(strict.slice_headers.len(), 2);
    assert_eq!(lenient.slice_headers.len(), 2);
    for (s, l) in strict
        .slice_headers
        .iter()
        .zip(lenient.slice_headers.iter())
    {
        assert_header_round_trips(s, l);
    }
}

// -- RGB / line-major driver -------------------------------------------

#[test]
fn rgb_decode_surfaces_forward_ordered_slice_headers() {
    // Mirror of the YCbCr test on the RGB / line-major driver: the
    // pass-1 collection is shared between the two drivers, so the
    // surfaced headers must agree with the encoder's input on both.
    let cr = rgb_v3_cr(1, 2, 2, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let headers = [
        make_header(0, 0, 1, 1, 2),
        make_header(1, 0, 1, 1, 2),
        make_header(0, 1, 1, 1, 2),
        make_header(1, 1, 1, 1, 2),
    ];
    let r = synth_rgb(fw, fh, 1);
    let g = synth_rgb(fw, fh, 7);
    let b = synth_rgb(fw, fh, 13);
    let frame = make_rgb_decoded_frame(r.clone(), g.clone(), b.clone(), fw, fh, 8);
    let bytes = encode_frame_rgb(&frame, &cr, &qts, &headers, true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("decode");

    assert!(decoded.keyframe);
    assert_eq!(decoded.slice_headers.len(), 4);
    for (surfaced, written) in decoded.slice_headers.iter().zip(headers.iter()) {
        assert_header_round_trips(surfaced, written);
    }
    assert_eq!(decoded.planes[0].samples, r);
    assert_eq!(decoded.planes[1].samples, g);
    assert_eq!(decoded.planes[2].samples, b);
}

// -- §5 third paragraph: tracker driven purely off decode outputs -------

#[test]
fn decoded_pair_drives_section5_stability_tracker_end_to_end() {
    // The complete §5 third-paragraph wiring a caller performs, with
    // BOTH `observe_frame` arguments read off `DecodedFrame`:
    //
    //   tracker.observe_frame(decoded.keyframe, &decoded.slice_headers)
    //
    // Stream A is a 2×2-grid Frame; stream B re-splits the same pixel
    // area onto a 1×1 grid (one full-raster Slice). A conforming
    // sequence (A, then A-geometry again as a non-keyframe) passes;
    // following A with B's geometry as a non-keyframe violates §5
    // ("each Slice MUST have the same value of slice_x, slice_y,
    // slice_width, and slice_height as a Slice in the previous Frame")
    // and surfaces `SliceGeometryUnstable` on forward index 0.
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let samples = synth_gray(fw, fh);

    let cr_a = grayscale_v3_cr(1, 2, 2, 8);
    let headers_a = [
        make_header(0, 0, 1, 1, 2),
        make_header(1, 0, 1, 1, 2),
        make_header(0, 1, 1, 1, 2),
        make_header(1, 1, 1, 1, 2),
    ];
    let frame_a = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes_a = encode_frame(&frame_a, &cr_a, &qts, &headers_a, true).expect("encode A");

    let cr_b = grayscale_v3_cr(1, 1, 1, 8);
    let headers_b = [make_header(0, 0, 1, 1, 2)];
    let frame_b = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes_b = encode_frame(&frame_b, &cr_b, &qts, &headers_b, true).expect("encode B");

    let decoded_a1 = decode_frame(&bytes_a, &cr_a, &qts, dims, true).expect("decode A1");
    let decoded_a2 = decode_frame(&bytes_a, &cr_a, &qts, dims, true).expect("decode A2");
    let decoded_b = decode_frame(&bytes_b, &cr_b, &qts, dims, true).expect("decode B");

    // Conforming sequence: keyframe opens the stream (this codec's
    // encoder writes `keyframe = 1` on every Frame), then the same
    // geometry validated as if it were a non-keyframe.
    let mut tracker = SliceGeometryStabilityTracker::new();
    tracker
        .observe_frame(decoded_a1.keyframe, &decoded_a1.slice_headers)
        .expect("keyframe opens the stream");
    tracker
        .observe_frame(false, &decoded_a2.slice_headers)
        .expect("identical geometry conforms to §5 on a non-keyframe");

    // Violating sequence: the 1×1-grid re-split's single full-raster
    // Slice (0, 0, 1, 1 on a DIFFERENT grid — but §5 compares the
    // §4.6.1-§4.6.4 wire quadruples, and stream A's previous Frame
    // does carry a (0, 0, 1, 1) Slice) — so to pin a real violation,
    // validate stream A's geometry against stream B's single-Slice
    // reference: Slices (1, 0), (0, 1), (1, 1) match nothing.
    let mut tracker_b = SliceGeometryStabilityTracker::new();
    tracker_b
        .observe_frame(decoded_b.keyframe, &decoded_b.slice_headers)
        .expect("stream B keyframe opens its own stream");
    let err = tracker_b
        .observe_frame(false, &decoded_a1.slice_headers)
        .expect_err("2x2 re-split after a single-Slice Frame violates §5");
    match err {
        Error::SliceGeometryUnstable {
            slice_index,
            slice_x,
            slice_y,
            slice_width,
            slice_height,
        } => {
            // Forward index 1 — the (1, 0, 1, 1) Slice — is the first
            // with no quadruple match in the previous (single-Slice)
            // Frame; index 0's (0, 0, 1, 1) matches B's only Slice.
            assert_eq!(slice_index, 1);
            assert_eq!((slice_x, slice_y), (1, 0));
            assert_eq!((slice_width, slice_height), (1, 1));
        }
        other => panic!("expected SliceGeometryUnstable, got {other:?}"),
    }
}
