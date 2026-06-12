//! End-to-end tests for the stateful multi-Frame
//! [`Ffv1DecodeSession`] (RFC 9043 §4.4 / §4.2.17 / §5).
//!
//! The session is the driver-level stitch round 279 queued: it owns
//! the per-stream decode inputs plus the cross-Frame state (the §5
//! third-paragraph [`SliceGeometryStabilityTracker`] walk and the
//! coded-order Frame counter), routes each Frame to the right
//! single-Frame driver on §4.2.5 `colorspace_type`, and applies the
//! two stream-scope conformance gates — §4.2.17 `intra` ("keyframe
//! MUST be 1 (keyframes only)") and §5 third paragraph ("each Slice
//! MUST have the same value of slice_x, slice_y, slice_width, and
//! slice_height as a Slice in the previous Frame") — that no
//! single-Frame stateless driver can own.
//!
//! Coverage:
//!
//! * multi-Frame YCbCr decode through the session is bit-exact
//!   against the stateless [`decode_frame`] driver, per Frame;
//! * RGB / line-major routing mirrors [`decode_frame_rgb`];
//! * [`DecodeOptions`] flow through to the routed driver (a §4.9.2
//!   `Uncorrectable` Slice aborts a strict session and decodes on a
//!   lenient one);
//! * the §4.2.17 / §5 gates fire through
//!   [`Ffv1DecodeSession::observe_decoded_frame`] on replayed decode
//!   outputs, and a violating Frame leaves the session state
//!   untouched.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, encode_frame, encode_frame_rgb,
    encode_slice_footer_with_raw_status, ColorspaceType, DecodeOptions, DecodedFrame,
    DecodedFramePlane, Error, Ffv1ConfigurationRecord, Ffv1DecodeSession, Ffv1SliceHeader,
    Ffv1Version, FramePixelDimensions, PictureStructure, QuantizationTableSet,
    MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS, SLICE_FOOTER_LEN_EC1,
};

// -- shared fixture helpers (mirror decoded_slice_headers.rs) -----------

fn grayscale_v3_cr(
    intra: Option<bool>,
    num_h: u32,
    num_v: u32,
    bits: u32,
) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1,
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
        intra,
        initial_state_delta: None,
    }
}

fn rgb_v3_cr(num_h: u32, num_v: u32, bits: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1,
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

fn make_header(slice_x: u32, slice_y: u32, slice_width: u32, slice_height: u32) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x,
        slice_y,
        slice_width,
        slice_height,
        quant_table_set_index_count: 2,
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

fn make_rgb_decoded_frame(r: Vec<i32>, g: Vec<i32>, b: Vec<i32>, w: u32, h: u32) -> DecodedFrame {
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
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

fn synth(w: u32, h: u32, seed: usize) -> Vec<i32> {
    (0..(w * h) as usize)
        .map(|i| (i.wrapping_mul(seed + 19) + 5) as i32 & 0xFF)
        .collect()
}

/// Rebuild a single-Slice `ec == 1` frame's §4.9 footer with an
/// explicit raw §4.9.2 status byte; the §4.9.3 parity is re-solved so
/// the CRC residue stays zero and the test isolates the §4.9.2 gate.
fn rewrite_single_slice_error_status(bytes: &[u8], raw_status: u8) -> Vec<u8> {
    assert!(bytes.len() > SLICE_FOOTER_LEN_EC1);
    let body = &bytes[..bytes.len() - SLICE_FOOTER_LEN_EC1];
    encode_slice_footer_with_raw_status(body, true, raw_status)
        .expect("re-solving the §4.9 footer must succeed for a valid body")
}

// -- multi-Frame decode through the session ------------------------------

#[test]
fn session_ycbcr_multi_frame_decode_is_bit_exact_against_stateless_driver() {
    // Three Frames in coded order through one session: every decode
    // matches the stateless `decode_frame` bit-for-bit, the §4.4
    // keyframe + §4.6 headers surface unchanged, and the session
    // counter / previous-Frame reference advance per Frame.
    let cr = grayscale_v3_cr(Some(false), 2, 2, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let headers = [
        make_header(0, 0, 1, 1),
        make_header(1, 0, 1, 1),
        make_header(0, 1, 1, 1),
        make_header(1, 1, 1, 1),
    ];

    let mut session = Ffv1DecodeSession::new(cr.clone(), qts.clone(), dims, true);
    assert_eq!(session.frames_observed(), 0);
    assert!(!session.has_previous_frame());

    for (i, seed) in [1usize, 7, 13].iter().enumerate() {
        let samples = synth(fw, fh, *seed);
        let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
        let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

        let via_session = session.decode_next_frame(&bytes).expect("session decode");
        let via_driver = decode_frame(&bytes, &cr, &qts, dims, true).expect("stateless decode");

        assert_eq!(via_session.planes[0].samples, samples);
        assert_eq!(via_session.planes[0].samples, via_driver.planes[0].samples);
        assert!(via_session.keyframe);
        assert_eq!(via_session.slice_headers.len(), 4);
        assert_eq!(session.frames_observed(), i as u64 + 1);
        assert!(session.has_previous_frame());
    }
}

#[test]
fn session_rgb_routes_line_major_driver() {
    // An RGB Configuration Record routes through the line-major
    // driver — same dispatch `encode_frame` performs on the write
    // side — and reproduces `decode_frame_rgb` bit-for-bit.
    let cr = rgb_v3_cr(1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let headers = [make_header(0, 0, 1, 1)];

    let r = synth(fw, fh, 1);
    let g = synth(fw, fh, 7);
    let b = synth(fw, fh, 13);
    let frame = make_rgb_decoded_frame(r.clone(), g.clone(), b.clone(), fw, fh);
    let bytes = encode_frame_rgb(&frame, &cr, &qts, &headers, true).expect("encode");

    let mut session = Ffv1DecodeSession::new(cr.clone(), qts.clone(), dims, true);
    let via_session = session.decode_next_frame(&bytes).expect("session decode");
    let via_driver = decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("stateless decode");

    assert_eq!(via_session.planes[0].samples, r);
    assert_eq!(via_session.planes[1].samples, g);
    assert_eq!(via_session.planes[2].samples, b);
    for (s, d) in via_session.planes.iter().zip(via_driver.planes.iter()) {
        assert_eq!(s.samples, d.samples);
    }
    assert_eq!(session.frames_observed(), 1);
}

// -- DecodeOptions flow-through ------------------------------------------

#[test]
fn session_options_flow_through_to_the_routed_driver() {
    // A §4.9.2 Table 16 `Uncorrectable` Slice aborts a strict session
    // (and does NOT advance it) but decodes on a lenient one — the
    // per-Frame policy gates flow through `with_options` unchanged.
    let cr = grayscale_v3_cr(Some(false), 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let headers = [make_header(0, 0, 1, 1)];
    let samples = synth(fw, fh, 3);
    let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");
    let bytes = rewrite_single_slice_error_status(&bytes, 2);

    let mut strict = Ffv1DecodeSession::with_options(
        cr.clone(),
        qts.clone(),
        dims,
        true,
        DecodeOptions::strict(),
    );
    assert!(matches!(
        strict.decode_next_frame(&bytes),
        Err(Error::SliceErrorStatus {
            slice_index: 0,
            status: 2
        })
    ));
    assert_eq!(strict.frames_observed(), 0, "aborted Frame is not counted");
    assert!(!strict.has_previous_frame());

    let mut lenient =
        Ffv1DecodeSession::with_options(cr, qts, dims, true, DecodeOptions::lenient());
    let decoded = lenient.decode_next_frame(&bytes).expect("lenient decode");
    // The body bytes are untouched (only the §4.9.2 byte + re-solved
    // parity changed), so the lenient decode is bit-exact.
    assert_eq!(decoded.planes[0].samples, samples);
    assert_eq!(lenient.frames_observed(), 1);
}

// -- stream-scope gates on replayed decode outputs ------------------------

#[test]
fn session_intra_gate_rejects_replayed_non_keyframe() {
    // `intra == 1` (RFC 9043 §4.2.17 Table 14: "keyframe MUST be 1")
    // — every in-tree encode writes keyframe = 1, so the violating
    // non-keyframe is replayed through `observe_decoded_frame` with
    // the §4.4 field flipped on the decode output. Geometry is kept
    // identical to the previous Frame so only the §4.2.17 gate can
    // fire.
    let cr = grayscale_v3_cr(Some(true), 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let headers = [make_header(0, 0, 1, 1)];
    let frame = make_gray_decoded_frame(synth(fw, fh, 3), fw, fh, 8);
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true).expect("encode");

    let mut session = Ffv1DecodeSession::new(cr.clone(), qts.clone(), dims, true);
    let decoded = session.decode_next_frame(&bytes).expect("keyframe decode");
    assert!(decoded.keyframe);
    assert_eq!(session.frames_observed(), 1);

    let mut non_keyframe = decoded.clone();
    non_keyframe.keyframe = false;
    assert_eq!(
        session.observe_decoded_frame(&non_keyframe),
        Err(Error::NonKeyframeInIntraStream { frame_index: 1 })
    );
    assert_eq!(session.frames_observed(), 1, "violating Frame not counted");

    // The same Frame as a keyframe still conforms — the gate keys on
    // the §4.4 value alone.
    session
        .observe_decoded_frame(&decoded)
        .expect("keyframe conforms to intra == 1");
    assert_eq!(session.frames_observed(), 2);
}

#[test]
fn session_geometry_gate_rejects_replayed_resplit_non_keyframe() {
    // §5 third paragraph through the session: a 2×2-grid decode opens
    // the stream; a non-keyframe carrying a different stream's 1×1
    // single-Slice geometry... — direction mirrors
    // decoded_slice_headers.rs: open with the single-Slice Frame,
    // then replay the 2×2 Frame as a non-keyframe. Slices (1,0),
    // (0,1), (1,1) match nothing in the previous Frame, pinning
    // `SliceGeometryUnstable` on forward index 1.
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (16u32, 8u32);
    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let samples = synth(fw, fh, 5);

    let cr_a = grayscale_v3_cr(Some(false), 1, 1, 8);
    let headers_a = [make_header(0, 0, 1, 1)];
    let frame_a = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes_a = encode_frame(&frame_a, &cr_a, &qts, &headers_a, true).expect("encode A");

    let cr_b = grayscale_v3_cr(Some(false), 2, 2, 8);
    let headers_b = [
        make_header(0, 0, 1, 1),
        make_header(1, 0, 1, 1),
        make_header(0, 1, 1, 1),
        make_header(1, 1, 1, 1),
    ];
    let frame_b = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes_b = encode_frame(&frame_b, &cr_b, &qts, &headers_b, true).expect("encode B");
    let decoded_b = decode_frame(&bytes_b, &cr_b, &qts, dims, true).expect("decode B");

    let mut session = Ffv1DecodeSession::new(cr_a, qts, dims, true);
    session
        .decode_next_frame(&bytes_a)
        .expect("single-Slice keyframe opens the stream");

    let mut resplit = decoded_b;
    resplit.keyframe = false;
    let err = session
        .observe_decoded_frame(&resplit)
        .expect_err("2x2 re-split after a single-Slice Frame violates §5");
    match err {
        Error::SliceGeometryUnstable {
            slice_index,
            slice_x,
            slice_y,
            slice_width,
            slice_height,
        } => {
            assert_eq!(slice_index, 1);
            assert_eq!((slice_x, slice_y), (1, 0));
            assert_eq!((slice_width, slice_height), (1, 1));
        }
        other => panic!("expected SliceGeometryUnstable, got {other:?}"),
    }
    assert_eq!(session.frames_observed(), 1, "violating Frame not counted");

    // The previous-Frame reference survived the violation: the
    // original single-Slice geometry still conforms as a non-keyframe.
    let mut conforming = make_gray_decoded_frame(vec![0; (fw * fh) as usize], fw, fh, 8);
    conforming.keyframe = false;
    conforming.slice_headers = vec![make_header(0, 0, 1, 1)];
    session
        .observe_decoded_frame(&conforming)
        .expect("original geometry still conforms");
    assert_eq!(session.frames_observed(), 2);
}
