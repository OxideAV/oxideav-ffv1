//! End-to-end tests for the per-Slice §4.9.3 CRC validation gate on
//! the frame-level decode drivers.
//!
//! Round 238 introduces [`oxideav_ffv1::DecodeOptions`] and the
//! options-aware [`oxideav_ffv1::decode_frame_with_options`] /
//! [`oxideav_ffv1::decode_frame_rgb_with_options`] entry points. The
//! gate governs how the drivers react to a non-zero RFC 9043 §4.9.3
//! whole-Slice CRC residue:
//!
//! * [`oxideav_ffv1::SliceCrcPolicy::Reject`] (default) — abort the
//!   frame decode with [`oxideav_ffv1::Error::SliceCrcMismatch`],
//!   matching the historical [`oxideav_ffv1::decode_frame`] behaviour
//!   on every prior round.
//! * [`oxideav_ffv1::SliceCrcPolicy::Accept`] — best-effort decode;
//!   per-Slice content is reconstructed and returned regardless of the
//!   residue.
//!
//! Structural failures (truncation, size mismatch) are policy-
//! independent: they always abort.
//!
//! The end-to-end test shape is:
//!
//! ```text
//!   DecodedFrame ── encode_frame ──▶ frame_bytes (clean, residue == 0)
//!     │
//!     │  flip a byte inside slice 0's body (residue != 0)
//!     ▼
//!   frame_bytes' ── decode_frame_with_options(Reject) ──▶ Err(SliceCrcMismatch)
//!                ── decode_frame_with_options(Accept) ──▶ Ok(DecodedFrame)
//! ```
//!
//! The Accept path's decoded samples are NOT compared to the original
//! — by definition a single body-byte flip in a range-coded SliceContent
//! cascades into a different per-Sample reconstruction. What we assert
//! is that:
//!   * Accept returns a structurally valid [`oxideav_ffv1::DecodedFrame`]
//!     with the right plane count + dimensions, and every recovered
//!     Sample lies in the §3.8 modular range `0 .. 2^bits_per_raw_sample`;
//!   * Reject errors with [`oxideav_ffv1::Error::SliceCrcMismatch`]
//!     exposing both the residue and the stored §4.9.3 parity word.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb_with_options, decode_frame_with_options, encode_frame,
    encode_frame_rgb, ColorspaceType, DecodeOptions, DecodedFrame, DecodedFramePlane, Error,
    Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

// -- shared fixture helpers --------------------------------------------

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

/// Flip exactly one bit deep inside the frame's first Slice body (well
/// before the trailing §4.9 footer) — this corrupts a content byte the
/// per-Slice §4.9.3 CRC covers, so the whole-Slice residue becomes
/// non-zero on re-parse. Picking a byte at index 12 puts the flip well
/// after the §4.6 SliceHeader's leading range-coded bytes (so the
/// header still parses) but inside the Slice's range-coded
/// SliceContent, so the per-Sample reconstruction diverges from the
/// original.
fn flip_body_byte_inside_first_slice(bytes: &mut [u8]) {
    let idx = 12usize.min(bytes.len() - 9); // stay well clear of the 8-byte footer
    bytes[idx] ^= 0x01;
}

fn samples_in_range(planes: &[DecodedFramePlane], bits: u32) -> bool {
    let upper = 1i32 << bits;
    planes
        .iter()
        .all(|p| p.samples.iter().all(|&s| (0..upper).contains(&s)))
}

// -- YCbCr / plane-major: per-Slice CRC gate --------------------------

#[test]
fn ycbcr_gate_reject_default_aborts_on_crc_failure() {
    // Build a clean range-coded grayscale frame, flip a content byte,
    // then assert the default-options entry point (Reject policy)
    // surfaces SliceCrcMismatch — and that this matches the historical
    // `decode_frame` behaviour bit-for-bit.
    let cr = grayscale_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 11 + 3) as i32 & 0xFF)
        .collect();
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let mut bytes =
        encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    flip_body_byte_inside_first_slice(&mut bytes);

    // Default policy (Reject) aborts. The legacy entry point matches.
    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    match decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::default()) {
        Err(Error::SliceCrcMismatch {
            residue,
            stored_parity,
        }) => {
            assert_ne!(
                residue, 0,
                "residue must be non-zero after a body byte flip"
            );
            // stored_parity carries the on-wire §4.9.3 parity word for
            // diagnostics — sanity: it's the same regardless of the flip.
            let _ = stored_parity;
        }
        other => panic!("expected SliceCrcMismatch on Reject path, got {other:?}"),
    }
    // The strict() convenience constructor is equivalent to default().
    assert!(matches!(
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict()),
        Err(Error::SliceCrcMismatch { .. })
    ));
    // And the legacy `decode_frame` (no options) must behave identically.
    assert!(matches!(
        decode_frame(&bytes, &cr, &qts, dims, true),
        Err(Error::SliceCrcMismatch { .. })
    ));
}

#[test]
fn ycbcr_gate_accept_partial_decode_returns_structurally_valid_frame() {
    // Same shape — flip a body byte — but the Accept policy must
    // return a `DecodedFrame` with the correct plane count, plane
    // dimensions, and in-range samples (§3.8 modular invariant). The
    // per-Sample values themselves will diverge from the original
    // input by construction (range-coded body flips cascade); the
    // contract is that decoding does NOT abort.
    let cr = grayscale_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 11 + 3) as i32 & 0xFF)
        .collect();
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let mut bytes =
        encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    flip_body_byte_inside_first_slice(&mut bytes);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded =
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient())
            .expect("Accept policy must not error on a non-zero CRC residue");
    assert_eq!(decoded.planes.len(), 1, "grayscale frame has one Plane");
    assert_eq!(decoded.width, fw);
    assert_eq!(decoded.height, fh);
    assert_eq!(decoded.bits_per_raw_sample, 8);
    assert_eq!(decoded.planes[0].width, fw);
    assert_eq!(decoded.planes[0].height, fh);
    assert_eq!(decoded.planes[0].samples.len(), (fw * fh) as usize);
    assert!(
        samples_in_range(&decoded.planes, 8),
        "every reconstructed Sample must lie in 0..2^8 (§3.8 modular)"
    );
}

#[test]
fn ycbcr_gate_clean_slice_both_policies_match_legacy_bit_exact() {
    // Regression: on a clean frame (no body mutation) all three entry
    // points — legacy `decode_frame`, Reject, Accept — must produce
    // the same bit-exact reconstruction. This pins that introducing
    // the gate did not perturb the happy path.
    let cr = grayscale_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 7 + 1) as i32 & 0xFF)
        .collect();
    let frame = make_gray_decoded_frame(samples.clone(), fw, fh, 8);
    let bytes =
        encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let legacy = decode_frame(&bytes, &cr, &qts, dims, true).expect("clean: legacy decode");
    let strict =
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict()).unwrap();
    let lenient =
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient()).unwrap();

    assert_eq!(legacy.planes[0].samples, samples);
    assert_eq!(strict.planes[0].samples, samples);
    assert_eq!(lenient.planes[0].samples, samples);
    // Each entry point must produce the same plane-buffer bytes — the
    // policy gate does not perturb decode behaviour on a clean Slice.
    assert_eq!(legacy.planes[0].samples, strict.planes[0].samples);
    assert_eq!(legacy.planes[0].samples, lenient.planes[0].samples);
}

// -- RGB / line-major: per-Slice CRC gate -----------------------------

#[test]
fn rgb_gate_reject_aborts_accept_partial_decodes() {
    // Mirror test on the RGB / line-major driver
    // (`decode_frame_rgb_with_options`). The §4.7 line-major
    // traversal goes through a different per-Slice plumbing path than
    // the YCbCr / plane-major driver, so the gate's wiring needs an
    // independent assertion.
    let cr = rgb_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(6)];
    let (fw, fh) = (6u32, 4u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 13 + 2) as i32 & 0xFF)
        .collect();
    let g: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 17 + 5) as i32 & 0xFF)
        .collect();
    let b: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 19 + 7) as i32 & 0xFF)
        .collect();
    let frame = make_rgb_decoded_frame(r, g, b, fw, fh, 8);
    let mut bytes =
        encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    flip_body_byte_inside_first_slice(&mut bytes);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();

    // Reject: aborts with SliceCrcMismatch.
    match decode_frame_rgb_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict()) {
        Err(Error::SliceCrcMismatch { residue, .. }) => {
            assert_ne!(residue, 0);
        }
        other => panic!("expected SliceCrcMismatch on RGB Reject path, got {other:?}"),
    }

    // Accept: returns a structurally valid R/G/B DecodedFrame.
    let decoded =
        decode_frame_rgb_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient())
            .expect("Accept policy must not error on a non-zero CRC residue");
    assert_eq!(decoded.planes.len(), 3, "RGB frame has three Planes");
    assert_eq!(decoded.colorspace, ColorspaceType::Rgb);
    for plane in &decoded.planes {
        assert_eq!(plane.width, fw);
        assert_eq!(plane.height, fh);
        assert_eq!(plane.samples.len(), (fw * fh) as usize);
    }
    assert!(samples_in_range(&decoded.planes, 8));
}
