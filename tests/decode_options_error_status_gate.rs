//! End-to-end tests for the per-Slice §4.9.2 `error_status` policy
//! gate on the frame-level decode drivers.
//!
//! Round 244 extends [`oxideav_ffv1::DecodeOptions`] with a second
//! independent integrity gate:
//! [`oxideav_ffv1::SliceErrorStatusPolicy`]. The gate governs how the
//! drivers react to a Slice whose §4.9.2 Table 16 `error_status` byte
//! is `Uncorrectable` (`2`):
//!
//! * [`oxideav_ffv1::SliceErrorStatusPolicy::Reject`] (default) —
//!   abort the frame decode with
//!   [`oxideav_ffv1::Error::SliceErrorStatus`], matching the
//!   historical [`oxideav_ffv1::decode_frame`] /
//!   [`oxideav_ffv1::decode_frame_rgb`] behaviour every prior round
//!   shipped before this round (the drivers parsed the footer +
//!   discarded the field — i.e. silently accepted every status, which
//!   round 244 closes the loop on by adding an actual policy
//!   gate).
//! * [`oxideav_ffv1::SliceErrorStatusPolicy::Accept`] — best-effort
//!   decode; per-Slice content is reconstructed regardless of the
//!   §4.9.2 status (the typed value is still surfaced on the parsed
//!   footer; the gate is purely about whether the driver aborts).
//!
//! Per §4.9.2 Table 16 only the `Uncorrectable` (`2`) wire byte is a
//! rejection target. `NoError` (`0`) is the clean path; `Correctable`
//! (`1`) declares damage the §4.9.3 CRC is expected to detect and
//! (per the encoder's contract) recover; `Reserved` (`>=3`) is unknown
//! and the gate treats it as "trust the bitstream" on both policies
//! — the §4.9.3 CRC residue is the stronger fixity signal for
//! reserved-range bytes.
//!
//! The fabrication shape is:
//!
//! ```text
//!   DecodedFrame ── encode_frame ──▶ frame_bytes (NoError, residue == 0)
//!     │
//!     │  splice in a fresh §4.9 footer with raw error_status = 2 and
//!     │  re-solve the §4.9.3 parity so the whole-Slice CRC residue is
//!     │  still zero. Now the §4.9.3 gate stays green and the §4.9.2
//!     │  gate is the one under test.
//!     ▼
//!   frame_bytes' ── decode_frame_with_options(Reject)  ──▶ Err(SliceErrorStatus)
//!                ── decode_frame_with_options(Accept)  ──▶ Ok(DecodedFrame)
//! ```
//!
//! Because we re-solve the parity, the §4.9.3 residue still validates
//! to zero — the per-Sample reconstruction reproduces the original
//! `DecodedFrame` bit-for-bit on the Accept path (the body bytes are
//! untouched). That's an extra invariant the §4.9.3 gate test cannot
//! pin (its body-byte flip cascades into the per-Sample stream).

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, decode_frame_rgb_with_options, decode_frame_with_options,
    encode_frame, encode_frame_rgb, encode_slice_footer_with_raw_status, ColorspaceType,
    DecodeOptions, DecodedFrame, DecodedFramePlane, Error, Ffv1ConfigurationRecord,
    Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure, QuantizationTableSet,
    SliceCrcPolicy, SliceErrorStatusPolicy, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS, SLICE_FOOTER_LEN_EC1,
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

/// Rewrite the trailing `ec == 1` Slice Footer of a *single-Slice*
/// frame so its §4.9.2 `error_status` byte takes a new raw value,
/// keeping the §4.9.3 whole-Slice CRC residue zero. Uses
/// [`encode_slice_footer_with_raw_status`] (which is the same solver
/// the encoder uses for every clean Slice) so the resulting frame
/// passes the §4.9.3 CRC gate on either policy — isolating the §4.9.2
/// gate under test.
///
/// `encode_slice_footer_with_raw_status` returns `body || footer` (the
/// full re-assembled Slice byte stream), not just the trailing 8
/// footer bytes — i.e. the result replaces `bytes` wholesale rather
/// than being appended to the body.
fn rewrite_single_slice_error_status(bytes: &[u8], raw_status: u8) -> Vec<u8> {
    assert!(
        bytes.len() > SLICE_FOOTER_LEN_EC1,
        "frame too short to carry an ec=1 footer"
    );
    let body_end = bytes.len() - SLICE_FOOTER_LEN_EC1;
    let body = &bytes[..body_end];
    encode_slice_footer_with_raw_status(body, true, raw_status)
        .expect("re-solving the §4.9 footer must succeed for a valid body")
}

fn samples_in_range(planes: &[DecodedFramePlane], bits: u32) -> bool {
    let upper = 1i32 << bits;
    planes
        .iter()
        .all(|p| p.samples.iter().all(|&s| (0..upper).contains(&s)))
}

// -- YCbCr / plane-major: per-Slice §4.9.2 gate -----------------------

#[test]
fn ycbcr_gate_reject_default_aborts_on_uncorrectable_status() {
    // Build a clean range-coded grayscale frame, rewrite its footer
    // with §4.9.2 Table 16 `Uncorrectable` (`2`), then assert that:
    //   * the default-options entry point (Reject policy) surfaces
    //     `Error::SliceErrorStatus { slice_index: 0, status: 2 }`;
    //   * the explicit `DecodeOptions::strict()` constructor matches;
    //   * the legacy `decode_frame` (no options) matches.
    let cr = grayscale_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 11 + 3) as i32 & 0xFF)
        .collect();
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes =
        encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let bytes = rewrite_single_slice_error_status(&bytes, 2);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();

    match decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::default()) {
        Err(Error::SliceErrorStatus {
            slice_index,
            status,
        }) => {
            assert_eq!(slice_index, 0, "single-Slice frame, slice index 0");
            assert_eq!(status, 2, "raw §4.9.2 Table 16 Uncorrectable byte");
        }
        other => panic!("expected SliceErrorStatus on Reject path, got {other:?}"),
    }
    assert!(matches!(
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict()),
        Err(Error::SliceErrorStatus { .. })
    ));
    assert!(matches!(
        decode_frame(&bytes, &cr, &qts, dims, true),
        Err(Error::SliceErrorStatus { .. })
    ));
}

#[test]
fn ycbcr_gate_accept_partial_decode_returns_bit_exact_frame() {
    // Same shape — splice an `Uncorrectable` footer in — but the
    // Accept policy must return a `DecodedFrame`. Because the body
    // bytes are untouched (only the footer's `error_status` byte +
    // re-solved parity changed), the recovered samples are
    // **bit-exact** against the original input: this is the
    // strictest invariant the §4.9.3 gate test cannot pin.
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

    let bytes = rewrite_single_slice_error_status(&bytes, 2);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded =
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient())
            .expect("Accept policy must not abort on §4.9.2 Uncorrectable");
    assert_eq!(decoded.planes.len(), 1, "grayscale frame has one Plane");
    assert_eq!(decoded.width, fw);
    assert_eq!(decoded.height, fh);
    assert_eq!(decoded.bits_per_raw_sample, 8);
    assert_eq!(
        decoded.planes[0].samples, samples,
        "body bytes intact => recovered Samples are bit-exact"
    );
    assert!(samples_in_range(&decoded.planes, 8));
}

#[test]
fn ycbcr_gate_clean_status_all_policies_match_legacy_bit_exact() {
    // Regression: on a frame whose §4.9.2 byte is the clean `0`
    // (`NoError`), every entry point produces the bit-exact original
    // reconstruction — introducing the gate must not perturb the
    // happy path.
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
    assert_eq!(legacy.planes[0].samples, strict.planes[0].samples);
    assert_eq!(legacy.planes[0].samples, lenient.planes[0].samples);
}

#[test]
fn ycbcr_gate_correctable_status_passes_under_reject() {
    // §4.9.2 Table 16 `Correctable` (`1`) is NOT a rejection target —
    // the encoder is asserting damage the §4.9.3 CRC is expected to
    // detect / recover. The Reject policy must let it through. The
    // body bytes are untouched, so the recovered Samples are bit-
    // exact against the original.
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

    let bytes = rewrite_single_slice_error_status(&bytes, 1);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict())
        .expect("Reject policy must accept §4.9.2 Correctable per Table 16");
    assert_eq!(decoded.planes[0].samples, samples);
}

#[test]
fn ycbcr_gate_reserved_status_passes_under_reject() {
    // §4.9.2 Table 16 `Reserved` (`>= 3`) is treated as "trust the
    // bitstream" on either policy — the §4.9.3 CRC is the stronger
    // fixity signal for reserved-range bytes. Reject must accept.
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

    // 0xAB is well inside the reserved range (3..=255).
    let bytes = rewrite_single_slice_error_status(&bytes, 0xAB);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded = decode_frame_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict())
        .expect("Reject policy must pass §4.9.2 Reserved (>=3) per the policy doc");
    assert_eq!(decoded.planes[0].samples, samples);
}

#[test]
fn ycbcr_gate_independent_of_crc_policy_field() {
    // The two gates are independent: Reject on the §4.9.2 gate must
    // fire even when the §4.9.3 gate is set to Accept, because the
    // body is intact (zero residue) but the status is Uncorrectable.
    let cr = grayscale_v3_cr(1, 1, 1, 8);
    let qts = vec![constant_context_qts(5)];
    let (fw, fh) = (8u32, 4u32);
    let header = make_header(0, 0, 1, 1, 2, 0);
    let samples: Vec<i32> = (0..(fw * fh) as usize)
        .map(|i| (i * 11 + 3) as i32 & 0xFF)
        .collect();
    let frame = make_gray_decoded_frame(samples, fw, fh, 8);
    let bytes =
        encode_frame(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let bytes = rewrite_single_slice_error_status(&bytes, 2);
    let dims = FramePixelDimensions::new(fw, fh).unwrap();

    // CRC=Accept, ErrorStatus=Reject (the "mixed" policy): still aborts.
    let mixed = DecodeOptions {
        slice_crc_policy: SliceCrcPolicy::Accept,
        slice_error_status_policy: SliceErrorStatusPolicy::Reject,
    };
    assert!(matches!(
        decode_frame_with_options(&bytes, &cr, &qts, dims, true, mixed),
        Err(Error::SliceErrorStatus { .. })
    ));

    // CRC=Reject, ErrorStatus=Accept: passes (CRC residue is zero by
    // construction, status is `Uncorrectable` but the gate is lenient).
    let inverse = DecodeOptions {
        slice_crc_policy: SliceCrcPolicy::Reject,
        slice_error_status_policy: SliceErrorStatusPolicy::Accept,
    };
    let decoded = decode_frame_with_options(&bytes, &cr, &qts, dims, true, inverse)
        .expect("ErrorStatus=Accept must not abort when CRC residue is zero");
    assert_eq!(decoded.planes.len(), 1);
}

// -- RGB / line-major: per-Slice §4.9.2 gate --------------------------

#[test]
fn rgb_gate_reject_aborts_accept_passes_with_bit_exact_decode() {
    // Mirror on the RGB / line-major driver
    // (`decode_frame_rgb_with_options`) — independent plumbing path
    // through §4.7 line-major traversal, so the gate's wiring needs an
    // independent assertion. The body is untouched, so the Accept
    // path decodes back to the original R/G/B Sample planes bit-exact.
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
    let frame = make_rgb_decoded_frame(r.clone(), g.clone(), b.clone(), fw, fh, 8);
    let bytes =
        encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let bytes = rewrite_single_slice_error_status(&bytes, 2);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();

    // Reject (default + strict + legacy entry point): aborts.
    match decode_frame_rgb_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::default()) {
        Err(Error::SliceErrorStatus {
            slice_index,
            status,
        }) => {
            assert_eq!(slice_index, 0);
            assert_eq!(status, 2);
        }
        other => panic!("expected SliceErrorStatus on RGB Reject path, got {other:?}"),
    }
    assert!(matches!(
        decode_frame_rgb_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict()),
        Err(Error::SliceErrorStatus { .. })
    ));
    assert!(matches!(
        decode_frame_rgb(&bytes, &cr, &qts, dims, true),
        Err(Error::SliceErrorStatus { .. })
    ));

    // Accept: returns a bit-exact R/G/B DecodedFrame.
    let decoded =
        decode_frame_rgb_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::lenient())
            .expect("Accept policy must not abort on §4.9.2 Uncorrectable");
    assert_eq!(decoded.planes.len(), 3, "RGB frame has three Planes");
    assert_eq!(decoded.colorspace, ColorspaceType::Rgb);
    assert_eq!(decoded.planes[0].samples, r);
    assert_eq!(decoded.planes[1].samples, g);
    assert_eq!(decoded.planes[2].samples, b);
    for plane in &decoded.planes {
        assert_eq!(plane.width, fw);
        assert_eq!(plane.height, fh);
    }
    assert!(samples_in_range(&decoded.planes, 8));
}

#[test]
fn rgb_gate_correctable_status_passes_under_reject() {
    // RGB mirror of the YCbCr `Correctable` (`1`) test — `1` is not a
    // rejection target per §4.9.2 Table 16.
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
    let frame = make_rgb_decoded_frame(r.clone(), g.clone(), b.clone(), fw, fh, 8);
    let bytes =
        encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).expect("encode");

    let bytes = rewrite_single_slice_error_status(&bytes, 1);

    let dims = FramePixelDimensions::new(fw, fh).unwrap();
    let decoded =
        decode_frame_rgb_with_options(&bytes, &cr, &qts, dims, true, DecodeOptions::strict())
            .expect("Reject must accept §4.9.2 Correctable per Table 16");
    assert_eq!(decoded.planes[0].samples, r);
    assert_eq!(decoded.planes[1].samples, g);
    assert_eq!(decoded.planes[2].samples, b);
}
