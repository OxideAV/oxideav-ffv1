//! Slice-footer parser fixture tests for FFV1 v3 streams (RFC 9043 §4.9).
//!
//! Each `<fixture>_FULL_SLICE<n>` constant below is the *whole* Slice
//! byte range — SliceHeader + SliceContent + (Golomb-Rice padding) +
//! the §4.9 Slice Footer (`u(24)` slice_size + `u(8)` error_status +
//! `u(32)` slice_crc_parity, since every fixture has `ec=1`).
//!
//! Extraction is purely black-box and matches the procedure in
//! `tests/fixture_slice_header.rs`, except the slice byte range here
//! *includes* the 8-byte footer:
//!
//! 1. `ffmpeg -i input.mkv -c copy -map 0:v -f rawvideo frame.bin`
//!    yields the raw FFV1 frame payload.
//! 2. Walk the §4.9.1 trailer-pointer chain backwards from the end of
//!    the frame: each iteration reads the `u(24)` `slice_size` field
//!    (the 3 bytes at offset `end - 8`) and carves off
//!    `slice_size + 8` bytes as the whole Slice.
//! 3. Each constant is one such whole-Slice byte range.
//!
//! Ground truth: each fixture's `trace.txt` `SLICE` event carries
//! `len=<total>` (the whole-Slice size including the 8-byte footer) and
//! `header_crc=0x<8-hex>` (the §4.9.3 `slice_crc_parity`). The parser
//! must (a) read the same `slice_size` field, (b) report the same
//! parity, and (c) confirm the §4.9.3 whole-Slice CRC residue is zero.

use oxideav_ffv1::{parse_slice_footer, Error, SliceErrorStatus};

include!("data/slice_footer_fixtures.rs");

/// Helper: assert a whole-Slice parses, has `ec=1` metadata, the
/// expected `slice_size` (body, footer-excluded) and parity, and a
/// zero §4.9.3 residue.
fn assert_ec1_slice(full: &[u8], want_size: u32, want_parity: u32) {
    let f = parse_slice_footer(full, true).expect("ec=1 slice footer validates");
    assert_eq!(
        f.slice_size, want_size,
        "slice_size (body, footer-excluded)"
    );
    assert_eq!(f.total_size, full.len() as u32, "total on-wire size");
    assert_eq!(f.slice_crc_parity, Some(want_parity), "slice_crc_parity");
    assert_eq!(f.footer_len(), 8, "ec=1 footer is 8 bytes");
    // testsrc → no decode error, so error_status is 0.
    assert_eq!(f.error_status, Some(SliceErrorStatus::NoError));
    assert_eq!(f.error_status_raw, Some(0));
}

// ---- v3-default (8-bit YUV 4:2:0, 2x2 slices) ----------------------

#[test]
fn v3_default_slice0_footer() {
    assert_ec1_slice(V3_DEFAULT_FULL_SLICE0, 229, 0xCB53_0827);
}

#[test]
fn v3_default_slice1_footer() {
    assert_ec1_slice(V3_DEFAULT_FULL_SLICE1, 308, 0xC930_79C7);
}

#[test]
fn v3_default_slice2_footer() {
    assert_ec1_slice(V3_DEFAULT_FULL_SLICE2, 552, 0xB892_3B4F);
}

#[test]
fn v3_default_slice3_footer() {
    assert_ec1_slice(V3_DEFAULT_FULL_SLICE3, 572, 0x42C8_841D);
}

/// Cross-fixture sweep: all four v3-default slices validate their
/// §4.9.3 CRC and reproduce the trace's `header_crc` parity values.
#[test]
fn v3_default_all_slice_crcs_match_trace() {
    let cases = [
        (V3_DEFAULT_FULL_SLICE0 as &[u8], 229u32, 0xCB53_0827u32),
        (V3_DEFAULT_FULL_SLICE1, 308, 0xC930_79C7),
        (V3_DEFAULT_FULL_SLICE2, 552, 0xB892_3B4F),
        (V3_DEFAULT_FULL_SLICE3, 572, 0x42C8_841D),
    ];
    for (full, size, parity) in cases {
        assert_ec1_slice(full, size, parity);
    }
}

// ---- Other colorspaces / bit depths -------------------------------

#[test]
fn v3_grayscale_slice0_footer() {
    assert_ec1_slice(V3_GRAYSCALE_FULL_SLICE0, 45, 0x44C7_D58E);
}

#[test]
fn v3_rgb_bgr0_slice0_footer() {
    assert_ec1_slice(V3_RGB_BGR0_FULL_SLICE0, 89, 0x3BBF_E098);
}

#[test]
fn v3_yuv444p16_slice0_footer() {
    assert_ec1_slice(V3_YUV444P16_FULL_SLICE0, 204, 0x0AD9_80DC);
}

// ---- Negative paths against real fixture bytes --------------------

/// Flipping one body byte of a real slice makes the §4.9.3 residue
/// non-zero; the validator surfaces both residue and the (unchanged)
/// stored parity.
#[test]
fn v3_default_slice0_corrupted_body_rejected() {
    let mut full = V3_DEFAULT_FULL_SLICE0.to_vec();
    full[10] ^= 0x01;
    match parse_slice_footer(&full, true) {
        Err(Error::SliceCrcMismatch {
            residue,
            stored_parity,
        }) => {
            assert_ne!(residue, 0);
            assert_eq!(stored_parity, 0xCB53_0827);
        }
        other => panic!("expected SliceCrcMismatch, got {other:?}"),
    }
}

/// Flipping a byte in the stored parity word is detected too.
#[test]
fn v3_default_slice0_corrupted_parity_rejected() {
    let mut full = V3_DEFAULT_FULL_SLICE0.to_vec();
    let last = full.len() - 1;
    full[last] ^= 0x80;
    match parse_slice_footer(&full, true) {
        Err(Error::SliceCrcMismatch { residue, .. }) => assert_ne!(residue, 0),
        other => panic!("expected SliceCrcMismatch, got {other:?}"),
    }
}

/// Dropping the last 3 bytes of a real slice shifts the footer window
/// so the `u(24)` `slice_size` field is read from the wrong bytes,
/// which no longer equals `buffer_len - 8`. The structural check
/// rejects it before any CRC work.
#[test]
fn v3_grayscale_slice0_truncated_footer_rejected() {
    let truncated = &V3_GRAYSCALE_FULL_SLICE0[..V3_GRAYSCALE_FULL_SLICE0.len() - 3];
    match parse_slice_footer(truncated, true) {
        Err(Error::SliceSizeOutOfRange { expected, .. }) => {
            // The misaligned size field can't match the (footer-
            // excluded) expected body length.
            assert_eq!(expected, (truncated.len() - 8) as u32);
        }
        other => panic!("expected SliceSizeOutOfRange, got {other:?}"),
    }
}

/// Parsing a real ec=1 slice as if `ec=0` reads the WRONG 3 bytes as
/// the slice_size (the parity tail, not the size field), so the
/// structural check rejects it. This documents that the caller MUST
/// pass the correct `ec` flag.
#[test]
fn v3_default_slice0_wrong_ec_flag_rejected() {
    // ec=0 would read the trailing 3 bytes (0x53, 0x08, 0x27 → a huge
    // value) as slice_size, which can't equal `len - 3`.
    assert!(matches!(
        parse_slice_footer(V3_DEFAULT_FULL_SLICE0, false),
        Err(Error::SliceSizeOutOfRange { .. })
    ));
}
