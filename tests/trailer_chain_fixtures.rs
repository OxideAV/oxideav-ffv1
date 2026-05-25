//! End-to-end fixture tests for the round-10 §4.9.1 trailer-pointer
//! chain walk ([`walk_trailer_chain`]).
//!
//! Where the in-crate `trailer_chain::tests` exercise the walker
//! against synthetic Slice byte sequences (constructed by hand to
//! announce specific `slice_size` values), these tests exercise it
//! against the **same** byte payloads the `fixture_slice_footer.rs`
//! suite parses one Slice at a time. The four v3-default whole-Slice
//! byte constants are concatenated end-to-end into a synthetic frame;
//! the chain walker is then asked to recover the four byte ranges
//! and each must match the `parse_slice_footer` ground truth (slice
//! size body, total length, and `slice_crc_parity`).
//!
//! Because the chain walker only reads the `u(24)` `slice_size` field
//! (the §4.9.1 trailer pointer), this is a structural cross-check: the
//! walker hands the same bytes back to `parse_slice_footer` that the
//! existing footer-parser fixture tests confirmed against the
//! `trace.txt` reference; if either side ever drifts, the test catches
//! it.
//!
//! The four v3-default slices report `slice_size` values of 229, 308,
//! 552, 572 — total body 1661 bytes, plus 4 × 8-byte footers = 1693
//! bytes of frame payload. `trace.txt`'s `SLICE` events report `len=237
//! / 316 / 560 / 580`, each being the body + the 8-byte footer; the
//! sum 237+316+560+580 also equals 1693, so the chain walker must
//! discover exactly 4 extents whose total spans the full frame.

use oxideav_ffv1::{parse_slice_footer, walk_trailer_chain};

include!("data/slice_footer_fixtures.rs");

/// v3-default frame: 4 slices, each (body + 8-byte ec=1 footer).
fn v3_default_frame() -> Vec<u8> {
    let mut frame = Vec::new();
    for s in [
        V3_DEFAULT_FULL_SLICE0,
        V3_DEFAULT_FULL_SLICE1,
        V3_DEFAULT_FULL_SLICE2,
        V3_DEFAULT_FULL_SLICE3,
    ] {
        frame.extend_from_slice(s);
    }
    frame
}

/// The chain walker recovers all four v3-default Slice byte ranges in
/// forward slice-index order. Each recovered range is then byte-for-byte
/// equal to the corresponding `*_FULL_SLICE*` constant.
#[test]
fn v3_default_chain_walk_recovers_four_slice_byte_ranges() {
    let frame = v3_default_frame();
    let extents = walk_trailer_chain(&frame, true).expect("v3-default frame chain walks");
    assert_eq!(extents.len(), 4, "v3-default has 4 slices (2x2 raster)");

    let expected_slices = [
        V3_DEFAULT_FULL_SLICE0,
        V3_DEFAULT_FULL_SLICE1,
        V3_DEFAULT_FULL_SLICE2,
        V3_DEFAULT_FULL_SLICE3,
    ];
    let expected_total_lens = [237usize, 316, 560, 580]; // trace.txt `len`s
    let mut expected_start = 0usize;
    for (i, ext) in extents.iter().enumerate() {
        assert_eq!(
            ext.start, expected_start,
            "slice {i} starts where the previous slice ended"
        );
        assert_eq!(
            ext.total_len, expected_total_lens[i],
            "slice {i} total_len matches trace.txt len"
        );
        let recovered = &frame[ext.start..ext.start + ext.total_len];
        assert_eq!(
            recovered, expected_slices[i],
            "byte range slice {i} matches the constant"
        );
        expected_start += ext.total_len;
    }
    assert_eq!(
        expected_start,
        frame.len(),
        "chain walk covers every byte of the frame"
    );
}

/// Each chain-walked byte range parses back to the same
/// `parse_slice_footer` results the `fixture_slice_footer.rs` suite
/// already validates. This is the round-10 ↔ round-7 integration
/// check: the walker delivers exactly what the footer parser needs.
#[test]
fn v3_default_chain_walked_slices_parse_under_slice_footer() {
    let frame = v3_default_frame();
    let extents = walk_trailer_chain(&frame, true).unwrap();
    // (body size, parity) per trace.txt header_crc for each slice.
    let expected = [
        (229u32, 0xCB53_0827u32),
        (308, 0xC930_79C7),
        (552, 0xB892_3B4F),
        (572, 0x42C8_841D),
    ];
    for (i, (ext, (want_size, want_parity))) in extents.iter().zip(expected).enumerate() {
        let buf = &frame[ext.start..ext.start + ext.total_len];
        let footer =
            parse_slice_footer(buf, true).expect("chain-walked slice parses under footer parser");
        assert_eq!(footer.slice_size, want_size, "slice {i} body length");
        assert_eq!(
            footer.slice_crc_parity,
            Some(want_parity),
            "slice {i} parity"
        );
        assert_eq!(
            footer.total_size as usize, ext.total_len,
            "slice {i} total_size echoes the buffer length"
        );
    }
}

/// Single-slice frames (one v3-grayscale slice, one v3-rgb-bgr0 slice,
/// one v3-yuv444p16 slice) round-trip too: the walker recovers exactly
/// one extent whose `start = 0` and `total_len = frame.len()`.
#[test]
fn v3_grayscale_single_slice_chain_walk() {
    let frame: Vec<u8> = V3_GRAYSCALE_FULL_SLICE0.to_vec();
    let extents = walk_trailer_chain(&frame, true).unwrap();
    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0].start, 0);
    assert_eq!(extents[0].total_len, frame.len());
    // Cross-check with the existing footer parser.
    let f = parse_slice_footer(&frame, true).unwrap();
    assert_eq!(f.total_size as usize, frame.len());
    assert_eq!(f.slice_crc_parity, Some(0x44C7_D58E));
}

#[test]
fn v3_rgb_bgr0_single_slice_chain_walk() {
    let frame: Vec<u8> = V3_RGB_BGR0_FULL_SLICE0.to_vec();
    let extents = walk_trailer_chain(&frame, true).unwrap();
    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0].start, 0);
    assert_eq!(extents[0].total_len, frame.len());
}

#[test]
fn v3_yuv444p16_single_slice_chain_walk() {
    let frame: Vec<u8> = V3_YUV444P16_FULL_SLICE0.to_vec();
    let extents = walk_trailer_chain(&frame, true).unwrap();
    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0].start, 0);
    assert_eq!(extents[0].total_len, frame.len());
}
