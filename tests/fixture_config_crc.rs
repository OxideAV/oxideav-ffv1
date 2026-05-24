//! Fixture-based validation of the RFC 9043 §4.3.2
//! `configuration_record_crc_parity` check against the workspace FFV1
//! corpus (`docs/video/ffv1/fixtures/`).
//!
//! Each extradata blob is the Matroska CodecPrivate of a fixture's
//! `input.mkv`. A conforming Configuration Record CRCs to `0` over the
//! whole blob (§4.3.2: "the Configuration Record as a whole has a CRC
//! remainder of zero"). The reference decoder reports this in the
//! `trace.txt` `GLOBAL_HEADER` event as `crcref=0x00000000` — the
//! residue it computes. A clean-room CRC that reproduces the same `0`
//! residue over the same bytes has the §4.9.3 generator (poly
//! `0x104C11DB7`, init 0, no inversion) exactly right.

use oxideav_ffv1::validate_configuration_record_crc;

/// `v3-default` extradata — Matroska CodecPrivate of
/// `docs/video/ffv1/fixtures/v3-default/input.mkv`. Trace
/// `GLOBAL_HEADER`: `crcref=0x00000000`.
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

/// `v3-grayscale` extradata. Trace: `crcref=0x00000000`.
const V3_GRAYSCALE_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x2f, 0xd3, 0xc8, 0x18, 0xce, 0x09, 0xeb, 0x7f, 0x68, 0x23, 0xd0, 0x46, 0xc2, 0x44,
    0x28, 0x0a, 0x38, 0x20, 0x41, 0x1c, 0x8f, 0xfd, 0x0b, 0xd7, 0xa0, 0xdd, 0x7d, 0xc7, 0xe2, 0xbe,
    0x16, 0x99, 0xb1, 0xe0, 0xb7, 0x06, 0x5a, 0x9c, 0x7e, 0x09,
];

/// `v3-rgb-bgr0` extradata (colorspace=1, RCT). Trace: `crcref=0x00000000`.
const V3_RGB_BGR0_EXTRADATA: &[u8] = &[
    0x55, 0xf6, 0x46, 0x87, 0xe6, 0xa9, 0xc1, 0x7b, 0x87, 0xbf, 0x82, 0x5e, 0xd8, 0x30, 0x2b, 0x95,
    0x12, 0x2e, 0xcf, 0x70, 0xe2, 0x0f, 0x76, 0xbc, 0x04, 0x17, 0x6c, 0xd6, 0x60, 0xd4, 0x99, 0xbf,
    0x4f, 0x95, 0xdf, 0x58, 0xfb, 0x51, 0xd1, 0x16, 0xf4, 0xad,
];

/// `v3-yuv444p16` extradata (16-bit, 52-byte). Trace: `crcref=0x00000000`.
/// Extracted black-box via `ffprobe -show_data` on `input.mkv`.
const V3_YUV444P16_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x68, 0x5d, 0x47, 0x52, 0x37, 0x13, 0xbc, 0x20, 0x1a, 0xc2, 0xff, 0xe7, 0x15, 0x23,
    0x7c, 0xf2, 0x14, 0xef, 0x39, 0xf4, 0xfb, 0xf7, 0xb2, 0x45, 0x8d, 0x81, 0x0f, 0x64, 0x37, 0x36,
    0x78, 0x68, 0xf8, 0xf5, 0x10, 0x25, 0xe0, 0x19, 0xff, 0x63, 0xbc, 0x94, 0xce, 0xbe, 0x50, 0xc7,
    0x12, 0x9a, 0xd7, 0xb8,
];

#[test]
fn v3_default_crc_is_valid() {
    validate_configuration_record_crc(V3_DEFAULT_EXTRADATA)
        .expect("v3-default CRC residue must be 0 (trace crcref=0x00000000)");
}

#[test]
fn v3_grayscale_crc_is_valid() {
    validate_configuration_record_crc(V3_GRAYSCALE_EXTRADATA)
        .expect("v3-grayscale CRC residue must be 0");
}

#[test]
fn v3_rgb_bgr0_crc_is_valid() {
    validate_configuration_record_crc(V3_RGB_BGR0_EXTRADATA)
        .expect("v3-rgb-bgr0 CRC residue must be 0");
}

#[test]
fn v3_yuv444p16_crc_is_valid() {
    validate_configuration_record_crc(V3_YUV444P16_EXTRADATA)
        .expect("v3-yuv444p16 CRC residue must be 0 (16-bit, 52-byte record)");
}

#[test]
fn flipped_byte_is_rejected() {
    // Corrupting any single byte of a valid record must make the CRC
    // residue non-zero — the whole point of the §4.3.2 fixity check.
    let mut corrupt = V3_DEFAULT_EXTRADATA.to_vec();
    corrupt[10] ^= 0x40;
    assert!(
        validate_configuration_record_crc(&corrupt).is_err(),
        "a corrupted Configuration Record must fail the CRC check"
    );
}

#[test]
fn truncated_parity_is_rejected() {
    // A record too short to even hold the 4-byte parity field is
    // structurally invalid.
    assert!(validate_configuration_record_crc(&[0x56, 0x00, 0x30]).is_err());
}
