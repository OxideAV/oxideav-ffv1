//! FFV1 **version 2** is reserved / experimental (RFC 9043 §4.2.1
//! Table 5): a conforming encoder never emits it. A decoder must
//! therefore reject a version-2 stream with a typed error rather than
//! mis-parsing it as a neighbouring version or panicking.
//!
//! This gate uses the real `v2-multislice-2x2` fixture's Configuration
//! Record (the one fixture in `docs/video/ffv1/fixtures/` deliberately
//! excluded from the decode corpus because FFV1 reserves version 2). The
//! 41-byte record was extracted black-box from the fixture's `input.mkv`
//! Matroska `CodecPrivate`; its first §3.8.1 range-coded `version` symbol
//! decodes to 2.

use oxideav_ffv1::{parse_configuration_record, parse_quantization_table_sets, Error};

/// `v2-multislice-2x2/input.mkv` Matroska `CodecPrivate` (the FFV1
/// Configuration Record). Extracted by black-box EBML parsing, which is
/// independent of the FFV1 bitstream.
const V2_CONFIG_RECORD: &[u8] = &[
    0x42, 0xc1, 0xa5, 0x5d, 0x67, 0x16, 0xf2, 0xa6, 0x96, 0x5d, 0x67, 0x0e, 0x17, 0xb3, 0x24, 0xfd,
    0x8e, 0xee, 0x64, 0xc9, 0xa4, 0xc0, 0xcb, 0xc2, 0xed, 0x02, 0xe9, 0xd0, 0x31, 0xc8, 0x6d, 0xa9,
    0x55, 0xf7, 0x73, 0x50, 0x86, 0xf1, 0xba, 0xdb, 0xab,
];

#[test]
fn v2_configuration_record_rejected_as_unsupported_version() {
    // The Parameters parser reads the §4.2.1 `version` symbol first; a
    // value of 2 is reserved (§4.2.1 Table 5) and must surface as the
    // typed `UnsupportedVersion(2)`, not a panic or a mis-mapped version.
    match parse_configuration_record(V2_CONFIG_RECORD) {
        Err(Error::UnsupportedVersion(2)) => {}
        other => panic!("expected UnsupportedVersion(2), got {other:?}"),
    }
}

#[test]
fn v2_quant_table_cascade_rejected_as_unsupported_version() {
    // The cascade-aware entry point (Configuration Record + §4.1
    // Quantization Table Sets) must reject the reserved version at the
    // same point, before touching the quant tables.
    match parse_quantization_table_sets(V2_CONFIG_RECORD) {
        Err(Error::UnsupportedVersion(2)) => {}
        other => panic!("expected UnsupportedVersion(2), got {other:?}"),
    }
}
