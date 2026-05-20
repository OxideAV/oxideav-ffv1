//! Test the configuration-record parser against real FFmpeg-encoded
//! FFV1 v3 fixtures extracted from `docs/video/ffv1/fixtures/`.
//!
//! Each extradata blob is the FFV1 Configuration Record from the
//! `input.mkv` file's Matroska CodecPrivate element. The expected
//! parsed values come from the fixture's `trace.txt` GLOBAL_HEADER
//! event (which is the authoritative reference for the bitstream's
//! header content).
//!
//! All bytes here were obtained by black-box invocations: the
//! Matroska container parser is independent of FFV1, and `ffprobe`'s
//! `extradata` is the raw Configuration Record bytes.

use oxideav_ffv1::{
    parse_configuration_record, ColorspaceType, Error, Ffv1ConfigurationRecord, Ffv1Version,
    PictureStructure,
};

/// FFV1 v3 / micro 4 / range coder default / 8-bit YUV 4:2:0 / 2x2
/// slices / 2 quant table sets.
///
/// Source: `docs/video/ffv1/fixtures/v3-default/input.mkv` Matroska
/// CodecPrivate. Trace ground truth:
///
/// ```text
/// GLOBAL_HEADER  extradata_size=42  version=3  micro=4  coder=1
///                colorspace=0  bits_per_raw_sample=8  chroma_planes=1
///                chroma_h_shift=1  chroma_v_shift=1  transparency=0
///                num_h_slices=2  num_v_slices=2  quant_table_count=2
///                ec=1  intra=0  flt=0  crcref=0x00000000
/// ```
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

/// FFV1 v3 / micro 4 / range coder default / 8-bit RGB (BGR0) / 1x1
/// slices.
///
/// Source: `docs/video/ffv1/fixtures/v3-rgb-bgr0/input.mkv` Matroska
/// CodecPrivate.
const V3_RGB_BGR0_EXTRADATA: &[u8] = &[
    0x55, 0xf6, 0x46, 0x87, 0xe6, 0xa9, 0xc1, 0x7b, 0x87, 0xbf, 0x82, 0x5e, 0xd8, 0x30, 0x2b, 0x95,
    0x12, 0x2e, 0xcf, 0x70, 0xe2, 0x0f, 0x76, 0xbc, 0x04, 0x17, 0x6c, 0xd6, 0x60, 0xd4, 0x99, 0xbf,
    0x4f, 0x95, 0xdf, 0x58, 0xfb, 0x51, 0xd1, 0x16, 0xf4, 0xad,
];

/// FFV1 v3 / micro 4 / 8-bit grayscale / single plane.
///
/// Source: `docs/video/ffv1/fixtures/v3-grayscale/input.mkv` Matroska
/// CodecPrivate.
const V3_GRAYSCALE_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x2f, 0xd3, 0xc8, 0x18, 0xce, 0x09, 0xeb, 0x7f, 0x68, 0x23, 0xd0, 0x46, 0xc2, 0x44,
    0x28, 0x0a, 0x38, 0x20, 0x41, 0x1c, 0x8f, 0xfd, 0x0b, 0xd7, 0xa0, 0xdd, 0x7d, 0xc7, 0xe2, 0xbe,
    0x16, 0x99, 0xb1, 0xe0, 0xb7, 0x06, 0x5a, 0x9c, 0x7e, 0x09,
];

#[test]
fn decodes_v3_default_extradata() {
    let cr: Ffv1ConfigurationRecord =
        parse_configuration_record(V3_DEFAULT_EXTRADATA).expect("v3-default parses");
    assert_eq!(cr.version, Ffv1Version::V3);
    assert_eq!(cr.micro_version, Some(4));
    assert_eq!(cr.coder_type, 1);
    assert_eq!(cr.colorspace_type, ColorspaceType::YCbCr);
    assert_eq!(cr.bits_per_raw_sample, 8);
    assert!(cr.chroma_planes);
    assert_eq!(cr.log2_h_chroma_subsample, 1);
    assert_eq!(cr.log2_v_chroma_subsample, 1);
    assert!(!cr.extra_plane);
    assert_eq!(cr.num_h_slices, Some(2));
    assert_eq!(cr.num_v_slices, Some(2));
    assert_eq!(cr.quant_table_set_count, Some(2));
    // Default coder → no custom state-transition deltas.
    assert!(cr.state_transition_delta.iter().all(|&d| d == 0));
}

#[test]
fn decodes_v3_rgb_bgr0_extradata() {
    let cr = parse_configuration_record(V3_RGB_BGR0_EXTRADATA).expect("v3-rgb-bgr0 parses");
    assert_eq!(cr.version, Ffv1Version::V3);
    assert_eq!(cr.micro_version, Some(4));
    assert_eq!(cr.coder_type, 1);
    assert_eq!(cr.colorspace_type, ColorspaceType::Rgb);
    assert_eq!(cr.bits_per_raw_sample, 8);
    // RGB requires chroma_planes=1 and zero chroma subsample per
    // RFC 9043 §4.2.5.
    assert!(cr.chroma_planes);
    assert_eq!(cr.log2_h_chroma_subsample, 0);
    assert_eq!(cr.log2_v_chroma_subsample, 0);
    // BGR0 — alpha is dropped, transparency=0 in trace.
    assert!(!cr.extra_plane);
}

#[test]
fn decodes_v3_grayscale_extradata() {
    let cr = parse_configuration_record(V3_GRAYSCALE_EXTRADATA).expect("v3-grayscale parses");
    assert_eq!(cr.version, Ffv1Version::V3);
    assert_eq!(cr.micro_version, Some(4));
    assert_eq!(cr.coder_type, 1);
    assert_eq!(cr.colorspace_type, ColorspaceType::YCbCr);
    assert_eq!(cr.bits_per_raw_sample, 8);
    // Grayscale → only the Y plane, chroma_planes=0.
    assert!(!cr.chroma_planes);
    assert!(!cr.extra_plane);
}

#[test]
fn picture_structure_enum_from_wire() {
    // §4.6.7 Table 15: 0=unknown, 1=TFF, 2=BFF, 3=progressive.
    assert_eq!(
        PictureStructure::from_wire(0),
        Ok(PictureStructure::Unknown)
    );
    assert_eq!(
        PictureStructure::from_wire(1),
        Ok(PictureStructure::TopFieldFirst)
    );
    assert_eq!(
        PictureStructure::from_wire(2),
        Ok(PictureStructure::BottomFieldFirst)
    );
    assert_eq!(
        PictureStructure::from_wire(3),
        Ok(PictureStructure::Progressive)
    );
    assert_eq!(PictureStructure::from_wire(42), Err(42));
}

#[test]
fn rejects_truncated_extradata() {
    // Less than the two bytes needed to seed the range coder.
    assert!(matches!(
        parse_configuration_record(&[]),
        Err(Error::TruncatedRangeCoder)
    ));
    assert!(matches!(
        parse_configuration_record(&[0x55]),
        Err(Error::TruncatedRangeCoder)
    ));
}
