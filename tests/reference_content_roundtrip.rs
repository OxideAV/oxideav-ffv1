//! Encode round-trip over **real reference-fixture content**.
//!
//! The self-round-trip suites ([`v0v1_roundtrip`], the v3 encode tests)
//! drive the encoder with synthetic SplitMix Samples. Those exercise the
//! predictor / context / entropy machinery, but their residual
//! distribution is white-noise-like: every §3.8.2 Golomb-Rice run region
//! is short and the §3.8.2.4 adaptive `k` parameter barely moves.
//!
//! Real footage (the `testsrc2` gradient/edge pattern the fixtures were
//! encoded from) produces long low-entropy runs, monotone gradients that
//! keep the §3.5 context near zero, and sharp edges that spike the Sample
//! Difference — a genuinely different residual profile. This test closes
//! the encode loop over that real content: it decodes a reference fixture
//! to recover the ground-truth Planes, re-encodes them with the crate's
//! encoder, decodes the re-encoded stream, and asserts the Planes survive
//! bit-exactly (RFC 9043 §1 lossless identity: `decode(encode(x)) == x`).

use oxideav_ffv1::{
    decode_frame_v0v1, encode_frame_v0v1, parse_quantization_table_sets, ColorspaceType,
    DecodedFrame, Ffv1ConfigurationRecord, Ffv1Version, FramePixelDimensions, QuantizationTableSet,
};

// This test only reads the `V1GR_*` fixtures from the shared module; the
// remaining fixture constants are consumed by `reference_fixture_decode.rs`.
#[allow(dead_code)]
#[path = "data/reference_fixtures.rs"]
mod fx;

/// `v3-default` extradata — its first parsed Quantization Table Set is a
/// real, well-formed §4.1 cascade (`context_count == 666`) reused as the
/// single implied v0/v1 Set (same source as `tests/v0v1_roundtrip.rs`).
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

fn real_quant_table_set() -> QuantizationTableSet {
    let parsed = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).expect("parse v3-default");
    parsed.quant_table_sets[0].clone()
}

/// Build a version-1 YUV 4:2:0 Configuration Record with the given coder
/// type (0 = Golomb-Rice, 1 = range default).
fn v1_yuv420_record(coder_type: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V1,
        micro_version: None,
        coder_type,
        state_transition_delta: [0i32; 256],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: 8,
        chroma_planes: true,
        log2_h_chroma_subsample: 1,
        log2_v_chroma_subsample: 1,
        extra_plane: false,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: None,
        ec: None,
        intra: None,
        initial_state_delta: None,
    }
}

/// Rebuild a keyframe `DecodedFrame` from the three reference Planes so
/// it can be handed back to the encoder.
fn frame_from_planes(y: &[i32], u: &[i32], v: &[i32]) -> DecodedFrame {
    use oxideav_ffv1::DecodedFramePlane;
    DecodedFrame {
        planes: vec![
            DecodedFramePlane {
                plane_index: 0,
                width: 64,
                height: 48,
                samples: y.to_vec(),
            },
            DecodedFramePlane {
                plane_index: 1,
                width: 32,
                height: 24,
                samples: u.to_vec(),
            },
            DecodedFramePlane {
                plane_index: 2,
                width: 32,
                height: 24,
                samples: v.to_vec(),
            },
        ],
        width: 64,
        height: 48,
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// Decode the v1-golomb reference fixture, re-encode the recovered Planes
/// through the crate's §3.8.2 Golomb-Rice encoder, decode the result, and
/// assert the Planes are bit-exact — the lossless identity over the real
/// `testsrc2` residual distribution, not synthetic noise.
#[test]
fn v1_golomb_real_content_reencodes_lossless() {
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let reference =
        decode_frame_v0v1(fx::V1GR_FRAME, dims).expect("decode v1-golomb reference fixture");

    let cr = v1_yuv420_record(0);
    let qts = real_quant_table_set();
    let frame = frame_from_planes(fx::V1GR_Y, fx::V1GR_U, fx::V1GR_V);

    let encoded = encode_frame_v0v1(&frame, &cr, &qts).expect("re-encode golomb");
    let redecoded = decode_frame_v0v1(&encoded, dims).expect("re-decode golomb");

    assert_eq!(redecoded.planes.len(), 3, "plane count after re-encode");
    assert_eq!(
        redecoded.planes[0].samples, reference.planes[0].samples,
        "Y"
    );
    assert_eq!(
        redecoded.planes[1].samples, reference.planes[1].samples,
        "Cb"
    );
    assert_eq!(
        redecoded.planes[2].samples, reference.planes[2].samples,
        "Cr"
    );
}

/// The same real content, re-encoded through the §3.8.1 range coder
/// (`coder_type == 1`) instead of Golomb-Rice, must also round-trip
/// bit-exactly — the two entropy back-ends over the identical real Planes.
#[test]
fn v1_golomb_real_content_reencodes_lossless_range() {
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let reference = decode_frame_v0v1(fx::V1GR_FRAME, dims).expect("decode v1-golomb reference");

    let cr = v1_yuv420_record(1);
    let qts = real_quant_table_set();
    let frame = frame_from_planes(fx::V1GR_Y, fx::V1GR_U, fx::V1GR_V);

    let encoded = encode_frame_v0v1(&frame, &cr, &qts).expect("re-encode range");
    let redecoded = decode_frame_v0v1(&encoded, dims).expect("re-decode range");

    assert_eq!(
        redecoded.planes[0].samples, reference.planes[0].samples,
        "Y"
    );
    assert_eq!(
        redecoded.planes[1].samples, reference.planes[1].samples,
        "Cb"
    );
    assert_eq!(
        redecoded.planes[2].samples, reference.planes[2].samples,
        "Cr"
    );
}
