//! End-to-end reference-fixture decode tests: decode each FFV1 fixture's
//! coded Frame and assert the reconstructed Planes are bit-exact against
//! the reference decoder's `expected.raw` (RFC 9043 — the fixtures were
//! produced by the FFV1 reference encoder/decoder pinned in
//! `docs/video/ffv1/ffv1-fixtures-and-traces.md`).
//!
//! These exercise the v3 driver ([`decode_frame`] / [`decode_frame_rgb`])
//! and the v0/v1 single-stream driver ([`decode_frame_v0v1`]) across
//! colour layouts and bit depths the unit / self-round-trip suites cover
//! only synthetically:
//!
//! * `v3-flat-color` — 8-bit YUV 4:2:0, range coder, extreme low entropy
//!   (run-mode / zero-residual range paths).
//! * `v3-yuv422p10` — 10-bit YUV 4:2:2 (`log2_h_chroma_subsample == 1`,
//!   `log2_v_chroma_subsample == 0`), range coder.
//! * `v3-yuv420p12` — 12-bit YUV 4:2:0, range coder.
//! * `v3-rgba` — 8-bit RGB + alpha (`transparency == 1`, four Planes),
//!   JPEG 2000 RCT, range coder.
//! * `v3-context-1` — 8-bit YUV 4:2:0, the large `-context 1`
//!   Quantization Table Set (5-input contexts, ~7563 contexts in set 1).
//! * `v0-yuv420-rangecoder` — FFV1 version 0, inline Parameters, range
//!   coder, single implied Slice.
//! * `v1-single-slice` — FFV1 version 1, inline Parameters, range coder,
//!   single Slice, 128×96.
//!
//! Fixture bytes (coded Frame + extradata + `expected.raw`) are inlined
//! from `tests/data/reference_fixtures.rs`; see that module header for the
//! black-box extraction procedure.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, decode_frame_v0v1, parse_quantization_table_sets,
    FramePixelDimensions,
};

#[path = "data/reference_fixtures.rs"]
mod fx;

/// Decode a v3 YCbCr / plane-major fixture and assert each Plane matches
/// the reference `expected.raw`.
fn assert_v3_ycbcr(extra: &[u8], frame: &[u8], w: u32, h: u32, expected: [&[i32]; 3], label: &str) {
    let parsed = parse_quantization_table_sets(extra)
        .unwrap_or_else(|e| panic!("{label}: extradata: {e:?}"));
    let dims = FramePixelDimensions::new(w, h).expect("dims");
    let decoded = decode_frame(
        frame,
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        parsed.record.ec.is_some(),
    )
    .unwrap_or_else(|e| panic!("{label}: decode: {e:?}"));
    assert_eq!(decoded.planes.len(), 3, "{label}: plane count");
    assert_eq!(decoded.planes[0].samples, expected[0], "{label}: Y plane");
    assert_eq!(decoded.planes[1].samples, expected[1], "{label}: Cb plane");
    assert_eq!(decoded.planes[2].samples, expected[2], "{label}: Cr plane");
}

#[test]
fn v3_flat_color_decodes_bit_exact() {
    assert_v3_ycbcr(
        fx::FLAT_EXTRA,
        fx::FLAT_FRAME,
        64,
        48,
        [fx::FLAT_Y, fx::FLAT_U, fx::FLAT_V],
        "v3-flat-color",
    );
}

#[test]
fn v3_yuv422p10_decodes_bit_exact() {
    assert_v3_ycbcr(
        fx::P10_EXTRA,
        fx::P10_FRAME,
        64,
        48,
        [fx::P10_Y, fx::P10_U, fx::P10_V],
        "v3-yuv422p10",
    );
}

#[test]
fn v3_yuv420p12_decodes_bit_exact() {
    assert_v3_ycbcr(
        fx::P12_EXTRA,
        fx::P12_FRAME,
        64,
        48,
        [fx::P12_Y, fx::P12_U, fx::P12_V],
        "v3-yuv420p12",
    );
}

#[test]
fn v3_context_1_decodes_bit_exact() {
    assert_v3_ycbcr(
        fx::CTX_EXTRA,
        fx::CTX_FRAME,
        128,
        96,
        [fx::CTX_Y, fx::CTX_U, fx::CTX_V],
        "v3-context-1",
    );
}

#[test]
fn v3_rgba_decodes_bit_exact() {
    // RGB + alpha: decode_frame_rgb returns R, G, B, A Planes (the
    // decoder's natural Plane order; expected.raw is packed BGRA, so the
    // data module unpacked each channel into a separate Plane buffer).
    let parsed = parse_quantization_table_sets(fx::RGBA_EXTRA).expect("rgba extradata");
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame_rgb(
        fx::RGBA_FRAME,
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        parsed.record.ec.is_some(),
    )
    .expect("rgba decode");
    assert_eq!(
        decoded.planes.len(),
        4,
        "v3-rgba: plane count (RGB + alpha)"
    );
    assert_eq!(decoded.planes[0].samples, fx::RGBA_R, "v3-rgba: R plane");
    assert_eq!(decoded.planes[1].samples, fx::RGBA_G, "v3-rgba: G plane");
    assert_eq!(decoded.planes[2].samples, fx::RGBA_B, "v3-rgba: B plane");
    assert_eq!(
        decoded.planes[3].samples,
        fx::RGBA_A,
        "v3-rgba: alpha plane"
    );
}

#[test]
fn v0_yuv420_rangecoder_decodes_bit_exact() {
    // FFV1 version 0: Parameters are inline in the Frame (no extradata),
    // so decode_frame_v0v1 takes only the Frame + pixel dimensions.
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame_v0v1(fx::V0RC_FRAME, dims).expect("v0 rangecoder decode");
    assert_eq!(decoded.planes.len(), 3, "v0-rangecoder: plane count");
    assert_eq!(
        decoded.planes[0].samples,
        fx::V0RC_Y,
        "v0-rangecoder: Y plane"
    );
    assert_eq!(
        decoded.planes[1].samples,
        fx::V0RC_U,
        "v0-rangecoder: Cb plane"
    );
    assert_eq!(
        decoded.planes[2].samples,
        fx::V0RC_V,
        "v0-rangecoder: Cr plane"
    );
}

#[test]
fn v1_single_slice_decodes_bit_exact() {
    let dims = FramePixelDimensions::new(128, 96).expect("dims");
    let decoded = decode_frame_v0v1(fx::V1SS_FRAME, dims).expect("v1 single-slice decode");
    assert_eq!(decoded.planes.len(), 3, "v1-single-slice: plane count");
    assert_eq!(
        decoded.planes[0].samples,
        fx::V1SS_Y,
        "v1-single-slice: Y plane"
    );
    assert_eq!(
        decoded.planes[1].samples,
        fx::V1SS_U,
        "v1-single-slice: Cb plane"
    );
    assert_eq!(
        decoded.planes[2].samples,
        fx::V1SS_V,
        "v1-single-slice: Cr plane"
    );
}
