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
//! * `v3-grayscale` — 8-bit single-plane luma-only (`chroma_planes == 0`,
//!   `plane_count == 1`), range coder, the no-chroma driver path.
//! * `v3-yuv444p16` — 16-bit YUV 4:4:4 (no subsampling,
//!   `bits_per_raw_sample == 16`), range coder, full-precision sample path.
//! * `v3-yuv422p10` — 10-bit YUV 4:2:2 (`log2_h_chroma_subsample == 1`,
//!   `log2_v_chroma_subsample == 0`), range coder.
//! * `v3-yuv420p12` — 12-bit YUV 4:2:0, range coder.
//! * `v3-rgba` — 8-bit RGB + alpha (`transparency == 1`, four Planes),
//!   JPEG 2000 RCT, range coder.
//! * `v3-rgb-bgr0` — 8-bit RGB, no alpha (`transparency == 0`, three
//!   Planes), JPEG 2000 RCT, range coder; the RGB driver without an
//!   alpha Plane.
//! * `v3-default` — 8-bit YUV 4:2:0, 128×96, 2×2 = 4 Slices, per-Slice
//!   CRC; the canonical multi-Slice fixture (slice-grid partition +
//!   §4.9.1 trailer chain).
//! * `v3-multislice-4x4` — 8-bit YUV 4:2:0, 128×96, 4×4 = 16 Slices;
//!   maximum reference-encoder-default slice count.
//! * `v3-frame-mt` — 8-bit YUV 4:2:0, 256×192, 4×4 = 16 Slices; larger
//!   frame, each luma Slice 64×48.
//! * `v3-context-1` — 8-bit YUV 4:2:0, the large `-context 1`
//!   Quantization Table Set (5-input contexts, ~7563 contexts in set 1).
//! * `v0-yuv420-rangecoder` — FFV1 version 0, inline Parameters, range
//!   coder, single implied Slice.
//! * `v0-yuv420-golomb-rice` — FFV1 version 0, inline Parameters,
//!   Golomb-Rice coder (`coder_type == 0`), single implied Slice — the
//!   §3.8.2 adaptive run-length / level-coding decode loop + §3.8.1.1.1
//!   Sentinel-mode range→Golomb byte handoff.
//! * `v1-single-slice` — FFV1 version 1, inline Parameters, range coder,
//!   single Slice, 128×96.
//! * `v1-golomb` — FFV1 version 1, inline Parameters, Golomb-Rice coder
//!   (`coder_type == 0`), 8-bit YUV 4:2:0, 64×48; the §3.8.2 residual
//!   path on a version-1 header (`bits_per_raw_sample` present).
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
fn v3_grayscale_decodes_bit_exact() {
    // FFV1 v3, single-plane luma-only (chroma_planes == 0, transparency == 0
    // -> plane_count == 1), 8-bit, range coder, default state-transition
    // table. Exercises the no-chroma path through the v3 YCbCr driver.
    let parsed = parse_quantization_table_sets(fx::GRAY_EXTRA).expect("gray extradata");
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame(
        fx::GRAY_FRAME,
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        parsed.record.ec.is_some(),
    )
    .expect("gray decode");
    assert_eq!(
        decoded.planes.len(),
        1,
        "v3-grayscale: plane count (luma only)"
    );
    assert_eq!(
        decoded.planes[0].samples,
        fx::GRAY_Y,
        "v3-grayscale: Y plane"
    );
}

#[test]
fn v3_yuv444p16_decodes_bit_exact() {
    // FFV1 v3, 16-bit YUV 4:4:4 (no chroma subsampling,
    // bits_per_raw_sample == 16), range coder, default state-transition
    // table. Exercises the full-precision 16-bit sample path.
    assert_v3_ycbcr(
        fx::Y444P16_EXTRA,
        fx::Y444P16_FRAME,
        64,
        48,
        [fx::Y444P16_Y, fx::Y444P16_U, fx::Y444P16_V],
        "v3-yuv444p16",
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
fn v3_default_multislice_2x2_decodes_bit_exact() {
    // FFV1 v3, 8-bit YUV 4:2:0, 128×96, num_h_slices == 2 /
    // num_v_slices == 2 -> 4 Slices, each with its own range coder + §4.9.3
    // per-Slice CRC trailer. The canonical multi-Slice fixture: the v3
    // driver walks the §4.9.1 trailer chain, validates each §4.9 footer,
    // parses each §4.6 Slice Header, and reassembles the §5 slice-grid
    // raster partition into the full frame.
    assert_v3_ycbcr(
        fx::MS2X2_EXTRA,
        fx::MS2X2_FRAME,
        128,
        96,
        [fx::MS2X2_Y, fx::MS2X2_U, fx::MS2X2_V],
        "v3-default (multi-slice 2x2)",
    );
}

#[test]
fn v3_multislice_4x4_decodes_bit_exact() {
    // FFV1 v3, 8-bit YUV 4:2:0, 128×96, num_h_slices == 4 /
    // num_v_slices == 4 -> 16 Slices, each with its own range coder + CRC
    // trailer. Stresses the §5 slice-grid partition at the maximum
    // reference-encoder-default slice count (16-way), plus the full 16-link §4.9.1
    // trailer chain.
    assert_v3_ycbcr(
        fx::MS4X4_EXTRA,
        fx::MS4X4_FRAME,
        128,
        96,
        [fx::MS4X4_Y, fx::MS4X4_U, fx::MS4X4_V],
        "v3-multislice-4x4",
    );
}

#[test]
fn v3_frame_mt_decodes_bit_exact() {
    // FFV1 v3, 8-bit YUV 4:2:0, 256×192, num_h_slices == 4 /
    // num_v_slices == 4 -> 16 Slices with per-Slice CRC. Larger than the
    // other multi-Slice fixtures (256×192 vs 128×96), so each luma Slice
    // is 64×48: exercises the §5 slice-grid partition + §4.9.1 trailer
    // chain at a non-trivial Slice geometry.
    assert_v3_ycbcr(
        fx::FMT_EXTRA,
        fx::FMT_FRAME,
        256,
        192,
        [fx::FMT_Y, fx::FMT_U, fx::FMT_V],
        "v3-frame-mt",
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
fn v3_rgb_bgr0_decodes_bit_exact() {
    // FFV1 v3, RGB (colorspace_type == 1, JPEG 2000 RCT), 8-bit packed
    // BGR0 (3 channels, transparency == 0 -> 3 Planes), range coder. The
    // RGB driver with no alpha Plane — distinct from v3-rgba. The R/G/B
    // expected Planes were unpacked from the reference bgr0 expected.raw.
    let parsed = parse_quantization_table_sets(fx::BGR0_EXTRA).expect("bgr0 extradata");
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame_rgb(
        fx::BGR0_FRAME,
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        parsed.record.ec.is_some(),
    )
    .expect("bgr0 decode");
    assert_eq!(
        decoded.planes.len(),
        3,
        "v3-rgb-bgr0: plane count (RGB, no alpha)"
    );
    assert_eq!(
        decoded.planes[0].samples,
        fx::BGR0_R,
        "v3-rgb-bgr0: R plane"
    );
    assert_eq!(
        decoded.planes[1].samples,
        fx::BGR0_G,
        "v3-rgb-bgr0: G plane"
    );
    assert_eq!(
        decoded.planes[2].samples,
        fx::BGR0_B,
        "v3-rgb-bgr0: B plane"
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
fn v0_yuv420_golomb_rice_decodes_bit_exact() {
    // FFV1 version 0, Golomb-Rice coder (`coder_type == 0`), 8-bit YUV
    // 4:2:0, single implied Slice. Inline Parameters are range-coded; the
    // §3.8.1.1.1 Sentinel-mode terminator hands off to the §3.8.2
    // Golomb-Rice Slice Content, decoded through the §3.8.2.2 adaptive
    // run-length + level-coding loop.
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame_v0v1(fx::V0GR_FRAME, dims).expect("v0 golomb-rice decode");
    assert_eq!(decoded.planes.len(), 3, "v0-golomb-rice: plane count");
    assert_eq!(
        decoded.planes[0].samples,
        fx::V0GR_Y,
        "v0-golomb-rice: Y plane"
    );
    assert_eq!(
        decoded.planes[1].samples,
        fx::V0GR_U,
        "v0-golomb-rice: Cb plane"
    );
    assert_eq!(
        decoded.planes[2].samples,
        fx::V0GR_V,
        "v0-golomb-rice: Cr plane"
    );
}

#[test]
fn v1_golomb_decodes_bit_exact() {
    // FFV1 version 1, Golomb-Rice coder (`coder_type == 0`), 8-bit YUV
    // 4:2:0, 64×48, single implied Slice, inline Parameters (no
    // Configuration Record). Complements `v0-yuv420-golomb-rice` (version
    // 0) by exercising the §3.8.2 Golomb-Rice residual decode on a
    // version-1 header (`bits_per_raw_sample` present in Parameters). The
    // Slice Header is range-coded even in Golomb mode; only the sample
    // residuals use §3.8.2, reached through the §3.8.1.1.1 Sentinel-mode
    // range → Golomb byte handoff.
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame_v0v1(fx::V1GR_FRAME, dims).expect("v1 golomb decode");
    assert_eq!(decoded.planes.len(), 3, "v1-golomb: plane count");
    assert_eq!(decoded.planes[0].samples, fx::V1GR_Y, "v1-golomb: Y plane");
    assert_eq!(decoded.planes[1].samples, fx::V1GR_U, "v1-golomb: Cb plane");
    assert_eq!(decoded.planes[2].samples, fx::V1GR_V, "v1-golomb: Cr plane");
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
