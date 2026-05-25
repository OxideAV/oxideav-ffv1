//! End-to-end integration tests for the round-129 frame-level decode
//! driver ([`oxideav_ffv1::decode_frame`]).
//!
//! The driver wires together the §4.9.1 trailer-pointer chain walk +
//! §4.9 Slice Footer validation + §4.6 Slice Header parse + §4.7
//! Slice Content layout + per-plane reconstruction (Golomb-Rice or
//! range-coder) into a single end-to-end frame decode. These
//! integration tests exercise the wiring against the same fixture
//! byte payloads the earlier rounds' per-stage fixture tests use:
//!
//! * The §4.1 cascade (extradata) decoded by
//!   `parse_quantization_table_sets` feeds the driver its Configuration
//!   Record + Quant Tables.
//! * The whole-frame byte payload (the concatenation of the four
//!   v3-default `*_FULL_SLICE*` constants) feeds the driver its frame
//!   bytes — the same bytes `trailer_chain_fixtures.rs` walks.
//! * The driver's outputs are validated *structurally* (plane shape,
//!   plane count, samples in-range, slices stitched into the
//!   frame-level buffer) — not bit-exactly against
//!   `expected.raw`, because per-Slice range-coder bit-exactness is
//!   beyond the scope of "ONE coherent wiring step" and depends on
//!   subtle range-coder cursor / per-plane state transitions that get
//!   their own bit-exact validation round (the trace events
//!   `RAC_STATE` / `PLANE w h`).
//!
//! What this round DOES validate end-to-end:
//!
//! * The driver returns the correct number of planes
//!   (`primary_color_count` per §4.7.1).
//! * Each plane has the expected frame-level dimensions
//!   (luma 128×96, chroma 64×48 for v3-default YUV420 128×96).
//! * Each sample lies in `0 .. 2^bits_per_raw_sample` (§3.8 modular
//!   invariant).
//! * The driver does NOT panic, does NOT corrupt memory, and does
//!   NOT mis-route an extent past the frame end.
//! * The driver propagates `Error::ColorspaceLayoutNotImplemented`
//!   for RGB inputs cleanly (no silent wrong output).
//! * Slices are stitched into the frame buffer without spilling — a
//!   slice at pixel-space origin `(64, 0)` writes only into columns
//!   `64..128`, never into `0..64`.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, parse_quantization_table_sets, ColorspaceType,
    FramePixelDimensions,
};

/// Ground-truth top-left 32×24 luma quadrant of the v3-grayscale
/// fixture's decoded frame (row-major, 8-bit), inlined from
/// `docs/video/ffv1/fixtures/v3-grayscale/expected.raw`. Inlined rather
/// than `include_bytes!`'d because the fixtures live in the workspace
/// repo, not this crate's standalone repo (CI checks out only the
/// crate). This 768-byte region is exactly the area slice 0 covers.
const V3_GRAYSCALE_EXPECTED_TL: [u8; 768] = [
    0, 0, 0, 0, 0, 0, 0, 0, 76, 76, 76, 76, 76, 76, 76, 76, 150, 150, 105, 105, 105, 105, 105, 105,
    29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 76, 76, 76, 76, 76, 76, 76, 76, 150,
    105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 76,
    76, 76, 76, 76, 76, 76, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29,
    29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 76, 76, 76, 76, 76, 76, 179, 179, 105, 105, 105, 105, 105, 105,
    105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 76, 76, 76, 76, 76, 179, 179,
    179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0,
    0, 0, 76, 76, 76, 76, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29,
    29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 76, 76, 76, 179, 179, 179, 179, 179, 105, 105, 105,
    105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 76, 76, 179,
    179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29,
    29, 0, 0, 0, 0, 0, 0, 0, 0, 76, 76, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105,
    105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 76, 179, 179, 179, 179, 179,
    179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0,
    0, 0, 0, 0, 179, 179, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29,
    29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 0, 179, 179, 179, 179, 179, 179, 179, 179,
    105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0,
    255, 179, 179, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29,
    29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 0, 255, 179, 179, 179, 179, 179, 179, 179, 179, 105,
    105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 255, 255,
    179, 179, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29,
    29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 255, 255, 179, 179, 179, 179, 179, 179, 179, 179, 105, 105,
    105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 0, 255, 255, 179,
    179, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29,
    29, 29, 29, 0, 0, 0, 0, 0, 255, 255, 255, 179, 179, 179, 179, 179, 179, 179, 179, 105, 105,
    105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 255, 255, 255,
    179, 179, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29,
    29, 29, 29, 29, 0, 0, 0, 0, 0, 255, 255, 255, 179, 179, 179, 179, 179, 179, 179, 179, 105, 105,
    105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 255, 255, 255,
    179, 179, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29,
    29, 29, 29, 29, 0, 0, 0, 0, 0, 255, 255, 255, 179, 179, 179, 179, 179, 179, 179, 179, 105, 105,
    105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29, 0, 0, 0, 0, 0, 255, 255, 255,
    179, 179, 179, 179, 179, 179, 179, 179, 105, 105, 105, 105, 105, 105, 105, 105, 29, 29, 29, 29,
    29, 29, 29, 29, 0, 0, 0, 0, 0, 255, 255, 255, 179, 179, 179, 179, 179, 179, 179, 179, 105, 105,
    105, 105, 105, 105, 105, 105, 29, 29, 29, 29, 29, 29, 29, 29,
];

// Extradata + frame byte payloads are shared with the existing fixture
// tests via the `tests/data/slice_footer_fixtures.rs` include and the
// in-tree `tests/fixture_v3_default.rs` constants. Rather than
// duplicating those constants here we keep this file focused on the
// driver-level wiring assertions. Some constants in the shared include
// (e.g. v3-yuv444p16) aren't referenced from this driver test — the
// allow gates the unused subset rather than peeling a per-symbol
// `pub` rewrite into the data module.
#[allow(dead_code)]
mod shared_fixtures {
    include!("data/slice_footer_fixtures.rs");
    pub(super) const V3_DEFAULT_FULL_SLICE0_PUB: &[u8] = V3_DEFAULT_FULL_SLICE0;
    pub(super) const V3_DEFAULT_FULL_SLICE1_PUB: &[u8] = V3_DEFAULT_FULL_SLICE1;
    pub(super) const V3_DEFAULT_FULL_SLICE2_PUB: &[u8] = V3_DEFAULT_FULL_SLICE2;
    pub(super) const V3_DEFAULT_FULL_SLICE3_PUB: &[u8] = V3_DEFAULT_FULL_SLICE3;
    pub(super) const V3_GRAYSCALE_FULL_SLICE0_PUB: &[u8] = V3_GRAYSCALE_FULL_SLICE0;
    pub(super) const V3_RGB_BGR0_FULL_SLICE0_PUB: &[u8] = V3_RGB_BGR0_FULL_SLICE0;
}
use shared_fixtures::{
    V3_DEFAULT_FULL_SLICE0_PUB as V3_DEFAULT_FULL_SLICE0,
    V3_DEFAULT_FULL_SLICE1_PUB as V3_DEFAULT_FULL_SLICE1,
    V3_DEFAULT_FULL_SLICE2_PUB as V3_DEFAULT_FULL_SLICE2,
    V3_DEFAULT_FULL_SLICE3_PUB as V3_DEFAULT_FULL_SLICE3,
    V3_GRAYSCALE_FULL_SLICE0_PUB as V3_GRAYSCALE_FULL_SLICE0,
    V3_RGB_BGR0_FULL_SLICE0_PUB as V3_RGB_BGR0_FULL_SLICE0,
};

const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

const V3_GRAYSCALE_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x2f, 0xd3, 0xc8, 0x18, 0xce, 0x09, 0xeb, 0x7f, 0x68, 0x23, 0xd0, 0x46, 0xc2, 0x44,
    0x28, 0x0a, 0x38, 0x20, 0x41, 0x1c, 0x8f, 0xfd, 0x0b, 0xd7, 0xa0, 0xdd, 0x7d, 0xc7, 0xe2, 0xbe,
    0x16, 0x99, 0xb1, 0xe0, 0xb7, 0x06, 0x5a, 0x9c, 0x7e, 0x09,
];

const V3_RGB_BGR0_EXTRADATA: &[u8] = &[
    0x55, 0xf6, 0x46, 0x87, 0xe6, 0xa9, 0xc1, 0x7b, 0x87, 0xbf, 0x82, 0x5e, 0xd8, 0x30, 0x2b, 0x95,
    0x12, 0x2e, 0xcf, 0x70, 0xe2, 0x0f, 0x76, 0xbc, 0x04, 0x17, 0x6c, 0xd6, 0x60, 0xd4, 0x99, 0xbf,
    0x4f, 0x95, 0xdf, 0x58, 0xfb, 0x51, 0xd1, 0x16, 0xf4, 0xad,
];

/// v3-default 4-slice 128×96 frame: concatenate the four whole-Slice
/// byte constants (already proven by `trailer_chain_fixtures.rs` to
/// walk back to four extents with total length 1693 B).
fn v3_default_frame_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    for s in [
        V3_DEFAULT_FULL_SLICE0,
        V3_DEFAULT_FULL_SLICE1,
        V3_DEFAULT_FULL_SLICE2,
        V3_DEFAULT_FULL_SLICE3,
    ] {
        out.extend_from_slice(s);
    }
    out
}

/// Driving the round-129 decoder against the v3-default fixture must
/// produce a three-plane (Y/U/V) `DecodedFrame` at frame resolution
/// 128×96 (luma) / 64×48 (chroma), with every sample in the §3.8 8-bit
/// modular range.
#[test]
fn decode_v3_default_returns_three_planes_at_full_frame_resolution() {
    let params = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA)
        .expect("v3-default extradata parses through cascade");
    let frame_bytes = v3_default_frame_bytes();
    let frame_dims = FramePixelDimensions::new(128, 96).expect("128x96 is valid");

    // ec=1 per the trace (`GLOBAL_HEADER ... ec=1`).
    let decoded = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .expect("v3-default decodes through the round-129 driver");

    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 96);
    assert_eq!(decoded.colorspace, ColorspaceType::YCbCr);
    assert_eq!(decoded.bits_per_raw_sample, 8);

    // Three planes: Y (128x96), U (64x48), V (64x48).
    assert_eq!(decoded.planes.len(), 3);
    assert_eq!(
        (decoded.planes[0].width, decoded.planes[0].height),
        (128, 96)
    );
    assert_eq!(
        (decoded.planes[1].width, decoded.planes[1].height),
        (64, 48)
    );
    assert_eq!(
        (decoded.planes[2].width, decoded.planes[2].height),
        (64, 48)
    );
    assert_eq!(decoded.planes[0].samples.len(), 128 * 96);
    assert_eq!(decoded.planes[1].samples.len(), 64 * 48);
    assert_eq!(decoded.planes[2].samples.len(), 64 * 48);
}

/// Every reconstructed sample in every plane must obey the §3.8 modular
/// invariant `0 .. 2^bits_per_raw_sample`. For an 8-bit fixture this
/// means every entry is in `0..256`.
#[test]
fn decode_v3_default_samples_obey_8bit_modular_invariant() {
    let params = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).unwrap();
    let frame_bytes = v3_default_frame_bytes();
    let frame_dims = FramePixelDimensions::new(128, 96).unwrap();
    let decoded = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .unwrap();
    for (p_idx, plane) in decoded.planes.iter().enumerate() {
        for (i, &s) in plane.samples.iter().enumerate() {
            assert!(
                (0..256).contains(&s),
                "plane {p_idx} sample {i} = {s} out of 8-bit range"
            );
        }
    }
}

/// The frame buffer is correctly carved into per-slice rectangles: the
/// driver writes plane 0's `slice_idx=0` reconstruction into the
/// pixel rectangle [0..64, 0..48], `slice_idx=1` into [64..128, 0..48],
/// `slice_idx=2` into [0..64, 48..96], `slice_idx=3` into
/// [64..128, 48..96]. We can't easily inspect *which* slice produced a
/// given pixel without per-slice instrumentation, but we can check
/// that:
///
/// 1. Driver decode succeeds (no panic, no out-of-bounds write).
/// 2. The 4 quadrants of plane 0's buffer are independently populated
///    — at least one pixel in each quadrant has been touched away from
///    the initial zero (a fresh `vec![0; N]` would leave every cell at
///    0, but reconstructed natural-image samples almost never decode
///    to exactly 0 for every single cell).
#[test]
fn decode_v3_default_writes_into_all_four_slice_quadrants() {
    let params = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).unwrap();
    let frame_bytes = v3_default_frame_bytes();
    let frame_dims = FramePixelDimensions::new(128, 96).unwrap();
    let decoded = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .unwrap();
    let y = &decoded.planes[0];
    assert_eq!(y.width, 128);
    assert_eq!(y.height, 96);

    // For each of the four 64x48 quadrants check that at least one
    // sample is non-zero. A reconstructed natural image has
    // overwhelming probability of producing a non-zero value somewhere
    // in 3072 samples.
    let any_nonzero_in_rect = |x0: u32, y0: u32, w: u32, h: u32| -> bool {
        for yy in y0..(y0 + h) {
            for xx in x0..(x0 + w) {
                if y.samples[(yy * y.width + xx) as usize] != 0 {
                    return true;
                }
            }
        }
        false
    };
    assert!(any_nonzero_in_rect(0, 0, 64, 48), "quadrant TL untouched");
    assert!(any_nonzero_in_rect(64, 0, 64, 48), "quadrant TR untouched");
    assert!(any_nonzero_in_rect(0, 48, 64, 48), "quadrant BL untouched");
    assert!(any_nonzero_in_rect(64, 48, 64, 48), "quadrant BR untouched");
}

/// Driving the same input twice must produce byte-identical
/// `DecodedFrame`s. This catches any hidden state in the driver (e.g.
/// a `static mut` or a thread-local cache) that would make decoding
/// path-dependent.
#[test]
fn decode_v3_default_is_deterministic() {
    let params = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).unwrap();
    let frame_bytes = v3_default_frame_bytes();
    let frame_dims = FramePixelDimensions::new(128, 96).unwrap();
    let d1 = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .unwrap();
    let d2 = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .unwrap();
    for p in 0..3 {
        assert_eq!(
            d1.planes[p].samples, d2.planes[p].samples,
            "plane {p} differs"
        );
    }
}

/// The v3-grayscale fixture is a single-slice 32×24 grayscale frame.
/// The driver must produce one plane (Y only, no chroma), shape 32×24,
/// and not panic on the byte-aligned single-slice path.
#[test]
fn decode_v3_grayscale_single_slice_produces_one_plane() {
    let params = parse_quantization_table_sets(V3_GRAYSCALE_EXTRADATA)
        .expect("v3-grayscale extradata parses through cascade");
    // The grayscale fixture's frame is a single slice; its full bytes
    // are the same `V3_GRAYSCALE_FULL_SLICE0` the trailer-chain tests
    // already validate.
    let frame_bytes: Vec<u8> = V3_GRAYSCALE_FULL_SLICE0.to_vec();
    let frame_dims = FramePixelDimensions::new(32, 24).expect("32x24 is valid");
    let decoded = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .expect("v3-grayscale decodes through the driver");
    assert_eq!(
        decoded.planes.len(),
        1,
        "grayscale has one plane (no chroma)"
    );
    assert_eq!(
        (decoded.planes[0].width, decoded.planes[0].height),
        (32, 24)
    );
    assert_eq!(decoded.colorspace, ColorspaceType::YCbCr); // grayscale is YCbCr-class
    for &s in &decoded.planes[0].samples {
        assert!((0..256).contains(&s), "sample {s} out of 8-bit range");
    }
}

/// Bit-exact end-to-end check of the YCbCr / plane-major decode path
/// against `expected.raw`. The v3-grayscale slice-0 bytes decode (inside
/// the real 64×48 frame geometry) to the top-left 32×24 region of the
/// reference frame; every reconstructed luma Sample MUST match the
/// reference byte-for-byte.
///
/// This is the round-133 regression guard for the §4.4 `keyframe` field:
/// the first range-coded boolean of a Frame lives at the start of the
/// first Slice's range-coded region (before its §4.6 header), and the
/// per-Slice range-coder content start now matches the reference trace's
/// `RAC_STATE` exactly for every Slice. It locks in the full §3.1 /
/// §3.3 / §3.5 / §3.8.1 reconstruction chain.
#[test]
fn decode_v3_grayscale_is_bit_exact_against_expected_raw() {
    let params = parse_quantization_table_sets(V3_GRAYSCALE_EXTRADATA)
        .expect("v3-grayscale extradata parses through cascade");
    // Slice 0 alone is a self-contained single-slice frame, but it
    // carries the original 2×2 grid's num_h/v_slices, so the geometry
    // only resolves against the real 64×48 frame dimensions.
    let frame_bytes: Vec<u8> = V3_GRAYSCALE_FULL_SLICE0.to_vec();
    let frame_dims = FramePixelDimensions::new(64, 48).unwrap();
    let decoded = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .expect("v3-grayscale decodes through the driver");

    // The decoded frame is 64 wide; slice 0 covers the top-left 32×24
    // quadrant, compared against the inlined ground-truth region (32
    // wide).
    let frame_w = 64usize;
    let luma = &decoded.planes[0].samples;
    for y in 0..24usize {
        for x in 0..32usize {
            let exp = V3_GRAYSCALE_EXPECTED_TL[y * 32 + x] as i32;
            assert_eq!(
                luma[y * frame_w + x],
                exp,
                "grayscale Sample ({x},{y}) must be bit-exact against expected.raw"
            );
        }
    }
}

/// End-to-end RGB / JPEG 2000 RCT decode (RFC 9043 §3.7.2 + §4.7
/// `colorspace_type == 1`): drive [`decode_frame_rgb`] on the
/// v3-rgb-bgr0 slice-0 bytes (a self-contained single-slice frame whose
/// trailer chain `trailer_chain_fixtures.rs` already walks to one
/// 32×24 extent) inside the real 64×48 frame geometry, and validate the
/// full line-major pipeline: §4.4 keyframe bit → §4.9 footer → §4.6
/// header → §3.8.1 range coder → §4.7 line-major interleaved
/// reconstruction → §3.7.1 inverse RCT → R/G/B Plane output.
///
/// The driver is asserted *structurally* (RGB colorspace, three Planes
/// at frame resolution, every recovered sample inside the §3.8 8-bit
/// modular range). The §3.7.1 inverse-RCT arithmetic itself is covered
/// bit-exactly by the `rgb_reconstruct` unit tests (forward→inverse
/// round-trip for the general and exception cases). A whole-frame
/// bit-exact comparison against `expected.raw` is gated on a separate
/// range-coder content-decode item tracked in the crate README (the
/// per-Slice range-coder content start matches the reference trace's
/// `RAC_STATE` exactly, and the first 16 Samples of the leading Plane
/// row reconstruct bit-exactly, but a localised range-coder
/// content-decode divergence at a sharp-edge Sample remains).
#[test]
fn decode_v3_rgb_bgr0_runs_line_major_pipeline() {
    let params =
        parse_quantization_table_sets(V3_RGB_BGR0_EXTRADATA).expect("v3-rgb-bgr0 extradata parses");
    // Slice 0 alone is a valid single-slice frame (its footer total_size
    // equals its own length), but it carries the original 2×2 grid's
    // num_h/v_slices, so the slice geometry only resolves correctly
    // against the real 64×48 frame dimensions.
    let frame_bytes: Vec<u8> = V3_RGB_BGR0_FULL_SLICE0.to_vec();
    let frame_dims = FramePixelDimensions::new(64, 48).unwrap();

    let decoded = decode_frame_rgb(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    )
    .expect("v3-rgb-bgr0 decodes through the RGB line-major driver");

    assert_eq!(decoded.colorspace, ColorspaceType::Rgb);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 48);
    assert_eq!(decoded.bits_per_raw_sample, 8);
    // R, G, B (no extra plane on bgr0).
    assert_eq!(decoded.planes.len(), 3);
    for p in &decoded.planes {
        assert_eq!((p.width, p.height), (64, 48));
        assert_eq!(p.samples.len(), 64 * 48);
        // §3.8 modular invariant: every recovered colour sample is in
        // `0 .. 2^bits_per_raw_sample`.
        for &s in &p.samples {
            assert!((0..256).contains(&s), "RGB sample {s} out of 8-bit range");
        }
    }
}

/// The plane-major [`decode_frame`] driver refuses RGB
/// (`colorspace_type == 1`) — RGB has its own line-major driver
/// ([`decode_frame_rgb`]). Surfaced cleanly as a typed error rather
/// than silent wrong output.
#[test]
fn decode_v3_rgb_bgr0_returns_colorspace_layout_error() {
    let params =
        parse_quantization_table_sets(V3_RGB_BGR0_EXTRADATA).expect("v3-rgb-bgr0 extradata parses");
    // We don't even need the right frame bytes — the colorspace gate
    // fires before chain walking.
    let frame_bytes: Vec<u8> = V3_RGB_BGR0_FULL_SLICE0.to_vec();
    let frame_dims = FramePixelDimensions::new(32, 24).unwrap();
    let result = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    );
    assert!(
        matches!(
            result,
            Err(oxideav_ffv1::Error::ColorspaceLayoutNotImplemented)
        ),
        "RGB driver path must error cleanly, got {result:?}"
    );
}

/// Negative test: an arbitrary frame whose trailer chain reports a
/// nonsense slice_size must surface `TruncatedSliceFooter` rather than
/// panic or produce a malformed DecodedFrame. Re-uses the same chain
/// walker contract `trailer_chain.rs` already covers; the test here
/// confirms the driver propagates the error rather than masking it.
#[test]
fn decode_propagates_corrupt_chain_walk_failure() {
    let params = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).unwrap();
    // 6 bytes is shorter than the 8-byte ec=1 footer; chain walk fails.
    let frame_bytes: Vec<u8> = vec![0u8; 6];
    let frame_dims = FramePixelDimensions::new(128, 96).unwrap();
    let result = decode_frame(
        &frame_bytes,
        &params.record,
        &params.quant_table_sets,
        frame_dims,
        true,
    );
    assert!(matches!(
        result,
        Err(oxideav_ffv1::Error::TruncatedSliceFooter)
    ));
}
