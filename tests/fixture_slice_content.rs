//! Slice Content scaffold (RFC 9043 §4.7 / §4.8) fixture tests.
//!
//! Each test extracts the parsed Slice Header from one slice's
//! range-coded region (the same byte slices that drive
//! `fixture_slice_header.rs` in this crate's `tests/`), then asks
//! [`compute_slice_content`] for the typed per-plane / per-line grid
//! and checks the resulting `slice_pixel_x` / `slice_pixel_y` /
//! `plane_pixel_width` / `plane_pixel_height` against the fixture's
//! `trace.txt` `SLICE` (raster coords) + `PLANE` (per-plane pixel
//! dimensions) events.
//!
//! No new bytes are extracted from the FFV1 frames — only the
//! already-extracted slice bodies the round-2 fixture file holds plus
//! the per-fixture container-level frame dimensions (Matroska
//! `PixelWidth`/`PixelHeight`, recorded verbatim in each fixture's
//! `notes.md`).

use oxideav_ffv1::{
    compute_slice_content, parse_configuration_record, parse_slice_header, FramePixelDimensions,
    PlaneTraversal,
};

// ---- Extradata blobs (identical to fixture_v3_default.rs) ---------

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

// ---- Slice byte ranges (identical to fixture_slice_header.rs) -----

const V3_DEFAULT_SLICE0: &[u8] = &[
    0xfc, 0x29, 0x80, 0x19, 0xdd, 0x14, 0xfc, 0x76, 0x98, 0xd7, 0x9e, 0xa8, 0x85, 0xb2, 0xdc, 0xff,
    0xfc, 0x15, 0xde, 0xff, 0x98, 0x71, 0x78, 0x5a, 0x73, 0x3a, 0x4b, 0xa8, 0x76, 0xb2, 0x4c, 0x96,
    0x14, 0xac, 0x16, 0x30, 0xaa, 0x21, 0xa5, 0x3c, 0xeb, 0x57, 0x2a, 0x05, 0xd6, 0x30, 0xb2, 0x9a,
    0x3d, 0xf1, 0x75, 0x7f, 0x80, 0xb4, 0x32, 0x30, 0x3e, 0x89, 0x5b, 0x1b, 0xc6, 0x8f, 0xbd, 0x5b,
    0x02, 0x30, 0x19, 0x31, 0x34, 0xd2, 0xb8, 0xe0, 0x91, 0x9b, 0xfe, 0x2f, 0x30, 0x96, 0xde, 0xef,
    0x24, 0xff, 0xff, 0xcf, 0xff, 0x5e, 0x69, 0x0a, 0x3b, 0x25, 0xf3, 0x9c, 0x06, 0xa3, 0xa2, 0x70,
    0xfb, 0xff, 0xe7, 0xba, 0x99, 0x04, 0x90, 0xc1, 0x0c, 0xbb, 0xbd, 0x98, 0x20, 0x23, 0x00, 0x77,
    0x71, 0x7e, 0x06, 0x5f, 0xfb, 0xeb, 0xef, 0x7f, 0x25, 0x06, 0x7f, 0x50, 0xf9, 0xac, 0xfb, 0x9a,
    0x6d, 0x4d, 0xbb, 0x76, 0x84, 0x92, 0xed, 0x6f, 0x73, 0xf2, 0xe3, 0x96, 0xbb, 0x94, 0xa4, 0x66,
    0xbc, 0x57, 0xaf, 0xf6, 0x85, 0xff, 0xfe, 0xf4, 0xe8, 0xe3, 0xed, 0x24, 0xbe, 0x3a, 0x9a, 0x74,
    0x0e, 0x01, 0x46, 0x00, 0xcd, 0x6b, 0x50, 0x16, 0x22, 0xc1, 0x5e, 0x37, 0xc8, 0x95, 0x00, 0x8a,
    0x0b, 0xbe, 0xbd, 0x15, 0x54, 0xb9, 0x44, 0x8a, 0x64, 0x66, 0x48, 0x4f, 0x63, 0x3f, 0x80, 0x92,
    0xd5, 0x48, 0x34, 0x3b, 0x5f, 0xe7, 0xe7, 0x55, 0xb2, 0xc9, 0x75, 0xa0, 0x11, 0x53, 0xca, 0x19,
    0xb7, 0xf9, 0x68, 0xa1, 0x59, 0x59, 0x81, 0x9d, 0xec, 0x41, 0x88, 0x36, 0xe6, 0x98, 0x43, 0xaa,
    0x60, 0xb6, 0xf7, 0xff, 0xf7,
];

const V3_GRAYSCALE_SLICE0: &[u8] = &[
    0xfc, 0x2c, 0x06, 0xcb, 0xd6, 0xe0, 0x7b, 0xc7, 0xc1, 0xcb, 0xc1, 0x77, 0xff, 0x9e, 0x5d, 0x1e,
    0x27, 0x5e, 0x24, 0x25, 0x2b, 0x47, 0x0d, 0xd6, 0x1b, 0x71, 0xbf, 0x40, 0xb1, 0xcd, 0x1e, 0x3b,
    0x98, 0x74, 0xda, 0xbd, 0xb0, 0xae, 0x56, 0xd3, 0x94, 0xcf, 0xcf, 0xff, 0xcb,
];

const V3_RGB_BGR0_SLICE0: &[u8] = &[
    0xfc, 0x2c, 0x06, 0xc9, 0x8e, 0x81, 0xe6, 0x02, 0xee, 0x2f, 0x60, 0x29, 0xeb, 0xdc, 0x90, 0xc6,
    0x51, 0xf1, 0xbe, 0xbb, 0x25, 0x52, 0x01, 0x88, 0xf3, 0x98, 0x95, 0x12, 0xff, 0xf8, 0x75, 0x67,
    0x9a, 0x02, 0xf1, 0x40, 0xf5, 0xec, 0x3b, 0xf1, 0xd4, 0x28, 0x9d, 0x5c, 0xfb, 0xa1, 0x1f, 0x69,
    0x73, 0x23, 0x3d, 0x69, 0xae, 0x3c, 0xfd, 0x8c, 0xb5, 0x99, 0x4f, 0xaf, 0xfd, 0x85, 0x0b, 0xbe,
    0x3a, 0x89, 0xf4, 0x6b, 0xca, 0xd8, 0xc0, 0x35, 0xc2, 0x51, 0xff, 0x8f, 0x0e, 0x99, 0xff, 0xff,
    0xb3, 0x0e, 0x2c, 0xf3, 0xff, 0xff, 0xff, 0xff, 0x04,
];

// ---- Per-fixture frame dimensions (per each fixture's notes.md) ---

/// `v3-default/notes.md` records `testsrc=size=128x96`. Trace
/// confirms slice_pixel_w = 64 = 128 / 2 (2x2 raster).
const V3_DEFAULT_FRAME_W: u32 = 128;
const V3_DEFAULT_FRAME_H: u32 = 96;

/// `v3-grayscale/notes.md` records `testsrc=size=64x48`. Trace
/// confirms slice_pixel_w = 32 = 64 / 2.
const V3_GRAYSCALE_FRAME_W: u32 = 64;
const V3_GRAYSCALE_FRAME_H: u32 = 48;

/// `v3-rgb-bgr0/notes.md` records `testsrc=size=64x48`. Trace
/// PLANE w=32 h=24 confirms.
const V3_RGB_BGR0_FRAME_W: u32 = 64;
const V3_RGB_BGR0_FRAME_H: u32 = 48;

// ---- Tests ---------------------------------------------------------

#[test]
fn v3_default_slice0_content_matches_trace_plane_dims() {
    // From trace.txt:
    //   SLICE slice_idx=0  slice_x=0  slice_y=0  slice_w=64 slice_h=48
    //   PLANE plane=Y  w=64  h=48
    //   PLANE plane=U  w=32  h=24
    //   PLANE plane=V  w=32  h=24
    let cr = parse_configuration_record(V3_DEFAULT_EXTRADATA).expect("cr parses");
    let header = parse_slice_header(V3_DEFAULT_SLICE0, &cr).expect("slice 0 header parses");
    let frame = FramePixelDimensions::new(V3_DEFAULT_FRAME_W, V3_DEFAULT_FRAME_H).unwrap();
    let sc = compute_slice_content(&header, &cr, frame).expect("slice content scaffold builds");

    // Pixel-space anchor + size.
    assert_eq!(sc.slice_pixel_x, 0);
    assert_eq!(sc.slice_pixel_y, 0);
    assert_eq!(sc.slice_pixel_width, 64);
    assert_eq!(sc.slice_pixel_height, 48);

    // 3 planes for YUV 4:2:0; YCbCr is plane-major.
    assert_eq!(sc.primary_color_count(), 3);
    assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

    // Plane sizes match the trace PLANE events.
    assert_eq!((sc.planes[0].width, sc.planes[0].height), (64, 48), "Y");
    assert_eq!((sc.planes[1].width, sc.planes[1].height), (32, 24), "U");
    assert_eq!((sc.planes[2].width, sc.planes[2].height), (32, 24), "V");

    // Each plane's lines vector is sized to its height with
    // identity-only placeholders.
    assert_eq!(sc.planes[0].lines.len(), 48);
    assert_eq!(sc.planes[1].lines.len(), 24);
    assert_eq!(sc.planes[2].lines.len(), 24);

    // Total line visits = 48 + 24 + 24 = 96 (per RFC §4.7).
    assert_eq!(sc.line_count(), 96);
}

#[test]
fn v3_grayscale_slice0_content_matches_trace_plane_dims() {
    // From trace.txt:
    //   SLICE slice_idx=0  slice_x=0  slice_y=0  slice_w=32 slice_h=24
    //   PLANE plane=Y  w=32  h=24
    let cr = parse_configuration_record(V3_GRAYSCALE_EXTRADATA).expect("cr parses");
    let header = parse_slice_header(V3_GRAYSCALE_SLICE0, &cr).expect("grayscale header parses");
    let frame = FramePixelDimensions::new(V3_GRAYSCALE_FRAME_W, V3_GRAYSCALE_FRAME_H).unwrap();
    let sc = compute_slice_content(&header, &cr, frame).expect("scaffold builds");

    assert_eq!(sc.slice_pixel_x, 0);
    assert_eq!(sc.slice_pixel_y, 0);
    assert_eq!(sc.slice_pixel_width, 32);
    assert_eq!(sc.slice_pixel_height, 24);

    // Single plane.
    assert_eq!(sc.primary_color_count(), 1);
    assert_eq!(sc.planes[0].width, 32);
    assert_eq!(sc.planes[0].height, 24);
    assert_eq!(sc.line_count(), 24);
}

#[test]
fn v3_rgb_bgr0_slice0_content_matches_trace_plane_dims() {
    // From trace.txt:
    //   SLICE slice_idx=0  slice_x=0  slice_y=0  slice_w=32 slice_h=24
    //   PLANE plane=RGB  w=32  h=24  transparency=0
    // RGB packs G/B/R as 3 internal planes (RFC §4.7.1
    // primary_color_count = 1 + 2 + 0 = 3); the trace emits one
    // PLANE per slice rather than three, but the per-plane pixel
    // grid is identical (RGB forbids chroma subsampling per §4.2.5).
    let cr = parse_configuration_record(V3_RGB_BGR0_EXTRADATA).expect("cr parses");
    let header = parse_slice_header(V3_RGB_BGR0_SLICE0, &cr).expect("rgb header parses");
    let frame = FramePixelDimensions::new(V3_RGB_BGR0_FRAME_W, V3_RGB_BGR0_FRAME_H).unwrap();
    let sc = compute_slice_content(&header, &cr, frame).expect("scaffold builds");

    assert_eq!(sc.slice_pixel_x, 0);
    assert_eq!(sc.slice_pixel_y, 0);
    assert_eq!(sc.slice_pixel_width, 32);
    assert_eq!(sc.slice_pixel_height, 24);

    // 3 internal planes (RCT G/B/R) — RGB iterates line-then-plane.
    assert_eq!(sc.primary_color_count(), 3);
    assert_eq!(sc.traversal, PlaneTraversal::LineMajor);
    for p in &sc.planes {
        assert_eq!((p.width, p.height), (32, 24));
    }
    // 24 rows × 3 planes = 72 line visits.
    assert_eq!(sc.line_count(), 72);
    let visits = sc.line_visits();
    assert_eq!(visits.len(), 72);
    // Line-major: the first three visits are (G,0)(B,0)(R,0).
    assert_eq!(visits[0].plane_index, 0);
    assert_eq!(visits[0].y, 0);
    assert_eq!(visits[1].plane_index, 1);
    assert_eq!(visits[1].y, 0);
    assert_eq!(visits[2].plane_index, 2);
    assert_eq!(visits[2].y, 0);
    // Then row y=1 begins.
    assert_eq!(visits[3].plane_index, 0);
    assert_eq!(visits[3].y, 1);
}

#[test]
fn v3_default_all_four_slices_tile_the_frame() {
    // Walking every slice of v3-default and summing pixel area must
    // cover the entire frame exactly once (RFC 9043 §5 restriction
    // "each position in the Slice raster MUST be filled by one and
    // only one Slice"). The trace's per-slice raster cells are
    // (0,0)(1,0)(0,1)(1,1), every cell a 1×1 raster footprint.
    let cr = parse_configuration_record(V3_DEFAULT_EXTRADATA).expect("cr parses");
    let frame = FramePixelDimensions::new(V3_DEFAULT_FRAME_W, V3_DEFAULT_FRAME_H).unwrap();

    // Synthesise the 4 slice headers by constructing them from the
    // known raster coords (the round-2 fixture file confirms each
    // bitstream parses to these). We don't need to re-decode their
    // bytes here — we want to test the geometry computation in
    // isolation given the raster coords trace-truth ground truth.
    let headers =
        [(0u32, 0u32), (1, 0), (0, 1), (1, 1)].map(|(sx, sy)| oxideav_ffv1::Ffv1SliceHeader {
            slice_x: sx,
            slice_y: sy,
            slice_width: 1,
            slice_height: 1,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 0, 0],
            picture_structure: oxideav_ffv1::PictureStructure::Progressive,
            picture_structure_raw: 3,
            sar_num: 0,
            sar_den: 0,
        });

    let mut total_y_pixels = 0u64;
    let mut total_uv_pixels = 0u64;
    for h in &headers {
        let sc = compute_slice_content(h, &cr, frame).expect("each slice builds");
        // Per-slice pixel rectangle.
        assert_eq!(sc.slice_pixel_width, 64);
        assert_eq!(sc.slice_pixel_height, 48);
        // Y plane area + 2 chroma plane areas.
        total_y_pixels += u64::from(sc.planes[0].width) * u64::from(sc.planes[0].height);
        total_uv_pixels += u64::from(sc.planes[1].width) * u64::from(sc.planes[1].height);
        total_uv_pixels += u64::from(sc.planes[2].width) * u64::from(sc.planes[2].height);
    }
    // 4 cells × 64 × 48 = 128 × 96 = 12288 luma samples.
    assert_eq!(
        total_y_pixels,
        u64::from(V3_DEFAULT_FRAME_W * V3_DEFAULT_FRAME_H)
    );
    // Chroma at 4:2:0 is 128 × 96 / 2 = 6144 across both U + V.
    assert_eq!(
        total_uv_pixels,
        u64::from(V3_DEFAULT_FRAME_W * V3_DEFAULT_FRAME_H / 2)
    );
}
