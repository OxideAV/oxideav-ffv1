//! Round-trip tests for the RGB / JPEG 2000 RCT frame encoder
//! (`encode_frame_rgb`).
//!
//! `encode_frame_rgb` is the symmetric inverse of `decode_frame_rgb`:
//! given an R / G / B (and optional alpha) DecodedFrame, it applies the
//! §3.7.1 forward RCT, then walks the §4.7 line-major traversal and
//! emits a Slice byte stream that `decode_frame_rgb` reconstructs back
//! to the original RGB Planes bit-for-bit.
//!
//! ```text
//!   DecodedFrame { R, G, B[, A] }
//!     ── encode_frame_rgb ─────▶ Vec<u8> frame_bytes
//!     ── decode_frame_rgb ─────▶ DecodedFrame { R, G, B[, A] }
//!     == original (bit-exact)
//! ```
//!
//! Scope is `coder_type == 1 || 2` (range-coded SliceContent). The
//! Golomb-Rice RGB encode path (`coder_type == 0`) is a follow-up
//! round (it needs a `PlaneReconstructor::encode_row` symmetric to its
//! current `reconstruct_row`).

use oxideav_ffv1::{
    decode_frame_rgb, encode_frame_rgb, ColorspaceType, DecodedFrame, DecodedFramePlane, Error,
    Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

/// Minimal RGB v3 Configuration Record pinned to `coder_type`. RGB has
/// three colour Planes; the optional `extra_plane` adds an alpha Plane
/// (untransformed) — controlled by the `extra` flag.
fn rgb_v3_cr(
    num_h: u32,
    num_v: u32,
    coder_type: u32,
    bits: u32,
    extra: bool,
) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::Rgb,
        bits_per_raw_sample: bits,
        chroma_planes: true, // RGB always has three Planes
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: extra,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(1),
        ec: Some(0),
        intra: Some(false),
        initial_state_delta: None,
    }
}

/// Single-context QTS — every neighbour configuration maps to context
/// `c`. With `c != 0` the §3.5 sign-flip path stays inactive so the
/// per-context state window's evolution is easy to reason about.
fn constant_context_qts(c: u32) -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    tables[0] = [c as i32; 256];
    QuantizationTableSet {
        tables,
        context_count: c + 1,
    }
}

fn make_header(
    slice_x: u32,
    slice_y: u32,
    slice_width: u32,
    slice_height: u32,
    quant_index_count: usize,
    quant_index: u32,
) -> Ffv1SliceHeader {
    let mut idx = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
    for slot in idx.iter_mut().take(quant_index_count) {
        *slot = quant_index;
    }
    Ffv1SliceHeader {
        slice_x,
        slice_y,
        slice_width,
        slice_height,
        quant_table_set_index_count: quant_index_count,
        quant_table_set_index: idx,
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

/// Build an RGB DecodedFrame with the three (or four, when `extra`)
/// colour Planes laid out as row-major `Vec<i32>` Sample buffers.
fn make_rgb_decoded_frame(
    r: Vec<i32>,
    g: Vec<i32>,
    b: Vec<i32>,
    a: Option<Vec<i32>>,
    w: u32,
    h: u32,
    bits: u32,
) -> DecodedFrame {
    let mut planes = vec![
        DecodedFramePlane {
            plane_index: 0,
            width: w,
            height: h,
            samples: r,
        },
        DecodedFramePlane {
            plane_index: 1,
            width: w,
            height: h,
            samples: g,
        },
        DecodedFramePlane {
            plane_index: 2,
            width: w,
            height: h,
            samples: b,
        },
    ];
    if let Some(alpha) = a {
        planes.push(DecodedFramePlane {
            plane_index: 3,
            width: w,
            height: h,
            samples: alpha,
        });
    }
    DecodedFrame {
        planes,
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

fn assert_rgb_round_trip(
    cr: &Ffv1ConfigurationRecord,
    qts: &[QuantizationTableSet],
    headers: &[Ffv1SliceHeader],
    frame: &DecodedFrame,
    ec: bool,
) {
    let bytes = encode_frame_rgb(frame, cr, qts, headers, ec)
        .expect("RGB encode must succeed for valid inputs");
    let decoded = decode_frame_rgb(
        &bytes,
        cr,
        qts,
        FramePixelDimensions::new(frame.width, frame.height).unwrap(),
        ec,
    )
    .expect("RGB encoded frame must round-trip through decode_frame_rgb");
    assert_eq!(
        decoded.planes.len(),
        frame.planes.len(),
        "Plane count diverged"
    );
    assert_eq!(decoded.colorspace, ColorspaceType::Rgb);
    assert_eq!(decoded.bits_per_raw_sample, frame.bits_per_raw_sample);
    for (got, want) in decoded.planes.iter().zip(frame.planes.iter()) {
        assert_eq!(
            got.samples, want.samples,
            "Plane {} samples diverged",
            got.plane_index
        );
        assert_eq!(got.width, want.width);
        assert_eq!(got.height, want.height);
    }
}

// ---- single-slice RGB round trips --------------------------------------

#[test]
fn rgb_encode_round_trips_single_slice_8bit_general() {
    let cr = rgb_v3_cr(1, 1, 1, 8, false);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..32).map(|i| (i * 7 + 3) & 0xFF).collect();
    let g: Vec<i32> = (0..32).map(|i| (i * 11 + 5) & 0xFF).collect();
    let b: Vec<i32> = (0..32).map(|i| (i * 13 + 7) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 8, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_single_slice_10bit_general() {
    // 10 bits ≥ 9 but extra_plane==false here also satisfies <=15, so the
    // §3.7.2.1 exception (Figure 9) applies. Round-trip both ways.
    let cr = rgb_v3_cr(1, 1, 1, 10, false);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..20).map(|i| (i * 47) % 1024).collect();
    let g: Vec<i32> = (0..20).map(|i| (i * 73 + 5) % 1024).collect();
    let b: Vec<i32> = (0..20).map(|i| (i * 113 + 11) % 1024).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 5, 4, 10);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_single_slice_9bit_exception() {
    // 9 bits is the *lower* boundary of the RFC 9043 §3.7.2.1 exception
    // window (9..=15 inclusive, extra_plane == 0). The exception predicate
    // fires, so both the forward and inverse RCT must use Figure 8 / 9.
    // The RCT coding width is bits + 1 == 10, so Sample Differences span a
    // 0..1024 modular window. No prior test exercised the 9-bit boundary.
    let cr = rgb_v3_cr(1, 1, 1, 9, false);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..20).map(|i| (i * 41) % 512).collect();
    let g: Vec<i32> = (0..20).map(|i| (i * 67 + 3) % 512).collect();
    let b: Vec<i32> = (0..20).map(|i| (i * 97 + 7) % 512).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 5, 4, 9);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_single_slice_15bit_exception() {
    // 15 bits is the *upper* boundary of the §3.7.2.1 exception window;
    // 16-bit (below) crosses out of the window into the general Figure 7
    // path. The RCT coding width is bits + 1 == 16 here, so the inverse
    // RCT runs at the largest width still inside the exception range.
    let cr = rgb_v3_cr(1, 1, 1, 15, false);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let m = 1i32 << 15;
    let r: Vec<i32> = (0..20).map(|i| (i * 1531) % m).collect();
    let g: Vec<i32> = (0..20).map(|i| (i * 2017 + 5) % m).collect();
    let b: Vec<i32> = (0..20).map(|i| (i * 2731 + 11) % m).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 5, 4, 15);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_single_slice_16bit_general() {
    // 16-bit RGB RCT is *outside* the §3.7.2.1 exception window (9..=15),
    // so the general Figure 6 / 7 RCT applies (per the RFC 9043 §3.7.2.1
    // Background note, 16-bit RCT was implemented without the GBR/BGR Plane
    // swap). The RCT coding width is bits + 1 == 17, so Sample Differences
    // span a 0..131072 modular window — the widest the RGB path reaches.
    // No prior test exercised 16-bit RGB.
    let cr = rgb_v3_cr(1, 1, 1, 16, false);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let m = 1i32 << 16;
    let r: Vec<i32> = (0..20).map(|i| (i * 3037) % m).collect();
    let g: Vec<i32> = (0..20).map(|i| (i * 4099 + 5) % m).collect();
    let b: Vec<i32> = (0..20).map(|i| (i * 5051 + 11) % m).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 5, 4, 16);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_with_alpha_plane() {
    // `extra_plane == true` adds an alpha Plane (Plane 3) carried
    // straight, no RCT. The exception predicate fires only when
    // `extra_plane == false`, so 12-bit with alpha takes the general
    // formula.
    let cr = rgb_v3_cr(1, 1, 1, 8, true);
    let qts = vec![constant_context_qts(7)];
    let header = make_header(0, 0, 1, 1, 3, 0);
    let r: Vec<i32> = (0..16).map(|i| (i * 17) & 0xFF).collect();
    let g: Vec<i32> = (0..16).map(|i| (i * 23) & 0xFF).collect();
    let b: Vec<i32> = (0..16).map(|i| (i * 29) & 0xFF).collect();
    let a: Vec<i32> = (0..16).map(|i| (i * 31 + 5) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, Some(a), 4, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_flat_planes() {
    // A flat RGB Plane (all Samples equal) collapses the forward RCT to
    // (Y=g, Cb=offset, Cr=offset), the lowest-entropy case. Validates
    // that the per-context state window initialisation + the constant
    // run-length of zero `sample_difference` values both encode + decode
    // bit-exactly.
    let cr = rgb_v3_cr(1, 1, 1, 8, false);
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r = vec![64; 16];
    let g = vec![128; 16];
    let b = vec![200; 16];
    let frame = make_rgb_decoded_frame(r, g, b, None, 4, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

// ---- slice grid --------------------------------------------------------

#[test]
fn rgb_encode_round_trips_2x2_slice_grid() {
    let cr = rgb_v3_cr(2, 2, 1, 8, false);
    let qts = vec![constant_context_qts(11)];
    let (fw, fh) = (6u32, 4u32);
    let pixels = |seed: u32| -> Vec<i32> {
        (0..(fw * fh))
            .map(|i| ((i * seed) ^ 0x5A) & 0xFF)
            .map(|v| v as i32)
            .collect()
    };
    let frame = make_rgb_decoded_frame(pixels(19), pixels(23), pixels(29), None, fw, fh, 8);
    let headers = vec![
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    assert_rgb_round_trip(&cr, &qts, &headers, &frame, true);
}

// ---- ec=0 (3-byte footer) ---------------------------------------------

#[test]
fn rgb_encode_round_trips_ec0_footer() {
    let cr = rgb_v3_cr(1, 1, 1, 8, false);
    let qts = vec![constant_context_qts(5)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r = vec![0, 1, 254, 255, 128, 64, 32, 16];
    let g = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let b = vec![200, 150, 100, 50, 25, 5, 1, 0];
    let frame = make_rgb_decoded_frame(r, g, b, None, 4, 2, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, false);
}

// ---- coder_type == 2 (alternative state-transition table) --------------

#[test]
fn rgb_encode_round_trips_coder_type_2() {
    // `coder_type == 2` overlays the Configuration Record's
    // `state_transition_delta` onto the default `one_state` table. The
    // round-179 plumbing already exercises the YCbCr path; here we add
    // the RGB symmetry.
    let mut cr = rgb_v3_cr(1, 1, 2, 8, false);
    // A small uniform +1 delta produces a recognisably distinct table.
    for d in cr.state_transition_delta.iter_mut() {
        *d = 1;
    }
    let qts = vec![constant_context_qts(7)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..16).map(|i| (i * 17 + 1) & 0xFF).collect();
    let g: Vec<i32> = (0..16).map(|i| (i * 19 + 3) & 0xFF).collect();
    let b: Vec<i32> = (0..16).map(|i| (i * 23 + 5) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 4, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

// ---- determinism -------------------------------------------------------

#[test]
fn rgb_encode_is_deterministic_across_calls() {
    let cr = rgb_v3_cr(1, 1, 1, 8, false);
    let qts = vec![constant_context_qts(4)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..24).map(|i| (i * 13 + 9) & 0xFF).collect();
    let g: Vec<i32> = (0..24).map(|i| (i * 17 + 11) & 0xFF).collect();
    let b: Vec<i32> = (0..24).map(|i| (i * 19 + 13) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 6, 4, 8);

    let a = encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    let b2 = encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(a, b2);
}

// ---- error-path guards -------------------------------------------------

#[test]
fn rgb_encode_rejects_v0() {
    let mut cr = rgb_v3_cr(1, 1, 1, 8, false);
    cr.version = Ffv1Version::V0;
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let frame = make_rgb_decoded_frame(vec![0; 4], vec![0; 4], vec![0; 4], None, 2, 2, 8);
    let r = encode_frame_rgb(&frame, &cr, &qts, &[header], true);
    assert!(matches!(r, Err(Error::SliceRequiresVersion3)));
}

#[test]
fn rgb_encode_rejects_ycbcr_config() {
    let mut cr = rgb_v3_cr(1, 1, 1, 8, false);
    cr.colorspace_type = ColorspaceType::YCbCr;
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let frame = make_rgb_decoded_frame(vec![0; 4], vec![0; 4], vec![0; 4], None, 2, 2, 8);
    let r = encode_frame_rgb(&frame, &cr, &qts, &[header], true);
    assert!(matches!(r, Err(Error::ColorspaceLayoutNotImplemented)));
}

#[test]
fn rgb_encode_round_trips_golomb_rice_single_slice_8bit() {
    // `coder_type == 0` Golomb-Rice RGB encode path (RFC 9043 §4.7
    // line-major + §4.8 Golomb-Rice content). The range-coded
    // SliceHeader bytes prefix the Golomb-Rice content tail at the
    // RangeEncoder's byte-boundary flush, exactly the layout
    // `decode_frame_rgb` walks for `coder_type == 0`.
    let cr = rgb_v3_cr(1, 1, 0, 8, false);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..32).map(|i| (i * 7 + 3) & 0xFF).collect();
    let g: Vec<i32> = (0..32).map(|i| (i * 11 + 5) & 0xFF).collect();
    let b: Vec<i32> = (0..32).map(|i| (i * 13 + 7) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 8, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_flat_planes() {
    // Flat RGB Plane collapses the forward RCT to (Y=g, Cb=offset,
    // Cr=offset); §3.8.2.2 run-mode dominates the per-row encode for
    // the long constant-zero `sample_difference` regions.
    let cr = rgb_v3_cr(1, 1, 0, 8, false);
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r = vec![64; 16];
    let g = vec![128; 16];
    let b = vec![200; 16];
    let frame = make_rgb_decoded_frame(r, g, b, None, 4, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_with_alpha_plane() {
    // `extra_plane == true` adds an untransformed alpha Plane; on the
    // Golomb-Rice path the alpha is encoded through the same
    // per-Plane `LineDecoderState` as the colour Planes.
    let cr = rgb_v3_cr(1, 1, 0, 8, true);
    let qts = vec![constant_context_qts(7)];
    let header = make_header(0, 0, 1, 1, 3, 0);
    let r: Vec<i32> = (0..16).map(|i| (i * 17) & 0xFF).collect();
    let g: Vec<i32> = (0..16).map(|i| (i * 23) & 0xFF).collect();
    let b: Vec<i32> = (0..16).map(|i| (i * 29) & 0xFF).collect();
    let a: Vec<i32> = (0..16).map(|i| (i * 31 + 5) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, Some(a), 4, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_10bit_exception() {
    // 10 bits + extra_plane==false fires the §3.7.2.1 exception
    // (Figure 8 forward / Figure 9 inverse) on the Golomb-Rice path.
    let cr = rgb_v3_cr(1, 1, 0, 10, false);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..20).map(|i| (i * 47) % 1024).collect();
    let g: Vec<i32> = (0..20).map(|i| (i * 73 + 5) % 1024).collect();
    let b: Vec<i32> = (0..20).map(|i| (i * 113 + 11) % 1024).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 5, 4, 10);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_16bit_general() {
    // 16-bit RGB is *outside* the §3.7.2.1 exception window (9..=15), so
    // the general Figure 6 / 7 RCT runs at coding width bits+1 == 17 — the
    // same width the range-coded `…_16bit_general` test exercises, but on
    // the §3.8.2 Golomb-Rice path (`coder_type == 0`). No prior
    // Golomb-Rice RGB test reached 16-bit: the matrix stopped at the
    // 10-bit exception case. High-entropy Samples force the per-context
    // §3.8.2 VLC window to evolve on every step and drive `bits+1`-wide
    // §3.7.1 inverse-RCT clamps, so an encode/decode width mismatch at the
    // top of the modular window surfaces as a Plane divergence.
    let cr = rgb_v3_cr(1, 1, 0, 16, false);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let m = 1i32 << 16;
    let r: Vec<i32> = (0..40).map(|i| (i * 3037 + 1) % m).collect();
    let g: Vec<i32> = (0..40).map(|i| (i * 4099 + 5) % m).collect();
    let b: Vec<i32> = (0..40).map(|i| (i * 5051 + 11) % m).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 8, 5, 16);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_2x2_slice_grid() {
    // Multi-slice on the Golomb-Rice path: each Slice carries its own
    // range-coded SliceHeader + Golomb-Rice content tail + footer; the
    // keyframe bit lives only in slice 0 per §4.4.
    let cr = rgb_v3_cr(2, 2, 0, 8, false);
    let qts = vec![constant_context_qts(11)];
    let (fw, fh) = (6u32, 4u32);
    let pixels = |seed: u32| -> Vec<i32> {
        (0..(fw * fh))
            .map(|i| ((i * seed) ^ 0x5A) & 0xFF)
            .map(|v| v as i32)
            .collect()
    };
    let frame = make_rgb_decoded_frame(pixels(19), pixels(23), pixels(29), None, fw, fh, 8);
    let headers = vec![
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    assert_rgb_round_trip(&cr, &qts, &headers, &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_ec0_footer() {
    // ec=0 selects the 3-byte §4.9 SliceFooter on the Golomb-Rice
    // path; the absence of the §4.9.3 CRC parity word is the only
    // structural delta.
    let cr = rgb_v3_cr(1, 1, 0, 8, false);
    let qts = vec![constant_context_qts(5)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r = vec![0, 1, 254, 255, 128, 64, 32, 16];
    let g = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let b = vec![200, 150, 100, 50, 25, 5, 1, 0];
    let frame = make_rgb_decoded_frame(r, g, b, None, 4, 2, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, false);
}

// ---- §4.6.6 slot-keyed VLC sharing on the Golomb-Rice RGB path -------
//
// Round 227 lifted the per-context VLC window out of the per-Plane
// state and into a per-§4.6.6-slot [`LineDecoderState`] on the
// Golomb-Rice RGB encode + decode paths, matching the §4.6.6 contract
// already enforced for the range-coded RGB path (round 220) and the
// YCbCr path (round 214). The run-mode triple
// (`run_index` / `run_mode` / `run_count`) stays per-Plane per
// §3.8.2.2.1 and is swapped into / out of the slot state around every
// row encode / decode.
//
// Encoder ↔ decoder are mutated in lockstep on every round-trip, so a
// wrong slot routing — wrong context-count sizing on the second-Plane
// first-touch, or a stray reset of the slot's VLC fields when the
// run-mode triple was supposed to reset — surfaces as a Plane-
// divergence assertion in any of the following round trips.

#[test]
fn rgb_encode_round_trips_golomb_rice_high_entropy_chroma_planes() {
    // High-entropy RGB content: every Sample distinct, every diff
    // distinct, so the per-context VLC window evolves on every Plane
    // step. With the slot-keyed VLC the window's evolution is shared
    // across G + B (both Planes route to the chroma slot 1 in the
    // §4.6.6 mapping `(0=luma, 1=chroma, 1=chroma)`); the per-Plane
    // run triple stays separate. A wrong (per-Plane) VLC allocation
    // would mis-key the VLC reads on Cb's second + later rows.
    let cr = rgb_v3_cr(1, 1, 0, 8, false);
    let qts = vec![constant_context_qts(11)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (w, h) = (12u32, 8u32);
    // xorshift-style pseudo-random with three distinct seeds per Plane.
    let pixels = |seed: u32| -> Vec<i32> {
        let mut x = seed;
        (0..(w * h))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 0xFF) as i32
            })
            .collect()
    };
    let r = pixels(0x1234_5678);
    let g = pixels(0x9ABC_DEF0);
    let b = pixels(0x0FED_CBA9);
    let frame = make_rgb_decoded_frame(r, g, b, None, w, h, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_distinct_per_slot_qts_indexes() {
    // `quant_table_set_index = [0, 1]` routes the luma slot through
    // QTS 0 and the chroma slot (G + B) through QTS 1. The two Sets
    // have distinct `context_count`, so the per-slot VLC window
    // sizing must follow the slot's resolved Set, not Plane 0's.
    // Tests the slot-to-QTS routing on the Golomb-Rice path.
    let cr = rgb_v3_cr(1, 1, 0, 8, false);
    let qts = vec![constant_context_qts(5), constant_context_qts(13)];
    let mut header = make_header(0, 0, 1, 1, 2, 0);
    // Luma slot -> QTS 0, chroma slot -> QTS 1.
    header.quant_table_set_index[0] = 0;
    header.quant_table_set_index[1] = 1;
    let r: Vec<i32> = (0..24).map(|i| (i * 19 + 1) & 0xFF).collect();
    let g: Vec<i32> = (0..24).map(|i| (i * 23 + 7) & 0xFF).collect();
    let b: Vec<i32> = (0..24).map(|i| (i * 29 + 11) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 6, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_extra_plane_distinct_slot() {
    // With `extra_plane == true` the §4.6.6 mapping is
    // `(0=luma, 1=chroma, 1=chroma, 2=alpha)`. The alpha Plane lands
    // in its own §4.6.6 slot, so its VLC window is independent of the
    // colour Planes. Run-triple is per-Plane on all four. A taller
    // shape (8 rows) means the run-mode triple gets exercised across
    // many row boundaries, where a slot-shared (wrongly) run-mode
    // triple would corrupt the second Plane's run accounting.
    let cr = rgb_v3_cr(1, 1, 0, 8, true);
    let qts = vec![constant_context_qts(7)];
    let header = make_header(0, 0, 1, 1, 3, 0);
    let (w, h) = (8u32, 8u32);
    let pixels = |seed: u32| -> Vec<i32> {
        let mut x = seed;
        (0..(w * h))
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 0xFF) as i32
            })
            .collect()
    };
    let r = pixels(0xDEAD_BEEF);
    let g = pixels(0xCAFE_F00D);
    let b = pixels(0xFEED_BABE);
    let a = pixels(0xBAAD_F00D);
    let frame = make_rgb_decoded_frame(r, g, b, Some(a), w, h, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_run_mode_dominates_per_plane() {
    // Each Plane is a constant (zero diff after RCT). Run mode
    // dominates the per-row encode for every Plane; if the run-mode
    // triple were (incorrectly) shared at the slot level the
    // chroma-slot Planes (G + B) would emit a non-keyframe-init
    // run state on G's first row when it should have just-reset.
    // Encoder ↔ decoder are in lockstep either way, but the
    // run-mode reset path is the most sensitive to per-Plane state
    // confusion; this is the boundary test for it.
    let cr = rgb_v3_cr(1, 1, 0, 8, false);
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (w, h) = (10u32, 6u32);
    let r = vec![100; (w * h) as usize];
    let g = vec![140; (w * h) as usize];
    let b = vec![180; (w * h) as usize];
    let frame = make_rgb_decoded_frame(r, g, b, None, w, h, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_encode_round_trips_golomb_rice_2x2_slice_grid_with_alpha() {
    // 2×2 slice grid combined with an extra alpha Plane on the
    // Golomb-Rice path. Each Slice instantiates its own per-slot
    // VLC windows + per-Plane run triples — keyframe-init contract.
    let cr = rgb_v3_cr(2, 2, 0, 8, true);
    let qts = vec![constant_context_qts(7)];
    let (fw, fh) = (6u32, 4u32);
    let pixels = |seed: u32| -> Vec<i32> {
        (0..(fw * fh))
            .map(|i| ((i * seed) ^ 0x5A) & 0xFF)
            .map(|v| v as i32)
            .collect()
    };
    let frame = make_rgb_decoded_frame(
        pixels(19),
        pixels(23),
        pixels(29),
        Some(pixels(31)),
        fw,
        fh,
        8,
    );
    let headers = vec![
        make_header(0, 0, 1, 1, 3, 0),
        make_header(1, 0, 1, 1, 3, 0),
        make_header(0, 1, 1, 1, 3, 0),
        make_header(1, 1, 1, 1, 3, 0),
    ];
    assert_rgb_round_trip(&cr, &qts, &headers, &frame, true);
}

#[test]
fn rgb_encode_rejects_out_of_range_coder_type() {
    // `coder_type` values outside `0..=2` still surface
    // `UnsupportedCoderType` (RFC 9043 §4.2.3 only defines 0/1/2).
    let cr = rgb_v3_cr(1, 1, 7, 8, false);
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let frame = make_rgb_decoded_frame(vec![0; 4], vec![0; 4], vec![0; 4], None, 2, 2, 8);
    let r = encode_frame_rgb(&frame, &cr, &qts, &[header], true);
    assert!(matches!(r, Err(Error::UnsupportedCoderType(7))));
}

#[test]
fn rgb_encode_rejects_zero_frame_dims() {
    let cr = rgb_v3_cr(1, 1, 1, 8, false);
    let qts = vec![constant_context_qts(3)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let mut frame = make_rgb_decoded_frame(vec![0; 4], vec![0; 4], vec![0; 4], None, 2, 2, 8);
    frame.width = 0;
    let r = encode_frame_rgb(&frame, &cr, &qts, &[header], true);
    assert!(matches!(r, Err(Error::InvalidFramePixelDimensions { .. })));
}

// -- genuine multi-context QTS (regression for the §3.5 routing bug) --
//
// The Golomb-Rice content encoder (`encode_line`, shared with the YCbCr
// path) used to route the §3.5 context off the per-pixel `diff` values;
// the production decoder routes it off the reconstructed *Sample*
// neighbours. A single-context table hid the divergence (constant
// context); a multi-context table whose context depends on `l - tl`
// surfaces it. These pin the RGB / line-major Golomb path round-trips
// with such a table.
//
// `tables[0]` magnitude is always ≥ 1 so the absolute context is never
// 0 and the §3.8.2.2 run mode never engages — isolating the §3.5
// routing fix from the orthogonal run-mode-encoder limitation (see the
// YCbCr `ramp_context_qts` doc-comment).
fn ramp_context_qts() -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    for (d, slot) in tables[0].iter_mut().enumerate() {
        let sv = if d < 128 { d as i32 } else { d as i32 - 256 };
        let mag = 1 + (sv.unsigned_abs() as i32).min(48) / 16; // 1..=4
        *slot = if sv < 0 { -mag } else { mag };
    }
    QuantizationTableSet {
        tables,
        context_count: 5,
    }
}

#[test]
fn rgb_golomb_multi_context_qts_single_slice_round_trips() {
    let cr = rgb_v3_cr(1, 1, 0, 8, false); // coder_type == 0 (Golomb-Rice)
    let qts = vec![ramp_context_qts()];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..32).map(|i| (i * 7 + 3) & 0xFF).collect();
    let g: Vec<i32> = (0..32).map(|i| (i * 11 + 5) & 0xFF).collect();
    let b: Vec<i32> = (0..32).map(|i| (i * 13 + 7) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 8, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn rgb_golomb_multi_context_qts_2x2_grid_round_trips() {
    let cr = rgb_v3_cr(2, 2, 0, 8, false);
    let qts = vec![ramp_context_qts()];
    let headers = [
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    let (w, h) = (8u32, 8u32);
    let r: Vec<i32> = (0..(w * h) as i32).map(|i| (i * 7 + 3) & 0xFF).collect();
    let g: Vec<i32> = (0..(w * h) as i32).map(|i| (i * 11 + 5) & 0xFF).collect();
    let b: Vec<i32> = (0..(w * h) as i32).map(|i| (i * 13 + 7) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, w, h, 8);
    assert_rgb_round_trip(&cr, &qts, &headers, &frame, true);
}

#[test]
fn rgb_range_multi_context_qts_single_slice_round_trips() {
    // The range-coded RGB path already handled multi-context tables;
    // this pins parity with the Golomb path on the same table.
    let cr = rgb_v3_cr(1, 1, 1, 8, false);
    let qts = vec![ramp_context_qts()];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let r: Vec<i32> = (0..32).map(|i| (i * 7 + 3) & 0xFF).collect();
    let g: Vec<i32> = (0..32).map(|i| (i * 11 + 5) & 0xFF).collect();
    let b: Vec<i32> = (0..32).map(|i| (i * 13 + 7) & 0xFF).collect();
    let frame = make_rgb_decoded_frame(r, g, b, None, 8, 4, 8);
    assert_rgb_round_trip(&cr, &qts, &[header], &frame, true);
}
