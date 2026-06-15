//! YCbCr chroma-planes round-trip tests for the unified [`encode_frame`]
//! dispatcher (RFC 9043 §4.2.7 / §4.2.8 / §4.2.9 / §4.7).
//!
//! The pre-existing `tests/range_encode_frame.rs` only covers
//! single-Plane grayscale (`chroma_planes = false`) on the range-coded
//! path; the Golomb-Rice driver had no integration test for
//! `chroma_planes = true` either. This suite drives every supported
//! combination of:
//!
//!   * §4.2.3 `coder_type` — `0` (Golomb-Rice), `1` (range coder,
//!     default state-transition table), `2` (range coder, deltas — here
//!     just the all-zero degenerate case that must round-trip the same
//!     bytes as `coder_type == 1`)
//!   * §4.2.7 / §4.2.8 / §4.2.9 chroma subsampling — 4:4:4 (h=0,v=0),
//!     4:2:2 (h=1,v=0), 4:2:0 (h=1,v=1)
//!   * §4.2.10 `extra_plane` — RGBA-style alpha Plane on top of the
//!     three YCbCr Planes (when `chroma_planes` is on)
//!
//! through the public [`encode_frame`] → [`decode_frame`] pipeline
//! end-to-end:
//!
//! ```text
//!   DecodedFrame { Y, Cb, Cr [, A] }
//!     ── encode_frame ──▶ Vec<u8> frame_bytes
//!     ── decode_frame ──▶ DecodedFrame
//!     == input (every Plane bit-exact)
//! ```
//!
//! The §3.3 median predictor + §3.8 modular wrap + per-Plane quant
//! routing (luma → slot 0, chroma → slot 1, extra → slot 2) wired by
//! `quant_index_slot` / `plane_origin` in `frame_encode.rs` is verified
//! here through the round-trip — a mismatched per-Plane width/height,
//! chroma origin, or quant slot routing on either side surfaces as a
//! decoded Plane that diverges from the input.

use oxideav_ffv1::{
    decode_frame, encode_frame, ColorspaceType, DecodedFrame, DecodedFramePlane,
    Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

// -- shared helpers -----------------------------------------------------

/// YCbCr-with-chroma config. Caller picks the entropy coder
/// (`coder_type ∈ {0, 1, 2}`) and the §4.2.7/§4.2.8/§4.2.9 chroma
/// subsample factors. `extra_plane` flips on the alpha Plane.
#[allow(clippy::too_many_arguments)]
fn ycbcr_v3_cr(
    coder_type: u32,
    num_h: u32,
    num_v: u32,
    bits: u32,
    log2_h_sub: u32,
    log2_v_sub: u32,
    extra_plane: bool,
    quant_table_set_count: u32,
) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: bits,
        chroma_planes: true,
        log2_h_chroma_subsample: log2_h_sub,
        log2_v_chroma_subsample: log2_v_sub,
        extra_plane,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(quant_table_set_count),
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

/// A genuine multi-context QTS that routes the §3.5 context off the
/// neighbour Sample values: `tables[0]` maps the signed low-8-bit
/// `l - tl` difference (RFC 9043 §3.5 Q0) to a quantized context that
/// varies with the `l` / `tl` Sample neighbours — distinguishing the
/// (correct) Sample-domain §3.5 routing from the (buggy) diff-domain
/// one, which a single-context table cannot. The mapping is
/// odd-symmetric (`q(-v) == -q(v)`), so the §3.5 negative-context
/// sign-flip path is exercised on both signs.
///
/// Only `tables[0]` is non-zero and its magnitude is always **≥ 1**, so
/// the summed absolute context is never 0 and the §3.8.2.2 run mode
/// (which engages only at context 0) never triggers. That isolates the
/// §3.5 routing fix this round delivers from the orthogonal §3.8.2.2.1
/// run-mode-encoder limitation (a non-zero Sample Difference at the very
/// first run-region pixel after a reset has no preceding zero-run pixel
/// to carry the run prefix, so it is not representable under the
/// per-call run state machine — a separate, larger follow-up).
fn ramp_context_qts() -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    for (d, slot) in tables[0].iter_mut().enumerate() {
        let sv = if d < 128 { d as i32 } else { d as i32 - 256 };
        let mag = 1 + (sv.unsigned_abs() as i32).min(48) / 16; // 1..=4
        *slot = if sv < 0 { -mag } else { mag };
    }
    QuantizationTableSet {
        tables,
        // Max |context| = 4 → contexts -4..=4 → index 0..=4 → 5 slots.
        context_count: 5,
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

/// Build a `DecodedFrame` with one Plane per supplied
/// `(width, height, samples)` triple. Frame-level
/// `(width, height, bits_per_raw_sample)` are taken from the first
/// Plane / arg list.
fn make_ycbcr_decoded_frame(
    bits: u32,
    frame_w: u32,
    frame_h: u32,
    planes_spec: Vec<(u32, u32, Vec<i32>)>,
) -> DecodedFrame {
    let planes: Vec<DecodedFramePlane> = planes_spec
        .into_iter()
        .enumerate()
        .map(|(p_idx, (w, h, samples))| {
            assert_eq!(
                samples.len(),
                (w as usize) * (h as usize),
                "plane {p_idx}: samples length must equal w*h"
            );
            DecodedFramePlane {
                plane_index: p_idx as u8,
                width: w,
                height: h,
                samples,
            }
        })
        .collect();
    DecodedFrame {
        planes,
        width: frame_w,
        height: frame_h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// xorshift-style deterministic pseudo-random Sample stream confined
/// to `[0, 1 << bits)`. Distinct `seed`s yield distinct Planes so the
/// per-Plane state windows don't accidentally cross-validate.
fn pseudo_random_samples(seed: u32, n: usize, bits: u32) -> Vec<i32> {
    let mask = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut s = seed.wrapping_mul(0x9E3779B1).wrapping_add(1);
    (0..n)
        .map(|_| {
            // xorshift32
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & mask) as i32
        })
        .collect()
}

fn assert_round_trip(
    cr: &Ffv1ConfigurationRecord,
    qts: &[QuantizationTableSet],
    headers: &[Ffv1SliceHeader],
    frame: &DecodedFrame,
    ec: bool,
) {
    let bytes = encode_frame(frame, cr, qts, headers, ec)
        .expect("YCbCr chroma encode must succeed for valid inputs");
    let decoded = decode_frame(
        &bytes,
        cr,
        qts,
        FramePixelDimensions::new(frame.width, frame.height).unwrap(),
        ec,
    )
    .expect("encoded YCbCr chroma frame must round-trip through decode_frame");
    assert_eq!(
        decoded.planes.len(),
        frame.planes.len(),
        "plane count must match"
    );
    for (p_idx, (got, want)) in decoded.planes.iter().zip(frame.planes.iter()).enumerate() {
        assert_eq!(got.width, want.width, "plane {p_idx}: width mismatch");
        assert_eq!(got.height, want.height, "plane {p_idx}: height mismatch");
        assert_eq!(
            got.samples, want.samples,
            "plane {p_idx}: samples diverged after round-trip"
        );
    }
    assert_eq!(decoded.width, frame.width);
    assert_eq!(decoded.height, frame.height);
    assert_eq!(decoded.bits_per_raw_sample, frame.bits_per_raw_sample);
}

// -- 4:4:4 — every chroma Plane is full-frame -------------------------

#[test]
fn range_yuv444_8bit_single_slice() {
    // 4:4:4 — chroma planes have the same dimensions as luma. The
    // chroma routing (`quant_index_slot` → slot 1) is exercised, but
    // not the subsample shifts.
    let cr = ycbcr_v3_cr(1, 1, 1, 8, 0, 0, false, 1);
    let qts = vec![constant_context_qts(7)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (6u32, 4u32);
    let y = pseudo_random_samples(1, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(2, (fw * fh) as usize, 8);
    let cr_p = pseudo_random_samples(3, (fw * fh) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (fw, fh, cb), (fw, fh, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn golomb_yuv444_8bit_single_slice() {
    // Same shape but on the Golomb-Rice (`coder_type == 0`) entropy
    // coder; `encode_frame` routes to `encode_frame_golomb_rice`.
    let cr = ycbcr_v3_cr(0, 1, 1, 8, 0, 0, false, 1);
    let qts = vec![constant_context_qts(7)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (6u32, 4u32);
    let y = pseudo_random_samples(4, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(5, (fw * fh) as usize, 8);
    let cr_p = pseudo_random_samples(6, (fw * fh) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (fw, fh, cb), (fw, fh, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// -- 4:2:2 — chroma half-width, full-height ---------------------------

#[test]
fn range_yuv422_8bit_single_slice() {
    // 4:2:2 — `log2_h_chroma_subsample == 1`. Chroma Planes are
    // half-width, full-height of luma; the §3.3 median predictor + §3.5
    // context are exercised at smaller dimensions independently of
    // luma. Round-tripping verifies `plane_origin` shifts by 1 on x for
    // chroma and that the chroma-Plane width/height math matches between
    // encode and decode.
    let cr = ycbcr_v3_cr(1, 1, 1, 8, 1, 0, false, 1);
    let qts = vec![constant_context_qts(8)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh); // 4:2:2
    let y = pseudo_random_samples(11, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(12, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(13, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn golomb_yuv422_8bit_single_slice() {
    let cr = ycbcr_v3_cr(0, 1, 1, 8, 1, 0, false, 1);
    let qts = vec![constant_context_qts(8)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh);
    let y = pseudo_random_samples(14, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(15, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(16, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// -- 4:2:0 — chroma half-width, half-height (the corpus v3-default) --

#[test]
fn range_yuv420_8bit_single_slice() {
    // 4:2:0 — `log2_h_chroma_subsample == 1`, `log2_v_chroma_subsample
    // == 1`. Chroma Planes are quarter-area of luma — the standard
    // YUV-MPEG storage; the same configuration the corpus
    // `v3-default` fixture uses (128x96 frame, 64x48 chroma).
    let cr = ycbcr_v3_cr(1, 1, 1, 8, 1, 1, false, 1);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(21, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(22, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(23, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn golomb_yuv420_8bit_single_slice() {
    let cr = ycbcr_v3_cr(0, 1, 1, 8, 1, 1, false, 1);
    let qts = vec![constant_context_qts(9)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(24, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(25, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(26, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// -- 4:2:0 + 2x2 slice grid -------------------------------------------

#[test]
fn range_yuv420_8bit_2x2_slice_grid() {
    // 2x2 slice grid on a 4:2:0 frame — `num_h_slices == 2`,
    // `num_v_slices == 2`. Each slice covers a 4x2 luma rectangle and
    // a 2x1 chroma rectangle (frame is 8x4 luma → 4x2 chroma). This
    // exercises the chroma `plane_origin` shift on per-slice origins
    // (slice (1, 0) lands at luma x=4, chroma x=2) and the
    // chroma-Plane blit math in `decode_frame`'s `blit_into`.
    let cr = ycbcr_v3_cr(1, 2, 2, 8, 1, 1, false, 1);
    let qts = vec![constant_context_qts(10)];
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(31, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(32, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(33, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    let headers = vec![
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    assert_round_trip(&cr, &qts, &headers, &frame, true);
}

#[test]
fn golomb_yuv420_8bit_2x2_slice_grid() {
    let cr = ycbcr_v3_cr(0, 2, 2, 8, 1, 1, false, 1);
    let qts = vec![constant_context_qts(10)];
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(34, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(35, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(36, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    let headers = vec![
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    assert_round_trip(&cr, &qts, &headers, &frame, true);
}

// -- 4:4:4 + extra (alpha) Plane --------------------------------------

#[test]
fn range_yuv444_with_extra_plane_8bit() {
    // `extra_plane = true` — a 4th Plane (alpha) at the same
    // resolution as luma sits at p_idx = 3. §4.6.5 derives
    // `quant_table_set_index_count = 1 + 1 + 1 = 3`; the extra Plane
    // selects slot 2. Round-tripping verifies the
    // `chroma_planes && extra_plane` quant routing matches between
    // encode and decode.
    let cr = ycbcr_v3_cr(1, 1, 1, 8, 0, 0, true, 1);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 3, 0);
    let (fw, fh) = (6u32, 4u32);
    let y = pseudo_random_samples(41, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(42, (fw * fh) as usize, 8);
    let cr_p = pseudo_random_samples(43, (fw * fh) as usize, 8);
    let alpha = pseudo_random_samples(44, (fw * fh) as usize, 8);
    let frame = make_ycbcr_decoded_frame(
        8,
        fw,
        fh,
        vec![(fw, fh, y), (fw, fh, cb), (fw, fh, cr_p), (fw, fh, alpha)],
    );
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn golomb_yuv444_with_extra_plane_8bit() {
    let cr = ycbcr_v3_cr(0, 1, 1, 8, 0, 0, true, 1);
    let qts = vec![constant_context_qts(6)];
    let header = make_header(0, 0, 1, 1, 3, 0);
    let (fw, fh) = (6u32, 4u32);
    let y = pseudo_random_samples(45, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(46, (fw * fh) as usize, 8);
    let cr_p = pseudo_random_samples(47, (fw * fh) as usize, 8);
    let alpha = pseudo_random_samples(48, (fw * fh) as usize, 8);
    let frame = make_ycbcr_decoded_frame(
        8,
        fw,
        fh,
        vec![(fw, fh, y), (fw, fh, cb), (fw, fh, cr_p), (fw, fh, alpha)],
    );
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// -- 4:2:0 + 10-bit ---------------------------------------------------

#[test]
fn range_yuv420_10bit_single_slice() {
    // 10-bit 4:2:0 — exercises the §3.8 modular wrap at the 10-bit
    // dynamic range across all three Planes.
    let cr = ycbcr_v3_cr(1, 1, 1, 10, 1, 1, false, 1);
    let qts = vec![constant_context_qts(5)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(51, (fw * fh) as usize, 10);
    let cb = pseudo_random_samples(52, (cw * ch) as usize, 10);
    let cr_p = pseudo_random_samples(53, (cw * ch) as usize, 10);
    let frame =
        make_ycbcr_decoded_frame(10, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// -- coder_type == 2 with all-zero deltas → identical to coder_type == 1

#[test]
fn range_yuv420_coder_type_2_zero_delta_matches_coder_type_1() {
    // `build_one_state` with an all-zero delta vector returns the
    // §3.8.1.5 default table, so a `coder_type == 2` encode with zero
    // deltas must produce exactly the same wire bytes as `coder_type
    // == 1` on the same input. This is the chroma-planes mirror of the
    // grayscale `range_round_trip_coder_type_2_*` checks in
    // `src/frame_encode.rs::tests`.
    let cr_one = ycbcr_v3_cr(1, 1, 1, 8, 1, 1, false, 1);
    let mut cr_two = cr_one.clone();
    cr_two.coder_type = 2;

    let qts = vec![constant_context_qts(7)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(61, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(62, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(63, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);

    let bytes_one =
        encode_frame(&frame, &cr_one, &qts, std::slice::from_ref(&header), true).unwrap();
    let bytes_two =
        encode_frame(&frame, &cr_two, &qts, std::slice::from_ref(&header), true).unwrap();
    assert_eq!(
        bytes_one, bytes_two,
        "all-zero state_transition_delta must produce identical bytes \
         under coder_type == 2 and coder_type == 1 (4:2:0)"
    );

    // And the `coder_type == 2` blob must still round-trip through
    // `decode_frame` — i.e. the decoder picks the same derived table.
    assert_round_trip(&cr_two, &qts, &[header], &frame, true);
}

// -- ec=0 footer + chroma planes --------------------------------------

#[test]
fn range_yuv420_ec0_footer_round_trips() {
    // `ec == 0` selects the 3-byte SliceFooter (no §4.9.3 CRC parity
    // word); chroma-Plane round-trip stays bit-exact.
    let cr = ycbcr_v3_cr(1, 1, 1, 8, 1, 1, false, 1);
    let qts = vec![constant_context_qts(8)];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(71, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(72, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(73, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, false);
}

// -- distinct per-plane QTS via slot routing --------------------------

#[test]
fn range_yuv420_distinct_qts_per_plane_category() {
    // Two distinct Quantization Table Sets in the cascade; the slice
    // header routes luma → set 0, chroma → set 1 via
    // `quant_table_set_index[..] = [0, 1]`. The round-trip verifies
    // that the encoder and decoder pick the same per-Plane qts AND that
    // distinct qts produce distinct (but still self-consistent)
    // per-context state windows on each Plane.
    let cr = ycbcr_v3_cr(1, 1, 1, 8, 1, 1, false, 2);
    let qts = vec![constant_context_qts(4), constant_context_qts(11)];
    let mut header = make_header(0, 0, 1, 1, 2, 0);
    header.quant_table_set_index[0] = 0;
    header.quant_table_set_index[1] = 1;
    let (fw, fh) = (8u32, 4u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(81, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(82, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(83, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

// -- genuine multi-context QTS (regression for the §3.5 routing bug) --
//
// Before this round the Golomb-Rice content encoder (`encode_line`)
// evaluated the §3.5 context from the per-pixel `diff` values it had
// pre-filled into the `current_row` buffer, while the production decoder
// (`PlaneReconstructor::reconstruct_row`) evaluates it from the
// reconstructed *Sample* neighbours. For a single-context table the
// routed context is constant, so the two agreed; for a genuinely
// multi-context table they desynced and the frame failed to round-trip.
// These tests pin the fix with a `ramp_context_qts` whose context
// depends on the `l - tl` neighbour difference.

#[test]
fn golomb_yuv444_multi_context_qts_round_trips() {
    let cr = ycbcr_v3_cr(0, 1, 1, 8, 0, 0, false, 1);
    let qts = vec![ramp_context_qts()];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (6u32, 4u32);
    let y = pseudo_random_samples(101, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(102, (fw * fh) as usize, 8);
    let cr_p = pseudo_random_samples(103, (fw * fh) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (fw, fh, cb), (fw, fh, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}

#[test]
fn golomb_yuv420_multi_context_qts_2x2_slice_grid_round_trips() {
    // Multi-slice + subsampled chroma on the Golomb path, so the
    // per-Slice / per-Plane state windows are all driven by the
    // multi-context routing.
    let cr = ycbcr_v3_cr(0, 2, 2, 8, 1, 1, false, 1);
    let qts = vec![ramp_context_qts()];
    let headers = [
        make_header(0, 0, 1, 1, 2, 0),
        make_header(1, 0, 1, 1, 2, 0),
        make_header(0, 1, 1, 1, 2, 0),
        make_header(1, 1, 1, 1, 2, 0),
    ];
    let (fw, fh) = (8u32, 8u32);
    let (cw, ch) = (fw / 2, fh / 2);
    let y = pseudo_random_samples(111, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(112, (cw * ch) as usize, 8);
    let cr_p = pseudo_random_samples(113, (cw * ch) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (cw, ch, cb), (cw, ch, cr_p)]);
    assert_round_trip(&cr, &qts, &headers, &frame, true);
}

#[test]
fn range_yuv444_multi_context_qts_round_trips() {
    // The range coder already handled multi-context tables; this pins
    // that the same `ramp_context_qts` round-trips on `coder_type == 1`
    // too, so the new helper is exercised on both entropy coders.
    let cr = ycbcr_v3_cr(1, 1, 1, 8, 0, 0, false, 1);
    let qts = vec![ramp_context_qts()];
    let header = make_header(0, 0, 1, 1, 2, 0);
    let (fw, fh) = (6u32, 4u32);
    let y = pseudo_random_samples(121, (fw * fh) as usize, 8);
    let cb = pseudo_random_samples(122, (fw * fh) as usize, 8);
    let cr_p = pseudo_random_samples(123, (fw * fh) as usize, 8);
    let frame =
        make_ycbcr_decoded_frame(8, fw, fh, vec![(fw, fh, y), (fw, fh, cb), (fw, fh, cr_p)]);
    assert_round_trip(&cr, &qts, &[header], &frame, true);
}
