//! Encode → decode self-round-trip for FFV1 **versions 0 and 1**
//! single-Slice YCbCr Frames (RFC 9043 §4.4 / §4.5 / §4.7).
//!
//! The v0/v1 path differs structurally from v3: the §4.2 Parameters and
//! the single §4.1 Quantization Table Set are carried **inline in the
//! Frame** (RFC 9043 §4.4), there is exactly one implied Slice over the
//! whole raster (no §4.6 Slice Header, no §4.9 Slice Footer, no §4.9.1
//! trailer chain), and the §4.2.16 `ec` / §4.2.17 `intra` tail is absent
//! (`version >= 3`-only).
//!
//! These tests build a v0/v1 Configuration Record + a real §4.1
//! Quantization Table Set (lifted from a parsed v3 fixture so the
//! cascade is genuinely well-formed), encode a frame of synthetic
//! Samples via [`encode_frame_v0v1`], and assert
//! [`decode_frame_v0v1`] recovers every Plane bit-exactly — the
//! milestone's lossless-recovery proof for the v0/v1 driver.

use oxideav_ffv1::{
    decode_frame_v0v1, decode_frame_v0v1_inter, encode_frame_v0v1, encode_frame_v0v1_inter,
    parse_quantization_table_sets, parse_v0v1_frame_prologue, ColorspaceType, Error,
    Ffv1ConfigurationRecord, Ffv1Version, FramePixelDimensions, QuantizationTableSet,
};

/// `v3-default` extradata (see `tests/fixture_config_encode.rs`). Its
/// first parsed Quantization Table Set is a real, well-formed §4.1
/// cascade (`context_count == 666`) we reuse as the single v0/v1 Set.
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

/// Pull a genuine, well-formed §4.1 Quantization Table Set out of the
/// parsed `v3-default` fixture cascade.
fn real_quant_table_set() -> QuantizationTableSet {
    let parsed = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).expect("parse v3-default");
    parsed.quant_table_sets[0].clone()
}

/// Build a v0 / v1 Configuration Record (single implied Slice, YCbCr,
/// range-coder default table). `chroma` toggles the Cb / Cr planes;
/// `extra` toggles the alpha plane.
fn v0v1_record(
    version: Ffv1Version,
    bits: u32,
    chroma: bool,
    extra: bool,
) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version,
        micro_version: None,
        coder_type: 1,
        state_transition_delta: [0i32; 256],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: bits,
        chroma_planes: chroma,
        log2_h_chroma_subsample: if chroma { 1 } else { 0 },
        log2_v_chroma_subsample: if chroma { 1 } else { 0 },
        extra_plane: extra,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: None,
        ec: None,
        intra: None,
        initial_state_delta: None,
    }
}

/// Deterministic pseudo-random Samples in `0 .. 2^bits` from a 64-bit
/// SplitMix-style generator — exercises real predictor / context paths
/// without a fixture file.
fn synth_samples(seed: u64, count: usize, bits: u32) -> Vec<i32> {
    let mask: u64 = (1u64 << bits) - 1;
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..count)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z & mask) as i32
        })
        .collect()
}

use oxideav_ffv1::{DecodedFrame, DecodedFramePlane};

fn plane(seed: u64, w: u32, h: u32, bits: u32, idx: u8) -> DecodedFramePlane {
    DecodedFramePlane {
        plane_index: idx,
        width: w,
        height: h,
        samples: synth_samples(seed, (w * h) as usize, bits),
    }
}

fn build_frame(w: u32, h: u32, bits: u32, cr: &Ffv1ConfigurationRecord, seed: u64) -> DecodedFrame {
    let mut planes = vec![plane(seed, w, h, bits, 0)];
    if cr.chroma_planes {
        let cw = w.div_ceil(1 << cr.log2_h_chroma_subsample);
        let ch = h.div_ceil(1 << cr.log2_v_chroma_subsample);
        planes.push(plane(seed ^ 0x1111, cw, ch, bits, 1));
        planes.push(plane(seed ^ 0x2222, cw, ch, bits, 2));
    }
    if cr.extra_plane {
        planes.push(plane(seed ^ 0x3333, w, h, bits, planes.len() as u8));
    }
    DecodedFrame {
        planes,
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// Build a v0/v1 Golomb-Rice (`coder_type == 0`) Configuration Record.
fn v0v1_record_golomb(
    version: Ffv1Version,
    bits: u32,
    chroma: bool,
    extra: bool,
) -> Ffv1ConfigurationRecord {
    let mut cr = v0v1_record(version, bits, chroma, extra);
    cr.coder_type = 0;
    cr
}

/// Build a frame whose every Plane's **top-left Sample is 0**. The §3.8.2
/// Golomb-Rice encode path cannot represent a non-zero Sample Difference
/// at the first Sample of a run region (absolute context 0, the
/// all-zero-neighbour first pixel) — this is a documented `coder_type ==
/// 0` encoder limitation (`Error::RunModeFirstPixelNonZero`); the range
/// coder carries such pixels without restriction. Forcing the corner to 0
/// keeps the synthetic Golomb round-trip inside the representable set
/// while still exercising the full predictor / context / run-mode
/// machinery across the rest of the Plane.
fn build_frame_golomb_safe(
    w: u32,
    h: u32,
    bits: u32,
    cr: &Ffv1ConfigurationRecord,
    seed: u64,
) -> DecodedFrame {
    let mut frame = build_frame(w, h, bits, cr, seed);
    for plane in frame.planes.iter_mut() {
        if let Some(first) = plane.samples.first_mut() {
            *first = 0;
        }
    }
    frame
}

fn assert_roundtrip_frame(cr: &Ffv1ConfigurationRecord, frame: &DecodedFrame) {
    let qts = real_quant_table_set();
    let bytes = encode_frame_v0v1(frame, cr, &qts).expect("encode v0/v1 frame");
    let dims = FramePixelDimensions::new(frame.width, frame.height).expect("dims");
    let decoded = decode_frame_v0v1(&bytes, dims).expect("decode v0/v1 frame");
    assert_eq!(decoded.planes.len(), frame.planes.len());
    for (p, (got, want)) in decoded.planes.iter().zip(frame.planes.iter()).enumerate() {
        assert_eq!(
            got.samples, want.samples,
            "plane {p} samples must be bit-exact lossless"
        );
    }
}

fn assert_roundtrip(cr: &Ffv1ConfigurationRecord, w: u32, h: u32, bits: u32, seed: u64) {
    let qts = real_quant_table_set();
    let frame = build_frame(w, h, bits, cr, seed);
    let bytes = encode_frame_v0v1(&frame, cr, &qts).expect("encode v0/v1 frame");
    let dims = FramePixelDimensions::new(w, h).expect("dims");
    let decoded = decode_frame_v0v1(&bytes, dims).expect("decode v0/v1 frame");

    assert_eq!(decoded.width, w);
    assert_eq!(decoded.height, h);
    assert!(decoded.keyframe);
    assert_eq!(decoded.colorspace, ColorspaceType::YCbCr);
    assert_eq!(decoded.bits_per_raw_sample, bits);
    assert_eq!(
        decoded.planes.len(),
        frame.planes.len(),
        "plane count must round-trip"
    );
    for (p, (got, want)) in decoded.planes.iter().zip(frame.planes.iter()).enumerate() {
        assert_eq!(got.width, want.width, "plane {p} width");
        assert_eq!(got.height, want.height, "plane {p} height");
        assert_eq!(
            got.samples, want.samples,
            "plane {p} samples must be bit-exact lossless"
        );
    }
}

#[test]
fn v1_gray_8bit_round_trips() {
    let cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    assert_roundtrip(&cr, 17, 13, 8, 0xABCD);
}

#[test]
fn v0_gray_8bit_round_trips() {
    // Version 0 omits bits_per_raw_sample on the wire (implied 8).
    let cr = v0v1_record(Ffv1Version::V0, 8, false, false);
    assert_roundtrip(&cr, 9, 7, 8, 0x1234);
}

#[test]
fn v1_yuv420_8bit_round_trips() {
    let cr = v0v1_record(Ffv1Version::V1, 8, true, false);
    assert_roundtrip(&cr, 16, 16, 8, 0x5555);
}

#[test]
fn v1_yuv444_8bit_round_trips() {
    let mut cr = v0v1_record(Ffv1Version::V1, 8, true, false);
    cr.log2_h_chroma_subsample = 0;
    cr.log2_v_chroma_subsample = 0;
    assert_roundtrip(&cr, 12, 10, 8, 0x7777);
}

#[test]
fn v1_yuva420_8bit_with_alpha_round_trips() {
    let cr = v0v1_record(Ffv1Version::V1, 8, true, true);
    assert_roundtrip(&cr, 16, 12, 8, 0x9999);
}

#[test]
fn v1_gray_10bit_round_trips() {
    let cr = v0v1_record(Ffv1Version::V1, 10, false, false);
    assert_roundtrip(&cr, 15, 11, 10, 0xC0DE);
}

#[test]
fn v1_gray_16bit_round_trips() {
    // 16-bit exercises the §3.3.1 alternate median predictor on the
    // range-coder path.
    let cr = v0v1_record(Ffv1Version::V1, 16, false, false);
    assert_roundtrip(&cr, 14, 9, 16, 0xFACE);
}

#[test]
fn v1_single_pixel_round_trips() {
    let cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    assert_roundtrip(&cr, 1, 1, 8, 0x4242);
}

#[test]
fn v1_tall_thin_round_trips() {
    let cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    assert_roundtrip(&cr, 1, 40, 8, 0x2468);
}

#[test]
fn v1_wide_short_round_trips() {
    let cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    assert_roundtrip(&cr, 40, 1, 8, 0x1357);
}

#[test]
fn v1_golomb_gray_8bit_round_trips() {
    let cr = v0v1_record_golomb(Ffv1Version::V1, 8, false, false);
    assert_roundtrip_frame(&cr, &build_frame_golomb_safe(17, 13, 8, &cr, 0xA1B2));
}

#[test]
fn v0_golomb_gray_8bit_round_trips() {
    let cr = v0v1_record_golomb(Ffv1Version::V0, 8, false, false);
    assert_roundtrip_frame(&cr, &build_frame_golomb_safe(11, 9, 8, &cr, 0xC3D4));
}

#[test]
fn v1_golomb_yuv420_8bit_round_trips() {
    let cr = v0v1_record_golomb(Ffv1Version::V1, 8, true, false);
    assert_roundtrip_frame(&cr, &build_frame_golomb_safe(16, 16, 8, &cr, 0xE5F6));
}

#[test]
fn v1_golomb_yuva420_8bit_round_trips() {
    let cr = v0v1_record_golomb(Ffv1Version::V1, 8, true, true);
    assert_roundtrip_frame(&cr, &build_frame_golomb_safe(16, 12, 8, &cr, 0x0718));
}

#[test]
fn v1_golomb_inter_non_keyframe_round_trips() {
    let cr = v0v1_record_golomb(Ffv1Version::V1, 8, false, false);
    let qts = real_quant_table_set();
    let key_frame = build_frame_golomb_safe(12, 10, 8, &cr, 0xAB01);
    let key_bytes = encode_frame_v0v1(&key_frame, &cr, &qts).expect("encode golomb keyframe");
    let prologue = parse_v0v1_frame_prologue(&key_bytes).expect("parse prologue");
    assert_eq!(prologue.record.coder_type, 0);

    let inter_frame = build_frame_golomb_safe(12, 10, 8, &cr, 0xCD02);
    let inter_bytes =
        encode_frame_v0v1_inter(&inter_frame, &prologue.record, &prologue.quant_table_set)
            .expect("encode golomb non-keyframe");
    let dims = FramePixelDimensions::new(12, 10).expect("dims");
    let decoded = decode_frame_v0v1_inter(
        &inter_bytes,
        &prologue.record,
        &prologue.quant_table_set,
        dims,
    )
    .expect("decode golomb non-keyframe");
    assert!(!decoded.keyframe);
    assert_eq!(decoded.planes[0].samples, inter_frame.planes[0].samples);
}

#[test]
fn prologue_parse_recovers_record_and_quant_set() {
    let cr = v0v1_record(Ffv1Version::V1, 8, true, false);
    let qts = real_quant_table_set();
    let frame = build_frame(8, 8, 8, &cr, 0xBEEF);
    let bytes = encode_frame_v0v1(&frame, &cr, &qts).expect("encode");

    let prologue = parse_v0v1_frame_prologue(&bytes).expect("parse prologue");
    assert!(prologue.keyframe);
    assert_eq!(prologue.record.version, Ffv1Version::V1);
    assert_eq!(prologue.record.coder_type, 1);
    assert!(prologue.record.chroma_planes);
    assert_eq!(prologue.record.num_h_slices, Some(1));
    assert_eq!(prologue.record.num_v_slices, Some(1));
    // The inline cascade reproduces the same context_count we encoded.
    assert_eq!(prologue.quant_table_set.context_count, qts.context_count);
    assert_eq!(prologue.quant_table_set.tables, qts.tables);
}

#[test]
fn v0_record_keeps_bits_at_8() {
    // Version 0 carries no bits_per_raw_sample; the decoder must imply 8.
    let cr = v0v1_record(Ffv1Version::V0, 8, false, false);
    let qts = real_quant_table_set();
    let frame = build_frame(6, 6, 8, &cr, 0x0F0F);
    let bytes = encode_frame_v0v1(&frame, &cr, &qts).expect("encode v0");
    let prologue = parse_v0v1_frame_prologue(&bytes).expect("parse v0 prologue");
    assert_eq!(prologue.record.version, Ffv1Version::V0);
    assert_eq!(prologue.record.bits_per_raw_sample, 8);
}

#[test]
fn encode_rejects_v3_record() {
    let mut cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    cr.version = Ffv1Version::V3;
    cr.micro_version = Some(4);
    cr.quant_table_set_count = Some(1);
    let qts = real_quant_table_set();
    let frame = build_frame(4, 4, 8, &cr, 0);
    assert!(matches!(
        encode_frame_v0v1(&frame, &cr, &qts),
        Err(Error::InFrameParametersForbiddenForVersion(3))
    ));
}

#[test]
fn encode_rejects_rgb() {
    let mut cr = v0v1_record(Ffv1Version::V1, 8, true, false);
    cr.colorspace_type = ColorspaceType::Rgb;
    let qts = real_quant_table_set();
    let frame = build_frame(4, 4, 8, &cr, 0);
    assert!(matches!(
        encode_frame_v0v1(&frame, &cr, &qts),
        Err(Error::ColorspaceLayoutNotImplemented)
    ));
}

#[test]
fn encode_rejects_coder_type_2() {
    let mut cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    cr.coder_type = 2;
    let qts = real_quant_table_set();
    let frame = build_frame(4, 4, 8, &cr, 0);
    assert!(matches!(
        encode_frame_v0v1(&frame, &cr, &qts),
        Err(Error::UnsupportedCoderType(2))
    ));
}

#[test]
fn decode_inter_rejects_keyframe_stream() {
    // A keyframe-encoded Frame fed to the inter entry must be rejected.
    let cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    let qts = real_quant_table_set();
    let frame = build_frame(8, 8, 8, &cr, 0x55AA);
    let bytes = encode_frame_v0v1(&frame, &cr, &qts).expect("encode");
    let dims = FramePixelDimensions::new(8, 8).expect("dims");
    assert!(matches!(
        decode_frame_v0v1_inter(&bytes, &cr, &qts, dims),
        Err(Error::UnexpectedKeyframeInInterDecode)
    ));
}

#[test]
fn keyframe_entry_rejects_non_keyframe_stream() {
    // decode_frame_v0v1 requires the §4.4 keyframe bit == 1 (a keyframe
    // carries the inline Parameters). A non-keyframe Frame must be routed
    // to decode_frame_v0v1_inter instead.
    let cr = v0v1_record(Ffv1Version::V1, 8, false, false);
    let qts = real_quant_table_set();
    let frame = build_frame(8, 8, 8, &cr, 0x1010);
    let bytes = encode_frame_v0v1_inter(&frame, &cr, &qts).expect("encode non-keyframe");
    let dims = FramePixelDimensions::new(8, 8).expect("dims");
    assert!(matches!(
        decode_frame_v0v1(&bytes, dims),
        Err(Error::NonKeyframeHasNoInFrameParameters)
    ));
}

#[test]
fn v1_inter_non_keyframe_round_trips() {
    // A v0/v1 non-keyframe carries no inline Parameters — it reuses the
    // governing keyframe's record + Quantization Table Set, recovered via
    // decode_frame_v0v1_inter. Encode a keyframe, parse its prologue to
    // obtain the reused config, then round-trip a non-keyframe body.
    let cr = v0v1_record(Ffv1Version::V1, 8, true, false);
    let qts = real_quant_table_set();

    // Keyframe establishes the configuration.
    let key_frame = build_frame(16, 12, 8, &cr, 0xAA01);
    let key_bytes = encode_frame_v0v1(&key_frame, &cr, &qts).expect("encode keyframe");
    let prologue = parse_v0v1_frame_prologue(&key_bytes).expect("parse keyframe prologue");

    // Non-keyframe with different content, reusing the keyframe's config.
    let inter_frame = build_frame(16, 12, 8, &cr, 0xBB02);
    let inter_bytes =
        encode_frame_v0v1_inter(&inter_frame, &prologue.record, &prologue.quant_table_set)
            .expect("encode non-keyframe");

    let dims = FramePixelDimensions::new(16, 12).expect("dims");
    let decoded = decode_frame_v0v1_inter(
        &inter_bytes,
        &prologue.record,
        &prologue.quant_table_set,
        dims,
    )
    .expect("decode non-keyframe");

    assert!(!decoded.keyframe);
    assert_eq!(decoded.planes.len(), inter_frame.planes.len());
    for (p, (got, want)) in decoded
        .planes
        .iter()
        .zip(inter_frame.planes.iter())
        .enumerate()
    {
        assert_eq!(
            got.samples, want.samples,
            "non-keyframe plane {p} must be bit-exact"
        );
    }
}
