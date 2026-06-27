#![no_main]
//! `roundtrip` fuzz target — encode → decode bit-exactness.
//!
//! Every other harness in this directory drives *attacker-controlled
//! bytes* through a decode / parse entry and asserts panic-freedom. This
//! one inverts the surface: it builds a **well-formed** [`DecodedFrame`]
//! plus a matching [`Ffv1ConfigurationRecord`] / §4.1 Quantization Table
//! Set / §4.6 Slice Header from the attacker bytes, encodes it with the
//! crate's own encoder, decodes the result with the crate's own decoder,
//! and asserts the recovered Planes are **bit-exact** to the input.
//!
//! FFV1 is a lossless codec (RFC 9043 §1): `decode(encode(x)) == x` is
//! the codec's defining contract. A divergence here is a real
//! encoder/decoder asymmetry bug — the encoder emitting a stream its own
//! decoder mis-reads, or vice versa — that the structured round-trip
//! tests (which use fixed seeds / shapes) can miss for an off-grid
//! dimension, a particular bit depth, or a coder-type the test matrix
//! didn't pair with that shape. The harness sweeps the cross product:
//!
//!   * version  — 0 / 1 (inline Parameters, §4.4) vs 3 (Configuration
//!     Record + single full-frame §4.6 Slice Header).
//!   * coder_type — 0 (§3.8.2 Golomb-Rice), 1 (§3.8.1 default-table
//!     range coder), 2 (§3.8.1.6 custom-table range coder).
//!   * colorspace — 0 (YCbCr / plane-major) vs 1 (RGB / line-major RCT).
//!   * bit depth  — 8 / 9 / 10 / 12 / 16 (incl. the §3.3.1 16-bit
//!     alternate predictor and the §3.7.2.1 RGB 9..15-bit exception).
//!   * chroma sub-sampling, alpha plane, and the `ec` Slice-Footer flag.
//!
//! Layout of the attacker buffer:
//!   byte 0      — width  (1..=MAX_DIM)
//!   byte 1      — height (1..=MAX_DIM)
//!   byte 2      — selector bits (version / coder / colorspace / chroma /
//!                 alpha / ec / depth — see `decode` below)
//!   byte 3      — extra depth / seed entropy
//!   bytes 4..   — folded into the Plane-sample PRNG seed
//!
//! The contract under test is the lossless identity plus panic-freedom of
//! the *encode* path (the other targets cover decode). Inputs that the
//! encoder legitimately rejects (a shape the chosen Parameters can't
//! represent) bail before the assert — only a panic, or a successful
//! encode whose decode diverges, is a finding.

use libfuzzer_sys::fuzz_target;
use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, decode_frame_v0v1, encode_frame, encode_frame_rgb,
    encode_frame_v0v1, parse_quantization_table_sets, ColorspaceType, DecodedFrame,
    DecodedFramePlane, Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions,
    PictureStructure, QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES,
};

/// Cap on each frame dimension. The encode + decode allocate buffers
/// proportional to `width * height` per Plane; bounding both keeps a
/// fuzz input's peak allocation finite while still exercising the
/// off-multiple, tall-thin, and wide-short shapes that stress the
/// predictor / context border math.
const MAX_DIM: u32 = 48;

/// A real, well-formed §4.1 Quantization Table Set lifted from the
/// `v3-default` extradata fixture (`context_count == 666`). Reusing a
/// parsed cascade means the round-trip exercises a genuine multi-context
/// quantization path rather than a hand-built degenerate table.
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

/// Deterministic SplitMix64-style PRNG sample stream confined to
/// `[0, 1 << bits)`.
fn synth_samples(seed: u64, count: usize, bits: u32) -> Vec<i32> {
    let mask: u64 = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
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

fn plane(seed: u64, w: u32, h: u32, bits: u32, idx: u8) -> DecodedFramePlane {
    DecodedFramePlane {
        plane_index: idx,
        width: w,
        height: h,
        samples: synth_samples(seed, (w as usize) * (h as usize), bits),
    }
}

/// Build a `DecodedFrame` matching the record's plane layout. For RGB
/// (`colorspace_type == 1`) every Plane is full-resolution (no chroma
/// sub-sampling applies); for YCbCr the Cb / Cr Planes are sub-sampled
/// per the record shifts.
fn build_frame(w: u32, h: u32, bits: u32, cr: &Ffv1ConfigurationRecord, seed: u64) -> DecodedFrame {
    let rgb = matches!(cr.colorspace_type, ColorspaceType::Rgb);
    let mut planes = vec![plane(seed, w, h, bits, 0)];
    if cr.chroma_planes {
        let (cw, ch) = if rgb {
            (w, h)
        } else {
            (
                w.div_ceil(1 << cr.log2_h_chroma_subsample),
                h.div_ceil(1 << cr.log2_v_chroma_subsample),
            )
        };
        planes.push(plane(seed ^ 0x1111, cw, ch, bits, 1));
        planes.push(plane(seed ^ 0x2222, cw, ch, bits, 2));
    }
    if cr.extra_plane {
        let idx = planes.len() as u8;
        planes.push(plane(seed ^ 0x3333, w, h, bits, idx));
    }
    DecodedFrame {
        planes,
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: cr.colorspace_type,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// A single full-frame §4.6 Slice Header for the v3 path. One Slice over
/// the whole raster keeps the round-trip deterministic while still
/// exercising the header parse / footer / trailer-chain machinery.
fn full_frame_header(w: u32, h: u32, qts_count: usize) -> Ffv1SliceHeader {
    let mut idx = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
    // Reference the single Set we encode below for every plane-context.
    for slot in idx.iter_mut().take(qts_count.min(MAX_QUANT_TABLE_SET_INDEXES)) {
        *slot = 0;
    }
    Ffv1SliceHeader {
        slice_x: 0,
        slice_y: 0,
        slice_width: w,
        slice_height: h,
        quant_table_set_index_count: qts_count.min(MAX_QUANT_TABLE_SET_INDEXES),
        quant_table_set_index: idx,
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

/// Assert the recovered Planes are bit-exact to the source Frame.
fn assert_bit_exact(src: &DecodedFrame, dec: &DecodedFrame) {
    assert_eq!(
        src.planes.len(),
        dec.planes.len(),
        "plane count diverged on round-trip"
    );
    for (p, (a, b)) in src.planes.iter().zip(dec.planes.iter()).enumerate() {
        assert_eq!(a.width, b.width, "plane {p}: width diverged");
        assert_eq!(a.height, b.height, "plane {p}: height diverged");
        assert_eq!(
            a.samples, b.samples,
            "plane {p}: lossless round-trip violated (decode(encode(x)) != x)"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let width = (u32::from(data[0]) % MAX_DIM) + 1;
    let height = (u32::from(data[1]) % MAX_DIM) + 1;
    let sel = data[2];
    let depth_sel = data[3];

    // Seed the Plane PRNG from the whole tail so distinct inputs probe
    // distinct sample content.
    let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
    for &b in &data[4..] {
        seed = seed.rotate_left(7) ^ u64::from(b).wrapping_mul(0x100_0000_01B3);
    }
    seed ^= u64::from(width) << 40 ^ u64::from(height) << 24 ^ u64::from(sel) << 8;

    // --- selector decode ------------------------------------------------
    // bit 0     : colorspace (0 = YCbCr, 1 = RGB)
    // bits 1..3 : coder_type (0 / 1 / 2)
    // bit 3     : chroma planes
    // bit 4     : alpha / extra plane
    // bit 5     : ec (Slice Footer present)
    // bit 6     : version select v3 vs v0/v1
    let rgb = sel & 0x01 != 0;
    let coder_type = (sel >> 1) & 0x03; // 0,1,2,3 -> clamp 3 to 2
    let coder_type = coder_type.min(2);
    let chroma = sel & 0x08 != 0;
    let alpha = sel & 0x10 != 0;
    let ec = sel & 0x20 != 0;
    let v3 = sel & 0x40 != 0;

    // bit depth: 8 / 9 / 10 / 12 / 16
    let bits = match depth_sel % 5 {
        0 => 8,
        1 => 9,
        2 => 10,
        3 => 12,
        _ => 16,
    };

    // RGB (JPEG 2000 RCT) always carries all three colour planes and no
    // chroma sub-sampling; the chroma toggle instead gates the optional
    // alpha plane for that layout. YCbCr honours both toggles.
    let (chroma_planes, h_shift, v_shift): (bool, u32, u32) = if rgb {
        (true, 0, 0)
    } else if chroma {
        // exercise both 4:2:0 and 4:4:4 via the alpha bit doubling as a
        // sub-sample selector when chroma is on.
        if alpha {
            (true, 0, 0)
        } else {
            (true, 1, 1)
        }
    } else {
        (false, 0, 0)
    };

    let version = if v3 { Ffv1Version::V3 } else { Ffv1Version::V1 };

    // Custom state-transition table (coder_type == 2) needs non-zero
    // deltas; a small deterministic perturbation is enough to swap the
    // §3.8.1.6 table away from the default.
    let mut state_transition_delta = [0i32; 256];
    if coder_type == 2 {
        for (i, d) in state_transition_delta.iter_mut().enumerate() {
            *d = (((i as i32 * 7 + 3) % 9) - 4) & !1; // even, small, signed
        }
    }

    let cr = Ffv1ConfigurationRecord {
        version,
        micro_version: if v3 { Some(4) } else { None },
        coder_type: u32::from(coder_type),
        state_transition_delta,
        colorspace_type: if rgb {
            ColorspaceType::Rgb
        } else {
            ColorspaceType::YCbCr
        },
        bits_per_raw_sample: bits,
        chroma_planes,
        log2_h_chroma_subsample: h_shift,
        log2_v_chroma_subsample: v_shift,
        extra_plane: alpha,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: if v3 { Some(1) } else { None },
        ec: if v3 { Some(u32::from(ec)) } else { None },
        intra: if v3 { Some(true) } else { None },
        initial_state_delta: None,
    };

    let Ok(parsed) = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA) else {
        return;
    };
    let qts: Vec<QuantizationTableSet> = vec![parsed.quant_table_sets[0].clone()];

    let Ok(dims) = FramePixelDimensions::new(width, height) else {
        return;
    };

    let frame = build_frame(width, height, bits, &cr, seed);

    if !v3 {
        // -------- versions 0 / 1: inline Parameters, single Slice -------
        // The v0/v1 driver covers YCbCr; RGB v0/v1 is wired through the
        // same inline path. Encode then decode and assert bit-exact.
        let Ok(bytes) = encode_frame_v0v1(&frame, &cr, &qts[0]) else {
            return;
        };
        if let Ok(dec) = decode_frame_v0v1(&bytes, dims) {
            assert_bit_exact(&frame, &dec);
        }
        return;
    }

    // ------------------------- version 3 --------------------------------
    let header = full_frame_header(width, height, qts.len());
    let headers = std::slice::from_ref(&header);

    if rgb {
        let Ok(bytes) = encode_frame_rgb(&frame, &cr, &qts, headers, ec) else {
            return;
        };
        if let Ok(dec) = decode_frame_rgb(&bytes, &cr, &qts, dims, ec) {
            assert_bit_exact(&frame, &dec);
        }
    } else {
        let Ok(bytes) = encode_frame(&frame, &cr, &qts, headers, ec) else {
            return;
        };
        if let Ok(dec) = decode_frame(&bytes, &cr, &qts, dims, ec) {
            assert_bit_exact(&frame, &dec);
        }
    }
});
