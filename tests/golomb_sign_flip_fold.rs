//! Regression gate for the r411 `roundtrip`-fuzz finding: a §3.5
//! sign-flipped context whose folded Sample Difference is exactly
//! `-2^(bits-1)` negates to `+2^(bits-1)` — one past the top of the §3.8
//! `bits`-wide signed window — and the §3.8.2 Golomb-Rice symbol coder
//! cannot carry that value in `bits` (its suffix arithmetic wraps
//! modulo `2^bits`, so a decoder reads a different value back). Since
//! §3.8 Sample reconstruction is modular, `±2^(bits-1)` code the same
//! Sample; the encoder must fold the out-of-window edge back to
//! `-2^(bits-1)` before emission (`fold_coded_diff` in
//! `src/sample_diff.rs`).
//!
//! This reproduces fuzz artifact
//! `crash-b4b103e485e4597a5b519e13e6f141f74260331d`: a 47×14 version-1
//! RGBA Golomb-Rice Frame (coded RCT space, `bits + 1 == 9`) whose
//! pseudo-random Planes hit the edge on a sign-flipped context. Before
//! the fix every Plane diverged from row 2 onward on self round-trip.

use oxideav_ffv1::{
    decode_frame_v0v1, encode_frame_v0v1, parse_quantization_table_sets, ColorspaceType,
    DecodedFrame, DecodedFramePlane, Ffv1ConfigurationRecord, Ffv1Version, FramePixelDimensions,
    QuantizationTableSet,
};

/// The `v3-default` fixture extradata — a real 666-context §4.1
/// Quantization Table Set (same cascade the fuzz target uses).
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

/// SplitMix64-style PRNG stream in `[0, 1 << bits)` — byte-for-byte the
/// fuzz target's sample synthesis, so the artifact reproduces exactly.
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

fn plane(seed: u64, w: u32, h: u32, bits: u32, idx: u8) -> DecodedFramePlane {
    DecodedFramePlane {
        plane_index: idx,
        width: w,
        height: h,
        samples: synth_samples(seed, (w as usize) * (h as usize), bits),
    }
}

#[test]
fn v0v1_rgba_golomb_sign_flip_edge_round_trips() {
    // Fuzz artifact bytes: 2e 3d 19 ff 1a 28 → 47×14, v1, RGB + alpha,
    // coder_type 0, 8-bit (9-bit coded RCT space), and this exact seed.
    let data: [u8; 6] = [0x2e, 0x3d, 0x19, 0xff, 0x1a, 0x28];
    let width = (u32::from(data[0]) % 48) + 1;
    let height = (u32::from(data[1]) % 48) + 1;
    let sel = data[2];
    let mut seed: u64 = 0xD1B5_4A32_D192_ED03;
    for &b in &data[4..] {
        seed = seed.rotate_left(7) ^ u64::from(b).wrapping_mul(0x100_0000_01B3);
    }
    seed ^= u64::from(width) << 40 ^ u64::from(height) << 24 ^ u64::from(sel) << 8;
    let bits = 8u32;

    let cr = Ffv1ConfigurationRecord {
        version: Ffv1Version::V1,
        micro_version: None,
        coder_type: 0,
        state_transition_delta: [0; 256],
        colorspace_type: ColorspaceType::Rgb,
        bits_per_raw_sample: bits,
        chroma_planes: true,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: true,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: None,
        ec: None,
        intra: None,
        initial_state_delta: None,
    };
    let parsed = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).expect("fixture extradata");
    let qts: QuantizationTableSet = parsed.quant_table_sets[0].clone();
    let dims = FramePixelDimensions::new(width, height).unwrap();

    let frame = DecodedFrame {
        planes: vec![
            plane(seed, width, height, bits, 0),
            plane(seed ^ 0x1111, width, height, bits, 1),
            plane(seed ^ 0x2222, width, height, bits, 2),
            plane(seed ^ 0x3333, width, height, bits, 3),
        ],
        width,
        height,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    };

    let bytes = encode_frame_v0v1(&frame, &cr, &qts).expect("encode");
    let dec = decode_frame_v0v1(&bytes, dims).expect("decode");
    for (p, (a, b)) in frame.planes.iter().zip(dec.planes.iter()).enumerate() {
        assert_eq!(
            a.samples, b.samples,
            "plane {p}: §3.8 lossless identity violated (sign-flip ±2^(bits-1) edge)"
        );
    }
}
