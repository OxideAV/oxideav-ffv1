//! Encoder byte-exactness pins for the depth-round optimization work.
//!
//! The round-386 bench+profile depth round lands performance changes in
//! the hot per-Sample loops (range coder, Golomb-Rice VLC, context
//! computation, plane reconstruction). Every one of those changes MUST
//! keep the produced bitstream **byte-identical** — FFV1 is a
//! deterministic codec and the crate's encoder output for a fixed input
//! is part of its observable contract (a stream hashed into an archive
//! manifest must re-hash identically after an internal refactor).
//!
//! This suite pins an FNV-1a-64 hash of the encoder's output across the
//! bench matrix (coder 0/1/2 × 8/10/16-bit × YCbCr/RGB × 1/4/16-slice
//! grids), plus the lossless decode-back invariant on each stream. The
//! pinned constants were recorded immediately after the round's
//! §3.8.2.2 run-mode encoder fix and BEFORE the first optimization
//! commit; any optimization that flips a single output byte fails here.
//!
//! Re-pinned in r411 after two deliberate wire-conformance changes
//! (both validated black-box against the external reference decoder):
//! the RFC 9043 §3.8.1.1.1 Sentinel-mode termination of every v3
//! range-coded Slice region (Slice end AND Slice-Header → Golomb
//! switch), and the Slice-scoped §3.8.2.2.1 run triple on the §4.7
//! line-major RGB Golomb path.
//!
//! Decode-side byte-exactness is separately pinned by
//! `reference_fixture_decode.rs` (bit-exact against the reference
//! decoder's `expected.raw` for every fixture under
//! `docs/video/ffv1/fixtures/`); together the two suites pin both
//! directions of the codec.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, encode_frame, ColorspaceType, DecodedFrame, DecodedFramePlane,
    Ffv1ConfigurationRecord, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

// ---------------------------------------------------------------------
// Deterministic input synthesis (identical logic to `benches/common`)
// ---------------------------------------------------------------------

/// xorshift32 — deterministic pseudo-random source, no external crates.
fn xorshift32(x: &mut u32) -> u32 {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    *x
}

/// Synthesise one Plane of mixed content: a flat left half (drives the
/// §3.8.2.2 run mode / the range coder's zero contexts) and a textured
/// right half (smooth gradient + 3-bit noise, drives the scalar paths),
/// scaled to the requested bit depth.
fn synth_plane(width: usize, height: usize, bits: u32, seed: u32) -> Vec<i32> {
    let mask = ((1u32 << bits) - 1) as i32;
    let shift = bits.saturating_sub(8);
    let mut rng = seed | 1;
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v = if x < width / 2 {
                // Flat region — constant per horizontal band.
                (((y / 8) as i32 * 17) << shift) & mask
            } else {
                let base = (((x * 13 + y * 31) >> 2) as i32) << shift;
                let noise = (xorshift32(&mut rng) & 7) as i32;
                (base + noise) & mask
            };
            out.push(v);
        }
    }
    out
}

// ---------------------------------------------------------------------
// Configuration / frame builders
// ---------------------------------------------------------------------

/// §4.1-style Quantization Table Set: 11 symmetric levels on the three
/// §3.5 Figure 5 primary differences (Q0/Q1/Q2), zero Q3/Q4 —
/// `context_count == (11^3 + 1) / 2 == 666`, the same shape the
/// registry's v0/v1 default uses. This is the realistic multi-context
/// regime (unlike the single-context tables most unit tests use).
fn default_like_qts() -> QuantizationTableSet {
    fn level(d: i32) -> i32 {
        let mag = d.unsigned_abs();
        let l = match mag {
            0 => 0,
            1..=2 => 1,
            3..=6 => 2,
            7..=14 => 3,
            15..=30 => 4,
            _ => 5,
        };
        if d < 0 {
            -l
        } else {
            l
        }
    }
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    let [t0, t1, t2, _, _] = &mut tables;
    for (idx, ((s0, s1), s2)) in t0
        .iter_mut()
        .zip(t1.iter_mut())
        .zip(t2.iter_mut())
        .enumerate()
    {
        // §3.4: the table is indexed by the low 8 bits of the signed
        // difference; interpret the index as signed two's complement.
        let d = if idx < 128 {
            idx as i32
        } else {
            idx as i32 - 256
        };
        let l = level(d);
        *s0 = l;
        *s1 = l * 11;
        *s2 = l * 121;
    }
    QuantizationTableSet {
        tables,
        context_count: 666,
    }
}

fn make_cr(
    colorspace: ColorspaceType,
    coder_type: u32,
    bits: u32,
    num_h: u32,
    num_v: u32,
) -> Ffv1ConfigurationRecord {
    // A sparse non-zero §4.2.4 delta pattern for `coder_type == 2` so
    // the custom-table path is pinned distinctly from the default table.
    let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
    if coder_type == 2 {
        for (i, slot) in deltas.iter_mut().enumerate().skip(1).step_by(7) {
            *slot = ((i % 5) as i32) - 2;
        }
    }
    let ycbcr = colorspace == ColorspaceType::YCbCr;
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: deltas,
        colorspace_type: colorspace,
        bits_per_raw_sample: bits,
        chroma_planes: true,
        log2_h_chroma_subsample: u32::from(ycbcr),
        log2_v_chroma_subsample: u32::from(ycbcr),
        extra_plane: false,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra: Some(false),
        initial_state_delta: None,
    }
}

fn make_headers(num_h: u32, num_v: u32) -> Vec<Ffv1SliceHeader> {
    let mut headers = Vec::new();
    for sy in 0..num_v {
        for sx in 0..num_h {
            headers.push(Ffv1SliceHeader {
                slice_x: sx,
                slice_y: sy,
                slice_width: 1,
                slice_height: 1,
                quant_table_set_index_count: 2,
                quant_table_set_index: [0u32; MAX_QUANT_TABLE_SET_INDEXES],
                picture_structure: PictureStructure::Progressive,
                picture_structure_raw: 0,
                sar_num: 0,
                sar_den: 0,
            });
        }
    }
    headers
}

/// YCbCr 4:2:0 three-Plane frame.
fn make_ycbcr420_frame(w: u32, h: u32, bits: u32) -> DecodedFrame {
    let (cw, ch) = (w / 2, h / 2);
    DecodedFrame {
        planes: vec![
            DecodedFramePlane {
                plane_index: 0,
                width: w,
                height: h,
                samples: synth_plane(w as usize, h as usize, bits, 0xC0FF_EE01),
            },
            DecodedFramePlane {
                plane_index: 1,
                width: cw,
                height: ch,
                samples: synth_plane(cw as usize, ch as usize, bits, 0xBEEF_0002),
            },
            DecodedFramePlane {
                plane_index: 2,
                width: cw,
                height: ch,
                samples: synth_plane(cw as usize, ch as usize, bits, 0xDEAD_0003),
            },
        ],
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// RGB three-Plane frame with correlated channels (so the §3.7.1
/// forward RCT produces realistic small chroma residuals).
fn make_rgb_frame(w: u32, h: u32, bits: u32) -> DecodedFrame {
    let mask = ((1u32 << bits) - 1) as i32;
    let g = synth_plane(w as usize, h as usize, bits, 0x0BAD_F00D);
    let mut rng = 0x1234_5679u32;
    let r: Vec<i32> = g
        .iter()
        .map(|&v| (v + (xorshift32(&mut rng) & 3) as i32) & mask)
        .collect();
    let b: Vec<i32> = g
        .iter()
        .map(|&v| (v + (xorshift32(&mut rng) & 3) as i32) & mask)
        .collect();
    DecodedFrame {
        planes: vec![
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
        ],
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

// ---------------------------------------------------------------------
// FNV-1a 64 — tiny, dependency-free, deterministic
// ---------------------------------------------------------------------

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------
// The pinned matrix
// ---------------------------------------------------------------------

const W: u32 = 64;
const H: u32 = 48;

struct Scenario {
    name: &'static str,
    colorspace: ColorspaceType,
    coder_type: u32,
    bits: u32,
    num_h: u32,
    num_v: u32,
    /// FNV-1a-64 of the encoded Frame bytes, recorded pre-optimization.
    pinned: u64,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "ycbcr420-golomb-8",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 0,
        bits: 8,
        num_h: 1,
        num_v: 1,
        pinned: 0xb81590b3e0519ebd,
    },
    Scenario {
        name: "ycbcr420-golomb-16",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 0,
        bits: 16,
        num_h: 1,
        num_v: 1,
        pinned: 0x8c07cf3eba3a5ad8,
    },
    Scenario {
        name: "ycbcr420-range-8",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 8,
        num_h: 1,
        num_v: 1,
        pinned: 0x6eae6be4ee5377a1,
    },
    Scenario {
        name: "ycbcr420-range-10",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 10,
        num_h: 1,
        num_v: 1,
        pinned: 0x99f3a7fc2407b123,
    },
    Scenario {
        name: "ycbcr420-range-16",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 16,
        num_h: 1,
        num_v: 1,
        pinned: 0xf2d0452e666a0c18,
    },
    Scenario {
        name: "ycbcr420-range2-8",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 2,
        bits: 8,
        num_h: 1,
        num_v: 1,
        pinned: 0x78328bd7cf290d55,
    },
    Scenario {
        name: "rgb-golomb-8",
        colorspace: ColorspaceType::Rgb,
        coder_type: 0,
        bits: 8,
        num_h: 1,
        num_v: 1,
        pinned: 0x408bdd0d39855ce5,
    },
    Scenario {
        name: "rgb-range-8",
        colorspace: ColorspaceType::Rgb,
        coder_type: 1,
        bits: 8,
        num_h: 1,
        num_v: 1,
        pinned: 0xd6798bf8e3f762ef,
    },
    Scenario {
        name: "rgb-range-16",
        colorspace: ColorspaceType::Rgb,
        coder_type: 1,
        bits: 16,
        num_h: 1,
        num_v: 1,
        pinned: 0x6da51abd923a22d3,
    },
    Scenario {
        name: "ycbcr420-range-8-2x2",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 8,
        num_h: 2,
        num_v: 2,
        pinned: 0xe96204b8d7c19fd6,
    },
    Scenario {
        name: "ycbcr420-range-8-4x4",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 8,
        num_h: 4,
        num_v: 4,
        pinned: 0xa758c48c1f7cb6be,
    },
    Scenario {
        name: "ycbcr420-golomb-8-2x2",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 0,
        bits: 8,
        num_h: 2,
        num_v: 2,
        pinned: 0x8310297b48ec7827,
    },
    Scenario {
        name: "rgb-range-8-2x2",
        colorspace: ColorspaceType::Rgb,
        coder_type: 1,
        bits: 8,
        num_h: 2,
        num_v: 2,
        pinned: 0x3d8f36bd5219abf2,
    },
];

fn run_scenario(s: &Scenario) -> (u64, usize) {
    let cr = make_cr(s.colorspace, s.coder_type, s.bits, s.num_h, s.num_v);
    let qts = vec![default_like_qts()];
    let headers = make_headers(s.num_h, s.num_v);
    let frame = match s.colorspace {
        ColorspaceType::YCbCr => make_ycbcr420_frame(W, H, s.bits),
        ColorspaceType::Rgb => make_rgb_frame(W, H, s.bits),
    };
    let bytes = encode_frame(&frame, &cr, &qts, &headers, true)
        .unwrap_or_else(|e| panic!("{}: encode failed: {e:?}", s.name));

    // Lossless decode-back invariant on the same stream.
    let dims = FramePixelDimensions::new(W, H).unwrap();
    let decoded = match s.colorspace {
        ColorspaceType::YCbCr => decode_frame(&bytes, &cr, &qts, dims, true),
        ColorspaceType::Rgb => decode_frame_rgb(&bytes, &cr, &qts, dims, true),
    }
    .unwrap_or_else(|e| panic!("{}: decode failed: {e:?}", s.name));
    assert_eq!(
        decoded.planes.len(),
        frame.planes.len(),
        "{}: plane count",
        s.name
    );
    for (got, want) in decoded.planes.iter().zip(frame.planes.iter()) {
        assert_eq!(got.samples, want.samples, "{}: lossless identity", s.name);
    }

    (fnv1a64(&bytes), bytes.len())
}

#[test]
fn encoder_output_bytes_are_pinned_across_the_matrix() {
    let mut failures = Vec::new();
    for s in SCENARIOS {
        let (hash, len) = run_scenario(s);
        if hash != s.pinned {
            failures.push(format!(
                "    name: {:28} got: 0x{hash:016x} (len {len}) pinned: 0x{:016x}",
                s.name, s.pinned
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "encoder output drifted from the pre-optimization pins:\n{}",
        failures.join("\n")
    );
}
