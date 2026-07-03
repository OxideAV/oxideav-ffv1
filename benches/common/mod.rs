//! Shared scenario matrix for the oxideav-ffv1 Criterion benches.
//!
//! Everything is synthesised on the fly from deterministic xorshift32
//! streams — no committed fixture files, no `docs/` access at run time,
//! no third-party samples. Inputs use a realistic §4.1-shaped
//! Quantization Table Set (11 symmetric levels on the three §3.5
//! Figure 5 primary differences → `context_count == 666`, the same
//! shape the registry's default set uses) and mixed flat + textured
//! content so both the §3.8.2.2 run mode and the scalar paths carry
//! realistic weight.
//!
//! The same synthesis logic (at 64×48) backs the byte-exactness pins in
//! `tests/optimization_pins.rs`, so every scenario benchmarked here is
//! also pinned there.

use oxideav_ffv1::{
    ColorspaceType, DecodedFrame, DecodedFramePlane, Ffv1ConfigurationRecord, Ffv1SliceHeader,
    Ffv1Version, PictureStructure, QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES,
    NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

/// Benchmark frame geometry. 320×240 keeps a full Criterion sweep of
/// the matrix in the low minutes while still being ~50× the pin-test
/// frame (so per-frame setup cost is negligible against the per-Sample
/// loops).
pub const W: u32 = 320;
/// See [`W`].
pub const H: u32 = 240;

/// xorshift32 — deterministic pseudo-random source.
pub fn xorshift32(x: &mut u32) -> u32 {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    *x
}

/// Synthesise one Plane of mixed content: a flat left half (drives the
/// §3.8.2.2 run mode / the range coder's zero contexts) and a textured
/// right half (smooth gradient + 3-bit noise, drives the scalar
/// paths), scaled to the requested bit depth.
pub fn synth_plane(width: usize, height: usize, bits: u32, seed: u32) -> Vec<i32> {
    let mask = ((1u32 << bits) - 1) as i32;
    let shift = bits.saturating_sub(8);
    let mut rng = seed | 1;
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v = if x < width / 2 {
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

/// §4.1-style Quantization Table Set: 11 symmetric levels on Q0/Q1/Q2,
/// zero Q3/Q4 — `context_count == (11^3 + 1) / 2 == 666`.
pub fn default_like_qts() -> QuantizationTableSet {
    fn level(d: i32) -> i32 {
        let l = match d.unsigned_abs() {
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

/// v3 Configuration Record for one scenario cell.
pub fn make_cr(
    colorspace: ColorspaceType,
    coder_type: u32,
    bits: u32,
    num_h: u32,
    num_v: u32,
) -> Ffv1ConfigurationRecord {
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

/// One §4.6 Slice Header per raster cell of the `num_h × num_v` grid.
pub fn make_headers(num_h: u32, num_v: u32) -> Vec<Ffv1SliceHeader> {
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

/// YCbCr 4:2:0 three-Plane frame at `W × H`.
pub fn make_ycbcr420_frame(bits: u32) -> DecodedFrame {
    let (w, h) = (W, H);
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

/// RGB three-Plane frame at `W × H` with correlated channels (so the
/// §3.7.1 forward RCT produces realistic small chroma residuals).
pub fn make_rgb_frame(bits: u32) -> DecodedFrame {
    let (w, h) = (W, H);
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

/// One scenario cell of the bench matrix.
pub struct Scenario {
    /// Criterion id, e.g. `ycbcr420/range/8bit/1slice`.
    pub name: &'static str,
    /// §4.2.5 colour layout.
    pub colorspace: ColorspaceType,
    /// §4.2.3 coder (0 Golomb-Rice, 1 range default, 2 range custom).
    pub coder_type: u32,
    /// §4.2.7 `bits_per_raw_sample`.
    pub bits: u32,
    /// §4.2.11 horizontal slice count.
    pub num_h: u32,
    /// §4.2.12 vertical slice count.
    pub num_v: u32,
}

/// The full decode+encode matrix: coder × depth × colorspace at one
/// slice, plus a slice-count sweep on the range-coded YCbCr and Golomb
/// YCbCr cells.
pub const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "ycbcr420/golomb/8bit/1slice",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 0,
        bits: 8,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "ycbcr420/golomb/16bit/1slice",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 0,
        bits: 16,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "ycbcr420/range/8bit/1slice",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 8,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "ycbcr420/range/10bit/1slice",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 10,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "ycbcr420/range/16bit/1slice",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 16,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "ycbcr420/range-custom/8bit/1slice",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 2,
        bits: 8,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "rgb/golomb/8bit/1slice",
        colorspace: ColorspaceType::Rgb,
        coder_type: 0,
        bits: 8,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "rgb/range/8bit/1slice",
        colorspace: ColorspaceType::Rgb,
        coder_type: 1,
        bits: 8,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "rgb/range/16bit/1slice",
        colorspace: ColorspaceType::Rgb,
        coder_type: 1,
        bits: 16,
        num_h: 1,
        num_v: 1,
    },
    Scenario {
        name: "ycbcr420/range/8bit/4slices",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 8,
        num_h: 2,
        num_v: 2,
    },
    Scenario {
        name: "ycbcr420/range/8bit/16slices",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 1,
        bits: 8,
        num_h: 4,
        num_v: 4,
    },
    Scenario {
        name: "ycbcr420/golomb/8bit/4slices",
        colorspace: ColorspaceType::YCbCr,
        coder_type: 0,
        bits: 8,
        num_h: 2,
        num_v: 2,
    },
];

/// Build the input frame for a scenario.
pub fn make_frame(s: &Scenario) -> DecodedFrame {
    match s.colorspace {
        ColorspaceType::YCbCr => make_ycbcr420_frame(s.bits),
        ColorspaceType::Rgb => make_rgb_frame(s.bits),
    }
}

/// Raw-sample payload of a frame in bytes (1 byte per Sample at ≤ 8
/// bits, 2 otherwise) — the Criterion `Throughput` denominator, so the
/// MiB/s figures are comparable across depths and layouts.
pub fn raw_bytes(frame: &DecodedFrame) -> u64 {
    let per = if frame.bits_per_raw_sample <= 8 { 1 } else { 2 };
    frame
        .planes
        .iter()
        .map(|p| u64::from(p.width) * u64::from(p.height) * per)
        .sum()
}
