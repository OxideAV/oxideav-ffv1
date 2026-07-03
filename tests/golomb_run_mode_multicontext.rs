//! Regression tests for the §3.8.2.2 run-mode encoder on genuinely
//! multi-context Quantization Table Sets (RFC 9043 §3.4 / §3.5 /
//! §3.8.2.2 / §3.8.2.4.1).
//!
//! Round 386 found the Golomb-Rice `encode_line` run-mode scanner
//! desynchronising from the decoder on any table where the §3.5 context
//! actually varies with the neighbours (every realistic table; the
//! zero-/single-context tables the unit suites use never vary):
//!
//! 1. The encoder's run scanner ended a run at the first
//!    nonzero-context Sample (a "predicate break"). No such break
//!    exists on the decode side: once run mode is entered at a
//!    context-0 Sample, the decoder's `run_count` countdown consumes
//!    Samples **without re-evaluating their context** — RFC 9043
//!    §3.8.2.2 leaves run mode only "as soon as a nonzero difference
//!    is found". The consequence was a *lossy* encode: a long-run "1"
//!    bit claimed a zero difference for a Sample whose actual
//!    difference was nonzero (the flat-region → textured-region
//!    boundary of every row), and the decoder reconstructed a repeat
//!    of the previous row instead of the input.
//! 2. The §3.8.2.4.1 level-coded break Sample was encoded against
//!    `state.vlc[0]` instead of the breaking Sample's own §3.5 context
//!    window — the decoder reads it from `state.vlc[abs_ctx.index]`,
//!    which need not be context 0 (see point 1: the break can land on
//!    a nonzero-context Sample).
//!
//! These tests drive the exact failing shape — a flat region entering
//! run mode followed by textured content inside the same Line — through
//! every Golomb-Rice frame driver on a realistic 666-context table.

use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, decode_frame_v0v1, encode_frame, encode_frame_rgb,
    encode_frame_v0v1, ColorspaceType, DecodedFrame, DecodedFramePlane, Ffv1ConfigurationRecord,
    Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure, QuantizationTableSet,
    MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

/// §4.1-style realistic table: 11 symmetric levels on Q0/Q1/Q2, zero
/// Q3/Q4 — `context_count == (11^3 + 1) / 2 == 666`. The context
/// genuinely varies with the neighbours, unlike the constant-context
/// tables elsewhere in the test suite.
fn multicontext_qts() -> QuantizationTableSet {
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

fn cr(colorspace: ColorspaceType, version: Ffv1Version, bits: u32) -> Ffv1ConfigurationRecord {
    let v3 = version == Ffv1Version::V3;
    Ffv1ConfigurationRecord {
        version,
        micro_version: if v3 { Some(4) } else { None },
        coder_type: 0, // Golomb-Rice
        state_transition_delta: [0i32; NUM_TRANSITION_DELTAS],
        colorspace_type: colorspace,
        bits_per_raw_sample: bits,
        chroma_planes: colorspace == ColorspaceType::Rgb,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: if v3 { Some(1) } else { None },
        ec: if v3 { Some(0) } else { None },
        intra: if v3 { Some(false) } else { None },
        initial_state_delta: None,
    }
}

fn header() -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x: 0,
        slice_y: 0,
        slice_width: 1,
        slice_height: 1,
        quant_table_set_index_count: 2,
        quant_table_set_index: [0u32; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

/// The killer shape: a flat left half (context 0 → run mode) and a
/// gradient right half (nonzero contexts + nonzero differences) in the
/// SAME Line. Rows past the first previously decoded as copies of row 0.
fn flat_then_textured(w: u32, h: u32, bits: u32) -> Vec<i32> {
    let mask = ((1u32 << bits) - 1) as i32;
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 {
                0
            } else {
                ((x * 13 + y * 31) >> 2) as i32 & mask
            };
            out.push(v);
        }
    }
    out
}

fn gray_frame(w: u32, h: u32, bits: u32) -> DecodedFrame {
    DecodedFrame {
        planes: vec![DecodedFramePlane {
            plane_index: 0,
            width: w,
            height: h,
            samples: flat_then_textured(w, h, bits),
        }],
        width: w,
        height: h,
        bits_per_raw_sample: bits,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

#[test]
fn v3_gray_golomb_multicontext_run_break_round_trips() {
    // Minimal repro of the round-386 finding: 8x4, flat|textured split.
    // Pre-fix, decoded rows 1..3 were copies of row 0 (the nonzero
    // boundary differences were swallowed by long runs).
    for (w, h) in [(8u32, 4u32), (16, 8), (33, 7)] {
        let record = cr(ColorspaceType::YCbCr, Ffv1Version::V3, 8);
        let qts = vec![multicontext_qts()];
        let frame = gray_frame(w, h, 8);
        let bytes = encode_frame(&frame, &record, &qts, &[header()], false)
            .expect("multi-context Golomb encode");
        let dec = decode_frame(
            &bytes,
            &record,
            &qts,
            FramePixelDimensions::new(w, h).unwrap(),
            false,
        )
        .expect("multi-context Golomb decode");
        assert_eq!(
            dec.planes[0].samples, frame.planes[0].samples,
            "{w}x{h}: run-mode break at a nonzero-context Sample must survive the round trip"
        );
    }
}

#[test]
fn v3_gray_golomb_multicontext_16bit_round_trips() {
    let record = cr(ColorspaceType::YCbCr, Ffv1Version::V3, 16);
    let qts = vec![multicontext_qts()];
    let frame = gray_frame(24, 6, 16);
    let bytes =
        encode_frame(&frame, &record, &qts, &[header()], false).expect("16-bit Golomb encode");
    let dec = decode_frame(
        &bytes,
        &record,
        &qts,
        FramePixelDimensions::new(24, 6).unwrap(),
        false,
    )
    .expect("16-bit Golomb decode");
    assert_eq!(dec.planes[0].samples, frame.planes[0].samples);
}

#[test]
fn v3_rgb_golomb_multicontext_round_trips() {
    // The RGB line-major driver reuses `encode_line`; the §3.7 RCT
    // widens `bits` by one, but the run-mode state machine is shared.
    let (w, h) = (16u32, 6u32);
    let record = cr(ColorspaceType::Rgb, Ffv1Version::V3, 8);
    let qts = vec![multicontext_qts()];
    let g = flat_then_textured(w, h, 8);
    let r: Vec<i32> = g.iter().map(|&v| (v + 1) & 0xFF).collect();
    let b: Vec<i32> = g.iter().map(|&v| (v + 2) & 0xFF).collect();
    let frame = DecodedFrame {
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
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    };
    let bytes =
        encode_frame_rgb(&frame, &record, &qts, &[header()], false).expect("RGB Golomb encode");
    let dec = decode_frame_rgb(
        &bytes,
        &record,
        &qts,
        FramePixelDimensions::new(w, h).unwrap(),
        false,
    )
    .expect("RGB Golomb decode");
    for (got, want) in dec.planes.iter().zip(frame.planes.iter()) {
        assert_eq!(got.samples, want.samples, "RGB plane {}", want.plane_index);
    }
}

#[test]
fn v0v1_gray_golomb_multicontext_round_trips() {
    // The v0/v1 inline-Parameters driver also routes its §4.7 Slice
    // Content through `encode_line` (after the §3.8.1.1.1 Sentinel
    // handoff), so the same regression shape must hold there.
    let (w, h) = (16u32, 6u32);
    let record = cr(ColorspaceType::YCbCr, Ffv1Version::V1, 8);
    let qts = multicontext_qts();
    let frame = gray_frame(w, h, 8);
    let bytes = encode_frame_v0v1(&frame, &record, &qts).expect("v0/v1 Golomb encode");
    let dec = decode_frame_v0v1(&bytes, FramePixelDimensions::new(w, h).unwrap())
        .expect("v0/v1 Golomb decode");
    assert_eq!(dec.planes[0].samples, frame.planes[0].samples);
}
