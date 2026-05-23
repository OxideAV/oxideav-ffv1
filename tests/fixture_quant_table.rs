//! Fixture-based validation of the RFC 9043 §4.1 Quantization Table
//! Set cascade against the workspace FFV1 corpus
//! (`docs/video/ffv1/fixtures/`).
//!
//! Each extradata blob is the Matroska CodecPrivate element of a
//! fixture's `input.mkv`. The ground truth is the fixture's
//! `trace.txt` `QUANT_TABLE` events, which carry the reference
//! decoder's `context_count` for each Quantization Table Set:
//!
//! ```text
//! GLOBAL_HEADER ... quant_table_count=2 ...
//! QUANT_TABLE  index=0  context_count=666
//! QUANT_TABLE  index=1  context_count=7563
//! ```
//!
//! A clean-room decoder that reproduces these `context_count` values
//! bit-exactly has correctly walked the §4.1 cascade: the run-length
//! `len - 1` symbols, the `scale *= 2*len_count - 1` accumulation, and
//! the `context_count = ceil(scale/2)` derivation (the trace says
//! nothing about the table *entries*, but `context_count` is a
//! function of every `len_count` across all five sub-tables, so a
//! single off-by-one in any run desynchronises it).

use oxideav_ffv1::parse_quantization_table_sets;

/// `v3-default` extradata — the Matroska CodecPrivate of
/// `docs/video/ffv1/fixtures/v3-default/input.mkv`.
///
/// Trace ground truth (`docs/video/ffv1/fixtures/v3-default/trace.txt`):
/// `quant_table_count=2`, `context_count` = 666 (index 0), 7563 (index 1).
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

/// `v3-grayscale` extradata — CodecPrivate of
/// `docs/video/ffv1/fixtures/v3-grayscale/input.mkv`. Same 8-bit
/// quant-table cascade as `v3-default` (chroma_planes differs but the
/// quant tables are identical): `context_count` = 666, 7563.
const V3_GRAYSCALE_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x2f, 0xd3, 0xc8, 0x18, 0xce, 0x09, 0xeb, 0x7f, 0x68, 0x23, 0xd0, 0x46, 0xc2, 0x44,
    0x28, 0x0a, 0x38, 0x20, 0x41, 0x1c, 0x8f, 0xfd, 0x0b, 0xd7, 0xa0, 0xdd, 0x7d, 0xc7, 0xe2, 0xbe,
    0x16, 0x99, 0xb1, 0xe0, 0xb7, 0x06, 0x5a, 0x9c, 0x7e, 0x09,
];

/// `v3-rgb-bgr0` extradata — CodecPrivate of
/// `docs/video/ffv1/fixtures/v3-rgb-bgr0/input.mkv` (colorspace=1, RCT
/// path). Same 8-bit cascade: `context_count` = 666, 7563.
const V3_RGB_BGR0_EXTRADATA: &[u8] = &[
    0x55, 0xf6, 0x46, 0x87, 0xe6, 0xa9, 0xc1, 0x7b, 0x87, 0xbf, 0x82, 0x5e, 0xd8, 0x30, 0x2b, 0x95,
    0x12, 0x2e, 0xcf, 0x70, 0xe2, 0x0f, 0x76, 0xbc, 0x04, 0x17, 0x6c, 0xd6, 0x60, 0xd4, 0x99, 0xbf,
    0x4f, 0x95, 0xdf, 0x58, 0xfb, 0x51, 0xd1, 0x16, 0xf4, 0xad,
];

/// `v3-yuv444p16` extradata — CodecPrivate of
/// `docs/video/ffv1/fixtures/v3-yuv444p16/input.mkv` (16-bit, the
/// larger 52-byte extradata). The >8-bit path produces a DIFFERENT
/// cascade: `context_count` = 365 (index 0), 5063 (index 1). Extracted
/// black-box via `ffprobe -show_data` on `input.mkv`.
const V3_YUV444P16_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x68, 0x5d, 0x47, 0x52, 0x37, 0x13, 0xbc, 0x20, 0x1a, 0xc2, 0xff, 0xe7, 0x15, 0x23,
    0x7c, 0xf2, 0x14, 0xef, 0x39, 0xf4, 0xfb, 0xf7, 0xb2, 0x45, 0x8d, 0x81, 0x0f, 0x64, 0x37, 0x36,
    0x78, 0x68, 0xf8, 0xf5, 0x10, 0x25, 0xe0, 0x19, 0xff, 0x63, 0xbc, 0x94, 0xce, 0xbe, 0x50, 0xc7,
    0x12, 0x9a, 0xd7, 0xb8,
];

#[test]
fn v3_default_context_counts_match_trace() {
    let p = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA)
        .expect("v3-default quant cascade parses");
    let counts: Vec<u32> = p.quant_table_sets.iter().map(|s| s.context_count).collect();
    // trace.txt: index=0 context_count=666, index=1 context_count=7563.
    assert_eq!(counts, vec![666, 7563], "v3-default context_count cascade");
    assert_eq!(p.record.quant_table_set_count, Some(2));
}

#[test]
fn v3_grayscale_context_counts_match_trace() {
    let p = parse_quantization_table_sets(V3_GRAYSCALE_EXTRADATA)
        .expect("v3-grayscale quant cascade parses");
    let counts: Vec<u32> = p.quant_table_sets.iter().map(|s| s.context_count).collect();
    assert_eq!(
        counts,
        vec![666, 7563],
        "v3-grayscale context_count cascade"
    );
}

#[test]
fn v3_rgb_bgr0_context_counts_match_trace() {
    let p = parse_quantization_table_sets(V3_RGB_BGR0_EXTRADATA)
        .expect("v3-rgb-bgr0 quant cascade parses");
    let counts: Vec<u32> = p.quant_table_sets.iter().map(|s| s.context_count).collect();
    assert_eq!(counts, vec![666, 7563], "v3-rgb-bgr0 context_count cascade");
}

#[test]
fn v3_yuv444p16_context_counts_match_trace() {
    let p = parse_quantization_table_sets(V3_YUV444P16_EXTRADATA)
        .expect("v3-yuv444p16 quant cascade parses");
    let counts: Vec<u32> = p.quant_table_sets.iter().map(|s| s.context_count).collect();
    // The 16-bit path has a different cascade than the 8-bit fixtures:
    // trace.txt: index=0 context_count=365, index=1 context_count=5063.
    assert_eq!(
        counts,
        vec![365, 5063],
        "v3-yuv444p16 context_count cascade (16-bit path)"
    );
}

#[test]
fn quant_tables_are_symmetric_with_flipped_sign() {
    // RFC 9043 §4.1: "The second half doesn't need to be stored as it
    // is identical to the first with flipped sign." Verify the
    // invariant on real decoded tables: table[256 - k] == -table[k]
    // for k in 1..128 and table[128] == -table[127].
    let p = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA)
        .expect("v3-default quant cascade parses");
    for set in &p.quant_table_sets {
        for sub in &set.tables {
            assert_eq!(sub[0], 0, "entry 0 is always 0 (§4.1 run starts at v=0)");
            for k in 1..128usize {
                assert_eq!(sub[256 - k], -sub[k], "§4.1 mirror invariant at k={k}");
            }
            assert_eq!(sub[128], -sub[127], "§4.1 midpoint invariant");
        }
    }
}

#[test]
fn quant_table_zero_neighbour_difference_maps_to_zero() {
    // §3.4: a quantized sample difference of 0 (index 0) must map to a
    // 0 quantization value in every sub-table (the run that starts at
    // v=0 always writes entry 0). This is what makes a flat region
    // decode into context 0.
    let p = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).expect("v3-default parses");
    for set in &p.quant_table_sets {
        for sub in &set.tables {
            assert_eq!(sub[0], 0);
        }
    }
}

#[test]
fn rejects_truncated_extradata() {
    // The range coder needs at least its two seed bytes; one byte must
    // fail rather than panic.
    assert!(parse_quantization_table_sets(&[0x55]).is_err());
    assert!(parse_quantization_table_sets(&[]).is_err());
}
