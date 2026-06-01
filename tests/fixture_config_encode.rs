//! Fixture-based round-trip validation of the §4.2 Parameters +
//! §4.1 Quantization Table Set cascade **encoder**
//! ([`encode_configuration_record_with_quant_tables`]) against the
//! workspace FFV1 corpus (`docs/video/ffv1/fixtures/`).
//!
//! Each fixture's extradata blob is the Matroska CodecPrivate of its
//! `input.mkv`. The encoder under test cannot reproduce those blobs
//! byte-for-byte (the corpus extradata carries the §4.2.14+
//! Parameters tail — `states_coded`, `initial_state_delta`, `ec`,
//! `intra` — which is blocked on the open #904 DOCS-GAP). What it can
//! do, however, is the **structural round-trip**:
//!
//! 1. Parse the corpus extradata via [`parse_quantization_table_sets`]
//!    to a [`ParametersWithQuantTables`].
//! 2. Re-encode that struct via
//!    [`encode_configuration_record_with_quant_tables`] to a fresh
//!    extradata blob.
//! 3. Parse the re-encoded blob and assert the
//!    `Ffv1ConfigurationRecord`, every `QuantizationTableSet.tables`,
//!    and every `QuantizationTableSet.context_count` is equal to the
//!    original parse.
//!
//! The §4.3.2 CRC residue of the re-encoded blob is asserted to be
//! zero via [`validate_configuration_record_crc`] — proving the
//! encoder solves the parity word correctly against the §4.9.3
//! generator.

use oxideav_ffv1::{
    encode_configuration_record_with_quant_tables, encode_parameters_with_quant_tables,
    parse_quantization_table_sets, validate_configuration_record_crc,
};

/// `v3-default` extradata — see `tests/fixture_quant_table.rs` for
/// the trace ground truth. Two Sets, context_count = 666, 7563.
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

/// `v3-grayscale` extradata. Two Sets, context_count = 666, 7563.
const V3_GRAYSCALE_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x2f, 0xd3, 0xc8, 0x18, 0xce, 0x09, 0xeb, 0x7f, 0x68, 0x23, 0xd0, 0x46, 0xc2, 0x44,
    0x28, 0x0a, 0x38, 0x20, 0x41, 0x1c, 0x8f, 0xfd, 0x0b, 0xd7, 0xa0, 0xdd, 0x7d, 0xc7, 0xe2, 0xbe,
    0x16, 0x99, 0xb1, 0xe0, 0xb7, 0x06, 0x5a, 0x9c, 0x7e, 0x09,
];

/// `v3-rgb-bgr0` extradata (colorspace=1, RCT). Two Sets, context_count = 666, 7563.
const V3_RGB_BGR0_EXTRADATA: &[u8] = &[
    0x55, 0xf6, 0x46, 0x87, 0xe6, 0xa9, 0xc1, 0x7b, 0x87, 0xbf, 0x82, 0x5e, 0xd8, 0x30, 0x2b, 0x95,
    0x12, 0x2e, 0xcf, 0x70, 0xe2, 0x0f, 0x76, 0xbc, 0x04, 0x17, 0x6c, 0xd6, 0x60, 0xd4, 0x99, 0xbf,
    0x4f, 0x95, 0xdf, 0x58, 0xfb, 0x51, 0xd1, 0x16, 0xf4, 0xad,
];

/// `v3-yuv444p16` extradata (16-bit, the larger 52-byte extradata).
/// Two Sets, context_count = 365, 5063.
const V3_YUV444P16_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x68, 0x5d, 0x47, 0x52, 0x37, 0x13, 0xbc, 0x20, 0x1a, 0xc2, 0xff, 0xe7, 0x15, 0x23,
    0x7c, 0xf2, 0x14, 0xef, 0x39, 0xf4, 0xfb, 0xf7, 0xb2, 0x45, 0x8d, 0x81, 0x0f, 0x64, 0x37, 0x36,
    0x78, 0x68, 0xf8, 0xf5, 0x10, 0x25, 0xe0, 0x19, 0xff, 0x63, 0xbc, 0x94, 0xce, 0xbe, 0x50, 0xc7,
    0x12, 0x9a, 0xd7, 0xb8,
];

/// Run the full encode → re-parse → equality cycle on `extradata`.
fn assert_extradata_round_trips(name: &str, extradata: &[u8]) {
    let parsed = parse_quantization_table_sets(extradata)
        .unwrap_or_else(|e| panic!("{name}: corpus extradata parses ({e:?})"));

    let reblob =
        encode_configuration_record_with_quant_tables(&parsed.record, &parsed.quant_table_sets)
            .unwrap_or_else(|e| panic!("{name}: round-trip encode ({e:?})"));

    // §4.3.2: the re-encoded blob is self-consistent (whole-blob CRC
    // residue zero). This proves the encoder's parity solver matches
    // the §4.9.3 generator the decoder validates against.
    validate_configuration_record_crc(&reblob).unwrap_or_else(|e| {
        panic!(
            "{name}: re-encoded blob fails §4.3.2 CRC check: {e:?}; reblob_len={}",
            reblob.len()
        )
    });

    let reparsed = parse_quantization_table_sets(&reblob)
        .unwrap_or_else(|e| panic!("{name}: re-encoded blob re-parses ({e:?})"));

    // Configuration Record field-for-field.
    assert_eq!(
        reparsed.record.version, parsed.record.version,
        "{name} version"
    );
    assert_eq!(
        reparsed.record.micro_version, parsed.record.micro_version,
        "{name} micro_version"
    );
    assert_eq!(
        reparsed.record.coder_type, parsed.record.coder_type,
        "{name} coder_type"
    );
    assert_eq!(
        reparsed.record.colorspace_type, parsed.record.colorspace_type,
        "{name} colorspace_type"
    );
    assert_eq!(
        reparsed.record.bits_per_raw_sample, parsed.record.bits_per_raw_sample,
        "{name} bits_per_raw_sample"
    );
    assert_eq!(
        reparsed.record.chroma_planes, parsed.record.chroma_planes,
        "{name} chroma_planes"
    );
    assert_eq!(
        reparsed.record.log2_h_chroma_subsample, parsed.record.log2_h_chroma_subsample,
        "{name} log2_h_chroma_subsample"
    );
    assert_eq!(
        reparsed.record.log2_v_chroma_subsample, parsed.record.log2_v_chroma_subsample,
        "{name} log2_v_chroma_subsample"
    );
    assert_eq!(
        reparsed.record.extra_plane, parsed.record.extra_plane,
        "{name} extra_plane"
    );
    assert_eq!(
        reparsed.record.num_h_slices, parsed.record.num_h_slices,
        "{name} num_h_slices"
    );
    assert_eq!(
        reparsed.record.num_v_slices, parsed.record.num_v_slices,
        "{name} num_v_slices"
    );
    assert_eq!(
        reparsed.record.quant_table_set_count, parsed.record.quant_table_set_count,
        "{name} quant_table_set_count"
    );

    // §4.2.4 sr loop: every state_transition_delta entry must
    // round-trip when coder_type > 1; the array is all-zero for
    // coder_type in {0, 1}.
    for i in 0..parsed.record.state_transition_delta.len() {
        assert_eq!(
            reparsed.record.state_transition_delta[i], parsed.record.state_transition_delta[i],
            "{name} state_transition_delta[{i}]"
        );
    }

    // §4.1 cascade.
    assert_eq!(
        reparsed.quant_table_sets.len(),
        parsed.quant_table_sets.len(),
        "{name} cascade length"
    );
    for (idx, (re_set, orig_set)) in reparsed
        .quant_table_sets
        .iter()
        .zip(parsed.quant_table_sets.iter())
        .enumerate()
    {
        assert_eq!(
            re_set.context_count, orig_set.context_count,
            "{name} set[{idx}] context_count"
        );
        for sub in 0..5usize {
            assert_eq!(
                re_set.tables[sub], orig_set.tables[sub],
                "{name} set[{idx}] sub-table[{sub}]"
            );
        }
    }
}

#[test]
fn v3_default_extradata_round_trips() {
    assert_extradata_round_trips("v3-default", V3_DEFAULT_EXTRADATA);
}

#[test]
fn v3_grayscale_extradata_round_trips() {
    assert_extradata_round_trips("v3-grayscale", V3_GRAYSCALE_EXTRADATA);
}

#[test]
fn v3_rgb_bgr0_extradata_round_trips() {
    assert_extradata_round_trips("v3-rgb-bgr0", V3_RGB_BGR0_EXTRADATA);
}

#[test]
fn v3_yuv444p16_extradata_round_trips() {
    assert_extradata_round_trips("v3-yuv444p16", V3_YUV444P16_EXTRADATA);
}

#[test]
fn re_encoded_blob_is_close_to_original_size() {
    // Sanity / regression check on the encoder output size against
    // the corpus extradata. The encoder omits the §4.2.14+ tail
    // (#904 DOCS-GAP), so the re-encoded blob encodes strictly fewer
    // symbols than the corpus encodes — but the closed-mode range
    // coder's final flush + the 4-byte parity word can land it at
    // any byte boundary in the same neighbourhood as the corpus. A
    // few bytes either way is fine; an order-of-magnitude difference
    // would mean the encoder is desynchronised from the parser.
    let parsed = parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).expect("parse");
    let reblob =
        encode_configuration_record_with_quant_tables(&parsed.record, &parsed.quant_table_sets)
            .expect("encode");
    let corpus_len = V3_DEFAULT_EXTRADATA.len();
    let reblob_len = reblob.len();
    let diff = corpus_len.abs_diff(reblob_len);
    assert!(
        diff <= 8,
        "re-encoded blob ({reblob_len} bytes) diverges from corpus extradata ({corpus_len} bytes) by {diff} bytes (>8)",
    );
}

#[test]
fn wrapper_api_agrees_with_direct_api() {
    // `encode_parameters_with_quant_tables(parsed)` is shorthand for
    // `encode_configuration_record_with_quant_tables(&parsed.record,
    // &parsed.quant_table_sets)`. Confirm they produce identical bytes
    // on each fixture.
    for (name, extradata) in [
        ("v3-default", V3_DEFAULT_EXTRADATA),
        ("v3-grayscale", V3_GRAYSCALE_EXTRADATA),
        ("v3-rgb-bgr0", V3_RGB_BGR0_EXTRADATA),
        ("v3-yuv444p16", V3_YUV444P16_EXTRADATA),
    ] {
        let parsed = parse_quantization_table_sets(extradata).expect(name);
        let direct =
            encode_configuration_record_with_quant_tables(&parsed.record, &parsed.quant_table_sets)
                .expect(name);
        let wrapped = encode_parameters_with_quant_tables(&parsed).expect(name);
        assert_eq!(direct, wrapped, "{name} direct vs wrapper API");
    }
}
