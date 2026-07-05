//! Conformance gate for the RFC 9043 §4.2.14/§4.2.15 explicit
//! initial-state path (`states_coded == 1`), pinned against the
//! hand-authored, reference-decoder-validated `states-coded-1` fixture
//! (docs/video/ffv1/fixtures/states-coded-1/, docs commit bb7e387).
//!
//! The fixture is the ONLY stream in the corpus that takes this path —
//! the reference encoder always writes `states_coded = 0` — so it pins
//! three things nothing else can:
//!
//! * the §4.2.15 delta block's symbol-coding layout (one dedicated
//!   32-slot window, fresh at the block start; see
//!   `quant_table::parse_parameters_tail`),
//! * the FFmpeg-interop row count (942 rows for the `[6,6,6,1,1]`
//!   table, NOT the §4.1 `context_count` of 666; see
//!   `QuantizationTableSet::initial_state_row_count`), and
//! * that the parse lands `ec` aligned afterwards (the fixture was
//!   built so a mis-sized delta block desynchronises everything
//!   downstream of it).

use oxideav_ffv1::{
    decode_frame, parse_quantization_table_sets, validate_configuration_record_crc,
    FramePixelDimensions,
};

#[path = "data/states_coded_1.rs"]
mod fx;

use fx::{STATES_CODED_1_CONFIG_RECORD, STATES_CODED_1_EXPECTED, STATES_CODED_1_FRAME};

/// §4.3.2: the whole 220-byte Configuration Record (parity included)
/// CRCs to residue 0 — the fixture's hand-solved parity word is intact.
#[test]
fn config_record_crc_validates() {
    validate_configuration_record_crc(STATES_CODED_1_CONFIG_RECORD)
        .expect("hand-solved §4.3.2 parity leaves residue 0");
}

/// The §4.2 Parameters parse walks the coded §4.2.15 triple-loop and
/// still lands every tail field where the fixture pinned it.
#[test]
fn parse_pins_initial_state_layout() {
    let parsed = parse_quantization_table_sets(STATES_CODED_1_CONFIG_RECORD)
        .expect("states_coded == 1 Parameters parse");
    let rec = &parsed.record;

    // Prefix fields per the fixture notes (docs commit bb7e387).
    assert_eq!(rec.micro_version, Some(4), "micro_version");
    assert_eq!(rec.coder_type, 1, "coder_type: default-table range coder");
    assert_eq!(rec.bits_per_raw_sample, 8, "bits_per_raw_sample");
    assert!(!rec.chroma_planes, "gray: no chroma planes");
    assert!(!rec.extra_plane, "no alpha");
    assert_eq!(rec.num_h_slices, Some(1), "1x1 slice grid");
    assert_eq!(rec.num_v_slices, Some(1), "1x1 slice grid");
    assert_eq!(rec.quant_table_set_count, Some(2), "two table sets");

    // §4.1 cascade: set 0 is the default 3-input table (len_count
    // [6,6,6,1,1], context_count 666), set 1 the default 5-input table
    // (len_count [6,6,3,3,3], context_count 7563).
    assert_eq!(parsed.quant_table_sets.len(), 2);
    let set0 = &parsed.quant_table_sets[0];
    let set1 = &parsed.quant_table_sets[1];
    assert_eq!(set0.context_count, 666, "§4.1 context_count, set 0");
    assert_eq!(set0.len_counts(), [6, 6, 6, 1, 1], "recovered len_count");
    assert_eq!(set1.context_count, 7563, "§4.1 context_count, set 1");
    assert_eq!(set1.len_counts(), [6, 6, 3, 3, 3], "recovered len_count");

    // The FFmpeg-interop §4.2.15 row counts: 942 for the pinned
    // [6,6,6,1,1] shape; set 1's shape is unpinned and falls back to
    // the RFC context_count.
    assert_eq!(set0.initial_state_row_count(), 942);
    assert_eq!(set1.initial_state_row_count(), 7563);

    // §4.2.14: set 0 coded, set 1 not.
    let deltas = rec
        .initial_state_delta
        .as_ref()
        .expect("set 0 has states_coded == 1");
    assert_eq!(deltas.len(), 2);
    let set0_deltas = deltas[0].as_ref().expect("set 0 coded");
    assert!(deltas[1].is_none(), "set 1 wrote states_coded == 0");
    assert_eq!(set0_deltas.len(), 942, "942 rows x 32 sr symbols");

    // Every delta is zero — every reconstructed initial state is 128 —
    // EXCEPT the very last symbol `[941][31]`, which RFC-exact decoding
    // reads as +1. The value sits in the region's final (flush-
    // redundant) bytes: re-emitting the whole Parameters with our
    // encoder reproduces the fixture's bytes through the last pre-flush
    // byte ONLY with this +1 in place, so it is genuinely coded, not a
    // tail-convention artifact. Row 941 is interop padding (>= the §4.1
    // context_count of 666), so the +1 never seeds a live context, and
    // the reference decoder's lossless decode of the fixture confirms
    // it is harmless there too.
    for (j, row) in set0_deltas.iter().enumerate() {
        for (k, &d) in row.iter().enumerate() {
            let expected = if (j, k) == (941, 31) { 1 } else { 0 };
            assert_eq!(d, expected, "initial_state_delta[0][{j}][{k}]");
        }
    }

    // §4.2.16 ec = 1 (slice CRCs present) — the alignment sentinel: a
    // mis-sized or mis-windowed delta block cannot reach this value.
    assert_eq!(rec.ec, Some(1), "§4.2.16 ec");
    // §4.2.17 intra: RFC-exact decoding reads 1 here (same final-bytes
    // provenance as the [941][31] delta — the symbol lives entirely in
    // the flush-redundant tail). Semantically inert for a single-
    // keyframe stream: Table 14's `intra == 1` only constrains later
    // frames to be keyframes.
    assert_eq!(rec.intra, Some(true), "§4.2.17 intra (see comment)");
}

/// THE conformance gate: the frame behind the `states_coded == 1`
/// record decodes bit-exactly to the reference decoder's output. The
/// transmitted deltas all reconstruct to 128 via Figures 29/30
/// (`(pred + 0) & 255`), so the pixels must equal the
/// `states_coded == 0` base decode.
#[test]
fn frame_decodes_bit_exact() {
    let parsed = parse_quantization_table_sets(STATES_CODED_1_CONFIG_RECORD).expect("parse");
    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame(
        STATES_CODED_1_FRAME,
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        parsed.record.ec.is_some(),
    )
    .expect("frame decode (ec == 1 slice CRC validated)");
    assert_eq!(decoded.planes.len(), 1, "gray: single plane");
    let expected: Vec<i32> = STATES_CODED_1_EXPECTED.iter().map(|&b| b as i32).collect();
    assert_eq!(
        decoded.planes[0].samples, expected,
        "64x48 gray plane bit-exact vs reference decode"
    );
}

/// Parse -> re-encode -> re-parse round-trip: the encode side emits the
/// coded §4.2.15 triple-loop under the same fixture-pinned layout
/// (dedicated fresh window, interop row count), so the re-encoded
/// record must re-parse to the identical tail and still decode the
/// frame bit-exactly. (Byte-identity with the fixture blob is NOT
/// asserted: range-coder flushes are redundant at the tail, and the
/// fixture's authoring flush differs from ours in the final two bytes
/// while decoding to the same symbols.)
#[test]
fn reencode_roundtrip_preserves_tail() {
    let parsed = parse_quantization_table_sets(STATES_CODED_1_CONFIG_RECORD).expect("parse");
    let blob = oxideav_ffv1::encode_configuration_record_with_quant_tables(
        &parsed.record,
        &parsed.quant_table_sets,
    )
    .expect("re-encode states_coded == 1 record");
    validate_configuration_record_crc(&blob).expect("re-encoded parity residue 0");

    let reparsed = parse_quantization_table_sets(&blob).expect("re-parse");
    assert_eq!(reparsed.record.ec, Some(1));
    assert_eq!(reparsed.record.intra, Some(true));
    assert_eq!(
        reparsed.record.initial_state_delta, parsed.record.initial_state_delta,
        "coded delta rows survive the round-trip exactly"
    );
    assert_eq!(reparsed.quant_table_sets[0].context_count, 666);
    assert_eq!(reparsed.quant_table_sets[1].context_count, 7563);

    let dims = FramePixelDimensions::new(64, 48).expect("dims");
    let decoded = decode_frame(
        STATES_CODED_1_FRAME,
        &reparsed.record,
        &reparsed.quant_table_sets,
        dims,
        reparsed.record.ec.is_some(),
    )
    .expect("frame decode against the round-tripped record");
    let expected: Vec<i32> = STATES_CODED_1_EXPECTED.iter().map(|&b| b as i32).collect();
    assert_eq!(decoded.planes[0].samples, expected);
}
