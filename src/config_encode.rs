//! FFV1 Configuration Record + §4.1 Quantization Table Set cascade
//! **encoder** — the symmetric inverse of
//! [`crate::config::parse_configuration_record`] +
//! [`crate::quant_table::parse_quantization_table_sets`] (RFC 9043
//! §4.1 / §4.2 / §4.3 / §4.3.2).
//!
//! The Configuration Record is the global per-stream header that lives
//! in the container's CodecPrivate / extradata area for FFV1 version 3
//! streams. On the decode side a single range-coder pass walks the
//! §4.2 `Parameters()` pseudocode (Figure 28), reads
//! `quant_table_set_count`, and *resumes* into the
//! `QuantizationTableSet( i )` cascade on the same arithmetic coder
//! and shared 32-slot context window. This encoder mirrors that walk
//! symbol-for-symbol on a [`RangeEncoder`], closes the range coder,
//! and appends a §4.3.2 `configuration_record_crc_parity` word solved
//! so the whole-blob CRC residue is zero.
//!
//! ## What is encoded
//!
//! The encoder emits everything the matching parser reads — the §4.2
//! Parameters prefix through `quant_table_set_count`, the §4.1
//! cascade, AND the §4.2.14-§4.2.17 tail (`states_coded` per set,
//! optional `initial_state_delta` triple-loop, `ec`, `intra`). The
//! produced `Vec<u8>` round-trips through
//! `parse_quantization_table_sets` to an equal
//! `Ffv1ConfigurationRecord` + `Vec<QuantizationTableSet>` including
//! the tail's `ec` and `intra` fields. The encoder always writes
//! `states_coded = 0` per set (the §4.2.14 default — "initial states
//! ... assumed to be all 128"), which matches every state buffer this
//! codec allocates today.
//!
//! ## Quantization Table inversion
//!
//! The §4.1 cascade encodes 256-entry tables whose first half (entries
//! `0..128`) is stored as a run-length stream of distinct
//! quantization levels: each run reads `len - 1` (`ur`) and writes
//! `len` consecutive copies of `scale * v` before incrementing `v`.
//! Inverting that is the §4.1 "stored values" view: walk the input
//! table's `0..128` slice, group consecutive equal entries into runs,
//! emit `len - 1` (`ur`) per run, and assert that each successive
//! group's value equals `scale * v` for `v = 0, 1, 2, …` (otherwise
//! the table does not fit the §4.1 wire schema).
//!
//! The §4.1 per-`QuantizationTable` state reset is mirrored from the
//! decoder (see [`crate::quant_table`] — reset the per-context state
//! window to 128 at the start of EACH of the five sub-tables, but do
//! NOT reset the arithmetic coder).

use crate::config::{Ffv1ConfigurationRecord, Ffv1Version};
use crate::crc::ffv1_crc32;
use crate::quant_table::{ParametersWithQuantTables, QuantizationTableSet};
use crate::range_coder::{RangeEncoder, PARAMETERS_INITIAL_STATE};
use crate::symbol::{put_br, put_sr, put_ur, SYMBOL_CONTEXT_SIZE};
use crate::Error;

/// Mirror of [`crate::config`]'s `PARAMETERS_STATE_LEN` — the cascade
/// shares that state buffer width. The §4.2 / §4.1 streams allocate one
/// 32-slot context window plus pad; the cursor stays at 0 throughout
/// (every Parameters symbol shares a single context window, see
/// `config::PARAMETERS_STATE_LEN`).
const PARAMETERS_STATE_LEN: usize = 64;

/// Upper bound on `quant_table_set_count` per RFC 9043 §4.2.13. Kept
/// internal to this module — the public surface validates against it
/// indirectly via [`Error::InvalidQuantTableSetCount`].
const MAX_QUANT_TABLE_SET_COUNT: u32 = 8;

/// Encode a Configuration Record + §4.1 Quantization Table Set cascade
/// into the §4.3 extradata byte stream, with the §4.3.2 CRC parity
/// word appended so the whole-blob CRC residue is zero.
///
/// The returned `Vec<u8>` is the container-level CodecPrivate /
/// extradata payload an FFV1 v3 stream needs. It parses back through
/// [`crate::parse_quantization_table_sets`] to an
/// `Ffv1ConfigurationRecord` `==` the input `record` and a sequence of
/// `QuantizationTableSet`s whose `tables` and `context_count` `==` the
/// input `qts`.
///
/// # Tail emission
///
/// The §4.2.14-§4.2.17 tail (`states_coded`, optional
/// `initial_state_delta`, `ec`, `intra`) is appended after the §4.1
/// cascade. Per-set `states_coded` is `1` iff
/// `record.initial_state_delta` is `Some(per_set)` and `per_set[i]` is
/// `Some(deltas)` of matching shape (`deltas.len() ==
/// qts[i].context_count`); the encoder then emits
/// `context_count[i] * 32` signed `sr` symbols on the shared 32-slot
/// Parameters state window per §4.2.15. When the field is `None`
/// (the codec's default), every set's `states_coded` is `0` and the
/// triple-loop is omitted (the §4.2.14 "initial states ... assumed
/// to be all 128" default). `ec` is taken from `record.ec`
/// (`None` = `0`) and `intra` from `record.intra` (`None` = `false`);
/// for v3 the parser (`parse_quantization_table_sets`) consumes the
/// same tail and restores all three fields.
///
/// # Errors
///
/// * [`Error::ConfigurationRecordForbiddenForVersion`] when `record.version`
///   is not `V3` — the §4.3 Configuration Record is a v3-only artefact
///   (§4.2.1 advisory).
/// * [`Error::UnsupportedCoderType`] when `record.coder_type > 2`.
/// * [`Error::UnsupportedColorspaceType`] is reserved for malformed
///   wire values, not unreachable here — every typed `ColorspaceType`
///   maps to a valid wire value.
/// * [`Error::InvalidChromaSubsample`] when either `log2_*_chroma_subsample`
///   exceeds 4.
/// * [`Error::InvalidQuantTableSetCount`] when `qts.len()` is 0 or > 8,
///   when `record.quant_table_set_count` is `Some(n)` and `n != qts.len()`,
///   or when `record.quant_table_set_count` is `None` (v3 requires it).
/// * [`Error::MalformedQuantTable`] when a `QuantizationTableSet`'s
///   first-half entries do not form the §4.1 monotonic `scale * v`
///   pattern the wire format can represent.
/// * [`Error::QuantContextCountOutOfRange`] when an encoded set's
///   derived `context_count` disagrees with the set's reported
///   `context_count` (defensive — a faithful caller will never trip it).
/// * [`Error::InitialStateDeltaShapeMismatch`] when
///   `record.initial_state_delta` is `Some(per_set)` and `per_set[i]`
///   is `Some(deltas)` whose length disagrees with `qts[i].context_count`
///   (the §4.2.15 loop count per set).
pub fn encode_configuration_record_with_quant_tables(
    record: &Ffv1ConfigurationRecord,
    qts: &[QuantizationTableSet],
) -> Result<Vec<u8>, Error> {
    if record.version != Ffv1Version::V3 {
        return Err(Error::ConfigurationRecordForbiddenForVersion(
            record.version.as_u32(),
        ));
    }
    if record.coder_type > 2 {
        return Err(Error::UnsupportedCoderType(record.coder_type));
    }
    if record.log2_h_chroma_subsample > 4 {
        return Err(Error::InvalidChromaSubsample(
            record.log2_h_chroma_subsample,
        ));
    }
    if record.log2_v_chroma_subsample > 4 {
        return Err(Error::InvalidChromaSubsample(
            record.log2_v_chroma_subsample,
        ));
    }

    // §4.2.13: 1 <= quant_table_set_count <= 8.
    let cascade_count = qts.len();
    if cascade_count == 0 || cascade_count as u32 > MAX_QUANT_TABLE_SET_COUNT {
        return Err(Error::InvalidQuantTableSetCount(cascade_count as u32));
    }
    // The Configuration Record field is what the decoder reads first
    // and loops on; it MUST agree with the cascade length the encoder
    // is about to emit. v3 requires the field to be present.
    let declared = record
        .quant_table_set_count
        .ok_or(Error::InvalidQuantTableSetCount(0))?;
    if declared as usize != cascade_count {
        return Err(Error::InvalidQuantTableSetCount(declared));
    }

    let mut re = RangeEncoder::new();
    // RFC 9043 §4.2: "Parameters has its own initial states, all set
    // to 128." A fresh state buffer starts every slot at 128.
    let mut state = [PARAMETERS_INITIAL_STATE; PARAMETERS_STATE_LEN];

    encode_parameters_prefix(&mut re, &mut state, record)?;
    for set in qts {
        encode_quantization_table_set(&mut re, &mut state, set)?;
    }

    // §4.2.14-§4.2.17 Parameters tail, on the same resumed range
    // coder + shared state buffer (Figure 28: the tail follows the
    // §4.1 cascade inside the same `version >= 3` block).
    encode_parameters_tail(&mut re, &mut state, record, qts)?;

    // Close the range coder so the produced bytes re-decode through a
    // fresh `RangeDecoder` (the only mode FFV1 ever uses, §3.8.1.1).
    let mut blob = re.finish();

    // §4.3.2 CRC parity: append a 4-byte word such that the §4.9.3
    // generator over the whole blob (prefix || parity) leaves
    // remainder zero. The same `CRC(M || CRC(M)) == 0` property
    // [`encode_slice_footer`] uses for §4.9.3 applies verbatim here:
    // the §4.9.3 generator is the same generator §4.3.2 invokes.
    let parity = ffv1_crc32(&blob);
    blob.extend_from_slice(&parity.to_be_bytes());

    Ok(blob)
}

/// Emit the §4.2 Parameters prefix through `quant_table_set_count`
/// onto `re`, walking the same field order [`parse_parameters`] reads.
///
/// The §4.1 cascade and the §4.2.14-§4.2.17 tail are written by
/// separate helpers (`encode_quantization_table_set` and
/// `encode_parameters_tail`) on the SAME resumed range coder + state
/// buffer.
fn encode_parameters_prefix(
    re: &mut RangeEncoder,
    state: &mut [u8],
    record: &Ffv1ConfigurationRecord,
) -> Result<(), Error> {
    debug_assert!(state.len() >= PARAMETERS_STATE_LEN);

    // ----- version (ur) ------------------------------------------------
    put_ur(
        re,
        &mut state[..SYMBOL_CONTEXT_SIZE],
        record.version.as_u32(),
    );

    // ----- micro_version (ur, v3 only) ---------------------------------
    // Mirror the decoder's policy: v3 always carries micro_version, the
    // wire field is present, the decoder reads it unconditionally.
    let micro = record
        .micro_version
        .ok_or(Error::UnsupportedVersion(record.version.as_u32()))?;
    put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], micro);

    // ----- coder_type (ur) ---------------------------------------------
    put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], record.coder_type);

    // ----- state_transition_delta[1..256] (sr, when coder_type > 1) ----
    if record.coder_type > 1 {
        // §4.2.4: 255 signed symbols, one shared context window — the
        // parser walks the same window with `get_sr` in the same order.
        for delta in record.state_transition_delta.iter().skip(1) {
            put_sr(re, &mut state[..SYMBOL_CONTEXT_SIZE], *delta);
        }
    }

    // ----- colorspace_type (ur) ----------------------------------------
    put_ur(
        re,
        &mut state[..SYMBOL_CONTEXT_SIZE],
        record.colorspace_type.as_u32(),
    );

    // ----- bits_per_raw_sample (ur) ------------------------------------
    //
    // §4.2.7 commentary: zero on the wire means "use 8". The parser
    // surfaces both 0 and 8 as `bits_per_raw_sample = 8`. The encoder
    // preserves the typed value by emitting `8` literally for 8-bit;
    // a caller that needs the wire-zero canonical encoding can pass
    // `bits_per_raw_sample = 0` explicitly through a future helper.
    put_ur(
        re,
        &mut state[..SYMBOL_CONTEXT_SIZE],
        record.bits_per_raw_sample,
    );

    // ----- chroma_planes (br) ------------------------------------------
    put_br(re, &mut state[..1], record.chroma_planes);

    // ----- log2_h_chroma_subsample (ur) --------------------------------
    put_ur(
        re,
        &mut state[..SYMBOL_CONTEXT_SIZE],
        record.log2_h_chroma_subsample,
    );

    // ----- log2_v_chroma_subsample (ur) --------------------------------
    put_ur(
        re,
        &mut state[..SYMBOL_CONTEXT_SIZE],
        record.log2_v_chroma_subsample,
    );

    // ----- extra_plane (br) --------------------------------------------
    put_br(re, &mut state[..1], record.extra_plane);

    // ----- v3 block: num_h_slices_minus_1 / num_v_slices_minus_1 /
    //                 quant_table_set_count (ur×3) --------------------
    //
    // Wire fields are `num_h_slices - 1` / `num_v_slices - 1`; the
    // decoder applies `wrapping_add(1)`.
    let h = record
        .num_h_slices
        .ok_or(Error::ConfigurationRecordForbiddenForVersion(
            record.version.as_u32(),
        ))?;
    let v = record
        .num_v_slices
        .ok_or(Error::ConfigurationRecordForbiddenForVersion(
            record.version.as_u32(),
        ))?;
    let qcount = record
        .quant_table_set_count
        .ok_or(Error::InvalidQuantTableSetCount(0))?;
    put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], h.wrapping_sub(1));
    put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], v.wrapping_sub(1));
    put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], qcount);

    Ok(())
}

/// Emit the §4.2.14-§4.2.17 Parameters tail onto the resumed range
/// coder, mirroring [`crate::quant_table::parse_parameters_tail`]
/// symbol-for-symbol.
///
/// Field order (Figure 28, `version >= 3` post-cascade block):
///
/// 1. For each of the `set_count` Quantization Table Sets: a `br`
///    `states_coded` flag. The flag is `1` iff the caller supplied a
///    matching-shape `Some(deltas)` for that set on
///    `record.initial_state_delta`; otherwise `0` (the §4.2.14
///    default — "initial states ... assumed to be all 128"). When the
///    flag is `1`, the §4.2.15 `initial_state_delta[i][j][k]`
///    triple-loop follows: `context_count[i] * 32` signed `sr`
///    symbols, in `j`-major / `k`-minor order, against the shared
///    32-slot Parameters state window. The encoder verifies the
///    supplied shape matches the §4.1 cascade's `context_count[i]`
///    and rejects mismatches with
///    [`Error::InitialStateDeltaShapeMismatch`].
/// 2. `ec` (`ur`), taken from `record.ec` (`None` = `0`).
/// 3. `intra` (`ur`), taken from `record.intra` (`None` = `false`).
fn encode_parameters_tail(
    re: &mut RangeEncoder,
    state: &mut [u8],
    record: &Ffv1ConfigurationRecord,
    qts: &[QuantizationTableSet],
) -> Result<(), Error> {
    let per_set_deltas: Option<&[Option<Vec<[i32; CONTEXT_SIZE]>>]> =
        record.initial_state_delta.as_deref();

    // Per-set `states_coded` + optional §4.2.15 triple-loop.
    for (i, set) in qts.iter().enumerate() {
        let coded_set = per_set_deltas
            .and_then(|all| all.get(i))
            .and_then(|s| s.as_ref());
        let states_coded = coded_set.is_some();
        put_br(re, &mut state[..1], states_coded);

        if let Some(deltas) = coded_set {
            // §4.1.2: context_count[i] is the per-set context count
            // the §4.2.15 loop iterates over. A mismatched shape would
            // desynchronise the symbol stream against the decoder.
            let expected = set.context_count as usize;
            if deltas.len() != expected {
                return Err(Error::InitialStateDeltaShapeMismatch {
                    set_index: i as u32,
                    expected_context_count: expected as u32,
                    actual_context_count: deltas.len() as u32,
                });
            }
            for row in deltas.iter() {
                for delta in row.iter() {
                    // `sr` against the shared 32-slot context window
                    // — matches the parser's loop in
                    // `quant_table::parse_parameters_tail`.
                    put_sr(re, &mut state[..SYMBOL_CONTEXT_SIZE], *delta);
                }
            }
        }
    }

    // Table 13 only defines `ec` values 0 and 1; "Other" is reserved
    // for future use. We pass the caller-supplied value through as a
    // `ur` symbol unchanged so the parsed-then-re-encoded round-trip
    // matches the input record bit-for-bit. Out-of-range values are a
    // caller responsibility; the parser will surface whatever the
    // resulting bitstream decodes to.
    let ec_value = record.ec.unwrap_or(0);
    put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], ec_value);

    let intra_value: u32 = if record.intra.unwrap_or(false) { 1 } else { 0 };
    put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], intra_value);

    Ok(())
}

/// Mirror of [`crate::quant_table`]'s `CONTEXT_SIZE`. Kept module-local
/// here so the encoder's §4.2.15 `[i32; CONTEXT_SIZE]` slot type stays
/// the same constant; equal to [`SYMBOL_CONTEXT_SIZE`] (§4.2 below
/// Figure 28).
const CONTEXT_SIZE: usize = 32;

/// Emit one `QuantizationTableSet( i )` (§4.1) onto the resumed
/// Parameters stream, mirroring [`decode_quantization_table_set`]
/// symbol-for-symbol.
fn encode_quantization_table_set(
    re: &mut RangeEncoder,
    state: &mut [u8],
    set: &QuantizationTableSet,
) -> Result<(), Error> {
    let mut scale: u64 = 1;
    for table in set.tables.iter() {
        let len_count = encode_quantization_table(re, state, scale, table)?;
        // §4.1: scale *= 2 * len_count - 1. Mirrors the decoder's
        // saturating walk so an over-large run on the encode side
        // would still produce a well-formed scale chain (the
        // round-trip check below will catch the divergence).
        scale = scale.saturating_mul(2u64.saturating_mul(len_count as u64).saturating_sub(1));
    }

    // §4.1.2: context_count[ i ] = ceil(scale / 2). Defensive
    // cross-check that the caller's `set.context_count` matches what
    // the freshly emitted cascade derives — otherwise the produced
    // blob would parse to a different `QuantizationTableSet` than the
    // caller handed us.
    let derived = scale.div_ceil(2);
    // §4.1.2: 1 <= context_count <= 32768.
    if derived == 0 || derived > 32768u64 {
        return Err(Error::QuantContextCountOutOfRange(
            derived.min(u32::MAX as u64) as u32,
        ));
    }
    if derived as u32 != set.context_count {
        return Err(Error::QuantContextCountOutOfRange(derived as u32));
    }
    Ok(())
}

/// Emit one `QuantizationTable(i, j, scale)` (§4.1) by deriving the
/// `len - 1` run-length stream from `table`'s first half.
///
/// Returns `len_count` (the number of distinct quantization levels
/// `v` reached), matching the decoder's return value.
///
/// # Errors
///
/// * [`Error::MalformedQuantTable`] when the input table's `0..128`
///   slice does not form a §4.1-valid sequence of runs of `scale * v`
///   for `v = 0, 1, 2, …`. This means the table either fails the
///   per-group "all entries equal `scale * v`" assertion or fails the
///   sign-flipped second-half reflection invariant.
fn encode_quantization_table(
    re: &mut RangeEncoder,
    state: &mut [u8],
    scale: u64,
    table: &[i32; 256],
) -> Result<i32, Error> {
    // Per-context state-window reset mirrors the decoder (#904
    // empirical resolution: reset per-table, NOT per-set; arithmetic
    // coder continues uninterrupted).
    for slot in state.iter_mut() {
        *slot = PARAMETERS_INITIAL_STATE;
    }

    // Validate the §4.1 sign-flipped reflection invariant the decoder
    // installs by construction. A table that violates it cannot round-trip.
    // table[256 - k] == -table[k] for k = 1..128, plus table[128] == -table[127].
    for k in 1..128usize {
        if table[256 - k] != table[k].wrapping_neg() {
            return Err(Error::MalformedQuantTable);
        }
    }
    if table[128] != table[127].wrapping_neg() {
        return Err(Error::MalformedQuantTable);
    }
    // The decoder ALWAYS writes table[0] = scale * 0 = 0 (the v = 0 run
    // begins at k = 0). A non-zero table[0] cannot be represented.
    if table[0] != 0 {
        return Err(Error::MalformedQuantTable);
    }

    // Walk the first-half slice, grouping equal-valued consecutive
    // entries into runs.
    //
    // Constraint: the n-th group's value MUST equal `(scale * n) as
    // i32` (saturating, to match the decoder's clamp). The decoder's
    // run-fill writes `scale * v` for v = 0, 1, 2, ...; the encoder
    // must produce the inverse, so each successive run jumps to the
    // next value in that sequence.
    let mut v: u32 = 0;
    let mut k: usize = 0;
    let mut len_count: i32 = 0;
    while k < 128 {
        let expected_value = (scale as i64)
            .saturating_mul(v as i64)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        if table[k] != expected_value {
            // The §4.1 wire schema cannot represent this table — the
            // run at position `k` does not have the value the
            // decoder's `scale * v` would have written.
            return Err(Error::MalformedQuantTable);
        }
        // Count the run length: consume consecutive equal entries
        // starting at `k`. The decoder allows a final run to push `k`
        // past 128 (the outer test is only between runs); we emit the
        // exact same `len - 1` symbol the decoder will read, so a
        // straddling-run encoding produces an identical `k`-walk.
        let mut len: u32 = 0;
        while k < 256 && table[k] == expected_value {
            len += 1;
            k += 1;
            // The decoder's outer `while k < 128` only tests between
            // runs; the inner `for n` is unbounded. To keep the
            // encoder a true inverse we mirror that: once we have
            // crossed `k >= 128` we stop counting consecutive equal
            // entries (the second-half sign-flipped reflection would
            // otherwise spuriously absorb mirror entries when v = 0).
            if k >= 128 {
                break;
            }
        }
        debug_assert!(len >= 1);

        // Emit `len - 1` as `ur` against the shared window.
        put_ur(re, &mut state[..SYMBOL_CONTEXT_SIZE], len - 1);

        v = v.checked_add(1).ok_or(Error::MalformedQuantTable)?;
        len_count = len_count.checked_add(1).ok_or(Error::MalformedQuantTable)?;

        if v > 128 {
            // A conforming first half has at most 128 distinct levels.
            return Err(Error::MalformedQuantTable);
        }
    }

    Ok(len_count)
}

/// Re-encode a parsed [`ParametersWithQuantTables`] into an extradata
/// blob, the natural composition of
/// [`encode_configuration_record_with_quant_tables`] for callers
/// holding a [`ParametersWithQuantTables`] returned by the parser.
///
/// `parse_quantization_table_sets(encode_parameters_with_quant_tables(p)?)`
/// returns a `ParametersWithQuantTables` whose `record` and
/// `quant_table_sets` are field-for-field equal to `p`.
pub fn encode_parameters_with_quant_tables(
    parsed: &ParametersWithQuantTables,
) -> Result<Vec<u8>, Error> {
    encode_configuration_record_with_quant_tables(&parsed.record, &parsed.quant_table_sets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ColorspaceType, Ffv1Version, NUM_TRANSITION_DELTAS};
    use crate::crc::validate_configuration_record_crc;
    use crate::predictor::{QuantTableSet, NUM_QUANT_SUBTABLES};
    use crate::quant_table::parse_quantization_table_sets;

    fn flat_qts(scale: u64, len_count: u32) -> QuantizationTableSet {
        // Build a Set whose every sub-table is a single
        // `(len_count)`-long zero run followed by `(127 -
        // (len_count - 1))` repeats of `scale * 1`, etc. The
        // simplest legal sub-table the decoder would produce is one
        // run filling all 128 entries: v = 0, len = 128.
        let _ = scale;
        let _ = len_count;
        let mut tables: QuantTableSet = [[0i32; 256]; NUM_QUANT_SUBTABLES];
        // All zeros: first-half all 0 (v = 0, len = 128); second-half
        // mirror is also 0 (sign-flip of 0).
        for table in tables.iter_mut() {
            for slot in table.iter_mut() {
                *slot = 0;
            }
        }
        // scale starts at 1, each sub-table contributes
        // `2 * len_count - 1 = 2 * 1 - 1 = 1`, so cumulative scale
        // stays at 1; context_count = ceil(1/2) = 1.
        QuantizationTableSet {
            tables,
            context_count: 1,
        }
    }

    fn dummy_v3_record() -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            version: Ffv1Version::V3,
            micro_version: Some(4),
            coder_type: 1,
            state_transition_delta: [0; NUM_TRANSITION_DELTAS],
            colorspace_type: ColorspaceType::YCbCr,
            bits_per_raw_sample: 8,
            chroma_planes: true,
            log2_h_chroma_subsample: 1,
            log2_v_chroma_subsample: 1,
            extra_plane: false,
            num_h_slices: Some(2),
            num_v_slices: Some(2),
            quant_table_set_count: Some(1),
            ec: Some(0),
            intra: Some(false),
            initial_state_delta: None,
        }
    }

    #[test]
    fn round_trip_minimal_v3_record_with_flat_qts() {
        let record = dummy_v3_record();
        let qts = vec![flat_qts(1, 1)];

        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");

        // §4.3.2: produced blob CRCs to zero.
        validate_configuration_record_crc(&blob)
            .expect("solved parity drives whole-blob residue to zero");

        // Parser yields the same record + qts.
        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.record.version, record.version);
        assert_eq!(parsed.record.micro_version, record.micro_version);
        assert_eq!(parsed.record.coder_type, record.coder_type);
        assert_eq!(parsed.record.colorspace_type, record.colorspace_type);
        assert_eq!(
            parsed.record.bits_per_raw_sample,
            record.bits_per_raw_sample
        );
        assert_eq!(parsed.record.chroma_planes, record.chroma_planes);
        assert_eq!(
            parsed.record.log2_h_chroma_subsample,
            record.log2_h_chroma_subsample
        );
        assert_eq!(
            parsed.record.log2_v_chroma_subsample,
            record.log2_v_chroma_subsample
        );
        assert_eq!(parsed.record.extra_plane, record.extra_plane);
        assert_eq!(parsed.record.num_h_slices, record.num_h_slices);
        assert_eq!(parsed.record.num_v_slices, record.num_v_slices);
        assert_eq!(
            parsed.record.quant_table_set_count,
            record.quant_table_set_count
        );
        assert_eq!(parsed.quant_table_sets.len(), 1);
        assert_eq!(parsed.quant_table_sets[0].context_count, 1);
        for (i, table) in parsed.quant_table_sets[0].tables.iter().enumerate() {
            assert_eq!(
                table, &qts[0].tables[i],
                "sub-table {i} re-decodes equal to the encoder input"
            );
        }
    }

    #[test]
    fn rejects_non_v3_version() {
        let mut record = dummy_v3_record();
        record.version = Ffv1Version::V0;
        let qts = vec![flat_qts(1, 1)];
        let err = encode_configuration_record_with_quant_tables(&record, &qts).unwrap_err();
        assert!(matches!(
            err,
            Error::ConfigurationRecordForbiddenForVersion(0)
        ));
    }

    #[test]
    fn rejects_coder_type_above_two() {
        let mut record = dummy_v3_record();
        record.coder_type = 3;
        let qts = vec![flat_qts(1, 1)];
        let err = encode_configuration_record_with_quant_tables(&record, &qts).unwrap_err();
        assert!(matches!(err, Error::UnsupportedCoderType(3)));
    }

    #[test]
    fn rejects_chroma_subsample_above_four() {
        let mut record = dummy_v3_record();
        record.log2_h_chroma_subsample = 5;
        let qts = vec![flat_qts(1, 1)];
        let err = encode_configuration_record_with_quant_tables(&record, &qts).unwrap_err();
        assert!(matches!(err, Error::InvalidChromaSubsample(5)));
    }

    #[test]
    fn rejects_cascade_count_zero() {
        let record = dummy_v3_record();
        let err = encode_configuration_record_with_quant_tables(&record, &[]).unwrap_err();
        assert!(matches!(err, Error::InvalidQuantTableSetCount(0)));
    }

    #[test]
    fn rejects_cascade_count_above_eight() {
        let record = dummy_v3_record();
        let qts: Vec<_> = (0..9).map(|_| flat_qts(1, 1)).collect();
        let err = encode_configuration_record_with_quant_tables(&record, &qts).unwrap_err();
        assert!(matches!(err, Error::InvalidQuantTableSetCount(9)));
    }

    #[test]
    fn rejects_mismatched_declared_count() {
        let mut record = dummy_v3_record();
        record.quant_table_set_count = Some(2);
        let qts = vec![flat_qts(1, 1)];
        let err = encode_configuration_record_with_quant_tables(&record, &qts).unwrap_err();
        assert!(matches!(err, Error::InvalidQuantTableSetCount(2)));
    }

    #[test]
    fn rejects_table_with_broken_sign_reflection() {
        // Construct a table that does NOT satisfy table[256-k] == -table[k].
        let mut tables: QuantTableSet = [[0i32; 256]; NUM_QUANT_SUBTABLES];
        tables[0][130] = 7; // table[256-126] should be -table[126] == 0, not 7.
        let set = QuantizationTableSet {
            tables,
            context_count: 1,
        };
        let record = dummy_v3_record();
        let err =
            encode_configuration_record_with_quant_tables(&record, std::slice::from_ref(&set))
                .unwrap_err();
        assert!(matches!(err, Error::MalformedQuantTable));
    }

    #[test]
    fn rejects_table_with_nonzero_at_index_zero() {
        let mut tables: QuantTableSet = [[0i32; 256]; NUM_QUANT_SUBTABLES];
        tables[0][0] = 1; // Decoder would always write 0 at index 0.
        let set = QuantizationTableSet {
            tables,
            context_count: 1,
        };
        let record = dummy_v3_record();
        let err =
            encode_configuration_record_with_quant_tables(&record, std::slice::from_ref(&set))
                .unwrap_err();
        assert!(matches!(err, Error::MalformedQuantTable));
    }

    #[test]
    fn round_trip_two_qts_sets_preserves_count() {
        let mut record = dummy_v3_record();
        record.quant_table_set_count = Some(2);
        let qts = vec![flat_qts(1, 1), flat_qts(1, 1)];

        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");
        validate_configuration_record_crc(&blob).expect("solved parity drives residue to zero");

        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.quant_table_sets.len(), 2);
        assert_eq!(parsed.record.quant_table_set_count, Some(2));
    }

    #[test]
    fn round_trip_with_state_transition_delta() {
        // coder_type = 2 forces the §4.2.4 sr loop for indices 1..256.
        let mut record = dummy_v3_record();
        record.coder_type = 2;
        // Sparse, signed delta pattern: +1 at index 5, -2 at index 200.
        record.state_transition_delta[5] = 1;
        record.state_transition_delta[200] = -2;

        let qts = vec![flat_qts(1, 1)];
        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");
        validate_configuration_record_crc(&blob).expect("CRC residue zero");

        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.record.coder_type, 2);
        assert_eq!(parsed.record.state_transition_delta[5], 1);
        assert_eq!(parsed.record.state_transition_delta[200], -2);
        // Indices we did not touch must round-trip as 0.
        assert_eq!(parsed.record.state_transition_delta[42], 0);
    }

    #[test]
    fn round_trip_via_parameters_wrapper() {
        let record = dummy_v3_record();
        let qts = vec![flat_qts(1, 1)];
        let blob = encode_configuration_record_with_quant_tables(&record, &qts).expect("encode");

        let parsed = parse_quantization_table_sets(&blob).expect("parse");
        // Re-encode through the typed wrapper; it should produce the
        // same byte stream the direct API does (modulo parity, which
        // is deterministic given the prefix).
        let reblob = encode_parameters_with_quant_tables(&parsed).expect("wrapper re-encode");
        assert_eq!(blob, reblob);
    }

    #[test]
    fn encoder_is_deterministic() {
        let record = dummy_v3_record();
        let qts = vec![flat_qts(1, 1)];
        let a = encode_configuration_record_with_quant_tables(&record, &qts).unwrap();
        let b = encode_configuration_record_with_quant_tables(&record, &qts).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn cascade_context_count_disagreement_rejected() {
        // Build a flat set but claim context_count = 99 (impossible
        // for an all-zero cascade whose derived count is 1).
        let mut set = flat_qts(1, 1);
        set.context_count = 99;
        let record = dummy_v3_record();
        let err =
            encode_configuration_record_with_quant_tables(&record, std::slice::from_ref(&set))
                .unwrap_err();
        assert!(matches!(err, Error::QuantContextCountOutOfRange(1)));
    }

    #[test]
    fn round_trip_tail_default_ec_intra() {
        // The §4.2.14-§4.2.17 tail with the default `states_coded == 0`
        // for every set, `ec == 0`, `intra == false`. Verifies the
        // encoder's tail emission AND the
        // `parse_quantization_table_sets` companion read.
        let mut record = dummy_v3_record();
        record.ec = Some(0);
        record.intra = Some(false);
        let qts = vec![flat_qts(1, 1)];
        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");
        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.record.ec, Some(0), "ec round-trips");
        assert_eq!(parsed.record.intra, Some(false), "intra round-trips");
    }

    #[test]
    fn round_trip_tail_ec_one_intra_true() {
        // §4.2.16 ec=1 (per-slice CRC) + §4.2.17 intra=1 (keyframe-only
        // stream) is the practical encoder configuration. Both bits
        // must traverse the tail unchanged.
        let mut record = dummy_v3_record();
        record.ec = Some(1);
        record.intra = Some(true);
        let qts = vec![flat_qts(1, 1)];
        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");
        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.record.ec, Some(1));
        assert_eq!(parsed.record.intra, Some(true));
    }

    #[test]
    fn round_trip_tail_none_defaults_to_zero() {
        // Caller may leave `record.ec` / `record.intra` as `None`
        // (e.g. when re-encoding a record that was parsed before this
        // round added tail support). The encoder writes 0 / false in
        // that case, and the re-parsed record surfaces `Some(0)` /
        // `Some(false)` — they are now always present on the wire.
        let mut record = dummy_v3_record();
        record.ec = None;
        record.intra = None;
        let qts = vec![flat_qts(1, 1)];
        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");
        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.record.ec, Some(0));
        assert_eq!(parsed.record.intra, Some(false));
    }

    #[test]
    fn round_trip_tail_multi_set_states_coded_zero() {
        // Two Sets: a per-set `states_coded` is written for each, both
        // `0`, then ec + intra. Verifies the per-set loop in the tail
        // (Figure 28: `for (i = 0; i < quant_table_set_count; i++)`).
        let mut record = dummy_v3_record();
        record.quant_table_set_count = Some(2);
        record.ec = Some(1);
        record.intra = Some(false);
        let qts = vec![flat_qts(1, 1), flat_qts(1, 1)];
        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");
        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.record.ec, Some(1));
        assert_eq!(parsed.record.intra, Some(false));
        assert_eq!(parsed.quant_table_sets.len(), 2);
    }

    #[test]
    fn round_trip_tail_with_state_transition_delta() {
        // Make sure the tail still round-trips when `coder_type == 2`
        // forces the §4.2.4 `state_transition_delta` `sr` loop into the
        // prefix — the tail picks up from the same resumed coder + state
        // buffer.
        let mut record = dummy_v3_record();
        record.coder_type = 2;
        record.state_transition_delta[10] = 3;
        record.ec = Some(1);
        record.intra = Some(true);
        let qts = vec![flat_qts(1, 1)];
        let blob =
            encode_configuration_record_with_quant_tables(&record, &qts).expect("encode succeeds");
        let parsed = parse_quantization_table_sets(&blob).expect("parse re-encoded blob");
        assert_eq!(parsed.record.coder_type, 2);
        assert_eq!(parsed.record.state_transition_delta[10], 3);
        assert_eq!(parsed.record.ec, Some(1));
        assert_eq!(parsed.record.intra, Some(true));
    }

    // -------------------------------------------------------------------
    // §4.2.15 initial_state_delta round-trips (round 241).
    //
    // These cover the new
    // `Ffv1ConfigurationRecord::initial_state_delta` field: a per-set
    // `Option<Vec<[i32; 32]>>` that drives `states_coded == 1` on the
    // wire and the §4.2.15 triple-loop emission. Each test
    // encodes → parses, checks the deltas are preserved, and verifies
    // the §4.3.2 whole-blob CRC residue is still zero so the parity
    // word is correctly solved against the new wire footprint.
    // -------------------------------------------------------------------

    /// Build a `[i32; CONTEXT_SIZE]` row whose entries follow a
    /// deterministic pseudo-pattern with both signs across the
    /// `sr`-symbol range, exercising the §4.2.15 triple-loop on
    /// non-trivial inputs.
    fn pseudo_delta_row(seed: i32) -> [i32; super::CONTEXT_SIZE] {
        let mut row = [0i32; super::CONTEXT_SIZE];
        for (k, slot) in row.iter_mut().enumerate() {
            // Small magnitudes (±k mod 8 with sign rotation) so the
            // `sr` encoder traverses Golomb-like small-symbol paths
            // and the residual state buffer remains in a well-defined
            // regime. The seed shifts the whole row to keep
            // distinct sets distinguishable.
            *slot = match (k as i32 + seed) % 7 {
                0 => 0,
                1 => 1,
                2 => -1,
                3 => 5,
                4 => -3,
                5 => 12,
                _ => -8,
            };
        }
        row
    }

    #[test]
    fn round_trip_initial_state_delta_zero_row_single_set() {
        // `states_coded == 1` but every coded delta is zero: the
        // §4.2.15 loop runs 1 * 32 = 32 `sr` symbols, all zero. The
        // wire footprint differs from the `None` case (extra symbols),
        // but the round-trip restores the same `Some(vec![Some(rows)])`
        // shape with all-zero rows.
        let mut record = dummy_v3_record();
        record.ec = Some(1);
        record.intra = Some(false);
        let qts = vec![flat_qts(1, 1)]; // context_count[0] == 1
        let zero_row = [0i32; super::CONTEXT_SIZE];
        record.initial_state_delta = Some(vec![Some(vec![zero_row])]);

        let blob = encode_configuration_record_with_quant_tables(&record, &qts)
            .expect("zero-row round-trip encodes");
        validate_configuration_record_crc(&blob)
            .expect("§4.3.2 parity word is solved against the new wire footprint");

        let parsed =
            parse_quantization_table_sets(&blob).expect("parse re-encoded blob with deltas");
        let surfaced = parsed
            .record
            .initial_state_delta
            .as_ref()
            .expect("parser surfaces Some(_) when at least one set wrote states_coded == 1");
        assert_eq!(surfaced.len(), 1);
        let set0 = surfaced[0].as_ref().expect("set 0 wrote states_coded == 1");
        assert_eq!(set0.len(), 1, "context_count[0] == 1 → one row");
        assert_eq!(set0[0], zero_row);
    }

    #[test]
    fn round_trip_initial_state_delta_nontrivial_row_single_set() {
        // §4.2.15 triple-loop with a non-trivial pseudo-pattern row of
        // mixed-sign small magnitudes. Each `sr` symbol must traverse
        // the shared 32-slot context window in the same order on both
        // sides; any drift would surface as a single wrong delta entry.
        let mut record = dummy_v3_record();
        record.ec = Some(1);
        record.intra = Some(true);
        let qts = vec![flat_qts(1, 1)];
        let row = pseudo_delta_row(0);
        record.initial_state_delta = Some(vec![Some(vec![row])]);

        let blob = encode_configuration_record_with_quant_tables(&record, &qts)
            .expect("non-trivial delta encodes");
        validate_configuration_record_crc(&blob).expect("§4.3.2 parity zero");

        let parsed = parse_quantization_table_sets(&blob).unwrap();
        let surfaced = parsed.record.initial_state_delta.as_ref().unwrap();
        assert_eq!(surfaced.len(), 1);
        let set0 = surfaced[0].as_ref().unwrap();
        assert_eq!(set0.len(), 1);
        assert_eq!(set0[0], row, "every `sr` symbol round-trips bit-exactly");
        // ec/intra still pick up correctly after the per-set tail.
        assert_eq!(parsed.record.ec, Some(1));
        assert_eq!(parsed.record.intra, Some(true));
    }

    #[test]
    fn round_trip_initial_state_delta_mixed_sets_states_coded() {
        // Two sets, first writes `states_coded == 1` with a coded row,
        // second writes `states_coded == 0` (no triple-loop). The
        // per-set `Option` distinguishes the two branches on
        // round-trip. The shared Parameters state buffer keeps the
        // state coherent so ec/intra still land in the right slots.
        let mut record = dummy_v3_record();
        record.quant_table_set_count = Some(2);
        record.ec = Some(0);
        record.intra = Some(false);
        let qts = vec![flat_qts(1, 1), flat_qts(1, 1)];
        let row = pseudo_delta_row(3);
        // Outer Some + per-set [Some(rows), None] — only set 0 carries
        // the wire loop.
        record.initial_state_delta = Some(vec![Some(vec![row]), None]);

        let blob = encode_configuration_record_with_quant_tables(&record, &qts)
            .expect("mixed-states encode");
        validate_configuration_record_crc(&blob).expect("§4.3.2 parity zero");

        let parsed = parse_quantization_table_sets(&blob).unwrap();
        let surfaced = parsed.record.initial_state_delta.as_ref().unwrap();
        assert_eq!(surfaced.len(), 2);
        let set0 = surfaced[0]
            .as_ref()
            .expect("set 0 carries states_coded == 1");
        assert!(
            surfaced[1].is_none(),
            "set 1 wrote states_coded == 0 (no triple-loop)"
        );
        assert_eq!(set0.len(), 1);
        assert_eq!(set0[0], row);
        assert_eq!(parsed.record.ec, Some(0));
        assert_eq!(parsed.record.intra, Some(false));
    }

    #[test]
    fn round_trip_initial_state_delta_all_unset_stays_none() {
        // Sanity inverse: the §4.2.14 default (every set `states_coded
        // == 0`) parses back as `record.initial_state_delta == None`,
        // so the field cleanly distinguishes "no triple-loop on the
        // wire" from "all-zero deltas on the wire".
        let mut record = dummy_v3_record();
        record.ec = Some(1);
        record.intra = Some(false);
        record.initial_state_delta = None; // explicit
        let qts = vec![flat_qts(1, 1)];
        let blob = encode_configuration_record_with_quant_tables(&record, &qts).unwrap();
        let parsed = parse_quantization_table_sets(&blob).unwrap();
        assert!(parsed.record.initial_state_delta.is_none());
    }

    #[test]
    fn initial_state_delta_shape_mismatch_rejected() {
        // The §4.2.15 loop encodes exactly `context_count[i] * 32`
        // symbols per coded set; a caller-supplied per-set vector
        // whose length disagrees with the §4.1 cascade's
        // `context_count[i]` would produce a desynchronised wire
        // stream. The encoder rejects it up-front so the corrupt blob
        // is never produced.
        let mut record = dummy_v3_record();
        record.ec = Some(0);
        record.intra = Some(false);
        // flat_qts(1, 1) has context_count == 1; supplying TWO rows
        // is one too many.
        let qts = vec![flat_qts(1, 1)];
        let row = pseudo_delta_row(0);
        record.initial_state_delta = Some(vec![Some(vec![row, row])]);

        let err = encode_configuration_record_with_quant_tables(&record, &qts)
            .expect_err("shape mismatch rejected");
        assert!(
            matches!(
                err,
                Error::InitialStateDeltaShapeMismatch {
                    set_index: 0,
                    expected_context_count: 1,
                    actual_context_count: 2,
                }
            ),
            "unexpected error: {err:?}",
        );
    }

    #[test]
    fn round_trip_initial_state_delta_preserves_signed_extremes() {
        // The `sr` symbol round-trips arbitrary i32 values per
        // [`crate::symbol::tests`]; verify the §4.2.15 loop preserves
        // the extremes so callers reading off-wire deltas surface what
        // the wire actually carried. Each row mixes `i32::MIN`,
        // `i32::MAX`, zero, and small magnitudes.
        let mut record = dummy_v3_record();
        record.ec = Some(1);
        record.intra = Some(true);
        let qts = vec![flat_qts(1, 1)];
        let mut row = [0i32; super::CONTEXT_SIZE];
        row[0] = i32::MIN;
        row[1] = i32::MAX;
        row[2] = -1;
        row[3] = 1;
        row[4] = 0;
        // Leave remaining slots at 0 — the loop iterates them too.
        record.initial_state_delta = Some(vec![Some(vec![row])]);

        let blob = encode_configuration_record_with_quant_tables(&record, &qts).unwrap();
        validate_configuration_record_crc(&blob).expect("§4.3.2 parity zero");

        let parsed = parse_quantization_table_sets(&blob).unwrap();
        let surfaced = parsed.record.initial_state_delta.as_ref().unwrap();
        assert_eq!(surfaced[0].as_ref().unwrap()[0], row);
    }

    #[test]
    fn round_trip_initial_state_delta_encoder_is_deterministic() {
        // Two back-to-back encodes of the same record + qts produce
        // byte-identical blobs. The §4.2.15 loop has no hidden state
        // beyond the shared Parameters context window, so determinism
        // here pins the per-symbol state-window evolution.
        let mut record = dummy_v3_record();
        record.ec = Some(1);
        record.intra = Some(false);
        let qts = vec![flat_qts(1, 1)];
        record.initial_state_delta = Some(vec![Some(vec![pseudo_delta_row(7)])]);

        let a = encode_configuration_record_with_quant_tables(&record, &qts).unwrap();
        let b = encode_configuration_record_with_quant_tables(&record, &qts).unwrap();
        assert_eq!(a, b);
    }
}
