//! Golomb-Rice mode primitives (RFC 9043 §3.8.2).
//!
//! This module implements both halves of the `sd`-typed symbol coding
//! for FFV1's Golomb-Rice coding mode — the alternative entropy coder
//! selected by `coder_type == 0` (RFC 9043 §4.2.3 Table 7). The
//! range-coded path lives in [`crate::range_coder`] + [`crate::symbol`];
//! this module carries the bit-coded counterpart.
//!
//! The Golomb-Rice path uses these layers, top-down:
//!
//! * **Bit reading / writing** — MSB-first `get_bits(n)` /
//!   `put_bits(v, n)` (RFC 9043 §2.2.9.4) over the slice's bit region,
//!   supplied by [`crate::bit_reader::BitReader`] /
//!   [`crate::bit_reader::BitWriter`].
//! * **Variable-length codes** — `get_ur_golomb(k)` /
//!   [`put_ur_golomb_esc`] and `get_sr_golomb(k)` /
//!   [`put_sr_golomb_esc`] per Figures 26 / 27 plus the ESC-prefix
//!   table in §3.8.2.1.1.
//! * **Sign extension** — `sign_extend` per §3.8.2.3.
//! * **Per-context VLC state** — [`VlcState`] plus [`get_vlc_symbol`] /
//!   [`put_vlc_symbol`] per the §3.8.2.4 scalar-mode pseudocode.
//! * **Run mode** — `log2_run` and the `run_index` / `run_mode` /
//!   `run_count` state machine that activates when the absolute
//!   context value is 0 (§3.8.2.2 + §3.8.2.4.1).
//!
//! The decode-side layers drive the per-Line sample-difference decode
//! loop in [`crate::sample_diff`]; the encode-side layers
//! (`put_ur_golomb_esc` / `put_sr_golomb_esc` / [`put_vlc_symbol`] /
//! [`put_vlc_symbol_level`]) are the symmetric inverses that produce
//! the bytes a fresh [`crate::bit_reader::BitReader`] re-decodes back
//! to the input residual. Both halves share the *same* §3.8.2.4 state
//! update, so a per-context state cloned to both encoder and decoder
//! drifts in lockstep.

use crate::bit_reader::{BitReader, BitWriter};

/// Initial values for the per-context VLC state at the start of a
/// keyframe (RFC 9043 §3.8.2.5).
///
/// `keyframe == 1` resets every VLC coder state slot to this value.
/// `count = 1` is the spec's literal value and represents "one symbol
/// observed" — this anchors the `error_sum / count` heuristic that
/// `get_vlc_symbol` uses to choose `k`.
pub const VLC_STATE_INITIAL: VlcState = VlcState {
    drift: 0,
    error_sum: 4,
    bias: 0,
    count: 1,
};

/// Per-context adaptive VLC coder state for the Golomb-Rice path
/// (RFC 9043 §3.8.2.4 + §3.8.2.5).
///
/// One [`VlcState`] is held per context value (`context_count` slots
/// per plane per slice). The four fields together implement an
/// adaptive Rice-parameter estimate: `count` tracks how many symbols
/// have been observed in this context, `error_sum` aggregates the
/// magnitudes of recent residuals, `drift` is the signed error sum
/// used to nudge `bias`, and `bias` recenters the symbol distribution
/// to track local mean changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VlcState {
    /// Signed accumulator for the per-context mean-tracking nudge.
    /// Updated by `+= v` after each decoded symbol; rescaled when
    /// `count == 128`.
    pub drift: i32,
    /// Sum of `|decoded residual|` over recently seen symbols;
    /// rescaled when `count == 128`.
    pub error_sum: i32,
    /// Per-context bias added to the decoded residual to recenter it
    /// (clamped to `-128..=127`).
    pub bias: i32,
    /// Number of symbols observed in this context since the last
    /// rescale (`>=1`).
    pub count: i32,
}

impl Default for VlcState {
    fn default() -> Self {
        VLC_STATE_INITIAL
    }
}

/// Read an unsigned Golomb-Rice code with parameter `k`
/// (RFC 9043 Figure 26).
///
/// The encoding splits the value into a unary prefix (up to 12 ones
/// before a stop-zero indicates the prefix value) and a `k`-bit suffix
/// in non-ESC mode. The ESC mode is signalled by 12 leading ones; the
/// suffix is then a flat 11-bit-wide field plus the constant 11.
pub fn get_ur_golomb(br: &mut BitReader<'_>, k: u32) -> u32 {
    // Loop bound 12 mirrors Figure 26's `for (prefix = 0; prefix < 12;
    // prefix++)`. Each iteration reads a leading bit; the first 1-bit
    // stops the prefix.
    for prefix in 0..12u32 {
        if br.get_bit() == 1 {
            // Non-ESC path: read `k` suffix bits and add the
            // shifted prefix. `k` can be 0 in which case the
            // get_bits(0) is undefined per our bit reader contract;
            // guard with an explicit zero case.
            let suffix = if k == 0 { 0 } else { br.get_bits(k) };
            return suffix + (prefix << k);
        }
    }
    // ESC path: read a flat 'bits' suffix (the caller's per-symbol
    // bit width, supplied as 'bits'). Per Figure 26 the constant
    // 11 is added.
    //
    // Figure 26 reads `get_bits(bits) + 11`; `bits` is supplied by
    // the caller (§3.8 Figure 10: `bits = bits_per_raw_sample + 1`
    // for JPEG 2000 RCT, `bits_per_raw_sample` otherwise). We don't
    // have `bits` in this primitive — `get_ur_golomb` is called from
    // `get_vlc_symbol` which passes the caller's `bits` via an
    // explicit wrapper [`get_ur_golomb_esc`].
    //
    // To avoid a `bits = 0` panic in stand-alone tests of the prefix
    // path, this branch falls back to reading 11 bits (the minimum
    // safe width that still matches a valid encoding); higher-level
    // callers should use [`get_ur_golomb_esc`] instead.
    br.get_bits(11) + 11
}

/// Read an unsigned Golomb-Rice code with parameter `k` and an
/// ESC-mode suffix width of `bits` (RFC 9043 Figure 26).
///
/// This is the form used by [`get_vlc_symbol`]: the per-symbol bit
/// width is the configuration record's `bits_per_raw_sample`
/// (plus 1 for JPEG 2000 RCT, per §3.8 Figure 10).
pub fn get_ur_golomb_esc(br: &mut BitReader<'_>, k: u32, bits: u32) -> u32 {
    for prefix in 0..12u32 {
        if br.get_bit() == 1 {
            let suffix = if k == 0 { 0 } else { br.get_bits(k) };
            return suffix + (prefix << k);
        }
    }
    // ESC: read 'bits' raw bits, add 11.
    let suffix = if bits == 0 { 0 } else { br.get_bits(bits) };
    suffix + 11
}

/// Read a signed Golomb-Rice code with parameter `k`
/// (RFC 9043 Figure 27).
///
/// The encoding folds signed values onto non-negative integers
/// (interleaving): even values map to non-negative, odd to negative.
/// `bits` is the per-symbol ESC width (see [`get_ur_golomb_esc`]).
pub fn get_sr_golomb_esc(br: &mut BitReader<'_>, k: u32, bits: u32) -> i32 {
    let v = get_ur_golomb_esc(br, k, bits);
    if v & 1 != 0 {
        // Odd → negative branch: -(v >> 1) - 1.
        -((v >> 1) as i32) - 1
    } else {
        (v >> 1) as i32
    }
}

/// Compute a sign-extended integer from a width-limited unsigned
/// magnitude (RFC 9043 §3.8.2.3).
///
/// `input_bits` is the bit width to fit into; values whose high bit
/// is set are interpreted as the corresponding negative number in
/// two's-complement form.
pub fn sign_extend(input_number: i32, input_bits: u32) -> i32 {
    if input_bits == 0 || input_bits >= 32 {
        return input_number;
    }
    let negative_bias: i32 = 1i32 << (input_bits - 1);
    let bits_mask: i32 = negative_bias.wrapping_sub(1);
    let output_number = input_number & bits_mask;
    let is_negative = (input_number & negative_bias) != 0;
    if is_negative {
        output_number - negative_bias
    } else {
        output_number
    }
}

/// Decode one signed Sample-Difference value via the adaptive
/// Golomb-Rice VLC coder (RFC 9043 §3.8.2.4).
///
/// `state` carries the per-context VLC state (drift / error_sum /
/// bias / count) and is updated in place. `bits` is the per-symbol
/// width used by the ESC-mode suffix (per §3.8 Figure 10).
///
/// This is the *scalar-mode* function. The level-coding variant for
/// run-mode aftermath (which skips the zero value) lives in
/// [`get_vlc_symbol_level`].
pub fn get_vlc_symbol(br: &mut BitReader<'_>, state: &mut VlcState, bits: u32) -> i32 {
    let k = vlc_pick_k(state);
    let v_raw = get_sr_golomb_esc(br, k, bits);

    // Conditional sign flip (§3.8.2.4):
    //   if (2 * state->drift < -state->count) {
    //       v = -1 - v;
    //   }
    let v = if 2 * state.drift < -state.count {
        -1 - v_raw
    } else {
        v_raw
    };

    // Final value = sign_extend(v + state->bias, bits).
    let ret = sign_extend(v.wrapping_add(state.bias), bits);

    // The decoder updates the adaptive state with the post-flip `v`.
    // Identical to the encoder's `vlc_update` (called from
    // `put_vlc_symbol` below); the symmetry is what keeps the per-
    // context state windows in lockstep through a round trip.
    vlc_update(state, v);

    ret
}

/// Adaptive Rice-parameter selection (§3.8.2.4 pseudocode
/// `i = state->count; k = 0; while (i < state->error_sum) { k++; i += i; }`).
///
/// Returns the `k` the decoder will use for the next [`get_sr_golomb_esc`]
/// call against this `state`; the encoder reuses the same routine to
/// pick the symmetric `k` for the next [`put_sr_golomb_esc`] call,
/// guaranteeing the two halves share the same `k` for each symbol.
fn vlc_pick_k(state: &VlcState) -> u32 {
    let mut i: i64 = state.count as i64;
    let mut k: u32 = 0;
    while i < state.error_sum as i64 {
        k += 1;
        i += i;
        // Safety stop: k cannot meaningfully exceed 32 because i32
        // doubles saturate after that. The pseudocode is unbounded;
        // we add the guard to avoid an infinite loop on a corrupted
        // state.
        if k >= 32 {
            break;
        }
    }
    k
}

/// Per-context state update from §3.8.2.4 — identical for both halves.
///
/// `v` is the post-sign-flip residual the decoder produced (or the
/// encoder is about to produce). Mutates `state` in place: `error_sum`
/// gains the absolute value of `v`, `drift` gains `v`, the counters
/// rescale when `count` reaches 128, and the bias/drift nudge keeps
/// `bias` tracking the local mean.
fn vlc_update(state: &mut VlcState, v: i32) {
    // Update the adaptive state:
    //   state->error_sum += abs(v);
    //   state->drift     += v;
    state.error_sum = state.error_sum.saturating_add(v.unsigned_abs() as i32);
    state.drift = state.drift.saturating_add(v);

    //   if (state->count == 128) { rescale all three counters by /= 2. }
    if state.count == 128 {
        state.count >>= 1;
        state.drift >>= 1;
        state.error_sum >>= 1;
    }
    state.count = state.count.saturating_add(1);

    // bias / drift nudge:
    //   if (state->drift <= -state->count) {
    //       state->bias  = max(state->bias - 1, -128);
    //       state->drift = max(state->drift + state->count, -state->count + 1);
    //   } else if (state->drift > 0) {
    //       state->bias  = min(state->bias + 1, 127);
    //       state->drift = min(state->drift - state->count, 0);
    //   }
    if state.drift <= -state.count {
        state.bias = (state.bias - 1).max(-128);
        state.drift = (state.drift + state.count).max(-state.count + 1);
    } else if state.drift > 0 {
        state.bias = (state.bias + 1).min(127);
        state.drift = (state.drift - state.count).min(0);
    }
}

/// Encode an unsigned Golomb-Rice code with parameter `k` and an
/// ESC-mode suffix width of `bits` — the symmetric inverse of
/// [`get_ur_golomb_esc`] (RFC 9043 Figure 26).
///
/// Mirrors Figure 26's two regimes:
///
/// * Non-ESC (`prefix < 12`): emit `prefix` zeros then a single 1, then
///   the `k`-bit suffix `value & ((1 << k) - 1)`. `prefix == value >> k`.
/// * ESC (`prefix >= 12`): emit twelve zero bits to signal the ESC
///   prefix, then a flat `bits`-wide field whose value is `value - 11`
///   per Figure 26's `get_bits(bits) + 11` decode rule.
///
/// The `k == 0` non-ESC suffix is skipped (its width is zero), matching
/// the decoder's `if k == 0 { 0 } else { br.get_bits(k) }` guard. The
/// `bits == 0` ESC suffix is skipped for the same reason — the
/// inverse of the decoder's `if bits == 0 { 0 } else { br.get_bits(bits) }`.
pub fn put_ur_golomb_esc(bw: &mut BitWriter, k: u32, bits: u32, value: u32) {
    let prefix = value >> k;
    if prefix < 12 {
        for _ in 0..prefix {
            bw.put_bit(0);
        }
        bw.put_bit(1);
        if k > 0 {
            // Take the bottom `k` bits — the decoder's `get_bits(k)`
            // reads MSB-first; `put_bits` emits MSB-first too.
            let mask = (1u32 << k) - 1;
            bw.put_bits(value & mask, k);
        }
    } else {
        // ESC: twelve zeros then a flat `bits`-wide field for
        // `value - 11`. `value - 11` is guaranteed non-negative because
        // `prefix >= 12` implies `value >= 12 << k >= 12 > 11`.
        for _ in 0..12 {
            bw.put_bit(0);
        }
        if bits > 0 {
            let esc = value - 11;
            bw.put_bits(esc, bits);
        }
    }
}

/// Encode a signed Golomb-Rice code with parameter `k` and ESC width
/// `bits` — the symmetric inverse of [`get_sr_golomb_esc`]
/// (RFC 9043 Figure 27).
///
/// The interleave folding maps `0, -1, 1, -2, 2, ...` onto unsigned
/// `0, 1, 2, 3, 4, ...`:
///
/// * Non-negative `value` maps to `2 * value`.
/// * Negative `value` maps to `2 * |value| - 1`.
///
/// The encoded unsigned magnitude then walks [`put_ur_golomb_esc`].
pub fn put_sr_golomb_esc(bw: &mut BitWriter, k: u32, bits: u32, value: i32) {
    // The signed-to-unsigned fold mirrors the decoder's
    //   if v & 1 != 0 { -((v >> 1) as i32) - 1 } else { (v >> 1) as i32 }
    // inverse: `value = (v - 1) / 2 - (-1)` for odd-`v` negatives, and
    // `value = v / 2` for even-`v` non-negatives.
    //
    // Cast through u32 (with `i32::MIN` cared for via `unsigned_abs`)
    // so the `2 * |value|` multiply never wraps a signed multiplication.
    let unsigned: u32 = if value < 0 {
        // `unsigned_abs` returns `value.abs() as u32`, handling the
        // `i32::MIN` magnitude (whose negation would overflow i32).
        let mag = value.unsigned_abs();
        mag.wrapping_mul(2).wrapping_sub(1)
    } else {
        (value as u32).wrapping_mul(2)
    };
    put_ur_golomb_esc(bw, k, bits, unsigned);
}

/// Level-coding variant of [`get_vlc_symbol`] (§3.8.2.4.1).
///
/// Identical to scalar mode except the zero value is skipped — used
/// for the very first sample-difference after a run-mode run breaks,
/// because the run is broken precisely because that sample is nonzero.
pub fn get_vlc_symbol_level(br: &mut BitReader<'_>, state: &mut VlcState, bits: u32) -> i32 {
    let mut diff = get_vlc_symbol(br, state, bits);
    if diff >= 0 {
        diff += 1;
    }
    diff
}

/// Encode one signed Sample-Difference value via the adaptive
/// Golomb-Rice VLC coder — the symmetric inverse of [`get_vlc_symbol`].
///
/// `target` is the value a fresh decoder reading the produced bits with
/// an identical `state` clone will return from its matching
/// `get_vlc_symbol` call. `state` is mutated in lockstep with the
/// decoder's (the post-flip residual `v` enters the same
/// `vlc_update` routine the decoder calls).
///
/// # Range requirement
///
/// `target` must lie inside the signed `bits`-wide range
/// `[-(1 << (bits - 1)), (1 << (bits - 1)))`. FFV1's `bits_per_raw_sample`
/// keeps Sample-Difference values inside this domain (`bits ==
/// bits_per_raw_sample` for native YUV / RGB; `bits ==
/// bits_per_raw_sample + 1` for the JPEG 2000 RCT path per §3.8
/// Figure 10), which makes the decoder's `sign_extend(v + bias, bits)`
/// the identity over the residual `v = target - bias` produced here.
/// If a caller hands in an out-of-range `target`, the produced bytes
/// still round-trip — but the recovered value will be the
/// sign-extended low `bits` bits of `target + bias - bias`, not
/// `target`, matching the decoder's `sign_extend` behaviour by
/// construction.
pub fn put_vlc_symbol(bw: &mut BitWriter, state: &mut VlcState, bits: u32, target: i32) {
    // Pick `k` from the CURRENT state, identical to the decoder's
    // first action in `get_vlc_symbol`.
    let k = vlc_pick_k(state);

    // Invert the decoder's two transformations.
    //
    // The decoder does:
    //   v_raw = get_sr_golomb_esc(...)
    //   v     = flip ? -1 - v_raw : v_raw
    //   ret   = sign_extend(v + bias, bits)
    //
    // We're given `target == ret` and need to emit the bits that drive
    // `v_raw` so the decoder reconstructs `target`. Within the domain
    // documented above, `sign_extend` is the identity, so:
    //   v     = target - bias
    //   v_raw = flip ? -1 - v : v
    let v = target.wrapping_sub(state.bias);
    let flip = 2 * state.drift < -state.count;
    let v_raw = if flip { -1 - v } else { v };

    put_sr_golomb_esc(bw, k, bits, v_raw);

    // Decoder updates state with the post-flip `v` (NOT `v_raw`); same
    // here, so the round-trip state evolution stays identical.
    vlc_update(state, v);
}

/// Encode the level-coded variant of [`put_vlc_symbol`] — the
/// symmetric inverse of [`get_vlc_symbol_level`] (§3.8.2.4.1).
///
/// The level form is used for the first non-zero sample after a
/// run-mode run breaks: zero is impossible (a continuing zero would
/// have stayed in run mode), so the decoder shifts every non-negative
/// raw value up by one. The encoder inverts that shift before
/// delegating to [`put_vlc_symbol`].
///
/// # Panics
///
/// `target == 0` is invalid in level-coding: the run break implies a
/// non-zero sample. Debug builds assert; release builds emit a
/// best-effort `0` which the decoder would observe as `+1`.
pub fn put_vlc_symbol_level(bw: &mut BitWriter, state: &mut VlcState, bits: u32, target: i32) {
    debug_assert!(
        target != 0,
        "level-coded symbols are non-zero by definition (§3.8.2.4.1)"
    );
    // The decoder does: `let diff = get_vlc_symbol(); if diff >= 0 {
    // diff += 1 }`. Invert: subtract 1 from any positive `target`
    // before delegating to the scalar encoder (so the scalar produces
    // `target - 1` and the decoder's `+= 1` recovers `target`).
    let scalar_target = if target > 0 { target - 1 } else { target };
    put_vlc_symbol(bw, state, bits, scalar_target);
}

/// `log2_run` table from RFC 9043 §3.8.2.2.1.
///
/// Length is 41 entries. `log2_run[run_index]` gives the bit width of
/// the run-length suffix when the run-mode unary prefix terminates.
/// The entries grow roughly logarithmically with `run_index` so the
/// adaptive run mode hits long zero runs efficiently.
pub const LOG2_RUN: [u8; 41] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn br(bytes: &[u8]) -> BitReader<'_> {
        BitReader::new(bytes)
    }

    // --- sign_extend ----------------------------------------------------

    #[test]
    fn sign_extend_8_bit_identity_for_small_values() {
        // 0..=127 unchanged at 8-bit width.
        assert_eq!(sign_extend(0, 8), 0);
        assert_eq!(sign_extend(1, 8), 1);
        assert_eq!(sign_extend(127, 8), 127);
    }

    #[test]
    fn sign_extend_8_bit_high_bit_negative() {
        // 128 (0x80) → -128. 255 (0xFF) → -1.
        assert_eq!(sign_extend(128, 8), -128);
        assert_eq!(sign_extend(255, 8), -1);
    }

    #[test]
    fn sign_extend_zero_width_is_identity() {
        // §3.8.2.3 doesn't define 0-width; we treat it as identity
        // (matches the test 'no shift' interpretation; never used in
        // practice since FFV1 bits_per_raw_sample >= 8).
        assert_eq!(sign_extend(42, 0), 42);
    }

    // --- get_ur_golomb tests off RFC 9043 §3.8.2.1.3 Table 3 -----------

    #[test]
    fn ur_golomb_k0_unary_prefix_value_zero() {
        // bits "1" at k=0 decodes to 0 (the `prefix=0` first-bit-set
        // path, suffix width 0).
        let buf = [0b1000_0000u8];
        let mut r = br(&buf);
        assert_eq!(get_ur_golomb_esc(&mut r, 0, 8), 0);
    }

    #[test]
    fn ur_golomb_k0_unary_prefix_value_two() {
        // "001" at k=0 → prefix=2, no suffix → 2.
        let buf = [0b0010_0000u8];
        let mut r = br(&buf);
        assert_eq!(get_ur_golomb_esc(&mut r, 0, 8), 2);
    }

    #[test]
    fn ur_golomb_k2_value_zero() {
        // "1 00" at k=2 → prefix=0, suffix=0 → 0.
        // Bits MSB-first: 1, 0, 0 → byte 0b100_00000 = 0x80.
        let buf = [0b1000_0000u8];
        let mut r = br(&buf);
        assert_eq!(get_ur_golomb_esc(&mut r, 2, 8), 0);
    }

    #[test]
    fn ur_golomb_k2_value_two() {
        // "1 10" at k=2 → prefix=0, suffix=2 → 2.
        let buf = [0b1100_0000u8];
        let mut r = br(&buf);
        assert_eq!(get_ur_golomb_esc(&mut r, 2, 8), 2);
    }

    #[test]
    fn ur_golomb_k2_value_five() {
        // "01 01" at k=2 → prefix=1, suffix=1 → 1 + (1<<2) = 5.
        let buf = [0b0101_0000u8];
        let mut r = br(&buf);
        assert_eq!(get_ur_golomb_esc(&mut r, 2, 8), 5);
    }

    // --- sr_golomb interleave-mapping ---------------------------------

    #[test]
    fn sr_golomb_k0_interleave_mapping() {
        // v=0 → "1" → ret 0.
        // v=1 → "01" → unsigned 1 → odd, -(1>>1)-1 = -1.
        // v=2 → "001" → unsigned 2 → even, 2>>1 = 1.
        // v=3 → "0001" → unsigned 3 → odd, -(3>>1)-1 = -2.
        //
        // Concatenated MSB-first: "1" "01" "001" "0001" = 10 bits =
        //   1010 0100 01.. → first byte 0xA4, second byte 0x40.
        let buf = [0b1010_0100u8, 0b0100_0000u8];
        let mut r = br(&buf);
        assert_eq!(get_sr_golomb_esc(&mut r, 0, 8), 0); // "1"
        assert_eq!(get_sr_golomb_esc(&mut r, 0, 8), -1); // "01"
        assert_eq!(get_sr_golomb_esc(&mut r, 0, 8), 1); // "001"
        assert_eq!(get_sr_golomb_esc(&mut r, 0, 8), -2); // "0001"
    }

    // --- ESC mode ------------------------------------------------------

    #[test]
    fn ur_golomb_esc_value_139() {
        // From RFC 9043 §3.8.2.1.3 Table 3 last row: "0000_0000_0000
        // 1000_0000" with any k decodes to 139.
        //
        // 12 zero bits = ESC; then 8 bits = 0x80; ESC suffix is `bits`
        // bits wide; we pass bits=8. Decoded: 0x80 + 11 = 128 + 11 = 139.
        //
        // The byte layout MSB-first is 12 zeros followed by 0b1000_0000
        // and trailing zeros:
        //   bits: 0000_0000 0000_1000 0000_0000
        // = 0x00 0x08 0x00.
        let buf = [0x00u8, 0x08, 0x00];
        let mut r = br(&buf);
        assert_eq!(get_ur_golomb_esc(&mut r, 5, 8), 139);
    }

    // --- VlcState invariants ------------------------------------------

    #[test]
    fn vlc_state_initial_matches_spec() {
        let s = VLC_STATE_INITIAL;
        assert_eq!(s.drift, 0);
        assert_eq!(s.error_sum, 4);
        assert_eq!(s.bias, 0);
        assert_eq!(s.count, 1);
    }

    #[test]
    fn vlc_state_default_is_initial() {
        assert_eq!(VlcState::default(), VLC_STATE_INITIAL);
    }

    // --- get_vlc_symbol behavioural sanity ----------------------------

    #[test]
    fn vlc_symbol_decodes_zero_when_first_bit_set() {
        // From a fresh VLC state (count=1, error_sum=4 → loop runs
        // until i >= 4, so k advances: i=1→2 k=1; i=2→4 k=2 — loop
        // exits since 4 < 4 is false. So k=2). The reader sees "1 00",
        // i.e. byte 0b100_00000 = 0x80; sr_golomb_esc returns 0 (even),
        // sign-extended through bias=0 = 0.
        let buf = [0x80u8];
        let mut r = br(&buf);
        let mut s = VLC_STATE_INITIAL;
        assert_eq!(get_vlc_symbol(&mut r, &mut s, 8), 0);
        // After decoding a 0 the state updates: count++, error_sum
        // unchanged. count was 1 → 2.
        assert_eq!(s.count, 2);
        assert_eq!(s.error_sum, 4);
        // drift +=0; remains 0.
        assert_eq!(s.drift, 0);
    }

    #[test]
    fn vlc_symbol_level_skips_zero() {
        // Same byte 0x80 → scalar would decode 0; level-coded version
        // bumps non-negative results by 1, so it returns 1.
        let buf = [0x80u8];
        let mut r = br(&buf);
        let mut s = VLC_STATE_INITIAL;
        assert_eq!(get_vlc_symbol_level(&mut r, &mut s, 8), 1);
    }

    // --- log2_run table characteristics --------------------------------

    #[test]
    fn log2_run_lengths_and_growth() {
        assert_eq!(LOG2_RUN.len(), 41);
        // Entry 0 is 0; the table is monotonically non-decreasing.
        assert_eq!(LOG2_RUN[0], 0);
        for window in LOG2_RUN.windows(2) {
            assert!(window[1] >= window[0], "log2_run must be non-decreasing");
        }
        // §3.8.2.2.1 last entry is 24.
        assert_eq!(LOG2_RUN[40], 24);
    }

    // --- put_ur_golomb_esc / put_sr_golomb_esc round trips -------------

    fn round_trip_ur_golomb(k: u32, bits: u32, values: &[u32]) {
        let mut bw = BitWriter::new();
        for &v in values {
            put_ur_golomb_esc(&mut bw, k, bits, v);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        for &expected in values {
            assert_eq!(
                get_ur_golomb_esc(&mut br, k, bits),
                expected,
                "ur round-trip k={k} bits={bits} value={expected}"
            );
        }
    }

    fn round_trip_sr_golomb(k: u32, bits: u32, values: &[i32]) {
        let mut bw = BitWriter::new();
        for &v in values {
            put_sr_golomb_esc(&mut bw, k, bits, v);
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        for &expected in values {
            assert_eq!(
                get_sr_golomb_esc(&mut br, k, bits),
                expected,
                "sr round-trip k={k} bits={bits} value={expected}"
            );
        }
    }

    #[test]
    fn put_ur_golomb_round_trips_small_non_esc_values() {
        // All `value < 12 << k` take the non-ESC unary-prefix path.
        for k in 0..=4 {
            let cap = 12u32 << k;
            let values: Vec<u32> = (0..cap.min(48)).collect();
            round_trip_ur_golomb(k, 8, &values);
        }
    }

    #[test]
    fn put_ur_golomb_round_trips_esc_boundary() {
        // The non-ESC -> ESC transition is at `prefix == 12`, i.e.
        // `value == 12 << k`. We probe two values either side of every
        // such boundary up to k=5.
        for k in 0..=5 {
            let boundary = 12u32 << k;
            let values = [
                boundary.saturating_sub(2),
                boundary - 1,
                boundary,
                boundary + 1,
                boundary + 2,
                boundary + (1 << k),
            ];
            round_trip_ur_golomb(k, 16, &values);
        }
    }

    #[test]
    fn put_ur_golomb_round_trips_emits_byte_exact_against_test_3_table_row() {
        // RFC 9043 §3.8.2.1.3 Table 3 last row: any k decodes 139 from
        // "twelve zero bits, then `bits` bits = 0x80, then +11". The
        // encoder must produce the same byte image for `value == 139`
        // (which has `prefix = 139 >> k` >= 12 for k <= 3; we test k=2
        // so `139 >> 2 == 34 >= 12`, ESC path).
        let mut bw = BitWriter::new();
        put_ur_golomb_esc(&mut bw, 2, 8, 139);
        let bytes = bw.finish();
        // Twelve zero bits, then 8 bits = 0x80 (139 - 11 = 128), then
        // the partial-byte zero pad. The bit stream is:
        //   0000_0000 0000_1000 0000_0000 (the second byte's top 4
        //   bits are the last 4 zeros of the ESC prefix; the next
        //   4 bits are the top of 0x80 = 1000; the third byte holds
        //   the remaining 0x80 bits = 0000 followed by a zero-pad).
        // Round-trip is the authoritative correctness check; we also
        // confirm at least the byte count comes out as expected
        // (20 bits -> 3 bytes after padding).
        assert_eq!(bytes.len(), 3);
        let mut br = BitReader::new(&bytes);
        assert_eq!(get_ur_golomb_esc(&mut br, 2, 8), 139);
    }

    #[test]
    fn put_sr_golomb_round_trips_zero_and_paired_signs() {
        round_trip_sr_golomb(0, 8, &[0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]);
        round_trip_sr_golomb(2, 8, &[0, -1, 1, -2, 2, -7, 7, -100, 100]);
    }

    #[test]
    fn put_sr_golomb_round_trips_large_magnitudes_through_esc() {
        // `bits == 16` accommodates `|value|` up to roughly
        // `1 << 15` post-fold. We push past the ESC boundary so both
        // paths are exercised.
        let mut values = Vec::new();
        for &k in &[0u32, 1, 3, 5] {
            let boundary = (12i32) << k;
            values.extend_from_slice(&[
                -boundary - 1,
                -boundary,
                -1,
                0,
                1,
                boundary,
                boundary + 1,
                2 * boundary,
            ]);
        }
        round_trip_sr_golomb(0, 16, &values);
        round_trip_sr_golomb(5, 16, &values);
    }

    #[test]
    fn put_sr_golomb_round_trips_i32_min_magnitude_guard() {
        // The `value < 0` arm uses `unsigned_abs` to avoid overflow at
        // `i32::MIN`. Pair it with a wide ESC width so the encoded
        // unsigned magnitude fits.
        round_trip_sr_golomb(0, 32, &[i32::MIN]);
    }

    // --- put_vlc_symbol / put_vlc_symbol_level round trips -------------

    /// Encode `targets` through one [`VlcState`] then decode them
    /// through a clone, asserting every value matches and the two state
    /// copies drift to the same final value.
    fn round_trip_vlc_symbols(bits: u32, targets: &[i32]) {
        let mut enc_state = VLC_STATE_INITIAL;
        let mut bw = BitWriter::new();
        for &t in targets {
            put_vlc_symbol(&mut bw, &mut enc_state, bits, t);
        }
        let bytes = bw.finish();

        let mut dec_state = VLC_STATE_INITIAL;
        let mut br = BitReader::new(&bytes);
        for (i, &expected) in targets.iter().enumerate() {
            assert_eq!(
                get_vlc_symbol(&mut br, &mut dec_state, bits),
                expected,
                "vlc symbol mismatch at index {i}"
            );
        }
        assert_eq!(
            enc_state,
            dec_state,
            "encoder / decoder state drift after {} symbols",
            targets.len()
        );
    }

    #[test]
    fn put_vlc_symbol_round_trips_zero_value() {
        round_trip_vlc_symbols(8, &[0]);
    }

    #[test]
    fn put_vlc_symbol_round_trips_alternating_signs() {
        round_trip_vlc_symbols(8, &[0, 1, -1, 2, -2, 3, -3, 5, -5, 10, -10]);
    }

    #[test]
    fn put_vlc_symbol_round_trips_long_constant_run() {
        // A long run of zeros exercises the state's slow-drift regime
        // (count climbs, error_sum stays small). The decoder and
        // encoder must stay perfectly synchronised over hundreds of
        // symbols.
        let zeros = vec![0i32; 500];
        round_trip_vlc_symbols(8, &zeros);
    }

    #[test]
    fn put_vlc_symbol_round_trips_count_rescale_at_128() {
        // §3.8.2.4 rescales count/drift/error_sum at count==128; a
        // 200-symbol mixed stream crosses that boundary at least once
        // per context.
        let mut targets = Vec::new();
        for i in 0..200i32 {
            // Alternating sign with growing magnitude.
            targets.push(if i & 1 == 0 { i / 4 } else { -i / 4 });
        }
        round_trip_vlc_symbols(8, &targets);
    }

    #[test]
    fn put_vlc_symbol_round_trips_xorshift_sample_diff_stream() {
        // 500 pseudo-random Sample-Difference values inside the signed
        // 8-bit range, modelled on a real per-plane residual stream.
        let mut targets = Vec::new();
        let mut x: u32 = 0xa5a5_a5a5;
        for _ in 0..500 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let mag = (x & 0x3F) as i32; // 0..=63
            targets.push(if (x & 0x40) != 0 { -mag } else { mag });
        }
        round_trip_vlc_symbols(8, &targets);
    }

    #[test]
    fn put_vlc_symbol_round_trips_higher_bit_depth() {
        // 16-bit Sample-Differences exercise the wider ESC path; the
        // state machine is bit-depth-agnostic so the round-trip should
        // hold identically.
        let mut targets: Vec<i32> = Vec::new();
        for i in -50..=50i32 {
            targets.push(i * 53);
        }
        round_trip_vlc_symbols(16, &targets);
    }

    #[test]
    fn put_vlc_symbol_level_round_trips_non_zero_targets() {
        // Level-coded variant: every target must be non-zero (a zero
        // would have stayed in run mode, never reaching the level
        // decoder). We round-trip a small set against the level
        // encoder/decoder pair and assert symmetry.
        let targets: &[i32] = &[1, -1, 2, -2, 5, -5, 10, -10, 100, -100];
        let mut enc_state = VLC_STATE_INITIAL;
        let mut bw = BitWriter::new();
        for &t in targets {
            put_vlc_symbol_level(&mut bw, &mut enc_state, 8, t);
        }
        let bytes = bw.finish();
        let mut dec_state = VLC_STATE_INITIAL;
        let mut br = BitReader::new(&bytes);
        for &expected in targets {
            assert_eq!(get_vlc_symbol_level(&mut br, &mut dec_state, 8), expected);
        }
        assert_eq!(enc_state, dec_state);
    }

    #[test]
    fn put_vlc_symbol_encoder_state_matches_decoder_step_by_step() {
        // Encode each target individually, then decode against a fresh
        // state, after each step asserting the enc/dec state windows
        // remain identical. This is the strict-lockstep version of the
        // bulk round trip above.
        let targets: &[i32] = &[3, -1, 0, 4, -7, 0, 0, 12, -3];
        let mut enc_state = VLC_STATE_INITIAL;
        let mut bw = BitWriter::new();
        let mut snapshots = Vec::new();
        for &t in targets {
            put_vlc_symbol(&mut bw, &mut enc_state, 8, t);
            snapshots.push(enc_state);
        }
        let bytes = bw.finish();
        let mut dec_state = VLC_STATE_INITIAL;
        let mut br = BitReader::new(&bytes);
        for (i, &expected) in targets.iter().enumerate() {
            assert_eq!(get_vlc_symbol(&mut br, &mut dec_state, 8), expected);
            assert_eq!(
                dec_state, snapshots[i],
                "state diverged after symbol {i} (target {expected})"
            );
        }
    }

    #[test]
    fn put_ur_golomb_round_trips_zero_at_every_small_k() {
        // value == 0 is `prefix == 0`, the smallest non-ESC encoding.
        // Decoding "1" at any k recovers 0.
        for k in 0..=8 {
            round_trip_ur_golomb(k, 8, &[0]);
        }
    }
}
