//! Golomb-Rice mode primitives (RFC 9043 §3.8.2).
//!
//! This module implements the `sd`-typed symbol decode for FFV1's
//! Golomb-Rice coding mode — the alternative entropy coder selected by
//! `coder_type == 0` (RFC 9043 §4.2.3 Table 7). The range-coded path
//! lives in [`crate::range_coder`] + [`crate::symbol`]; this module
//! carries the bit-coded counterpart.
//!
//! The Golomb-Rice path uses these layers, top-down:
//!
//! * **Bit reading** — MSB-first `get_bits(n)` (RFC 9043 §2.2.9.4) over
//!   the slice's bit region, supplied by [`crate::bit_reader::BitReader`].
//! * **Variable-length codes** — `get_ur_golomb(k)` and
//!   `get_sr_golomb(k)` per Figures 26 / 27 plus the ESC-prefix table
//!   in §3.8.2.1.1.
//! * **Sign extension** — `sign_extend` per §3.8.2.3.
//! * **Per-context VLC state** — [`VlcState`] plus
//!   [`get_vlc_symbol`] per the §3.8.2.4 scalar-mode pseudocode.
//! * **Run mode** — `log2_run` and the `run_index` / `run_mode` /
//!   `run_count` state machine that activates when the absolute
//!   context value is 0 (§3.8.2.2 + §3.8.2.4.1).
//!
//! All five layers are needed to drive the per-Line sample-difference
//! decode loop in [`crate::sample_diff`]; the layers themselves are
//! quant-table-independent, so they can be unit-tested with synthetic
//! contexts before the quant-table cascade lands.

use crate::bit_reader::BitReader;

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
    // Adaptive Rice parameter k: smallest k such that `i * 2 >=
    // error_sum`, starting from `i = count`.
    //
    // The pseudocode in §3.8.2.4 is:
    //
    //   i = state->count;
    //   k = 0;
    //   while (i < state->error_sum) {
    //       k++;
    //       i += i;
    //   }
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

    let v_raw = get_sr_golomb_esc(br, k, bits);

    // Conditional sign flip (§3.8.2.4):
    //   if (2 * state->drift < -state->count) {
    //       v = -1 - v;
    //   }
    let mut v = v_raw;
    if 2 * state.drift < -state.count {
        v = -1 - v;
    }

    // Final value = sign_extend(v + state->bias, bits).
    let ret = sign_extend(v.wrapping_add(state.bias), bits);

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

    ret
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
}
