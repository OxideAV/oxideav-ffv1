//! Range-coded scalar symbol coding for FFV1 (`ur` / `sr` / `br`).
//!
//! `ur` (unsigned scalar), `sr` (signed scalar), and `br` (boolean)
//! symbols are coded directly via a [`RangeDecoder`] / [`RangeEncoder`]
//! using the state-machine described in RFC 9043 §3.8.1.2 Figure 21.
//!
//! Each symbol shares a 32-slot context window inside a larger state
//! buffer; the layout (offsets 0, 1..=10, 11..=21, 22..=31) is the
//! contract drawn from Figure 21 and is preserved verbatim here.
//!
//! Decode side: [`get_ur`] / [`get_sr`] / [`get_br`].
//! Encode side: [`put_ur`] / [`put_sr`] / [`put_br`] — symmetric
//! inverses that walk the same context-slot layout in the same order,
//! so every round trip
//! (`put_ur(enc, ctx, v); ...; get_ur(dec, ctx2) == v`) holds
//! bit-exactly when the encoder uses [`RangeEncoder`] and the decoder
//! uses a fresh [`RangeDecoder`] over the encoder's output.

use crate::range_coder::{DecodeWindow, RangeDecoder, RangeEncoder, decode_bit};

/// Number of state slots consumed by one scalar-symbol context window.
///
/// RFC 9043 §3.8.1.2 (Figure 21) reads from offsets 0, 1..=10, 11..=21,
/// and 22..=31 — a total of 32 slots per symbol. The Parameters
/// section keeps several such windows back-to-back in its state buffer.
pub const SYMBOL_CONTEXT_SIZE: usize = 32;

/// Decode one `ur` (unsigned scalar range) symbol from `rc`, using
/// `ctx[0..32]` as the 32-slot state window described by RFC 9043
/// Figure 21.
pub fn get_ur(rc: &mut RangeDecoder<'_>, ctx: &mut [u8]) -> u32 {
    debug_assert!(ctx.len() >= SYMBOL_CONTEXT_SIZE);
    let win: &mut [u8; SYMBOL_CONTEXT_SIZE] = (&mut ctx[..SYMBOL_CONTEXT_SIZE])
        .try_into()
        .expect("sliced to SYMBOL_CONTEXT_SIZE");
    decode_symbol(rc, win, false) as u32
}

/// Decode one `sr` (signed scalar range) symbol from `rc`. Returns
/// `i32` because §3.8.1.2 permits a sign bit; the value range is
/// `i32::MIN..=i32::MAX` but in practice FFV1 keeps scalars well below
/// 2^16 in magnitude.
pub fn get_sr(rc: &mut RangeDecoder<'_>, ctx: &mut [u8]) -> i32 {
    debug_assert!(ctx.len() >= SYMBOL_CONTEXT_SIZE);
    let win: &mut [u8; SYMBOL_CONTEXT_SIZE] = (&mut ctx[..SYMBOL_CONTEXT_SIZE])
        .try_into()
        .expect("sliced to SYMBOL_CONTEXT_SIZE");
    decode_symbol(rc, win, true)
}

/// Fixed-window variant of [`put_sr`] — the encode-side mirror of
/// [`decode_symbol_window`].
#[inline]
pub(crate) fn put_sr_window(re: &mut RangeEncoder, ctx: &mut [u8; SYMBOL_CONTEXT_SIZE], v: i32) {
    encode_symbol(re, ctx, true, v);
}

/// Decode one `br` (single-bit range-coded boolean) using the first
/// slot of `ctx` only. `br` is functionally a one-context `get_rac`
/// and is described in RFC 9043 §3.8.1.1 (the binary layer).
pub fn get_br(rc: &mut RangeDecoder<'_>, ctx: &mut [u8]) -> bool {
    debug_assert!(!ctx.is_empty());
    rc.get_rac(&mut ctx[0]) == 1
}

/// §3.8.1.2 symbol decode (Figure 21 / `get_symbol`) through the
/// decoder object. See [`decode_symbol_window`].
fn decode_symbol(
    rc: &mut RangeDecoder<'_>,
    ctx: &mut [u8; SYMBOL_CONTEXT_SIZE],
    is_signed: bool,
) -> i32 {
    let mut w = rc.window();
    let (buf, next) = rc.tables();
    let v = decode_symbol_window(&mut w, buf, next, ctx, is_signed);
    rc.set_window(w);
    v
}

/// §3.8.1.2 symbol decode over a detached [`DecodeWindow`] — the form
/// the per-row reconstruction loops call so the arithmetic window stays
/// in registers for a whole row.
///
/// Context layout inside the 32-byte window (RFC 9043 §3.8.1.2): `[0]`
/// is the zero flag, `[1 ..= 10]` the unary exponent (index clamped at
/// 9), `[22 ..= 31]` the mantissa bits (index clamped at 9), and
/// `[11 ..= 21]` the sign (index clamped at 10).
#[inline(always)]
pub(crate) fn decode_symbol_window(
    w: &mut DecodeWindow,
    buf: &[u8],
    next: &[[u8; 256]; 2],
    ctx: &mut [u8; SYMBOL_CONTEXT_SIZE],
    is_signed: bool,
) -> i32 {
    if decode_bit(w, buf, next, &mut ctx[0]) != 0 {
        return 0;
    }
    let mut e: usize = 0;
    while decode_bit(w, buf, next, &mut ctx[1 + e.min(9)]) != 0 {
        e += 1;
        if e >= 32 {
            break;
        }
    }
    let mut a: u32 = 1;
    for i in (0..e).rev() {
        let bit = decode_bit(w, buf, next, &mut ctx[22 + i.min(9)]);
        a = a.wrapping_mul(2).wrapping_add(bit);
    }
    if !is_signed {
        return a as i32;
    }
    let negative = decode_bit(w, buf, next, &mut ctx[11 + e.min(10)]);
    let sign_mask = 0u32.wrapping_sub(negative);
    ((a ^ sign_mask).wrapping_sub(sign_mask)) as i32
}

/// Encode one `ur` (unsigned scalar range) symbol to `re`, using
/// `ctx[0..32]` as the 32-slot state window described by RFC 9043
/// Figure 21. The symmetric inverse of [`get_ur`].
pub fn put_ur(re: &mut RangeEncoder, ctx: &mut [u8], v: u32) {
    debug_assert!(ctx.len() >= SYMBOL_CONTEXT_SIZE);
    let win: &mut [u8; SYMBOL_CONTEXT_SIZE] = (&mut ctx[..SYMBOL_CONTEXT_SIZE])
        .try_into()
        .expect("sliced to SYMBOL_CONTEXT_SIZE");
    encode_symbol(re, win, false, v as i32);
}

/// Encode one `sr` (signed scalar range) symbol to `re`. The symmetric
/// inverse of [`get_sr`]. Negative values use the sign-bit slots at
/// offsets `11..=21`; the magnitude `|v|` walks the same `is_zero` /
/// exponent / mantissa layout as `ur`.
pub fn put_sr(re: &mut RangeEncoder, ctx: &mut [u8], v: i32) {
    debug_assert!(ctx.len() >= SYMBOL_CONTEXT_SIZE);
    let win: &mut [u8; SYMBOL_CONTEXT_SIZE] = (&mut ctx[..SYMBOL_CONTEXT_SIZE])
        .try_into()
        .expect("sliced to SYMBOL_CONTEXT_SIZE");
    encode_symbol(re, win, true, v);
}

/// Encode one `br` (single-bit range-coded boolean) using the first
/// slot of `ctx` only. The symmetric inverse of [`get_br`]; functionally
/// a one-context [`RangeEncoder::put_rac`].
pub fn put_br(re: &mut RangeEncoder, ctx: &mut [u8], bit: bool) {
    debug_assert!(!ctx.is_empty());
    re.put_rac(&mut ctx[0], if bit { 1 } else { 0 });
}

/// Shared body of `put_ur` / `put_sr` — the symmetric inverse of
/// [`decode_symbol`] walked in the same context-slot order Figure 21
/// reads.
///
/// Emits (in order): an is-zero bit at offset 0 (early-exit when
/// `v == 0`), a unary exponent terminated by a 0-bit using offsets
/// `1..=10` with saturation at index 9, the MSB-first mantissa using
/// offsets `22..=31` with saturation at index 9, and (if `is_signed`)
/// a sign bit using offsets `11..=21` with saturation at index 10.
#[inline(always)]
fn encode_symbol(
    re: &mut RangeEncoder,
    ctx: &mut [u8; SYMBOL_CONTEXT_SIZE],
    is_signed: bool,
    v: i32,
) {
    if v == 0 {
        re.put_rac(&mut ctx[0], 1);
        return;
    }
    re.put_rac(&mut ctx[0], 0);
    // Magnitude as a 32-bit value (`i32::MIN` maps to 2^31).
    let abs: u32 = v.unsigned_abs();
    let e = (31 - abs.leading_zeros()) as usize;
    // Unary exponent: `e` ones then a zero (context index clamped at 9).
    for idx in 0..e {
        re.put_rac(&mut ctx[1 + idx.min(9)], 1);
    }
    re.put_rac(&mut ctx[1 + e.min(9)], 0);
    // Mantissa below the leading one, MSB first.
    for i in (0..e).rev() {
        re.put_rac(&mut ctx[22 + i.min(9)], ((abs >> i) & 1) as u8);
    }
    if is_signed {
        re.put_rac(&mut ctx[11 + e.min(10)], u8::from(v < 0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a sequence of unsigned symbols through one shared context
    /// and round-trip through a fresh decoder using the same context
    /// reset; assert the decoded values match the originals exactly.
    fn round_trip_ur(values: &[u32]) {
        let mut enc = RangeEncoder::new();
        let mut ctx_enc = [128u8; SYMBOL_CONTEXT_SIZE];
        for &v in values {
            put_ur(&mut enc, &mut ctx_enc, v);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("two-byte minimum");
        let mut ctx_dec = [128u8; SYMBOL_CONTEXT_SIZE];
        let decoded: Vec<u32> = (0..values.len())
            .map(|_| get_ur(&mut dec, &mut ctx_dec))
            .collect();
        assert_eq!(decoded, values, "ur round-trip mismatch");
        // Context drift after the round trip must match too — the
        // decoder and encoder mutate `ctx` via the same transition
        // tables on the same bit sequence.
        assert_eq!(ctx_enc, ctx_dec, "context-window drift after round trip");
    }

    /// Encode a sequence of signed symbols through one shared context
    /// and round-trip through a fresh decoder; assert exact match.
    fn round_trip_sr(values: &[i32]) {
        let mut enc = RangeEncoder::new();
        let mut ctx_enc = [128u8; SYMBOL_CONTEXT_SIZE];
        for &v in values {
            put_sr(&mut enc, &mut ctx_enc, v);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("two-byte minimum");
        let mut ctx_dec = [128u8; SYMBOL_CONTEXT_SIZE];
        let decoded: Vec<i32> = (0..values.len())
            .map(|_| get_sr(&mut dec, &mut ctx_dec))
            .collect();
        assert_eq!(decoded, values, "sr round-trip mismatch");
        assert_eq!(ctx_enc, ctx_dec, "context-window drift after round trip");
    }

    #[test]
    fn put_ur_round_trips_zero() {
        round_trip_ur(&[0]);
    }

    #[test]
    fn put_ur_round_trips_small_values() {
        round_trip_ur(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn put_ur_round_trips_power_of_two_boundaries() {
        // 2^k - 1, 2^k, 2^k + 1 across every exponent up to the
        // saturated `e == 9` and a bit beyond, to cover the
        // `e.min(9)` saturation on both halves.
        let mut vs = vec![0u32];
        for k in 0..16 {
            let p = 1u32 << k;
            vs.push(p.saturating_sub(1));
            vs.push(p);
            vs.push(p.saturating_add(1));
        }
        round_trip_ur(&vs);
    }

    #[test]
    fn put_ur_round_trips_saturated_exponent() {
        // 2^16 and 2^20 land in the saturated-exponent regime where
        // both the exponent and mantissa loops are pinned at offset 9.
        round_trip_ur(&[1 << 16, (1 << 20) - 1, 1 << 24, u32::MAX >> 1]);
    }

    #[test]
    fn put_sr_round_trips_zero_and_positive_negative_pairs() {
        round_trip_sr(&[0, 1, -1, 2, -2, 7, -7, 64, -64, 1024, -1024]);
    }

    #[test]
    fn put_sr_round_trips_around_int_boundaries() {
        // The `i32::MIN` case is the magnitude-overflow guard the
        // encoder mirrors from the decoder's `-(a as i64)` cast.
        // `i16::MIN` is the more typical FFV1 worst case.
        round_trip_sr(&[i16::MIN as i32, -(1 << 20), 1 << 20, i32::MAX, i32::MIN]);
    }

    #[test]
    fn put_br_round_trips_alternating() {
        let bits: Vec<bool> = (0..1024).map(|i| (i & 1) == 0).collect();
        let mut enc = RangeEncoder::new();
        let mut ctx_enc = [128u8; SYMBOL_CONTEXT_SIZE];
        for &b in &bits {
            put_br(&mut enc, &mut ctx_enc, b);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("two-byte minimum");
        let mut ctx_dec = [128u8; SYMBOL_CONTEXT_SIZE];
        let decoded: Vec<bool> = (0..bits.len())
            .map(|_| get_br(&mut dec, &mut ctx_dec))
            .collect();
        assert_eq!(decoded, bits);
        assert_eq!(ctx_enc, ctx_dec);
    }

    #[test]
    fn put_ur_round_trips_mixed_sequence_with_state_persistence() {
        // The Configuration Record reads many `ur` symbols against a
        // single 32-slot Parameters window; replicate that
        // history-dependent walk here to make sure the encoder's
        // state mutation matches the decoder's bit-for-bit even after
        // hundreds of symbols.
        let mut vs = Vec::new();
        let mut x: u32 = 0xa5a5a5a5;
        for _ in 0..500 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // Keep most values small (the FFV1 Configuration Record's
            // typical regime) with a sprinkle of mid-range values.
            vs.push(x & 0xFF);
        }
        round_trip_ur(&vs);
    }

    #[test]
    fn put_sr_round_trips_mixed_signed_sequence() {
        // Mirror the unsigned mixed test for signed symbols.
        let mut vs = Vec::new();
        let mut x: u32 = 0x5a5a5a5a;
        for _ in 0..500 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let mag = (x & 0x3F) as i32;
            vs.push(if (x & 0x40) != 0 { -mag } else { mag });
        }
        round_trip_sr(&vs);
    }

    #[test]
    fn put_ur_independent_per_symbol_contexts_round_trip() {
        // The Slice Header decodes each `ur` against the *same* shared
        // 32-slot window; the §4.2 quant-table state-buffer cascade is
        // similar. But the §4.2.13 cascade also exposes a path where
        // each symbol uses its own fresh window — exercise that pattern
        // by handing every symbol an unshared 32-slot context.
        let values: Vec<u32> = (0..40).map(|i| i * 37 + 1).collect();
        let mut enc = RangeEncoder::new();
        let mut ctx_enc = vec![128u8; values.len() * SYMBOL_CONTEXT_SIZE];
        for (i, &v) in values.iter().enumerate() {
            let slot = &mut ctx_enc[i * SYMBOL_CONTEXT_SIZE..(i + 1) * SYMBOL_CONTEXT_SIZE];
            put_ur(&mut enc, slot, v);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("two-byte minimum");
        let mut ctx_dec = vec![128u8; values.len() * SYMBOL_CONTEXT_SIZE];
        let decoded: Vec<u32> = (0..values.len())
            .map(|i| {
                let slot = &mut ctx_dec[i * SYMBOL_CONTEXT_SIZE..(i + 1) * SYMBOL_CONTEXT_SIZE];
                get_ur(&mut dec, slot)
            })
            .collect();
        assert_eq!(decoded, values);
        assert_eq!(ctx_enc, ctx_dec);
    }
}
