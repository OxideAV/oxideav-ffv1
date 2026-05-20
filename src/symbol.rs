//! Range-coded scalar symbol decoding for FFV1 (`ur` / `sr` / `br`).
//!
//! `ur` (unsigned scalar), `sr` (signed scalar), and `br` (boolean)
//! symbols are decoded directly from a [`RangeDecoder`] using the
//! state-machine described in RFC 9043 §3.8.1.2 Figure 21.
//!
//! Each symbol shares a 32-slot context window inside a larger state
//! buffer; the layout (offsets 0, 1..=10, 11..=21, 22..=31) is the
//! contract drawn from Figure 21 and is preserved verbatim here.

use crate::range_coder::RangeDecoder;

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
    decode_symbol(rc, ctx, false) as u32
}

/// Decode one `sr` (signed scalar range) symbol from `rc`. Returns
/// `i32` because §3.8.1.2 permits a sign bit; the value range is
/// `i32::MIN..=i32::MAX` but in practice FFV1 keeps scalars well below
/// 2^16 in magnitude.
pub fn get_sr(rc: &mut RangeDecoder<'_>, ctx: &mut [u8]) -> i32 {
    debug_assert!(ctx.len() >= SYMBOL_CONTEXT_SIZE);
    decode_symbol(rc, ctx, true)
}

/// Decode one `br` (single-bit range-coded boolean) using the first
/// slot of `ctx` only. `br` is functionally a one-context `get_rac`
/// and is described in RFC 9043 §3.8.1.1 (the binary layer).
pub fn get_br(rc: &mut RangeDecoder<'_>, ctx: &mut [u8]) -> bool {
    debug_assert!(!ctx.is_empty());
    rc.get_rac(&mut ctx[0]) == 1
}

/// Shared body of `ur` / `sr` from RFC 9043 §3.8.1.2 Figure 21.
///
/// Reads (in order): a single is-zero bit at offset 0, an exponent
/// using offsets 1..=10 with saturation at 9, a mantissa MSB-first
/// using offsets 22..=31 with saturation at 9, and (if `is_signed`)
/// a sign bit using offsets 11..=21 with saturation at 10.
fn decode_symbol(rc: &mut RangeDecoder<'_>, ctx: &mut [u8], is_signed: bool) -> i32 {
    // Offset 0: "is the value zero?"
    if rc.get_rac(&mut ctx[0]) == 1 {
        return 0;
    }

    // Exponent: count leading 1-bits at offsets 1..=10 (saturated).
    // The loop in Figure 21 runs `while get_rac(state + 1 + min(e,9))`,
    // i.e. the bit at `1 + min(e,9)` is 1 every step until a 0 stops it.
    let mut e: u32 = 0;
    while rc.get_rac(&mut ctx[1 + e.min(9) as usize]) == 1 {
        e += 1;
        // RFC 9043 doesn't strictly bound `e`, but FFV1 scalars in the
        // configuration record never exceed `e = 31` (single-symbol
        // u32). Cap defensively at 31 so we never overflow the
        // mantissa shift below.
        if e >= 32 {
            break;
        }
    }

    // Mantissa: MSB-first. Initial `a = 1` so the implicit
    // most-significant bit is set; subsequent bits use offsets
    // 22..=31 (saturated). Figure 21 indexes `state + 22 + min(i,9)`.
    let mut a: u32 = 1;
    if e > 0 {
        // `i` walks from `e - 1` down to `0` per Figure 21.
        for i in (0..e).rev() {
            let bit = rc.get_rac(&mut ctx[22 + i.min(9) as usize]) as u32;
            a = a.wrapping_mul(2).wrapping_add(bit);
        }
    }

    if !is_signed {
        return a as i32;
    }

    // Sign bit: offsets 11..=21 (saturated at 10 because the exponent
    // tops out at 9 above + the per-spec `min(e, 10)`).
    let sign_bit = rc.get_rac(&mut ctx[11 + e.min(10) as usize]);
    if sign_bit == 1 {
        // Negative — RFC says `return -a`. Cast through i64 to avoid
        // an `i32::MIN` overflow if a downstream symbol ever pushes
        // `a` to `0x80000000`.
        -(a as i64) as i32
    } else {
        a as i32
    }
}
