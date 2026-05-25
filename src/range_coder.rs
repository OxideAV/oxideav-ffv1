//! FFV1 range coder — binary value layer.
//!
//! Implements the binary range coder defined by RFC 9043 §3.8.1.1
//! (Figures 11–20). The configuration record uses *Closed* mode
//! (RFC 9043 §3.8.1.1.1): the byte length is supplied by the
//! container, and any read past the slice end MUST appear to the
//! coder as a zero byte.
//!
//! Both halves of the coder live here:
//!
//! * [`RangeDecoder`] — the decode side (init / refill / `get_rac`)
//!   driven by Figures 18 / 19 / 20.
//! * [`RangeEncoder`] — the encode side
//!   ([`RangeEncoder::put_rac`] / [`RangeEncoder::finish`]), the
//!   symmetric inverse of [`RangeDecoder`]. Used by the (still-WIP)
//!   FFV1 encoder path and by round-trip tests that need to manufacture
//!   wire bitstreams the decoder side already knows how to consume.
//!
//! Scalar `ur` / `sr` / `br` symbol decoding (and the matching
//! `put_ur` / `put_sr` / `put_br` symbol encoding) are built on top
//! of these in [`crate::symbol`].

use crate::Error;

/// Number of contexts visible to one [`RangeDecoder`] / context array.
///
/// The Parameters section uses 128 contexts (RFC 9043 §4.2: "Parameters
/// has its own initial states, all set to 128"). Other call sites use
/// the same numeric width, so the table itself is a plain `u8` slice
/// and callers size their state buffer to match.
pub const PARAMETERS_INITIAL_STATE: u8 = 128;

/// The default state-transition table from RFC 9043 §3.8.1.5 (Figure 24).
///
/// Indexed by the current state value (0..=255) and the binary
/// symbol just decoded: `next_state = ONE_STATE[state]` if the bit
/// was 1, otherwise `next_state = ZERO_STATE[state]`. The table
/// below is the published `one_state` half; `zero_state[i]` is
/// derived as `256 - one_state[256 - i]` per §3.8.1.4 (Figures 22–23).
pub const DEFAULT_ONE_STATE: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 190, 191, 192, 194,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 202, 204, 205, 206, 207, 208, 209, 209, 210, 211,
    212, 213, 215, 215, 216, 217, 218, 219, 220, 220, 222, 223, 224, 225, 226, 227, 227, 229, 229,
    230, 231, 232, 234, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,
    248, 0, 0, 0, 0, 0, 0, 0,
];

/// Derive `zero_state[i]` from the published `one_state` table.
///
/// Per RFC 9043 §3.8.1.4 the two halves of the transition function
/// are mirror images: `zero_state[i] = 256 - one_state[256 - i]`. We
/// build it once at decoder construction so the call sites just index
/// twin tables.
fn derive_zero_state(one: &[u8; 256]) -> [u8; 256] {
    let mut zs = [0u8; 256];
    // `i = 0` is unused (states are 1..=255 in practice); leave it 0
    // to match the reference layout.
    for (i, slot) in zs.iter_mut().enumerate().skip(1) {
        // 256 - i fits the table index when i >= 1 (so 256 - i is in
        // 1..=255); `one[256 - i]` is also in 0..=255 so the wrapping
        // subtraction is a plain `256 - x` modulo 256.
        let mirror = one[256 - i] as u16;
        *slot = (256u16.wrapping_sub(mirror) & 0xFF) as u8;
    }
    zs
}

/// Binary-mode FFV1 range decoder operating in *Closed* mode.
///
/// "Closed" means the byte-length of the range-coded region is known
/// from the container header (`NumBytes`), and reads past the end
/// return zero bytes (RFC 9043 §3.8.1.1.1). The Configuration Record
/// always uses this mode (`§3.8.1.1.1` first bullet).
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    buf: &'a [u8],
    pos: usize,
    low: u32,
    range: u32,
    /// Closed-mode "ran past end" sticky flag.
    ///
    /// Tracks whether the buffer pointer has consumed every byte of
    /// the range-coded region. Subsequent refills SHALL inject zero
    /// bytes per RFC 9043 §3.8.1.1.1 ("Bytes beyond the length are
    /// read as 0 by the range decoder").
    end: bool,
    one_state: [u8; 256],
    zero_state: [u8; 256],
}

impl<'a> RangeDecoder<'a> {
    /// Construct a Closed-mode decoder over `buf` using the default
    /// state-transition table (RFC 9043 §3.8.1.5).
    pub fn new(buf: &'a [u8]) -> Result<Self, Error> {
        Self::with_one_state(buf, &DEFAULT_ONE_STATE)
    }

    /// Construct a Closed-mode decoder over `buf` using a caller-supplied
    /// `one_state` table. The `zero_state` half is derived from it per
    /// RFC 9043 §3.8.1.4 (Figures 22–23).
    pub fn with_one_state(buf: &'a [u8], one_state: &[u8; 256]) -> Result<Self, Error> {
        // Per Figure 18: range = 0xFF00, low = get_bits(16). Closed
        // mode requires at least two bytes to seed `low`.
        if buf.len() < 2 {
            return Err(Error::TruncatedRangeCoder);
        }
        let zero_state = derive_zero_state(one_state);
        let range = 0xFF00u32;
        let low = ((buf[0] as u32) << 8) | (buf[1] as u32);
        // Per Figure 18: if (low >= range) { low = range; end = 1; }
        // (clamp + sticky end-of-stream flag). The two seed bytes are
        // already consumed, so `pos = 2`.
        let (low, end) = if low >= range {
            (range, true)
        } else {
            (low, false)
        };
        Ok(RangeDecoder {
            buf,
            pos: 2,
            low,
            range,
            end,
            one_state: *one_state,
            zero_state,
        })
    }

    /// Re-fill the working window after a symbol consumed the bottom
    /// 8 bits of `range` (RFC 9043 Figure 19).
    fn refill(&mut self) {
        if self.range < 256 {
            self.range = self.range.wrapping_mul(256);
            self.low = self.low.wrapping_mul(256);
            // Closed mode (RFC 9043 §3.8.1.1.1 + Figure 19): past-end
            // bytes are read as 0. The byte cursor advances
            // unconditionally so the renorm byte cadence stays identical
            // to the encoder's even once the cursor walks past the
            // (footer-excluded) body — every such read injects a zero.
            let byte = self.buf.get(self.pos).copied().unwrap_or(0) as u32;
            self.pos += 1;
            if self.pos >= self.buf.len() {
                self.end = true;
            }
            self.low = self.low.wrapping_add(byte);
        }
    }

    /// Decode one binary symbol against `state` and update `state` to
    /// the next state per the active transition table (RFC 9043
    /// Figure 20).
    pub fn get_rac(&mut self, state: &mut u8) -> u8 {
        let s = *state as u32;
        let rangeoff = (self.range.wrapping_mul(s)) / 256;
        self.range = self.range.wrapping_sub(rangeoff);
        if self.low < self.range {
            *state = self.zero_state[*state as usize];
            self.refill();
            0
        } else {
            self.low = self.low.wrapping_sub(self.range);
            *state = self.one_state[*state as usize];
            self.range = rangeoff;
            self.refill();
            1
        }
    }

    /// Bytes consumed from the input buffer so far. Useful only for
    /// callers that want to sanity-check the configuration record CRC
    /// region; the range decoder itself does not depend on it.
    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.pos
    }
}

/// Binary-mode FFV1 range encoder (Closed mode), symmetric inverse of
/// [`RangeDecoder`].
///
/// Mirrors the decoder's Figure-18/19/20 state machine: `range` is a
/// 16-bit quantity initialised to `0xFF00`, each symbol partitions
/// `range` at `(range * state) / 256`, and a renormalisation loop
/// emits one byte every time `range` would drop below `0x100`.
///
/// Byte emission uses the classic *delayed-byte* / *pending-0xFF carry*
/// technique: the most recent emitted byte is held in a one-byte cache
/// so a later renormalisation can fold a carry into it, and runs of
/// `0xFF` bytes (which a carry would propagate through) are tracked
/// separately and only emitted once a non-`0xFF` byte fixes whether
/// the carry happened. This is the standard technique for any
/// arithmetic / range coder that emits bytes from the top of a wider
/// internal register; the decoder's `low = (b0 << 8) | b1` seed +
/// per-byte refill consumes whatever the encoder produces here.
///
/// `RangeEncoder` is the encoder-side counterpart to [`RangeDecoder`];
/// no `Open` mode is implemented because every FFV1 range-coded region
/// (Configuration Record, Slice Header, Slice Content) uses Closed
/// mode (RFC 9043 §3.8.1.1.1: "the size is supplied by the container
/// or by [a] previously-decoded length field").
#[derive(Debug)]
pub struct RangeEncoder {
    /// 17-bit working register: bits 0..=15 are the live `low`, bit 16
    /// is the carry produced by `low + range` overflowing the 16-bit
    /// window. The renormalisation shift folds bit 16 back through the
    /// cached byte + the pending-0xFF run.
    low: u32,
    range: u32,
    out: Vec<u8>,
    /// Cached previous emitted byte. `-1` means "no byte cached yet"
    /// (i.e. we have not produced any output past the initial pair of
    /// renorms); a non-negative value is the byte awaiting a possible
    /// carry-in from the next renorm.
    cache: i32,
    pending_ff: u32,
    one_state: [u8; 256],
    zero_state: [u8; 256],
}

impl RangeEncoder {
    /// Construct a Closed-mode encoder using the default
    /// state-transition table (RFC 9043 §3.8.1.5).
    pub fn new() -> Self {
        Self::with_one_state(&DEFAULT_ONE_STATE)
    }

    /// Construct a Closed-mode encoder using a caller-supplied
    /// `one_state` table. The `zero_state` half is derived from it per
    /// RFC 9043 §3.8.1.4 (Figures 22–23).
    pub fn with_one_state(one_state: &[u8; 256]) -> Self {
        let zero_state = derive_zero_state(one_state);
        RangeEncoder {
            low: 0,
            range: 0xFF00,
            out: Vec::new(),
            cache: -1,
            pending_ff: 0,
            one_state: *one_state,
            zero_state,
        }
    }

    /// Emit the byte leaving the top of `low` (bits 8..=15), folding
    /// any carry (bit 16) into the previously-cached byte and any
    /// pending 0xFF run. This is the inverse of the decoder's
    /// `refill()`: the decoder pulls one byte off the input on every
    /// `range < 256`; the encoder pushes one byte onto the output on
    /// the same condition.
    fn shift(&mut self) {
        let carry = (self.low >> 16) & 0xFF;
        let byte = (self.low >> 8) & 0xFF;
        if byte != 0xFF {
            if self.cache >= 0 {
                self.out.push(((self.cache as u32 + carry) & 0xFF) as u8);
            }
            while self.pending_ff > 0 {
                // A carry through a 0xFF byte yields 0x00 (the 0xFF
                // plus 1 overflows the 8-bit slot). Either way the
                // result is `(0xFF + carry) & 0xFF`.
                self.out.push(((0xFF + carry) & 0xFF) as u8);
                self.pending_ff -= 1;
            }
            self.cache = byte as i32;
        } else {
            // Defer 0xFF emission: a later carry might flip every
            // pending 0xFF into 0x00, so we cannot commit them yet.
            self.pending_ff += 1;
        }
        // Drop bits 8..=16 — bit 16 has been consumed as `carry` above,
        // bits 8..=15 have been moved into `cache` (or counted toward
        // `pending_ff`).
        self.low = (self.low << 8) & 0xFFFF;
    }

    /// Renormalise: while `range < 0x100`, shift in 8 more bits per
    /// the inverse of Figure 19. The encoder shifts left (`* 256`)
    /// rather than right because it is consuming the top byte of
    /// `low` rather than appending a fresh bottom byte.
    fn renorm(&mut self) {
        while self.range < 0x100 {
            self.range = self.range.wrapping_mul(256);
            self.shift();
        }
    }

    /// Encode one binary symbol against `state` and update `state` to
    /// the next state per the active transition table — the symmetric
    /// inverse of [`RangeDecoder::get_rac`] (Figure 20). The state
    /// table mutation is identical to the decoder's (the transition
    /// is keyed on the encoded bit, not on the coder's internal state).
    pub fn put_rac(&mut self, state: &mut u8, bit: u8) {
        let s = *state as u32;
        let rangeoff = (self.range.wrapping_mul(s)) / 256;
        // Figure 20 inverted: the decoder's `if low < range { ... 0 }
        // else { low -= range; range = rangeoff; ... 1 }` becomes:
        // bit 0 → range -= rangeoff (decoder will take the `low <
        // range` branch); bit 1 → low += range - rangeoff (i.e. the
        // new `low` lands inside the high-side partition the decoder
        // detects with `low >= range`).
        self.range = self.range.wrapping_sub(rangeoff);
        if bit == 0 {
            *state = self.zero_state[*state as usize];
        } else {
            // The decoder reads `low >= range` and then does
            // `low -= range`; we add `range` (the post-split low value)
            // before that subtraction so the decoder sees the original
            // pre-split low. The carry handling in `shift()` propagates
            // the 17-bit add correctly back through the byte cache.
            self.low = self.low.wrapping_add(self.range);
            *state = self.one_state[*state as usize];
            self.range = rangeoff;
        }
        self.renorm();
    }

    /// Flush the coder, emitting the remaining bytes from `low` so the
    /// produced byte stream re-decodes to exactly the encoded symbols.
    ///
    /// Two trailing `shift()` calls drain bits 8..=15 of `low` followed
    /// by bits 0..=7 (shifted into the byte slot by the first call);
    /// any cached byte and pending-0xFF run are flushed after. The
    /// resulting `Vec<u8>` is the byte sequence a fresh
    /// [`RangeDecoder`] will replay.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..2 {
            self.shift();
        }
        if self.cache >= 0 {
            self.out.push(self.cache as u8);
        }
        while self.pending_ff > 0 {
            self.out.push(0xFF);
            self.pending_ff -= 1;
        }
        self.out
    }
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_low_and_range_from_first_two_bytes() {
        // Per RFC 9043 Figure 18 the initial Low is `get_bits(16)`
        // from the first two bytes of the bytestream. Pad with zero
        // bytes so refill() doesn't see EOF immediately.
        let buf = [0x12u8, 0x34, 0, 0, 0, 0];
        let rc = RangeDecoder::new(&buf).expect("two-byte minimum met");
        assert_eq!(rc.low, 0x1234);
        assert_eq!(rc.range, 0xFF00);
        assert!(!rc.end);
    }

    #[test]
    fn rejects_buffer_shorter_than_two_bytes() {
        assert!(matches!(
            RangeDecoder::new(&[]),
            Err(Error::TruncatedRangeCoder)
        ));
        assert!(matches!(
            RangeDecoder::new(&[0xAB]),
            Err(Error::TruncatedRangeCoder)
        ));
    }

    #[test]
    fn closed_mode_past_end_reads_as_zero() {
        // Provide exactly the two seed bytes plus nothing; every
        // refill after that must inject zero bytes and never panic.
        let buf = [0x80u8, 0x00];
        let mut rc = RangeDecoder::new(&buf).expect("two-byte minimum");
        let mut state = PARAMETERS_INITIAL_STATE;
        for _ in 0..256 {
            let _ = rc.get_rac(&mut state);
        }
    }

    #[test]
    fn deterministic_output_for_fixed_input() {
        // Same input bytes → same decoded bit sequence (no internal
        // randomness). Cheap sanity check on the state machine.
        let buf = [0x56u8, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60];
        let mut rc1 = RangeDecoder::new(&buf).unwrap();
        let mut rc2 = RangeDecoder::new(&buf).unwrap();
        let mut st1 = PARAMETERS_INITIAL_STATE;
        let mut st2 = PARAMETERS_INITIAL_STATE;
        let bits1: Vec<u8> = (0..32).map(|_| rc1.get_rac(&mut st1)).collect();
        let bits2: Vec<u8> = (0..32).map(|_| rc2.get_rac(&mut st2)).collect();
        assert_eq!(bits1, bits2);
    }

    #[test]
    fn encoder_round_trips_constant_zeros() {
        // Encode a long run of zero bits; the decoder must produce the
        // same run back. Exercises the `range -= rangeoff` branch and
        // its renormalisation cadence without the high-side carry path.
        let mut enc = RangeEncoder::new();
        let mut st = PARAMETERS_INITIAL_STATE;
        for _ in 0..256 {
            enc.put_rac(&mut st, 0);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("flush yields at least two bytes");
        let mut st2 = PARAMETERS_INITIAL_STATE;
        for _ in 0..256 {
            assert_eq!(dec.get_rac(&mut st2), 0);
        }
    }

    #[test]
    fn encoder_round_trips_constant_ones() {
        // Encode a long run of one bits; this drives the carry path
        // (every `bit == 1` add into `low`) and forces the pending-0xFF
        // delayed-byte machinery as `range` saturates the high side.
        let mut enc = RangeEncoder::new();
        let mut st = PARAMETERS_INITIAL_STATE;
        for _ in 0..256 {
            enc.put_rac(&mut st, 1);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("flush yields at least two bytes");
        let mut st2 = PARAMETERS_INITIAL_STATE;
        for _ in 0..256 {
            assert_eq!(dec.get_rac(&mut st2), 1);
        }
    }

    #[test]
    fn encoder_round_trips_alternating_pattern() {
        // Alternating bits stress the state-transition tables on both
        // halves; this is the cheapest "no fixed pattern" round-trip.
        let pattern: Vec<u8> = (0..1024).map(|i| (i & 1) as u8).collect();
        let mut enc = RangeEncoder::new();
        let mut st = PARAMETERS_INITIAL_STATE;
        for &b in &pattern {
            enc.put_rac(&mut st, b);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("flush yields at least two bytes");
        let mut st2 = PARAMETERS_INITIAL_STATE;
        let decoded: Vec<u8> = (0..pattern.len()).map(|_| dec.get_rac(&mut st2)).collect();
        assert_eq!(decoded, pattern);
    }

    #[test]
    fn encoder_round_trips_pseudo_random_pattern() {
        // Deterministic pseudo-random stream (xorshift32) so this test
        // is reproducible. The decoder must reconstruct the exact
        // bitstream; any single-bit divergence would tear off into a
        // long miscompare from that point on.
        let mut x: u32 = 0xdeadbeef;
        let pattern: Vec<u8> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 1) as u8
            })
            .collect();
        let mut enc = RangeEncoder::new();
        let mut st = PARAMETERS_INITIAL_STATE;
        for &b in &pattern {
            enc.put_rac(&mut st, b);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("flush yields at least two bytes");
        let mut st2 = PARAMETERS_INITIAL_STATE;
        let decoded: Vec<u8> = (0..pattern.len()).map(|_| dec.get_rac(&mut st2)).collect();
        assert_eq!(decoded, pattern);
    }

    #[test]
    fn encoder_round_trips_with_independent_per_bit_states() {
        // Each bit uses its own state slot — exercises the encoder when
        // the transition table can't "warm up" on prior bits, which is
        // the regime the §3.8.1.2 scalar symbols hit (every is-zero /
        // exponent / mantissa bit reads a fresh state).
        let mut enc = RangeEncoder::new();
        let mut states = [PARAMETERS_INITIAL_STATE; 64];
        let bits: Vec<u8> = (0..states.len()).map(|i| (i & 1) as u8).collect();
        for (i, &b) in bits.iter().enumerate() {
            enc.put_rac(&mut states[i], b);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).expect("flush yields at least two bytes");
        let mut states2 = [PARAMETERS_INITIAL_STATE; 64];
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(dec.get_rac(&mut states2[i]), b);
        }
        // State updates are key-mutation-deterministic, so the encoder
        // and decoder must arrive at the same final states.
        assert_eq!(states, states2);
    }

    #[test]
    fn encoder_finish_produces_at_least_two_bytes() {
        // Even with zero encoded symbols, `finish()` must produce
        // enough bytes for `RangeDecoder::new()` to seed `low` (the
        // two-byte minimum from Figure 18).
        let enc = RangeEncoder::new();
        let bytes = enc.finish();
        assert!(
            bytes.len() >= 2,
            "finish() emitted {} bytes — need >= 2 for the decoder seed",
            bytes.len()
        );
        RangeDecoder::new(&bytes).expect("empty stream must still flush a seedable buffer");
    }

    #[test]
    fn zero_state_table_matches_figure_23() {
        // RFC 9043 Figure 23: `zero_state_i = 256 - one_state_{256 - i}`.
        // Verify the helper's output for a spot-check of indices that
        // exercise non-trivial values in the default table.
        let zs = derive_zero_state(&DEFAULT_ONE_STATE);
        // i=1: zero_state[1] = (256 - one_state[255]) & 0xFF
        //                    = (256 - 0) & 0xFF = 0.
        assert_eq!(zs[1], 0);
        // i=128: zero_state[128] = (256 - one_state[128]) & 0xFF.
        //   DEFAULT_ONE_STATE[128] (per Figure 24 row 8) = 134.
        //   256 - 134 = 122.
        assert_eq!(zs[128], 122);
        // i=255: zero_state[255] = (256 - one_state[1]) & 0xFF.
        //   one_state[1] = 0, so zero_state[255] = 256 & 0xFF = 0.
        assert_eq!(zs[255], 0);
    }
}
