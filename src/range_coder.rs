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

use crate::config::NUM_TRANSITION_DELTAS;
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

/// Construct a `one_state` table for `coder_type == 2` from
/// [`DEFAULT_ONE_STATE`] and the Configuration Record's
/// `state_transition_delta[1..=255]` (RFC 9043 §3.8.1.4 Figure 22).
///
/// Per Figure 22: `one_state[i] = default_state_transition[i] +
/// state_transition_delta[i]`, taken modulo 256 (the table is `u8`).
/// The `coder_type == 2` (§3.8.1.6) variant uses this derived table in
/// place of the default; the `zero_state` half is then re-derived from
/// it per [`derive_zero_state`].
///
/// Entry `[0]` of the delta array is unused (Figure 28's loop starts at
/// `i = 1`); the corresponding `one_state[0]` mirrors the default (zero).
/// The wrap-around on the `u8` sum is what the RFC implies — there is no
/// clamp — but in practice deltas are tiny relative to 256 and the
/// derived table stays monotonically close to the default.
pub fn build_one_state(deltas: &[i32; NUM_TRANSITION_DELTAS]) -> [u8; 256] {
    let mut out = DEFAULT_ONE_STATE;
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        let base = *slot as i32;
        let combined = base.wrapping_add(deltas[i]);
        *slot = (combined & 0xFF) as u8;
    }
    out
}

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

    /// Terminate a range-coded region in **Sentinel mode** and return the
    /// byte offset at which the Golomb-Rice bit stream that follows begins
    /// (RFC 9043 §3.8.1.1.1).
    ///
    /// FFV1 switches from a range-coded region (for versions 0/1, the §4.4
    /// inline Parameters) to a Golomb-coded one using Sentinel mode: "the
    /// end of the range-coded bytestream is a binary symbol with state 129,
    /// which value SHALL be discarded. After reading this symbol, the range
    /// decoder will have read one byte beyond the end of the range-coded
    /// bytestream. This way the byte position of the end can be
    /// determined." The boundary is therefore `pos - 1` after the sentinel
    /// read. Clamped to `buf.len()`.
    pub fn terminate_sentinel(&mut self) -> usize {
        // Read and discard the §3.8.1.1.1 sentinel symbol (state 129). Its
        // renormalisation deterministically advances `pos` to the byte
        // boundary at which the Golomb-Rice bit stream begins — exactly the
        // length the encoder's `terminate_sentinel` emitted. The boundary
        // is the resulting `pos` (clamped to the buffer length).
        let mut sentinel_state: u8 = 129;
        let _ = self.get_rac(&mut sentinel_state);
        self.pos.saturating_sub(1).min(self.buf.len())
    }

    /// Replace the active state-transition table in place, preserving the
    /// current byte-window state (`low` / `range` / `pos` / `end`).
    ///
    /// This exists for the **versions 0/1 `coder_type == 2`** single-stream
    /// case (RFC 9043 §4.4): there the §4.2 Parameters (including the
    /// §4.2.4 `state_transition_delta`) and the §4.7 Slice Content share
    /// one continuous range-coder pass. The Parameters — like the v3
    /// Configuration Record (§4.3), which is a *separate* range-coder pass
    /// always seeded with the §3.8.1.5 default table — MUST be read with
    /// the default transition table (a custom table cannot apply to the
    /// very deltas that define it). Once the deltas are known, the §4.7
    /// Slice Content uses the §3.8.1.6 custom table built by
    /// [`build_one_state`]. The transition table only governs how a
    /// context's probability byte *evolves* (`get_rac`'s `*state =
    /// one_state[*state]` / `zero_state[*state]` update); it does not
    /// touch the `low` / `range` arithmetic-window mechanics, so swapping
    /// it mid-pass — exactly at the Parameters → Slice-Content boundary —
    /// is well-defined and leaves the byte cursor untouched.
    ///
    /// The `zero_state` half is re-derived from `one_state` per
    /// §3.8.1.4 (Figures 22–23), matching [`Self::with_one_state`].
    pub fn set_one_state(&mut self, one_state: &[u8; 256]) {
        self.one_state = *one_state;
        self.zero_state = derive_zero_state(one_state);
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

    /// Terminate a range-coded region in **Sentinel mode** (RFC 9043
    /// §3.8.1.1.1), returning the bytes up to the boundary where an
    /// appended Golomb-Rice bit stream begins.
    ///
    /// The symmetric inverse of [`RangeDecoder::terminate_sentinel`]: a
    /// state-129 symbol is written (the decoder reads and discards it),
    /// the coder is flushed, and the single trailing flush byte past the
    /// boundary is dropped so the appended Golomb-Rice stream begins
    /// exactly at the `pos - 1` offset the decoder recovers. The sentinel
    /// symbol absorbs the decoder's one-byte over-read, so no real symbol
    /// before it depends on the dropped byte.
    pub fn terminate_sentinel(mut self) -> Vec<u8> {
        let mut sentinel_state: u8 = 129;
        self.put_rac(&mut sentinel_state, 0);
        let mut bytes = self.finish();
        bytes.pop();
        bytes
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

    /// Replace the active state-transition table in place, preserving the
    /// current byte-window state (`low` / `range` / `cache` /
    /// `pending_ff` / `out`). The symmetric inverse of
    /// [`RangeDecoder::set_one_state`] — used by the v0/v1 `coder_type ==
    /// 2` encoder, where the §4.2 Parameters are emitted with the
    /// §3.8.1.5 default table and the §4.7 Slice Content with the
    /// §3.8.1.6 custom table on the same continuous pass.
    ///
    /// The `zero_state` half is re-derived from `one_state` per
    /// §3.8.1.4 (Figures 22–23), matching [`Self::with_one_state`].
    pub fn set_one_state(&mut self, one_state: &[u8; 256]) {
        self.one_state = *one_state;
        self.zero_state = derive_zero_state(one_state);
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
    fn build_one_state_with_zero_delta_matches_default() {
        // RFC 9043 §3.8.1.4 Figure 22: `one_state = default + delta`.
        // An all-zero delta vector (the common case — every shipped v3
        // fixture is `coder_type == 1` and has no delta on the wire)
        // must yield exactly the default table.
        let deltas = [0i32; NUM_TRANSITION_DELTAS];
        let derived = build_one_state(&deltas);
        assert_eq!(derived, DEFAULT_ONE_STATE);
    }

    #[test]
    fn build_one_state_with_positive_delta_adds_per_index() {
        // A `+1` delta everywhere shifts every entry up by 1 (the entry
        // at index 0 stays at 0 by convention — Figure 28's loop starts
        // at i=1; we mirror that by skipping index 0). u8 wrap is
        // exposed for the indices the default would overflow.
        let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
        for d in deltas.iter_mut().skip(1) {
            *d = 1;
        }
        let derived = build_one_state(&deltas);
        assert_eq!(derived[0], 0);
        for i in 1..256 {
            let expected = DEFAULT_ONE_STATE[i].wrapping_add(1);
            assert_eq!(
                derived[i], expected,
                "at index {i}: default {}, expected {}, got {}",
                DEFAULT_ONE_STATE[i], expected, derived[i],
            );
        }
    }

    #[test]
    fn build_one_state_with_negative_delta_subtracts_modularly() {
        // A `-2` delta everywhere subtracts modulo 256. The first low
        // default entries are 0, so the modular subtraction wraps them
        // up to the high end.
        let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
        for d in deltas.iter_mut().skip(1) {
            *d = -2;
        }
        let derived = build_one_state(&deltas);
        for i in 1..256 {
            let expected = DEFAULT_ONE_STATE[i].wrapping_sub(2);
            assert_eq!(derived[i], expected, "at index {i}");
        }
    }

    #[test]
    fn build_one_state_round_trips_through_decoder_encoder() {
        // A non-trivial delta must produce a coder pair whose round
        // trip is still bit-exact: the encoder's `with_one_state` and
        // the decoder's `with_one_state` agree on the derived table.
        let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
        // Sparse non-zero pattern; small magnitudes so the wrap is rare
        // and the per-bit decisions still resolve normally.
        for (i, slot) in deltas.iter_mut().enumerate().skip(1).step_by(7) {
            *slot = ((i % 5) as i32) - 2;
        }
        let one_state = build_one_state(&deltas);

        let mut enc = RangeEncoder::with_one_state(&one_state);
        // Encode a deterministic per-symbol-independent-context bit
        // stream. Each symbol carries its own state slot, so the result
        // exercises every state transition the table covers.
        let bits: Vec<u8> = (0..32).map(|i| ((i * 5 + 3) & 1) as u8).collect();
        let mut enc_states = vec![PARAMETERS_INITIAL_STATE; bits.len()];
        for (b, s) in bits.iter().zip(enc_states.iter_mut()) {
            enc.put_rac(s, *b);
        }
        let bytes = enc.finish();

        let mut dec = RangeDecoder::with_one_state(&bytes, &one_state).unwrap();
        let mut dec_states = vec![PARAMETERS_INITIAL_STATE; bits.len()];
        for (i, (want, s)) in bits.iter().zip(dec_states.iter_mut()).enumerate() {
            let got = dec.get_rac(s);
            assert_eq!(got, *want, "bit mismatch at {i}");
        }
        // Post-trip state vectors must agree symbol-for-symbol: the
        // shared `one_state` table drove the same transitions on both
        // sides.
        assert_eq!(enc_states, dec_states);
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

    #[test]
    fn set_one_state_mid_pass_swap_round_trips() {
        // The versions-0/1 `coder_type == 2` model (RFC 9043 §4.4): a
        // prefix of symbols is coded with the §3.8.1.5 default table (the
        // §4.2 Parameters, incl. the §4.2.4 deltas), then the live coder
        // swaps onto the §3.8.1.6 custom table for the remaining symbols
        // (the §4.7 Slice Content) WITHOUT restarting the byte window.
        // The decoder mirrors the swap at the same symbol boundary and
        // must recover every bit.
        let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
        for (i, slot) in deltas.iter_mut().enumerate().skip(1) {
            *slot = ((i as i32 * 5 + 1) % 7) - 3;
        }
        let custom = build_one_state(&deltas);

        // 12 "default-table" prefix bits, then 20 "custom-table" bits.
        let prefix: Vec<u8> = (0..12).map(|i| ((i * 3 + 1) & 1) as u8).collect();
        let tail: Vec<u8> = (0..20).map(|i| ((i * 5 + 2) & 1) as u8).collect();

        let mut enc = RangeEncoder::new();
        let mut enc_pre = vec![PARAMETERS_INITIAL_STATE; prefix.len()];
        for (b, s) in prefix.iter().zip(enc_pre.iter_mut()) {
            enc.put_rac(s, *b);
        }
        enc.set_one_state(&custom);
        let mut enc_tail = vec![PARAMETERS_INITIAL_STATE; tail.len()];
        for (b, s) in tail.iter().zip(enc_tail.iter_mut()) {
            enc.put_rac(s, *b);
        }
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(&bytes).unwrap();
        let mut dec_pre = vec![PARAMETERS_INITIAL_STATE; prefix.len()];
        for (i, (want, s)) in prefix.iter().zip(dec_pre.iter_mut()).enumerate() {
            assert_eq!(dec.get_rac(s), *want, "prefix bit {i}");
        }
        dec.set_one_state(&custom);
        let mut dec_tail = vec![PARAMETERS_INITIAL_STATE; tail.len()];
        for (i, (want, s)) in tail.iter().zip(dec_tail.iter_mut()).enumerate() {
            assert_eq!(dec.get_rac(s), *want, "tail bit {i}");
        }
        // Both sides drove identical transitions, so the post-trip state
        // vectors match across the swap boundary.
        assert_eq!(enc_pre, dec_pre);
        assert_eq!(enc_tail, dec_tail);
    }
}
