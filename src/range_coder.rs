//! FFV1 range coder — binary value layer.
//!
//! Implements the binary range decoder defined by RFC 9043 §3.8.1.1
//! (Figures 11–20). The configuration record uses *Closed* mode
//! (RFC 9043 §3.8.1.1.1): the byte length is supplied by the
//! container, and any read past the slice end MUST appear to the
//! coder as a zero byte.
//!
//! Only the binary-mode primitives (init / refill / get_rac) live
//! here. Scalar `ur` / `sr` symbol decoding is built on top of these
//! in [`crate::symbol`].

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
