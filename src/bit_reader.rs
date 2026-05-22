//! MSB-first bit reader for the Golomb-Rice coding path (RFC 9043 §3.8.2).
//!
//! The Golomb-Rice mode reads VLC codes from the bitstream as a
//! continuous bit-stream after the slice's range-coded portion ends.
//! RFC 9043 §2.2.9.4 defines `get_bits(i)` as "read the next i bits in
//! the bitstream, from most significant bit to least significant bit,
//! and to return the corresponding value." This module implements that
//! contract over a byte slice.
//!
//! No reader state is shared with the range decoder in
//! [`crate::range_coder`] — they consume non-overlapping byte regions
//! of a Slice (the slice header is range-coded, the slice content in
//! Golomb-Rice mode is bit-coded, the slice footer is byte-aligned per
//! RFC 9043 §4.9 note). The per-slice byte split is the caller's
//! responsibility.

/// MSB-first bit reader over a borrowed byte slice.
///
/// The reader holds an internal cursor pointing at the next byte plus
/// a small accumulator carrying the leftover bits of the current byte.
/// `get_bits(n)` consumes `n` bits MSB-first; reading past the end of
/// the slice returns zero bits (matching the RFC's "padded with zeroes"
/// language for end-of-frame padding in §3.8.2).
#[derive(Debug)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Next byte index to refill from.
    pos: usize,
    /// Bit accumulator. Bits are added MSB-first, so the most recently
    /// loaded byte's MSB occupies the highest still-unread bit position.
    acc: u64,
    /// Number of unread bits currently held in `acc`.
    nbits: u32,
}

impl<'a> BitReader<'a> {
    /// Construct an MSB-first bit reader over `buf`. The reader starts
    /// with no bits buffered; the first `get_bits` call refills from
    /// `buf[0]`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            acc: 0,
            nbits: 0,
        }
    }

    /// Number of bits remaining in the bit stream (rounded up to the
    /// next byte boundary). RFC 9043 §3.8.2 pads the bitstream with
    /// zeroes "until the bitstream contains a multiple of eight bits",
    /// so a caller asking for `bits_left` past the slice end will see
    /// the padding count as well as the live data.
    pub fn bits_left(&self) -> u64 {
        // `nbits` is bounded by 64, `pos` by buf.len(); both fit in
        // u64 without overflow on a 64-bit host.
        self.nbits as u64 + ((self.buf.len() - self.pos) as u64) * 8
    }

    /// Read the next `n` bits MSB-first (1 <= n <= 32) and return them
    /// as a `u32`. Reads past the buffer end inject zero bits.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `n == 0` or `n > 32`. The §3.8.2
    /// callers never request 0 bits or more than 32; the bound exists
    /// to keep the accumulator math simple (32-bit-at-a-time refills).
    pub fn get_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0 && n <= 32, "get_bits supports 1..=32 bits");
        while self.nbits < n {
            // Refill one byte. End-of-buffer reads inject zero bytes
            // per §3.8.2's "padded with zeroes" rule.
            let byte: u64 = if self.pos < self.buf.len() {
                let b = self.buf[self.pos] as u64;
                self.pos += 1;
                b
            } else {
                0
            };
            // Shift accumulator up by 8 and OR in the new byte at the
            // bottom; high-order bits remain the still-unread leading
            // bits of earlier bytes.
            self.acc = (self.acc << 8) | byte;
            self.nbits += 8;
        }
        // Extract the top `n` bits of the accumulator.
        let shift = self.nbits - n;
        // `1u64 << n` is safe because `n <= 32` here.
        let mask: u64 = (1u64 << n) - 1;
        let value = (self.acc >> shift) & mask;
        self.nbits -= n;
        // Clear the bits we just consumed; future reads see only the
        // lower `shift` bits as the new accumulator content.
        self.acc &= (1u64 << shift) - 1;
        value as u32
    }

    /// Read a single bit MSB-first. Convenience wrapper around
    /// `get_bits(1)`; the FFV1 Golomb-Rice prefix decoder calls
    /// `get_bits(1)` in a hot loop.
    pub fn get_bit(&mut self) -> u32 {
        self.get_bits(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_msb_first_byte_aligned() {
        // 0b1010_0101 = 0xA5. Reading 8 bits returns 0xA5.
        let buf = [0xA5u8];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.get_bits(8), 0xA5);
    }

    #[test]
    fn reads_msb_first_one_at_a_time() {
        // 0xA5 = 1010_0101 MSB-first.
        let buf = [0xA5u8];
        let mut br = BitReader::new(&buf);
        let bits: Vec<u32> = (0..8).map(|_| br.get_bit()).collect();
        assert_eq!(bits, vec![1, 0, 1, 0, 0, 1, 0, 1]);
    }

    #[test]
    fn reads_across_byte_boundaries() {
        // 0xA5 0x3C = 1010_0101 0011_1100. Read 4 + 8 + 4 bits.
        let buf = [0xA5u8, 0x3C];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.get_bits(4), 0b1010);
        assert_eq!(br.get_bits(8), 0b0101_0011);
        assert_eq!(br.get_bits(4), 0b1100);
    }

    #[test]
    fn past_end_reads_zero_bits() {
        // Empty buffer: every read returns zero.
        let buf = [];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.get_bits(8), 0);
        assert_eq!(br.get_bits(8), 0);
    }

    #[test]
    fn bits_left_drains_with_consumption() {
        let buf = [0xFFu8, 0xFF];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.bits_left(), 16);
        br.get_bits(3);
        assert_eq!(br.bits_left(), 13);
        br.get_bits(13);
        assert_eq!(br.bits_left(), 0);
    }

    #[test]
    fn reads_wide_value_up_to_32_bits() {
        // 0xDEADBEEF read as one 32-bit value MSB-first.
        let buf = [0xDEu8, 0xADu8, 0xBEu8, 0xEFu8];
        let mut br = BitReader::new(&buf);
        assert_eq!(br.get_bits(32), 0xDEADBEEF);
    }
}
