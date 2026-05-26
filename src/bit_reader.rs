//! MSB-first bit reader / writer for the Golomb-Rice coding path
//! (RFC 9043 §3.8.2).
//!
//! The Golomb-Rice mode reads VLC codes from the bitstream as a
//! continuous bit-stream after the slice's range-coded portion ends.
//! RFC 9043 §2.2.9.4 defines `get_bits(i)` as "read the next i bits in
//! the bitstream, from most significant bit to least significant bit,
//! and to return the corresponding value." This module implements that
//! contract over a byte slice — both halves:
//!
//! * [`BitReader`] consumes MSB-first bits from a borrowed byte slice;
//!   reads past the buffer end inject zero bits (RFC 9043 §3.8.2
//!   "padded with zeroes" rule).
//! * [`BitWriter`] is the symmetric inverse: it accumulates MSB-first
//!   bits into an owned `Vec<u8>` and, on [`BitWriter::finish`], pads
//!   the final partial byte with zero bits the way the Golomb-Rice
//!   content section is required to be padded so the §4.9 Slice Footer
//!   begins on a byte boundary.
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

/// MSB-first bit writer — the symmetric inverse of [`BitReader`].
///
/// Bits are accumulated MSB-first into an internal `u64` accumulator;
/// once eight bits have been buffered, the high byte is committed to
/// the output vector. [`BitWriter::finish`] flushes any remaining
/// fractional byte by left-shifting it into the byte slot's MSB region
/// and zero-padding the bottom bits — the same padding rule
/// [`BitReader`] reads back as zero bits, so round trips through the
/// `BitWriter -> BitReader` pair are bit-exact regardless of how many
/// bits were written.
///
/// The §3.8.2 Golomb-Rice content section of a Slice ends on a byte
/// boundary so the §4.9 Slice Footer can begin byte-aligned; the
/// caller is expected to call [`BitWriter::finish`] before appending
/// the footer, which is exactly what this writer's flush rule
/// produces.
#[derive(Debug, Default)]
pub struct BitWriter {
    /// Accumulator. Bits are appended at the LSB end; the writer
    /// shifts left by one before each new bit so the accumulator's
    /// low `nbits` positions hold the still-unflushed bits in
    /// most-significant-first order.
    acc: u64,
    /// Number of unflushed bits currently in `acc` (0..8).
    nbits: u32,
    /// Output byte vector. One byte is appended every time `nbits`
    /// would exceed 8 (in [`put_bits`]) or 8 (in [`put_bit`]).
    out: Vec<u8>,
}

impl BitWriter {
    /// Construct an empty MSB-first bit writer with no buffered bits
    /// and an empty output vector.
    pub fn new() -> Self {
        BitWriter::default()
    }

    /// Append one bit MSB-first. `bit` is taken mod 2.
    pub fn put_bit(&mut self, bit: u32) {
        self.acc = (self.acc << 1) | (bit as u64 & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.nbits = 0;
        }
    }

    /// Append `n` bits of `value` MSB-first (1 <= n <= 32). The bits
    /// emitted are the bottom `n` of `value`, most-significant first.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `n == 0` or `n > 32`. The §3.8.2
    /// callers never request 0 bits or more than 32; the bound exists
    /// to keep the round-trip with [`BitReader::get_bits`] symmetric
    /// over the same domain.
    pub fn put_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n > 0 && n <= 32, "put_bits supports 1..=32 bits");
        // Walk from the most-significant requested bit down. `value`
        // is taken mod `2^n`; higher bits are ignored.
        for i in (0..n).rev() {
            self.put_bit((value >> i) & 1);
        }
    }

    /// Number of bits currently buffered but not yet flushed to the
    /// output vector (0..8).
    pub fn bits_buffered(&self) -> u32 {
        self.nbits
    }

    /// Drain the writer, flushing the final partial byte (if any) by
    /// left-shifting its bits into the MSB region of a fresh byte and
    /// zero-padding the bottom. Returns the byte sequence a fresh
    /// [`BitReader`] will replay exactly the bits this writer received.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            // Left-shift so the `nbits` bits become the high bits of the
            // new byte. The bottom `8 - nbits` bits stay zero — the
            // RFC 9043 §3.8.2 "padded with zeroes" rule.
            self.acc <<= 8 - self.nbits;
            self.out.push(self.acc as u8);
        }
        self.out
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

    // ----- BitWriter ---------------------------------------------------

    #[test]
    fn writer_emits_msb_first_byte_aligned() {
        // Writing 0xA5 as 8 bits produces the same byte 0xA5.
        let mut bw = BitWriter::new();
        bw.put_bits(0xA5, 8);
        assert_eq!(bw.finish(), vec![0xA5]);
    }

    #[test]
    fn writer_emits_msb_first_one_bit_at_a_time() {
        // 1010_0101 MSB-first -> byte 0xA5.
        let mut bw = BitWriter::new();
        for b in [1, 0, 1, 0, 0, 1, 0, 1] {
            bw.put_bit(b);
        }
        assert_eq!(bw.finish(), vec![0xA5]);
    }

    #[test]
    fn writer_round_trips_across_byte_boundaries() {
        // Mirror of `reads_across_byte_boundaries`: write 4 + 8 + 4
        // bits, expect the same two bytes the reader test consumes.
        let mut bw = BitWriter::new();
        bw.put_bits(0b1010, 4);
        bw.put_bits(0b0101_0011, 8);
        bw.put_bits(0b1100, 4);
        let out = bw.finish();
        assert_eq!(out, vec![0xA5, 0x3C]);
        // Re-read what we just wrote and confirm bit-exact recovery.
        let mut br = BitReader::new(&out);
        assert_eq!(br.get_bits(4), 0b1010);
        assert_eq!(br.get_bits(8), 0b0101_0011);
        assert_eq!(br.get_bits(4), 0b1100);
    }

    #[test]
    fn writer_pads_partial_final_byte_with_zero_bits() {
        // Write 3 bits "111"; finish() pads the bottom 5 bits with
        // zeroes — high three bits set, low five clear: 0b1110_0000.
        let mut bw = BitWriter::new();
        bw.put_bits(0b111, 3);
        assert_eq!(bw.finish(), vec![0b1110_0000]);
    }

    #[test]
    fn writer_emits_wide_value_up_to_32_bits() {
        // Mirror of `reads_wide_value_up_to_32_bits`.
        let mut bw = BitWriter::new();
        bw.put_bits(0xDEADBEEF, 32);
        assert_eq!(bw.finish(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn writer_round_trip_pseudo_random_bit_run() {
        // Write a long deterministic bit sequence and confirm a fresh
        // reader recovers it bit-exactly. The accumulator's high-water
        // mark + boundary crossings are exercised by 100+ bits split
        // across varying widths.
        let mut bw = BitWriter::new();
        let mut x: u32 = 0xa5a5_a5a5;
        let mut original_bits: Vec<u32> = Vec::new();
        for _ in 0..30 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let width = (x & 0xF) + 1; // 1..=16
            let value = x & ((1u32 << width) - 1);
            bw.put_bits(value, width);
            for i in (0..width).rev() {
                original_bits.push((value >> i) & 1);
            }
        }
        let bytes = bw.finish();
        let mut br = BitReader::new(&bytes);
        for (i, expected) in original_bits.iter().enumerate() {
            let got = br.get_bit();
            assert_eq!(got, *expected, "bit {i} mismatch");
        }
    }

    #[test]
    fn writer_bits_buffered_reports_partial_state() {
        let mut bw = BitWriter::new();
        assert_eq!(bw.bits_buffered(), 0);
        bw.put_bits(0b101, 3);
        assert_eq!(bw.bits_buffered(), 3);
        bw.put_bits(0b00, 2);
        assert_eq!(bw.bits_buffered(), 5);
        bw.put_bits(0b110, 3);
        // 8 bits in -> flushed; nothing buffered.
        assert_eq!(bw.bits_buffered(), 0);
    }
}
