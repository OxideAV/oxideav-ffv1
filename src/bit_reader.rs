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
/// The reader keeps up to 64 unread bits left-aligned in an accumulator
/// (the next bit to read is bit 63) and refills it four bytes at a time.
/// `get_bits(n)` consumes `n` bits MSB-first; reading past the end of
/// the slice returns zero bits (matching the RFC's "padded with zeroes"
/// language for end-of-frame padding in §3.8.2).
#[derive(Debug)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Next byte of `buf` to load into the accumulator.
    pos: usize,
    /// Unread bits, left-aligned; every bit below the top `nbits` is 0.
    acc: u64,
    /// Number of valid bits in `acc`.
    nbits: u32,
    /// Zero-padding bits loaded past the end of `buf` so far.
    padding: u64,
}

impl<'a> BitReader<'a> {
    /// Wrap a byte slice in an MSB-first bit reader.
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            acc: 0,
            nbits: 0,
            padding: 0,
        }
    }

    /// Bits of the slice not yet consumed (zero padding excluded).
    pub fn bits_left(&self) -> u64 {
        // Padding is loaded behind every real bit, so it is the last
        // thing left in the accumulator.
        (self.nbits as u64).saturating_sub(self.padding) + ((self.buf.len() - self.pos) as u64) * 8
    }

    /// Top the accumulator up. Called with `nbits < 32`; leaves at
    /// least 32 valid bits (zero bits once the slice is exhausted).
    #[inline(always)]
    fn refill(&mut self) {
        if let Some(word) = self.buf.get(self.pos..self.pos + 4) {
            let w = u32::from_be_bytes(word.try_into().expect("sliced to 4 bytes"));
            self.acc |= u64::from(w) << (32 - self.nbits);
            self.nbits += 32;
            self.pos += 4;
        } else {
            self.refill_tail();
        }
    }

    /// Byte-wise refill for the last (fewer than four) bytes, then zero
    /// padding.
    #[cold]
    fn refill_tail(&mut self) {
        while self.nbits <= 56 {
            let byte = match self.buf.get(self.pos) {
                Some(&b) => {
                    self.pos += 1;
                    u64::from(b)
                }
                None => {
                    self.padding += 8;
                    0
                }
            };
            self.acc |= byte << (56 - self.nbits);
            self.nbits += 8;
        }
    }

    /// Read `n` bits MSB-first (`n <= 32`; `n == 0` reads nothing and
    /// returns 0).
    #[inline(always)]
    pub fn get_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32, "get_bits supports up to 32 bits");
        if self.nbits < n {
            self.refill();
        }
        // Two shifts so `n == 0` is a well-defined shift by 64 in total.
        let value = ((self.acc >> 1) >> (63 - n)) as u32;
        self.acc <<= n;
        self.nbits -= n;
        value
    }

    /// Look at the next `n` bits (`1 ..= 32`) without consuming them.
    #[inline(always)]
    pub(crate) fn peek_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0 && n <= 32, "peek_bits supports 1..=32 bits");
        if self.nbits < n {
            self.refill();
        }
        (self.acc >> (64 - n)) as u32
    }

    /// Consume `n` bits already made available by [`Self::peek_bits`].
    #[inline(always)]
    pub(crate) fn skip_bits(&mut self, n: u32) {
        debug_assert!(n <= self.nbits, "skip_bits beyond the buffered window");
        self.acc <<= n;
        self.nbits -= n;
    }

    /// Read one bit.
    #[inline(always)]
    pub fn get_bit(&mut self) -> u32 {
        self.get_bits(1)
    }
}

/// MSB-first bit writer — the inverse of [`BitReader`].
///
/// Pending bits sit in the low end of a 64-bit accumulator and are
/// flushed to the byte vector four bytes at a time.
#[derive(Debug, Default)]
pub struct BitWriter {
    acc: u64,
    /// Pending bit count (`< 32` between calls).
    nbits: u32,
    out: Vec<u8>,
}

impl BitWriter {
    /// A writer with an empty output.
    pub fn new() -> Self {
        BitWriter::default()
    }

    /// Reserve room for `additional` more output bytes.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.out.reserve(additional);
    }

    /// Append one bit (the low bit of `bit`).
    #[inline(always)]
    pub fn put_bit(&mut self, bit: u32) {
        self.put_bits(bit & 1, 1);
    }

    /// Append the low `n` bits of `value`, MSB-first (`n <= 32`).
    #[inline(always)]
    pub fn put_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32, "put_bits supports up to 32 bits");
        let field = u64::from(value) & ((1u64 << n) - 1);
        self.acc = (self.acc << n) | field;
        self.nbits += n;
        if self.nbits >= 32 {
            self.nbits -= 32;
            let word = (self.acc >> self.nbits) as u32;
            self.out.extend_from_slice(&word.to_be_bytes());
        }
    }

    /// Bits written since the last byte boundary (`0 ..= 7`).
    pub fn bits_buffered(&self) -> u32 {
        self.nbits & 7
    }

    /// Flush, zero-padding the final partial byte, and return the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.out.push((self.acc >> self.nbits) as u8);
        }
        if self.nbits > 0 {
            self.out.push((self.acc << (8 - self.nbits)) as u8);
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
