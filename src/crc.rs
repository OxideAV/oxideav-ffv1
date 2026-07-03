//! FFV1 Configuration Record CRC validation (RFC 9043 §4.3.2 / §4.9.3).
//!
//! RFC 9043 §4.3.1 / §4.3.2 (Figure, `ConfigurationRecord( NumBytes )`)
//! end the Configuration Record with a 32-bit `u(32)`
//! `configuration_record_crc_parity` chosen so that the Configuration
//! Record *as a whole* — every byte of the extradata blob, including
//! those four parity bytes — has a CRC remainder of zero (§4.3.2:
//! "configuration_record_crc_parity is 32 bits that are chosen so that
//! the Configuration Record as a whole has a CRC remainder of zero").
//!
//! §4.3.2 defers the generator polynomial to §4.9.3 (the Slice CRC):
//! "the standard IEEE CRC polynomial (0x104C11DB7) with initial value 0,
//! without pre-inversion, and without post-inversion." This is the
//! MSB-first ("big-endian" / `Crc-32/MPEG-2`-family) form: bytes are
//! shifted into the high end of the accumulator and the 33-bit
//! polynomial's low 32 bits (`0x04C11DB7`) are XORed when the top bit
//! pops out.
//!
//! Verifying the CRC therefore reduces to: run the register over the
//! entire extradata and assert the residue is `0`.
//!
//! ## Table-driven implementation (round 386)
//!
//! The §4.9.3 CRC runs over **every byte of every Slice** on both the
//! decode gate (`parse_slice_footer`) and the encode parity solver
//! (`encode_slice_footer`), so its cost is per-Frame-linear, not just a
//! header check. The r386 profile showed the original bit-at-a-time
//! loop at ~4% of a range-coded decode and ~8% of a Golomb decode, and
//! a plain single-table form did not beat it (one serial
//! load-to-use-latency-bound table lookup per byte is no faster than
//! eight branchless ALU steps on a wide core). The implementation is
//! therefore the classic **slicing-by-8** form: eight tables — where
//! `TABLE[k][b]` is the register after clocking byte `b` and then `k`
//! zero bytes through an all-zero register — let one step consume eight
//! input bytes with eight *independent* loads, exploiting the CRC's
//! GF(2) linearity (the register update is a XOR-sum of each byte's
//! independently advanced contribution). All tables are derived at
//! compile time from the polynomial by the same shift-and-XOR
//! recurrence the bit loop used, so the forms are equal by
//! construction; a unit test pins table-form == bit-serial-reference on
//! every single byte and on multi-byte streams at every alignment, and
//! every fixture CRC pin covers the wire behaviour.

use crate::Error;

/// The low 32 bits of the IEEE CRC generator polynomial `0x104C11DB7`
/// (RFC 9043 §4.9.3). The implicit bit 32 (`1 << 32`) is handled by the
/// MSB-first shift-and-test in the table builder.
const FFV1_CRC_POLY: u32 = 0x04C1_1DB7;

/// Slicing-by-8 tables. `CRC_TABLES[0][b]` is the plain byte table (the
/// register after clocking byte `b` through an all-zero register);
/// `CRC_TABLES[k][b]` advances that value through `k` further zero
/// bytes. Built at compile time from [`FFV1_CRC_POLY`].
const CRC_TABLES: [[u32; 256]; 8] = build_crc_tables();

const fn build_crc_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    // Table 0: eight bit-serial steps per byte value.
    let mut b = 0usize;
    while b < 256 {
        let mut crc = (b as u32) << 24;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ FFV1_CRC_POLY
            } else {
                crc << 1
            };
            bit += 1;
        }
        tables[0][b] = crc;
        b += 1;
    }
    // Table k: advance table k-1 through one zero byte
    // (`A(R) = (R << 8) ^ T0[R >> 24]`, the standard byte step with a
    // zero input byte).
    let mut k = 1usize;
    while k < 8 {
        let mut b = 0usize;
        while b < 256 {
            let prev = tables[k - 1][b];
            tables[k][b] = (prev << 8) ^ tables[0][(prev >> 24) as usize];
            b += 1;
        }
        k += 1;
    }
    tables
}

/// One classic byte step: `R' = (R << 8) ^ T0[(R >> 24) ^ byte]`.
#[inline]
fn crc_byte_step(crc: u32, byte: u8) -> u32 {
    (crc << 8) ^ CRC_TABLES[0][(((crc >> 24) as u8) ^ byte) as usize]
}

/// Compute the FFV1 / RFC 9043 §4.9.3 CRC-32 over `data`.
///
/// Polynomial `0x104C11DB7`, initial value `0`, no pre-inversion, no
/// post-inversion, MSB-first (each input byte enters at bit 31). When
/// `data` is the *whole* Configuration Record (parity bytes included)
/// the result is `0` for a well-formed record (§4.3.2).
///
/// Exposed crate-wide because the §4.9.3 Slice Footer CRC uses the
/// identical generator over the Slice bytes.
pub(crate) fn ffv1_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        // Fold the register into the first four data bytes, then XOR
        // the eight independently advanced per-byte contributions. The
        // register update is linear over GF(2), so this equals eight
        // sequential byte steps — but the eight loads have no
        // dependency chain between them.
        let v = crc ^ u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        crc = CRC_TABLES[7][(v >> 24) as usize]
            ^ CRC_TABLES[6][((v >> 16) & 0xFF) as usize]
            ^ CRC_TABLES[5][((v >> 8) & 0xFF) as usize]
            ^ CRC_TABLES[4][(v & 0xFF) as usize]
            ^ CRC_TABLES[3][chunk[4] as usize]
            ^ CRC_TABLES[2][chunk[5] as usize]
            ^ CRC_TABLES[1][chunk[6] as usize]
            ^ CRC_TABLES[0][chunk[7] as usize];
    }
    for &byte in chunks.remainder() {
        crc = crc_byte_step(crc, byte);
    }
    crc
}

/// Validate the §4.3.2 `configuration_record_crc_parity` of an FFV1
/// Configuration Record.
///
/// `extradata` is the *entire* Configuration Record blob (the container
/// CodecPrivate / extradata payload), including the trailing four parity
/// bytes. Per RFC 9043 §4.3.2 a conforming record CRCs to `0` over the
/// whole blob.
///
/// Returns [`Error::TruncatedRangeCoder`] when the blob is too short to
/// even contain the 4-byte parity field, and
/// [`Error::ConfigurationRecordCrcMismatch`] (carrying the non-zero
/// residue) when the CRC check fails.
///
/// A passing check proves the Configuration Record was received intact;
/// it does *not* re-derive the stored parity word — the §4.3.2 "remainder
/// of zero" property makes the residue-over-the-whole-blob the canonical
/// check, identical to how a conforming encoder chose the parity bytes.
pub fn validate_configuration_record_crc(extradata: &[u8]) -> Result<(), Error> {
    // The parity field is `u(32)`: the blob must hold at least those
    // 4 bytes (in practice the Parameters prefix makes it much longer,
    // but this is the hard structural minimum for the §4.3.2 field).
    if extradata.len() < 4 {
        return Err(Error::TruncatedRangeCoder);
    }
    let residue = ffv1_crc32(extradata);
    if residue != 0 {
        return Err(Error::ConfigurationRecordCrcMismatch(residue));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real Configuration Record (here the v3-default fixture's
    /// CodecPrivate) CRCs to exactly `0` over the whole blob, parity
    /// bytes included (RFC 9043 §4.3.2). This is the canonical fixity
    /// property; the encoder solved for the four parity bytes so that
    /// this residue is zero. We embed one small known-valid blob so the
    /// `crc.rs` unit suite is self-contained (the cross-fixture sweep
    /// lives in `tests/fixture_config_crc.rs`).
    const VALID_RECORD: &[u8] = &[
        0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86,
        0x2f, 0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89,
        0xac, 0x8f, 0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
    ];

    /// A zero-length input has CRC 0 (the register starts at 0 and never
    /// advances). A single `0x00` byte likewise stays at 0 — zeros never
    /// set the top bit, so the polynomial is never applied.
    #[test]
    fn crc_of_all_zero_input_is_zero() {
        assert_eq!(ffv1_crc32(&[]), 0);
        assert_eq!(ffv1_crc32(&[0x00]), 0);
        assert_eq!(ffv1_crc32(&[0x00, 0x00, 0x00, 0x00]), 0);
    }

    /// A well-formed Configuration Record has a §4.3.2 residue of zero
    /// over the whole blob.
    #[test]
    fn valid_record_residue_is_zero() {
        assert_eq!(
            ffv1_crc32(VALID_RECORD),
            0,
            "a conforming record CRCs to 0 (RFC 9043 §4.3.2)"
        );
        assert!(validate_configuration_record_crc(VALID_RECORD).is_ok());
    }

    /// A single flipped bit anywhere in an otherwise-valid blob makes the
    /// residue non-zero, so validation fails with the residue attached.
    #[test]
    fn corrupted_blob_fails() {
        let mut full = VALID_RECORD.to_vec();
        // Flip one bit of an interior byte.
        full[5] ^= 0x01;
        match validate_configuration_record_crc(&full) {
            Err(Error::ConfigurationRecordCrcMismatch(residue)) => assert_ne!(residue, 0),
            other => panic!("expected CRC mismatch, got {other:?}"),
        }
    }

    /// Flipping a bit in the trailing parity word is detected too.
    #[test]
    fn corrupted_parity_fails() {
        let mut full = VALID_RECORD.to_vec();
        let last = full.len() - 1;
        full[last] ^= 0x80;
        assert!(validate_configuration_record_crc(&full).is_err());
    }

    /// A blob shorter than the 4-byte parity field is structurally
    /// invalid.
    #[test]
    fn too_short_blob_rejected() {
        assert!(matches!(
            validate_configuration_record_crc(&[0x01, 0x02, 0x03]),
            Err(Error::TruncatedRangeCoder)
        ));
        assert!(matches!(
            validate_configuration_record_crc(&[]),
            Err(Error::TruncatedRangeCoder)
        ));
    }

    /// Known-answer vectors for the §4.9.3 generator (poly `0x104C11DB7`,
    /// init 0, MSB-first, no inversion), hand-derived from the bit loop:
    ///
    /// * `0x80`: `0x80 << 24 = 0x8000_0000`. Eight shift-XOR steps over
    ///   `0x04C11DB7` give `0x690C_E0EE`.
    /// * `0xFF`: a full byte of ones through the same loop gives
    ///   `0xB1F7_40B4`.
    ///
    /// These are the canonical CRC-32 results (init 0, refin/refout
    /// false, xorout 0) for a single byte and lock the polynomial
    /// orientation and the absence of inversion.
    #[test]
    fn known_answer_single_bytes() {
        assert_eq!(ffv1_crc32(&[0x80]), 0x690C_E0EE);
        assert_eq!(ffv1_crc32(&[0xFF]), 0xB1F7_40B4);
    }

    /// The pre-r386 bit-at-a-time register, kept as the reference
    /// implementation the table form must equal on every input.
    fn ffv1_crc32_bitwise(data: &[u8]) -> u32 {
        let mut crc: u32 = 0;
        for &byte in data {
            crc ^= (byte as u32) << 24;
            for _ in 0..8 {
                if crc & 0x8000_0000 != 0 {
                    crc = (crc << 1) ^ FFV1_CRC_POLY;
                } else {
                    crc <<= 1;
                }
            }
        }
        crc
    }

    /// The compile-time table form equals the bit-serial reference on
    /// every single-byte input (which spans the whole table) and on a
    /// multi-byte pseudo-random stream (which exercises the register
    /// feedback across byte steps).
    #[test]
    fn table_form_matches_bit_serial_reference() {
        for b in 0..=255u8 {
            assert_eq!(
                ffv1_crc32(&[b]),
                ffv1_crc32_bitwise(&[b]),
                "single byte 0x{b:02x}"
            );
        }
        let mut x: u32 = 0xdead_beef;
        let stream: Vec<u8> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x & 0xFF) as u8
            })
            .collect();
        for len in [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1024, 4095, 4096,
        ] {
            assert_eq!(
                ffv1_crc32(&stream[..len]),
                ffv1_crc32_bitwise(&stream[..len]),
                "stream prefix len {len}"
            );
        }
    }
}
