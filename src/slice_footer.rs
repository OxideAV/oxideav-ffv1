//! FFV1 Slice Footer parser (RFC 9043 §4.9).
//!
//! The Slice Footer sits at the *end* of every FFV1 v3 Slice. Per the
//! §4.9 pseudocode:
//!
//! ```text
//! SliceFooter( ) {
//!     slice_size                                   | u(24)
//!     if (ec) {
//!         error_status                             | u(8)
//!         slice_crc_parity                         | u(32)
//!     }
//! }
//! ```
//!
//! When the Configuration Record's `ec` is 1 the footer is 8 bytes
//! long; when `ec` is 0 it is 3 bytes long. `slice_size` (§4.9.1) is
//! the size of the rest of the Slice in bytes — i.e. the SliceHeader +
//! SliceContent + (Golomb-Rice) padding, **excluding** the footer
//! itself. `error_status` (§4.9.2) is 0 / 1 / 2 with the table-16
//! meaning. `slice_crc_parity` (§4.9.3) is chosen so the entire Slice
//! (footer included) CRCs to zero under the §4.9.3 generator — the same
//! polynomial / orientation as the §4.3.2 Configuration Record CRC, so
//! we reuse the [`crate::crc::ffv1_crc32`] implementation.
//!
//! The Slice Footer is always byte-aligned (§4.9 preamble).
//!
//! # What this module implements
//!
//! `parse_slice_footer(full_slice_bytes, ec)` extracts the three (or
//! one, for `ec=0`) fields and validates them. When `ec == 1` it also
//! verifies the §4.9.3 whole-Slice CRC residue is zero and reports the
//! non-zero residue otherwise. The caller passes the *whole* Slice
//! buffer (header + content + footer) — typically the byte range
//! produced by walking the §4.9.1 trailer-pointer chain backwards from
//! the end of the FFV1 frame.
//!
//! `encode_slice_footer(body, ec, error_status)` is the **symmetric
//! inverse**: given a Slice body (SliceHeader + SliceContent + any
//! Golomb-Rice padding), it writes the §4.9 footer trailer — 3 bytes
//! for `ec == 0` (`slice_size` u(24)), 8 bytes for `ec == 1`
//! (`slice_size` u(24) + `error_status` u(8) + a `slice_crc_parity`
//! u(32) solved so the whole-Slice CRC residue is zero per §4.9.3).
//! This is the first frame-level FFV1 encoder primitive shipping in
//! `src/`; the §4.9 footer round-trip composes directly with the
//! §3.8.1 binary `RangeEncoder` + §3.8.1.2 scalar `put_ur` / `put_sr`
//! / `put_br` primitives from round 137 — building the body the
//! footer wraps.
//!
//! # What this module does NOT do
//!
//! It does not parse OR write the SliceHeader / SliceContent inside
//! the buffer (`crate::slice_header` and `crate::slice_content` own
//! the SliceHeader side; the SliceContent encode side is a future
//! round) and it does not consult or alter the range / arithmetic
//! coder state — the footer is a wholly out-of-band 3- or 8-byte
//! trailer.

use crate::crc::ffv1_crc32;
use crate::Error;

/// Size in bytes of a `ec == 0` Slice Footer (just the `u(24)`
/// `slice_size` field).
pub const SLICE_FOOTER_LEN_EC0: usize = 3;

/// Size in bytes of a `ec == 1` Slice Footer (`u(24)` `slice_size` +
/// `u(8)` `error_status` + `u(32)` `slice_crc_parity`).
pub const SLICE_FOOTER_LEN_EC1: usize = 8;

/// Typed mirror of the RFC 9043 §4.9.2 Table 16 `error_status` values.
///
/// "Reserved for future use" wire values fold to
/// [`SliceErrorStatus::Reserved`]; the raw wire byte is preserved on
/// [`Ffv1SliceFooter::error_status_raw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceErrorStatus {
    /// 0 — "no error" (§4.9.2 Table 16).
    NoError,
    /// 1 — "Slice contains a correctable error" (§4.9.2 Table 16).
    Correctable,
    /// 2 — "Slice contains an uncorrectable error" (§4.9.2 Table 16).
    Uncorrectable,
    /// Any value `>= 3` — "reserved for future use" (§4.9.2 Table 16).
    Reserved,
}

impl SliceErrorStatus {
    /// Maps the on-wire `u(8)` byte to the typed enum per §4.9.2.
    pub fn from_wire(byte: u8) -> Self {
        match byte {
            0 => Self::NoError,
            1 => Self::Correctable,
            2 => Self::Uncorrectable,
            _ => Self::Reserved,
        }
    }

    /// True for the §4.9.2 `Uncorrectable` value, the only one a
    /// conformant decoder must refuse to render. `Correctable` is
    /// recoverable by definition; `Reserved` is unknown and treated
    /// as "trust the bitstream" (the slice CRC, if present, is the
    /// stronger fixity signal).
    pub fn is_uncorrectable(self) -> bool {
        matches!(self, Self::Uncorrectable)
    }
}

/// Parsed contents of an FFV1 v3 Slice Footer (RFC 9043 §4.9).
///
/// For `ec == 0` only `slice_size` is meaningful; `error_status_raw`
/// / `error_status` / `slice_crc_parity` are `None`.
#[derive(Debug, Clone)]
pub struct Ffv1SliceFooter {
    /// `slice_size` (§4.9.1): size of the rest of the Slice in bytes,
    /// i.e. SliceHeader + SliceContent + (Golomb-Rice) padding —
    /// EXCLUDING the footer itself.
    pub slice_size: u32,
    /// Total on-wire Slice length (`slice_size` + footer length). This
    /// is a convenience derived value: it equals the byte length of
    /// the buffer the parser was handed.
    pub total_size: u32,
    /// `error_status` (§4.9.2) as the typed enum, when `ec == 1`.
    pub error_status: Option<SliceErrorStatus>,
    /// Raw on-wire `error_status` byte (preserved for diagnostic
    /// logging of `Reserved` values), when `ec == 1`.
    pub error_status_raw: Option<u8>,
    /// `slice_crc_parity` (§4.9.3) as the on-wire `u(32)` value (the
    /// "stored parity"), when `ec == 1`.
    ///
    /// Round-trip identity: `crc(slice_bytes[..size_field])` over the
    /// §4.9.3 generator equals this value (the encoder chooses it so
    /// the *whole-Slice* CRC residue is zero).
    pub slice_crc_parity: Option<u32>,
}

impl Ffv1SliceFooter {
    /// Footer length in bytes (3 for `ec == 0`, 8 for `ec == 1`).
    pub fn footer_len(&self) -> usize {
        if self.slice_crc_parity.is_some() {
            SLICE_FOOTER_LEN_EC1
        } else {
            SLICE_FOOTER_LEN_EC0
        }
    }
}

/// Parse a §4.9 Slice Footer from `full_slice_bytes` and (when
/// `ec == 1`) validate the §4.9.3 Slice CRC.
///
/// `full_slice_bytes` is the *whole* Slice byte range — SliceHeader +
/// SliceContent + (Golomb-Rice padding if `coder_type == 0`) + the
/// footer itself. The caller obtains this by walking the §4.9.1
/// trailer-pointer chain backwards from the end of the FFV1 frame:
/// each iteration reads the last `slice_size` u(24) field and slices
/// off `slice_size + footer_len` bytes.
///
/// `ec` is `true` when the Configuration Record carries
/// `error_correction == 1` (i.e. the §4.9 `if (ec)` branch is taken
/// and the trailing `error_status` + `slice_crc_parity` are present).
///
/// # Returns
///
/// A populated [`Ffv1SliceFooter`]. For `ec == 1` the function also
/// asserts that the §4.9.3 whole-Slice CRC residue is zero, returning
/// [`Error::SliceCrcMismatch`] (with the non-zero residue) on failure.
///
/// # Errors
///
/// * [`Error::TruncatedSliceFooter`] when the buffer is shorter than
///   the footer (`< 3` for `ec == 0`, `< 8` for `ec == 1`).
/// * [`Error::SliceSizeOutOfRange`] when the on-wire `slice_size`
///   field disagrees with `full_slice_bytes.len() - footer_len`.
/// * [`Error::SliceCrcMismatch`] when `ec == 1` and the §4.9.3 CRC
///   residue is non-zero.
pub fn parse_slice_footer(full_slice_bytes: &[u8], ec: bool) -> Result<Ffv1SliceFooter, Error> {
    let footer_len = if ec {
        SLICE_FOOTER_LEN_EC1
    } else {
        SLICE_FOOTER_LEN_EC0
    };
    if full_slice_bytes.len() < footer_len {
        return Err(Error::TruncatedSliceFooter);
    }
    let total_len = full_slice_bytes.len();
    // The footer sits at the END of the slice, byte-aligned (§4.9
    // preamble). For `ec == 1` the trailing 8 bytes are
    // [size_hi, size_mid, size_lo, error_status, parity[3], parity[2],
    // parity[1], parity[0]] — i.e. parity is the trailing u(32), error
    // status the byte preceding it, and slice_size the u(24) preceding
    // error_status. For `ec == 0` only slice_size's 3 bytes are
    // present and they ARE the trailing 3 bytes.
    let size_off = total_len - footer_len;
    let s0 = full_slice_bytes[size_off];
    let s1 = full_slice_bytes[size_off + 1];
    let s2 = full_slice_bytes[size_off + 2];
    let slice_size = (u32::from(s0) << 16) | (u32::from(s1) << 8) | u32::from(s2);

    // Cross-check the size field against the buffer length: every
    // conforming slice satisfies `slice_size == total_len - footer_len`
    // (§4.9.1: "size of the Slice in bytes" — i.e. of the rest of the
    // slice; the footer itself is excluded by virtue of being what
    // *announces* the size). If the caller mis-walked the trailer
    // chain, this catches it immediately rather than letting the
    // header parser run off the end of the buffer.
    let expected_size = (total_len - footer_len) as u32;
    if slice_size != expected_size {
        return Err(Error::SliceSizeOutOfRange {
            field: slice_size,
            expected: expected_size,
        });
    }

    if !ec {
        return Ok(Ffv1SliceFooter {
            slice_size,
            total_size: total_len as u32,
            error_status: None,
            error_status_raw: None,
            slice_crc_parity: None,
        });
    }

    let error_status_raw = full_slice_bytes[size_off + 3];
    let error_status = SliceErrorStatus::from_wire(error_status_raw);
    let p0 = full_slice_bytes[size_off + 4];
    let p1 = full_slice_bytes[size_off + 5];
    let p2 = full_slice_bytes[size_off + 6];
    let p3 = full_slice_bytes[size_off + 7];
    let slice_crc_parity =
        (u32::from(p0) << 24) | (u32::from(p1) << 16) | (u32::from(p2) << 8) | u32::from(p3);

    // §4.9.3: "the Slice as a whole has a CRC remainder of 0." The
    // generator is the §4.9.3 IEEE poly (0x104C11DB7) with init 0
    // and no inversion — the same one the Configuration Record CRC
    // (§4.3.2) uses, hence the shared `ffv1_crc32`.
    let residue = ffv1_crc32(full_slice_bytes);
    if residue != 0 {
        return Err(Error::SliceCrcMismatch {
            residue,
            stored_parity: slice_crc_parity,
        });
    }

    Ok(Ffv1SliceFooter {
        slice_size,
        total_size: total_len as u32,
        error_status: Some(error_status),
        error_status_raw: Some(error_status_raw),
        slice_crc_parity: Some(slice_crc_parity),
    })
}

/// Encode a §4.9 Slice Footer onto an already-built Slice body
/// (RFC 9043 §4.9 / §4.9.1 / §4.9.2 / §4.9.3) — the symmetric inverse
/// of [`parse_slice_footer`].
///
/// `body` is the SliceHeader + SliceContent + any Golomb-Rice padding —
/// i.e. exactly the byte range whose length §4.9.1 calls `slice_size`.
/// The function returns the **full Slice** byte sequence: `body`'s
/// bytes followed by either:
///
/// * 3 footer bytes for `ec == false` — just `slice_size` as `u(24)`
///   big-endian; `error_status` is ignored (the §4.9 `if (ec)` branch
///   is not taken).
/// * 8 footer bytes for `ec == true` — `slice_size` u(24),
///   `error_status` u(8) (taken from the typed [`SliceErrorStatus`]
///   argument per §4.9.2 Table 16), then a `slice_crc_parity` u(32)
///   chosen so the whole-Slice §4.9.3 CRC residue is zero. The
///   `Reserved` variant maps back to the raw byte `3` (the lowest
///   reserved wire value per Table 16); callers needing a specific
///   reserved value should call [`encode_slice_footer_with_raw_status`]
///   instead.
///
/// # Solver
///
/// §4.9.3 specifies the parity word as "32 bits that are chosen so
/// that the Slice as a whole has a CRC remainder of 0". The §4.9.3
/// CRC generator is `MSB-first`, initial value `0`, no inversion. For
/// such a CRC, a property of the polynomial division is that
/// `CRC(M || CRC(M)) == 0` when the appended bytes are interpreted in
/// the same MSB-first orientation the generator uses. The encoder
/// therefore runs the generator over `body || size_u24 || error_status`
/// and emits the resulting 32-bit CRC as the `slice_crc_parity` field
/// (big-endian, top byte first) — by construction the whole-Slice CRC
/// residue is then `0`, which is exactly the value
/// `validate_configuration_record_crc` (§4.3.2) and the `ec == 1`
/// branch of [`parse_slice_footer`] check for.
///
/// The resulting byte sequence, re-parsed through [`parse_slice_footer`]
/// with the same `ec` argument, reproduces the inputs bit-exactly
/// (`slice_size == body.len() as u32`, `error_status` round-trips
/// through [`SliceErrorStatus::from_wire`], and the §4.9.3 CRC check
/// passes).
///
/// # Errors
///
/// * [`Error::SliceSizeOutOfRange`] when `body.len()` does not fit in
///   the `u(24)` `slice_size` field (i.e. `body.len() >= 1 << 24`).
///   `field` carries the saturated low 24 bits; `expected` carries the
///   `1 << 24` boundary the body must stay below.
pub fn encode_slice_footer(
    body: &[u8],
    ec: bool,
    error_status: SliceErrorStatus,
) -> Result<Vec<u8>, Error> {
    let error_status_byte = match error_status {
        SliceErrorStatus::NoError => 0u8,
        SliceErrorStatus::Correctable => 1,
        SliceErrorStatus::Uncorrectable => 2,
        // Lowest reserved wire value per §4.9.2 Table 16; callers that
        // need a specific reserved byte go through the raw-status API.
        SliceErrorStatus::Reserved => 3,
    };
    encode_slice_footer_with_raw_status(body, ec, error_status_byte)
}

/// Encode a §4.9 Slice Footer with an explicit `error_status` raw byte.
///
/// Same as [`encode_slice_footer`] but accepts the `error_status`
/// field as a raw `u8` — useful for emitting a specific reserved-range
/// value (3..=255 per §4.9.2 Table 16) the typed
/// [`SliceErrorStatus::Reserved`] cannot distinguish on its own.
///
/// For `ec == false` the `error_status_raw` argument is unused (the
/// §4.9 `if (ec)` branch is not taken). For `ec == true` it lands on
/// the wire verbatim; round-tripping through [`SliceErrorStatus::from_wire`]
/// folds 3..=255 to [`SliceErrorStatus::Reserved`] per Table 16.
///
/// See [`encode_slice_footer`] for the §4.9.3 parity-solver derivation
/// and error-condition semantics.
pub fn encode_slice_footer_with_raw_status(
    body: &[u8],
    ec: bool,
    error_status_raw: u8,
) -> Result<Vec<u8>, Error> {
    // §4.9.1: slice_size is u(24); a body whose length escapes that
    // bound cannot be expressed on the wire. The §4.9.1 trailer-chain
    // walk also reads back at most 0xFFFFFF, so honouring the bound
    // keeps the round-trip well-defined.
    const SIZE_BOUND: usize = 1 << 24;
    if body.len() >= SIZE_BOUND {
        return Err(Error::SliceSizeOutOfRange {
            field: (body.len() & (SIZE_BOUND - 1)) as u32,
            expected: SIZE_BOUND as u32,
        });
    }
    let slice_size = body.len() as u32;
    let footer_len = if ec {
        SLICE_FOOTER_LEN_EC1
    } else {
        SLICE_FOOTER_LEN_EC0
    };
    let mut out = Vec::with_capacity(body.len() + footer_len);
    out.extend_from_slice(body);

    // §4.9.1 slice_size u(24), big-endian (MSB-first / "network order"
    // matches the §2.2.9 bitstream convention every other u(N) field
    // in the spec uses).
    out.push(((slice_size >> 16) & 0xFF) as u8);
    out.push(((slice_size >> 8) & 0xFF) as u8);
    out.push((slice_size & 0xFF) as u8);

    if !ec {
        return Ok(out);
    }

    // §4.9.2 error_status u(8).
    out.push(error_status_raw);

    // §4.9.3 slice_crc_parity u(32), MSB-first.
    //
    // The §4.9.3 generator is MSB-first, initial value 0, no inversion.
    // For that family, appending `CRC(M)` to `M` (with the four parity
    // bytes interpreted in the same MSB-first orientation) yields a
    // whole-buffer CRC residue of 0. Running the generator over the
    // body + the just-appended size(3) + error_status(1) therefore
    // produces a parity word that drives the whole 8-byte footer's
    // CRC residue to 0, satisfying §4.9.3 ("the Slice as a whole has
    // a CRC remainder of 0") exactly.
    let parity = crate::crc::ffv1_crc32(&out);
    out.push(((parity >> 24) & 0xFF) as u8);
    out.push(((parity >> 16) & 0xFF) as u8);
    out.push(((parity >> 8) & 0xFF) as u8);
    out.push((parity & 0xFF) as u8);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ec == 0`: a 3-byte buffer holding `slice_size = 0` parses to
    /// a footer-only "empty body" record. This is the degenerate
    /// boundary: a slice can't really be 0 bytes long in practice
    /// (the header alone consumes bytes), but the parser must accept
    /// it because the structural check (`slice_size == len - 3`)
    /// holds.
    #[test]
    fn ec0_size_zero_footer_only() {
        let buf = [0x00, 0x00, 0x00];
        let f = parse_slice_footer(&buf, false).expect("3-byte ec=0 footer parses");
        assert_eq!(f.slice_size, 0);
        assert_eq!(f.total_size, 3);
        assert!(f.error_status.is_none());
        assert!(f.slice_crc_parity.is_none());
        assert_eq!(f.footer_len(), 3);
    }

    /// `ec == 0`: `slice_size` is the on-wire u(24) big-endian read
    /// from the trailing 3 bytes. Build a 1-byte "body" + 3-byte
    /// footer (size_field = 1) and check the round-trip.
    #[test]
    fn ec0_size_one_one_byte_body() {
        let buf = [0xAB, 0x00, 0x00, 0x01];
        let f = parse_slice_footer(&buf, false).expect("4-byte buffer parses");
        assert_eq!(f.slice_size, 1);
        assert_eq!(f.total_size, 4);
    }

    /// `ec == 0`: the on-wire `slice_size` must match the buffer
    /// length minus the 3-byte footer; a mismatch is the
    /// trailer-chain-walk-was-wrong error.
    #[test]
    fn ec0_size_mismatch_rejected() {
        // 5-byte buffer, but size_field = 10 → mismatch.
        let buf = [0xAA, 0xBB, 0x00, 0x00, 0x0A];
        match parse_slice_footer(&buf, false) {
            Err(Error::SliceSizeOutOfRange { field, expected }) => {
                assert_eq!(field, 10);
                assert_eq!(expected, 2);
            }
            other => panic!("expected SliceSizeOutOfRange, got {other:?}"),
        }
    }

    /// `ec == 1`: parsing without the 8-byte footer is the
    /// truncated-buffer error.
    #[test]
    fn ec1_truncated_rejected() {
        for short in [0usize, 1, 2, 3, 4, 5, 6, 7] {
            let buf = vec![0u8; short];
            assert!(matches!(
                parse_slice_footer(&buf, true),
                Err(Error::TruncatedSliceFooter)
            ));
        }
    }

    /// `ec == 0`: parsing without the 3-byte footer is the
    /// truncated-buffer error.
    #[test]
    fn ec0_truncated_rejected() {
        for short in [0usize, 1, 2] {
            let buf = vec![0u8; short];
            assert!(matches!(
                parse_slice_footer(&buf, false),
                Err(Error::TruncatedSliceFooter)
            ));
        }
    }

    /// `ec == 1`: a body of zeros + a footer whose parity is the §4.9.3
    /// CRC of the body has whole-slice residue zero. This is the
    /// canonical "encoder solved for parity" path.
    #[test]
    fn ec1_residue_zero_for_solved_parity() {
        // Body: 1 byte 0x80 (chosen so CRC is the known-answer
        // 0x690CE0EE from `crc.rs::known_answer_single_bytes`).
        // Footer: size=0x000001 + error=0 + parity = CRC of "body +
        // size+error" (so the whole 9-byte slice residue is 0).
        let body = [0x80u8];
        // Pre-footer bytes that go into the CRC alongside the body:
        // size_field big-endian and error_status.
        let size = 0x00_00_01u32;
        let footer_size = [
            ((size >> 16) & 0xFF) as u8,
            ((size >> 8) & 0xFF) as u8,
            (size & 0xFF) as u8,
        ];
        let error_status = 0u8;

        // Compute the parity that drives the whole-slice CRC to 0.
        // CRC is a linear function over GF(2) extended in 0; the
        // simplest way is to run the generator over body + size +
        // error_status, then append the resulting CRC as parity —
        // the generator's "residue of CRC's body || CRC(body) == 0"
        // property is exactly the construction the encoder uses.
        let mut buf = Vec::new();
        buf.extend_from_slice(&body);
        buf.extend_from_slice(&footer_size);
        buf.push(error_status);
        let parity = ffv1_crc32(&buf);
        buf.extend_from_slice(&parity.to_be_bytes());

        let f = parse_slice_footer(&buf, true).expect("solved-parity slice validates");
        assert_eq!(f.slice_size, 1);
        assert_eq!(f.total_size, 9);
        assert_eq!(f.error_status, Some(SliceErrorStatus::NoError));
        assert_eq!(f.error_status_raw, Some(0));
        assert_eq!(f.slice_crc_parity, Some(parity));
        assert_eq!(f.footer_len(), 8);
    }

    /// `ec == 1`: flipping a body byte makes the residue non-zero and
    /// the validator surfaces both the residue and the stored parity
    /// (so callers can diagnose ec=1 vs ec=2 recovery decisions).
    #[test]
    fn ec1_corrupted_body_rejected() {
        // Same construction as above…
        let body = [0x80u8];
        let size = 0x00_00_01u32;
        let footer_size = [
            ((size >> 16) & 0xFF) as u8,
            ((size >> 8) & 0xFF) as u8,
            (size & 0xFF) as u8,
        ];
        let mut buf = Vec::new();
        buf.extend_from_slice(&body);
        buf.extend_from_slice(&footer_size);
        buf.push(0u8);
        let parity = ffv1_crc32(&buf);
        buf.extend_from_slice(&parity.to_be_bytes());

        // …then flip the body byte.
        buf[0] ^= 0x01;

        match parse_slice_footer(&buf, true) {
            Err(Error::SliceCrcMismatch {
                residue,
                stored_parity,
            }) => {
                assert_ne!(residue, 0);
                assert_eq!(stored_parity, parity);
            }
            other => panic!("expected SliceCrcMismatch, got {other:?}"),
        }
    }

    /// `error_status` enum mapping covers §4.9.2 Table 16 verbatim,
    /// including the catch-all reserved range.
    #[test]
    fn error_status_table_16_mapping() {
        assert_eq!(SliceErrorStatus::from_wire(0), SliceErrorStatus::NoError);
        assert_eq!(
            SliceErrorStatus::from_wire(1),
            SliceErrorStatus::Correctable
        );
        assert_eq!(
            SliceErrorStatus::from_wire(2),
            SliceErrorStatus::Uncorrectable
        );
        for reserved in [3u8, 4, 7, 100, 255] {
            assert_eq!(
                SliceErrorStatus::from_wire(reserved),
                SliceErrorStatus::Reserved
            );
        }
        assert!(SliceErrorStatus::Uncorrectable.is_uncorrectable());
        assert!(!SliceErrorStatus::Correctable.is_uncorrectable());
        assert!(!SliceErrorStatus::NoError.is_uncorrectable());
        assert!(!SliceErrorStatus::Reserved.is_uncorrectable());
    }

    // ---- §4.9 encoder round-trips (encode → parse symmetry) ----------

    /// `ec == 0` round trip: a body of arbitrary bytes gains a 3-byte
    /// `slice_size` u(24) footer and re-parses to the same `slice_size`
    /// and total length. No CRC validation is performed in the `ec == 0`
    /// branch.
    #[test]
    fn encode_ec0_round_trips_through_parse() {
        for body in [
            vec![],
            vec![0x00u8],
            vec![0xFFu8; 7],
            (0..=255u8).collect::<Vec<u8>>(),
        ] {
            let wire = encode_slice_footer(&body, false, SliceErrorStatus::NoError)
                .expect("body length fits in u(24)");
            assert_eq!(wire.len(), body.len() + SLICE_FOOTER_LEN_EC0);
            assert_eq!(&wire[..body.len()], &body[..]);

            let parsed = parse_slice_footer(&wire, false).expect("encoded ec=0 footer parses");
            assert_eq!(parsed.slice_size as usize, body.len());
            assert_eq!(parsed.total_size as usize, wire.len());
            assert!(parsed.error_status.is_none());
            assert!(parsed.slice_crc_parity.is_none());
            assert_eq!(parsed.footer_len(), SLICE_FOOTER_LEN_EC0);
        }
    }

    /// `ec == 1` round trip: a body of arbitrary bytes gains an 8-byte
    /// footer whose `slice_crc_parity` (§4.9.3) drives the whole-Slice
    /// CRC residue to zero — i.e. `parse_slice_footer(..., true)`
    /// returns Ok with no `SliceCrcMismatch`.
    #[test]
    fn encode_ec1_round_trips_through_parse_with_zero_residue() {
        for body in [
            vec![],
            vec![0x80u8],
            vec![0xDEu8, 0xAD, 0xBE, 0xEF],
            (0..200u8).collect::<Vec<u8>>(),
        ] {
            let wire = encode_slice_footer(&body, true, SliceErrorStatus::NoError)
                .expect("body fits in u(24)");
            assert_eq!(wire.len(), body.len() + SLICE_FOOTER_LEN_EC1);
            assert_eq!(&wire[..body.len()], &body[..]);

            // §4.9.3 whole-Slice CRC residue is zero by construction.
            assert_eq!(ffv1_crc32(&wire), 0, "body len {}", body.len());

            let parsed = parse_slice_footer(&wire, true).expect("encoded ec=1 footer parses");
            assert_eq!(parsed.slice_size as usize, body.len());
            assert_eq!(parsed.total_size as usize, wire.len());
            assert_eq!(parsed.error_status, Some(SliceErrorStatus::NoError));
            assert_eq!(parsed.error_status_raw, Some(0));
            assert!(parsed.slice_crc_parity.is_some());
            assert_eq!(parsed.footer_len(), SLICE_FOOTER_LEN_EC1);
        }
    }

    /// `ec == 1` round trip: every §4.9.2 Table 16 typed value
    /// (NoError / Correctable / Uncorrectable / Reserved) survives the
    /// encode→parse round-trip with the wire byte preserved.
    #[test]
    fn encode_ec1_error_status_table_16_round_trips() {
        let body = [0xAAu8, 0xBB, 0xCC];
        for (status, raw) in [
            (SliceErrorStatus::NoError, 0u8),
            (SliceErrorStatus::Correctable, 1),
            (SliceErrorStatus::Uncorrectable, 2),
            // Encoder maps the typed `Reserved` to byte 3 (the lowest
            // reserved wire value per Table 16).
            (SliceErrorStatus::Reserved, 3),
        ] {
            let wire = encode_slice_footer(&body, true, status).expect("body fits in u(24)");
            let parsed = parse_slice_footer(&wire, true).expect("parses");
            assert_eq!(parsed.error_status, Some(status));
            assert_eq!(parsed.error_status_raw, Some(raw));
            assert_eq!(ffv1_crc32(&wire), 0);
        }
    }

    /// `encode_slice_footer_with_raw_status`: every reserved wire byte
    /// (3..=255) is preserved verbatim and decodes to the typed
    /// `Reserved` variant per §4.9.2 Table 16.
    #[test]
    fn encode_ec1_raw_status_preserves_reserved_bytes() {
        let body = [0x01u8, 0x02, 0x03];
        for raw in [3u8, 7, 99, 254, 255] {
            let wire =
                encode_slice_footer_with_raw_status(&body, true, raw).expect("body fits in u(24)");
            let parsed = parse_slice_footer(&wire, true).expect("parses");
            assert_eq!(parsed.error_status, Some(SliceErrorStatus::Reserved));
            assert_eq!(parsed.error_status_raw, Some(raw));
            assert_eq!(ffv1_crc32(&wire), 0);
        }
    }

    /// `ec == 0`: `encode_slice_footer_with_raw_status` ignores the
    /// `error_status` argument when `ec == false` — the §4.9 `if (ec)`
    /// branch is not taken so the trailer is just the 3-byte
    /// `slice_size` field.
    #[test]
    fn encode_ec0_ignores_error_status_argument() {
        let body = [0x42u8; 5];
        let wire_a = encode_slice_footer(&body, false, SliceErrorStatus::NoError).unwrap();
        let wire_b = encode_slice_footer(&body, false, SliceErrorStatus::Uncorrectable).unwrap();
        let wire_c = encode_slice_footer_with_raw_status(&body, false, 99).unwrap();
        assert_eq!(wire_a, wire_b);
        assert_eq!(wire_a, wire_c);
        assert_eq!(wire_a.len(), body.len() + SLICE_FOOTER_LEN_EC0);
    }

    /// §4.9.1: `slice_size` is `u(24)`; a body whose length escapes
    /// that bound cannot be expressed on the wire. The encoder surfaces
    /// the structural overflow with the `1 << 24` boundary as the
    /// `expected` field of [`Error::SliceSizeOutOfRange`].
    #[test]
    fn encode_rejects_body_too_long_for_u24_slice_size() {
        // 16 MiB exactly — first length that does NOT fit.
        let body = vec![0u8; 1 << 24];
        for ec in [false, true] {
            match encode_slice_footer(&body, ec, SliceErrorStatus::NoError) {
                Err(Error::SliceSizeOutOfRange { field, expected }) => {
                    assert_eq!(expected, 1 << 24);
                    assert_eq!(field, 0); // 16 MiB wraps to 0 mod 2^24.
                }
                other => panic!("expected SliceSizeOutOfRange, got {other:?}"),
            }
        }
        // One byte below the bound is fine (this is the largest valid
        // body the wire format admits).
        let max_body = vec![0u8; (1 << 24) - 1];
        let wire = encode_slice_footer(&max_body, false, SliceErrorStatus::NoError)
            .expect("(1<<24)-1 fits in u(24)");
        assert_eq!(wire.len(), max_body.len() + SLICE_FOOTER_LEN_EC0);
        // No need to verify the parse here — the round-trip tests above
        // already cover that path. The point of this case is just to
        // pin down that the boundary is `<`, not `<=`.
    }

    /// Encoder + parser symmetry: corrupting *any* body byte after a
    /// successful encode causes the parser's §4.9.3 residue check to
    /// fail (and the stored parity is reported back unchanged so the
    /// caller can log the expected-vs-computed mismatch).
    #[test]
    fn encoded_ec1_corrupted_body_fails_parser_residue_check() {
        let body = vec![0x10u8, 0x20, 0x30, 0x40];
        let mut wire = encode_slice_footer(&body, true, SliceErrorStatus::NoError).unwrap();
        // §4.9.3 sanity precondition: a clean encode is residue-zero.
        assert_eq!(ffv1_crc32(&wire), 0);
        let stored_parity_pre = {
            let off = wire.len() - 4;
            (u32::from(wire[off]) << 24)
                | (u32::from(wire[off + 1]) << 16)
                | (u32::from(wire[off + 2]) << 8)
                | u32::from(wire[off + 3])
        };
        // Flip the first body byte; the §4.9.3 residue is no longer 0.
        wire[0] ^= 0x01;
        match parse_slice_footer(&wire, true) {
            Err(Error::SliceCrcMismatch {
                residue,
                stored_parity,
            }) => {
                assert_ne!(residue, 0);
                assert_eq!(stored_parity, stored_parity_pre);
            }
            other => panic!("expected SliceCrcMismatch, got {other:?}"),
        }
    }

    /// `ec == 1`: the §4.9.3 parity solver is **deterministic** —
    /// re-encoding the same body produces the same parity word
    /// bit-exactly. (Trivially true given the CRC is a pure function,
    /// but the regression guard pins down that no path-dependent
    /// state leaks into the encoder.)
    #[test]
    fn encode_ec1_is_deterministic() {
        let body = vec![0xC3u8; 17];
        let wire_a = encode_slice_footer(&body, true, SliceErrorStatus::NoError).unwrap();
        let wire_b = encode_slice_footer(&body, true, SliceErrorStatus::NoError).unwrap();
        assert_eq!(wire_a, wire_b);
    }

    /// `ec == 1`: re-encoding a body whose contents change in exactly
    /// one bit produces a different parity word — the §4.9.3 generator
    /// has the bit-mixing property a meaningful CRC needs (a "trivial
    /// always-zero parity" implementation would silently pass).
    #[test]
    fn encode_ec1_parity_responds_to_body_changes() {
        let mut body = vec![0u8; 8];
        let wire_zero = encode_slice_footer(&body, true, SliceErrorStatus::NoError).unwrap();
        body[3] = 0x01;
        let wire_one = encode_slice_footer(&body, true, SliceErrorStatus::NoError).unwrap();
        // The body bytes themselves differ in exactly one byte; the
        // parity (last 4 bytes) must differ too.
        let parity_zero = &wire_zero[wire_zero.len() - 4..];
        let parity_one = &wire_one[wire_one.len() - 4..];
        assert_ne!(parity_zero, parity_one);
    }

    /// `ec == 1`: the encoder's parity solver also responds to changes
    /// in the `error_status` byte (the byte sits inside the §4.9.3
    /// CRC's coverage by §4.9 — the spec's pseudocode places it BEFORE
    /// `slice_crc_parity` in the SliceFooter struct, so the §4.9.3
    /// "Slice as a whole" CRC sees it).
    #[test]
    fn encode_ec1_parity_responds_to_error_status_changes() {
        let body = vec![0xAAu8; 5];
        let wire_no_err = encode_slice_footer(&body, true, SliceErrorStatus::NoError).unwrap();
        let wire_corr = encode_slice_footer(&body, true, SliceErrorStatus::Correctable).unwrap();
        let parity_no_err = &wire_no_err[wire_no_err.len() - 4..];
        let parity_corr = &wire_corr[wire_corr.len() - 4..];
        assert_ne!(parity_no_err, parity_corr);
        // Both still satisfy §4.9.3 residue zero — that's the whole
        // point: the parity adapts to keep the residue at zero whatever
        // the preceding bytes are.
        assert_eq!(ffv1_crc32(&wire_no_err), 0);
        assert_eq!(ffv1_crc32(&wire_corr), 0);
    }

    /// `ec == 1`: the parity word the encoder picks equals the §4.9.3
    /// CRC of `body || size(3) || error_status(1)` exactly. Pins the
    /// solver shape (and surfaces any future change that breaks the
    /// `CRC(M) appended to M => CRC == 0` derivation).
    #[test]
    fn encode_ec1_parity_equals_crc_of_pre_parity_prefix() {
        let body = vec![0x01u8, 0x02, 0x03, 0x04, 0x05];
        let wire = encode_slice_footer(&body, true, SliceErrorStatus::Correctable).unwrap();
        let pre = &wire[..wire.len() - 4];
        let expected_parity = ffv1_crc32(pre);
        let actual_parity = (u32::from(wire[wire.len() - 4]) << 24)
            | (u32::from(wire[wire.len() - 3]) << 16)
            | (u32::from(wire[wire.len() - 2]) << 8)
            | u32::from(wire[wire.len() - 1]);
        assert_eq!(actual_parity, expected_parity);
    }

    /// `ec == 1`: a `slice_size` field that disagrees with the buffer
    /// length is rejected BEFORE the CRC check. That ordering matters
    /// for debugging: a mis-walked trailer chain should surface as a
    /// structural error, not as a (downstream) CRC failure.
    #[test]
    fn ec1_size_mismatch_rejected_before_crc() {
        // 16-byte buffer, but size_field = 99 → mismatch.
        let mut buf = vec![0xAAu8; 16];
        let size_off = buf.len() - SLICE_FOOTER_LEN_EC1;
        buf[size_off] = 0;
        buf[size_off + 1] = 0;
        buf[size_off + 2] = 99;
        match parse_slice_footer(&buf, true) {
            Err(Error::SliceSizeOutOfRange { field, expected }) => {
                assert_eq!(field, 99);
                assert_eq!(expected, 8);
            }
            other => panic!("expected SliceSizeOutOfRange, got {other:?}"),
        }
    }
}
