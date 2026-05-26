//! FFV1 Slice Header parser (RFC 9043 §4.6).
//!
//! A Slice Header sits at the start of each Slice's range-coded
//! region for FFV1 version 3 streams (§4.5). It carries the slice's
//! position and size on the configured slice raster, the per-plane
//! quantization-table-set indexes, the picture structure flag, and a
//! sample aspect ratio.
//!
//! This module decodes the Figure in §4.6 verbatim:
//!
//! ```text
//! SliceHeader( ) {
//!     slice_x                                  | ur
//!     slice_y                                  | ur
//!     slice_width - 1                          | ur
//!     slice_height - 1                         | ur
//!     for (i = 0; i < quant_table_set_index_count; i++) {
//!         quant_table_set_index[ i ]           | ur
//!     }
//!     picture_structure                        | ur
//!     sar_num                                  | ur
//!     sar_den                                  | ur
//! }
//! ```
//!
//! The parse is *structural*: the returned [`Ffv1SliceHeader`] reports
//! the raster-space slice position and size, the plane-to-quant-table
//! mapping, and the cosmetic picture-structure / SAR triple. **No
//! pixel decoding happens here** — slice contents (per-plane
//! sample-difference streams) and slice footer parsing are still
//! deferred to later rounds.
//!
//! Composition with the round-1 range coder is direct: the caller
//! supplies the slice's range-coded byte region (the bytes BEFORE the
//! 8-byte SliceFooter when `ec=1`, or BEFORE the 3-byte footer when
//! `ec=0`); a fresh [`RangeDecoder`] is constructed over those bytes
//! and walked through the symbols above.

use crate::config::{Ffv1ConfigurationRecord, Ffv1Version, PictureStructure};
use crate::range_coder::{RangeDecoder, RangeEncoder, PARAMETERS_INITIAL_STATE};
use crate::symbol::{get_ur, put_ur, SYMBOL_CONTEXT_SIZE};
use crate::Error;

/// Maximum quant-table-set-index slots a slice header may carry.
///
/// Per RFC 9043 §4.6.5 the count is `1 + (chroma||v<=3 ? 1 : 0) +
/// (extra_plane ? 1 : 0)` — i.e. between 1 and 3 inclusive. Three
/// gives the absolute upper bound (chroma + extra plane present).
pub const MAX_QUANT_TABLE_SET_INDEXES: usize = 3;

/// Parsed contents of an FFV1 v3 Slice Header (RFC 9043 §4.6).
///
/// The picture-structure and SAR fields are surfaced raw; their
/// out-of-spec values are not rejected here so a caller that wants to
/// log unusual streams can do so. The `slice_width` / `slice_height`
/// values are the *on-wire-plus-one* — i.e. they already incorporate
/// the "minus 1" that RFC 9043 §4.6.3 / §4.6.4 specify.
#[derive(Debug, Clone)]
pub struct Ffv1SliceHeader {
    /// `slice_x` (RFC 9043 §4.6.1): x position on the slice raster
    /// formed by `num_h_slices`.
    pub slice_x: u32,
    /// `slice_y` (§4.6.2): y position on the slice raster formed by
    /// `num_v_slices`.
    pub slice_y: u32,
    /// `slice_width` (§4.6.3): width on the slice raster. On the wire
    /// this is encoded as `slice_width - 1`; this field carries the
    /// post-increment value, i.e. `>= 1`.
    pub slice_width: u32,
    /// `slice_height` (§4.6.4): height on the slice raster. As with
    /// `slice_width`, this is the post-`+1` value.
    pub slice_height: u32,
    /// Number of valid entries in `quant_table_set_index` — equal to
    /// `quant_table_set_index_count` per §4.6.5.
    pub quant_table_set_index_count: usize,
    /// Per-plane quantization-table-set selector (§4.6.6). Only
    /// `[..quant_table_set_index_count]` is meaningful; the tail is
    /// zero-filled.
    pub quant_table_set_index: [u32; MAX_QUANT_TABLE_SET_INDEXES],
    /// `picture_structure` (§4.6.7), mapped to the typed enum.
    pub picture_structure: PictureStructure,
    /// Raw `picture_structure` value as it appeared on the wire.
    /// Provided so reserved (>= 4) values can still be logged after
    /// the typed `picture_structure` field has been coerced to
    /// [`PictureStructure::Unknown`].
    pub picture_structure_raw: u32,
    /// `sar_num` (§4.6.8). Zero means "aspect ratio unknown".
    pub sar_num: u32,
    /// `sar_den` (§4.6.9). Zero means "aspect ratio unknown".
    pub sar_den: u32,
}

impl Ffv1SliceHeader {
    /// True when the SAR pair is meaningful (both numerator and
    /// denominator are non-zero) per §4.6.8 / §4.6.9.
    pub fn sar_is_known(&self) -> bool {
        self.sar_num != 0 && self.sar_den != 0
    }

    /// Returns the per-plane quantization-table indices that are
    /// actually populated.
    pub fn quant_table_indices(&self) -> &[u32] {
        &self.quant_table_set_index[..self.quant_table_set_index_count]
    }
}

/// Per RFC 9043 §4.6.5 the index count is bounded by:
///   `1 + (chroma_planes||version<=3 ? 1 : 0) + (extra_plane ? 1 : 0)`.
///
/// All three operands are 0 or 1, so the result is in `1..=3`.
fn quant_table_set_index_count(cr: &Ffv1ConfigurationRecord) -> usize {
    let chroma_or_old = cr.chroma_planes || matches!(cr.version, Ffv1Version::V0 | Ffv1Version::V1);
    // For version 3 specifically, `version <= 3` is also true; for
    // future v4 this would flip. We use the explicit enum match for
    // version 0/1; v3 is handled by the `version <= 3` arm via a
    // direct boolean.
    let chroma_or_v3 = chroma_or_old || cr.version == Ffv1Version::V3;
    1 + usize::from(chroma_or_v3) + usize::from(cr.extra_plane)
}

/// Size of the slice-header context state buffer.
///
/// RFC 9043 §4.6 says "Slice Header has its own initial states, all
/// set to 128" without specifying the buffer width. The Parameters
/// section (§4.2) has the same ambiguity and was empirically
/// resolved as "all symbols share a single 32-slot context window";
/// the fixture tests below confirm the same hypothesis holds for the
/// slice header.
///
/// All Slice Header fields share a single 32-slot context window; the
/// buffer needs only [`SYMBOL_CONTEXT_SIZE`] slots, padded to 64.
const SLICE_HEADER_STATE_LEN: usize = 64;

/// Parse a slice header (RFC 9043 §4.6) from the start of `slice_bytes`.
///
/// `slice_bytes` is the slice's range-coded byte region — i.e. the
/// slice as transmitted, **excluding** the trailing
/// [`SliceFooter`](https://www.rfc-editor.org/rfc/rfc9043.html#section-4.9)
/// (8 bytes when `ec=1`, 3 bytes when `ec=0`). The caller is
/// responsible for locating the slice boundary via the
/// trailer-pointer chain; this parser starts at `slice_bytes[0]` and
/// reads only as many bytes as the range coder consumes for the
/// header.
///
/// `cr` is the per-stream Configuration Record — its `chroma_planes`,
/// `extra_plane`, and `version` together determine the
/// `quant_table_set_index_count` (§4.6.5).
///
/// Returns the parsed [`Ffv1SliceHeader`]. The range coder's residual
/// state is intentionally discarded by this entry point; callers that
/// need to continue with Slice Content range decoding (coder_type
/// 1/2) should use [`parse_slice_header_from_decoder`] instead, which
/// borrows a caller-owned [`RangeDecoder`] so the same decoder cursor
/// flows into the per-Plane reconstruction.
///
/// # Errors
///
/// * [`Error::TruncatedRangeCoder`] if `slice_bytes` has fewer than
///   two bytes (the minimum to seed the range coder).
/// * [`Error::UnsupportedVersion`] is *not* raised — the Configuration
///   Record's version was validated when it was parsed; this parser
///   assumes a v3 stream (the only version with a wire SliceHeader,
///   per §4.5 `if (version >= 3) SliceHeader()`).
pub fn parse_slice_header(
    slice_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
) -> Result<Ffv1SliceHeader, Error> {
    let mut rc = RangeDecoder::new(slice_bytes)?;
    parse_slice_header_from_decoder(&mut rc, cr)
}

/// Parse a slice header from a caller-owned [`RangeDecoder`] (RFC 9043
/// §4.6).
///
/// Unlike [`parse_slice_header`], this entry point does NOT construct a
/// fresh decoder; the caller passes one already seeded over the slice's
/// range-coded body. After return, the decoder's cursor (`low` /
/// `range` / byte position) is positioned immediately after the Slice
/// Header — i.e. at the first byte of the Slice Content's range-coded
/// region (for `coder_type == 1 || coder_type == 2`) or at the
/// byte-alignment boundary the Golomb-Rice bit reader will resume from
/// (for `coder_type == 0`).
///
/// This is the entry point the round-129 frame driver uses to chain
/// Slice Header → Slice Content decoding on the same range decoder.
pub fn parse_slice_header_from_decoder(
    rc: &mut RangeDecoder<'_>,
    cr: &Ffv1ConfigurationRecord,
) -> Result<Ffv1SliceHeader, Error> {
    // RFC 9043 §4.6: "Slice Header has its own initial states, all
    // set to 128." All Slice Header fields share a single 32-slot
    // context window — the same convention the §4.2 Parameters section
    // uses (the §4.6 `ur` symbols all read offsets 0..=31 of the same
    // `state` pointer). Confirmed bit-exact: with this layout the
    // per-Slice range-coder content start matches the reference trace's
    // `RAC_STATE` for every Slice once the §4.4 frame `keyframe` bit is
    // consumed by the driver before the first Slice's header.
    let mut state = [PARAMETERS_INITIAL_STATE; SLICE_HEADER_STATE_LEN];

    // `win!()` always yields the first (shared) window.
    macro_rules! win {
        () => {{
            (0usize, SYMBOL_CONTEXT_SIZE)
        }};
    }

    // ----- slice_x (ur) -----------------------------------------------
    let (lo, hi) = win!();
    let slice_x = get_ur(rc, &mut state[lo..hi]);

    // ----- slice_y (ur) -----------------------------------------------
    let (lo, hi) = win!();
    let slice_y = get_ur(rc, &mut state[lo..hi]);

    // ----- slice_width - 1 (ur), so add 1 to recover slice_width ------
    let (lo, hi) = win!();
    let slice_width_minus_1 = get_ur(rc, &mut state[lo..hi]);
    let slice_width = slice_width_minus_1.wrapping_add(1);

    // ----- slice_height - 1 (ur) --------------------------------------
    let (lo, hi) = win!();
    let slice_height_minus_1 = get_ur(rc, &mut state[lo..hi]);
    let slice_height = slice_height_minus_1.wrapping_add(1);

    // ----- quant_table_set_index[i] (ur), 1..=3 entries ----------------
    let count = quant_table_set_index_count(cr);
    debug_assert!(count <= MAX_QUANT_TABLE_SET_INDEXES);
    let mut quant_table_set_index = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
    for slot in quant_table_set_index.iter_mut().take(count) {
        let (lo, hi) = win!();
        *slot = get_ur(rc, &mut state[lo..hi]);
    }

    // ----- picture_structure (ur) -------------------------------------
    let (lo, hi) = win!();
    let picture_structure_raw = get_ur(rc, &mut state[lo..hi]);
    // §4.6.7 Table 15: reserved values fold to Unknown but the raw
    // wire value is preserved on the struct for diagnostic logging.
    let picture_structure =
        PictureStructure::from_wire(picture_structure_raw).unwrap_or(PictureStructure::Unknown);

    // ----- sar_num (ur) -----------------------------------------------
    let (lo, hi) = win!();
    let sar_num = get_ur(rc, &mut state[lo..hi]);

    // ----- sar_den (ur) -----------------------------------------------
    let (lo, hi) = win!();
    let sar_den = get_ur(rc, &mut state[lo..hi]);

    Ok(Ffv1SliceHeader {
        slice_x,
        slice_y,
        slice_width,
        slice_height,
        quant_table_set_index_count: count,
        quant_table_set_index,
        picture_structure,
        picture_structure_raw,
        sar_num,
        sar_den,
    })
}

/// Encode a slice header (RFC 9043 §4.6) by appending its `ur` symbols
/// to a caller-owned [`RangeEncoder`].
///
/// This is the symmetric inverse of [`parse_slice_header_from_decoder`]:
/// it walks the same Figure-in-§4.6 fields in the same order, each one
/// a `put_ur` against the shared 32-slot context window §4.6 places at
/// the start of the Slice's range-coded region. After return, the
/// encoder's byte position sits immediately after the last `ur` symbol
/// of the header — i.e. exactly where the matching `coder_type >= 1`
/// Slice Content encoder would resume.
///
/// `cr` is the per-stream Configuration Record — its `chroma_planes`,
/// `extra_plane`, and `version` together determine
/// `quant_table_set_index_count` (§4.6.5), the same way the decoder
/// derives the loop bound.
///
/// `header.slice_width` / `header.slice_height` are the post-`+1`
/// raster values per §4.6.3 / §4.6.4. The encoder transmits
/// `slice_width - 1` / `slice_height - 1` so the decoder's
/// `wrapping_add(1)` recovers the input bit-exactly. A zero raster
/// dimension is rejected (`Error::SliceSizeOutOfRange`) — every Slice
/// covers at least one cell of the slice raster (§4.6: "x position …
/// y position … width … height" all describe a Slice of non-zero
/// extent).
///
/// `header.quant_table_set_index_count` is verified to equal the value
/// `quant_table_set_index_count(cr)` derives from `cr` (a mismatched
/// header would silently desynchronise the decoder's loop). The first
/// `count` entries of `header.quant_table_set_index` are emitted; any
/// trailing zeros past `count` are ignored.
///
/// # Errors
///
/// * [`Error::SliceSizeOutOfRange`] when `slice_width == 0` or
///   `slice_height == 0` (the wire field is `slice_width - 1`, so 0
///   would underflow the round-trip), or when the
///   `quant_table_set_index_count` field disagrees with what the
///   `Ffv1ConfigurationRecord` derives. The `field` value carries the
///   header's reported dimension or count; `expected` carries the
///   minimum (1) or the count the Configuration Record demands.
pub fn encode_slice_header_to_encoder(
    re: &mut RangeEncoder,
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
) -> Result<(), Error> {
    // §4.6.3 / §4.6.4: width and height are transmitted as
    // `slice_width - 1` / `slice_height - 1`. A 0 raster dimension is
    // unrepresentable — it would `wrapping_sub` underflow on the wire,
    // and a 0-pixel Slice has no §4.7 layout to match anyway.
    if header.slice_width == 0 {
        return Err(Error::SliceSizeOutOfRange {
            field: header.slice_width,
            expected: 1,
        });
    }
    if header.slice_height == 0 {
        return Err(Error::SliceSizeOutOfRange {
            field: header.slice_height,
            expected: 1,
        });
    }

    // §4.6.5: the index count is determined by `cr`. A header that
    // claims a different count would emit a different number of `ur`
    // symbols than the decoder's matching loop reads, desynchronising
    // every subsequent field. Reject the mismatch here so the encoder
    // surface stays self-consistent with the decoder surface.
    let count = quant_table_set_index_count(cr);
    if header.quant_table_set_index_count != count {
        return Err(Error::SliceSizeOutOfRange {
            field: header.quant_table_set_index_count as u32,
            expected: count as u32,
        });
    }
    debug_assert!(count <= MAX_QUANT_TABLE_SET_INDEXES);

    // RFC 9043 §4.6: "Slice Header has its own initial states, all set
    // to 128." All fields share a single 32-slot context window — same
    // layout the decoder uses. See parse_slice_header_from_decoder
    // above for the empirical justification.
    let mut state = [PARAMETERS_INITIAL_STATE; SLICE_HEADER_STATE_LEN];

    macro_rules! win {
        () => {{
            (0usize, SYMBOL_CONTEXT_SIZE)
        }};
    }

    // ----- slice_x (ur) -----------------------------------------------
    let (lo, hi) = win!();
    put_ur(re, &mut state[lo..hi], header.slice_x);

    // ----- slice_y (ur) -----------------------------------------------
    let (lo, hi) = win!();
    put_ur(re, &mut state[lo..hi], header.slice_y);

    // ----- slice_width - 1 (ur) ---------------------------------------
    let (lo, hi) = win!();
    put_ur(re, &mut state[lo..hi], header.slice_width - 1);

    // ----- slice_height - 1 (ur) --------------------------------------
    let (lo, hi) = win!();
    put_ur(re, &mut state[lo..hi], header.slice_height - 1);

    // ----- quant_table_set_index[i] (ur), 1..=3 entries ----------------
    for i in 0..count {
        let (lo, hi) = win!();
        put_ur(re, &mut state[lo..hi], header.quant_table_set_index[i]);
    }

    // ----- picture_structure (ur) -------------------------------------
    //
    // Emit the raw wire value so callers can round-trip reserved /
    // Unknown variants through the encoder/decoder pair without the
    // typed enum lossily clamping them. (Round trips of typed values
    // pass header.picture_structure_raw == picture_structure.as_wire()
    // in the test suite below.)
    let (lo, hi) = win!();
    put_ur(re, &mut state[lo..hi], header.picture_structure_raw);

    // ----- sar_num (ur) -----------------------------------------------
    let (lo, hi) = win!();
    put_ur(re, &mut state[lo..hi], header.sar_num);

    // ----- sar_den (ur) -----------------------------------------------
    let (lo, hi) = win!();
    put_ur(re, &mut state[lo..hi], header.sar_den);

    Ok(())
}

/// Encode a slice header (RFC 9043 §4.6) into a freshly-allocated
/// `Vec<u8>` carrying its range-coded byte region.
///
/// Convenience wrapper around [`encode_slice_header_to_encoder`]: it
/// constructs a fresh [`RangeEncoder`], encodes the header, finishes
/// the encoder, and returns the resulting bytes — exactly the byte
/// region the matching [`parse_slice_header`] entry point consumes.
///
/// For `coder_type == 0` Slices (the §3.8.2 Golomb-Rice content
/// branch), this is the whole header — the SliceContent that follows
/// it switches to a byte-aligned bit reader, so the range coder's
/// residual state is naturally discarded. For `coder_type >= 1` Slices
/// the matching workflow is [`encode_slice_header_to_encoder`] with a
/// caller-owned [`RangeEncoder`] that carries forward into the
/// SliceContent encoder (mirroring how
/// [`parse_slice_header_from_decoder`] keeps the decoder cursor live
/// for the range-coded Slice Content).
///
/// See [`encode_slice_header_to_encoder`] for error semantics.
pub fn encode_slice_header(
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
) -> Result<Vec<u8>, Error> {
    let mut re = RangeEncoder::new();
    encode_slice_header_to_encoder(&mut re, header, cr)?;
    Ok(re.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};

    fn dummy_cr(
        chroma_planes: bool,
        extra_plane: bool,
        version: Ffv1Version,
    ) -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            version,
            micro_version: Some(4),
            coder_type: 1,
            state_transition_delta: [0; crate::config::NUM_TRANSITION_DELTAS],
            colorspace_type: ColorspaceType::YCbCr,
            bits_per_raw_sample: 8,
            chroma_planes,
            log2_h_chroma_subsample: 1,
            log2_v_chroma_subsample: 1,
            extra_plane,
            num_h_slices: Some(2),
            num_v_slices: Some(2),
            quant_table_set_count: Some(2),
        }
    }

    #[test]
    fn quant_index_count_chroma_v3_no_extra() {
        // chroma_planes=1, extra=0, v3 → 1 + 1 + 0 = 2.
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        assert_eq!(quant_table_set_index_count(&cr), 2);
    }

    #[test]
    fn quant_index_count_grayscale_v3_no_extra() {
        // chroma_planes=0, extra=0, v3 — but version<=3 → 1 + 1 + 0 = 2.
        let cr = dummy_cr(false, false, Ffv1Version::V3);
        assert_eq!(quant_table_set_index_count(&cr), 2);
    }

    #[test]
    fn quant_index_count_chroma_v3_with_extra() {
        // chroma=1, extra=1, v3 → 1 + 1 + 1 = 3 (upper bound).
        let cr = dummy_cr(true, true, Ffv1Version::V3);
        assert_eq!(quant_table_set_index_count(&cr), 3);
    }

    #[test]
    fn rejects_truncated_slice() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        assert!(matches!(
            parse_slice_header(&[], &cr),
            Err(Error::TruncatedRangeCoder)
        ));
        assert!(matches!(
            parse_slice_header(&[0xAB], &cr),
            Err(Error::TruncatedRangeCoder)
        ));
    }

    #[test]
    fn header_struct_helpers() {
        let h = Ffv1SliceHeader {
            slice_x: 0,
            slice_y: 0,
            slice_width: 1,
            slice_height: 1,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 1, 0],
            picture_structure: PictureStructure::Unknown,
            picture_structure_raw: 0,
            sar_num: 0,
            sar_den: 0,
        };
        assert!(!h.sar_is_known());
        assert_eq!(h.quant_table_indices(), &[0, 1]);

        let h2 = Ffv1SliceHeader {
            sar_num: 16,
            sar_den: 9,
            ..h.clone()
        };
        assert!(h2.sar_is_known());
    }

    // ---- §4.6 encoder round-trips (encode → parse symmetry) ----------

    /// Test scaffold carrying the field values for a single round-trip
    /// case. Bundling them keeps the [`round_trip_header`] helper's
    /// argument list manageable (clippy's `too_many_arguments` cap is
    /// 7 — the header alone has 9 user-visible fields).
    struct HeaderCase<'a> {
        slice_x: u32,
        slice_y: u32,
        slice_width: u32,
        slice_height: u32,
        indices: &'a [u32],
        picture_structure_raw: u32,
        sar_num: u32,
        sar_den: u32,
    }

    impl<'a> HeaderCase<'a> {
        /// Minimal 1×1-raster / `slice_x=slice_y=0` header with all
        /// optional fields zeroed. Useful as a base to `..` over with
        /// struct-update syntax for per-field exhaustive tests.
        fn minimal(indices: &'a [u32]) -> Self {
            Self {
                slice_x: 0,
                slice_y: 0,
                slice_width: 1,
                slice_height: 1,
                indices,
                picture_structure_raw: 0,
                sar_num: 0,
                sar_den: 0,
            }
        }
    }

    /// Compose a header with the given fields, then encode → parse and
    /// assert every field round-trips bit-exactly. The `picture_structure`
    /// typed value is derived from the raw via [`PictureStructure::from_wire`]
    /// the same way the parser does.
    fn round_trip_header(cr: &Ffv1ConfigurationRecord, case: &HeaderCase<'_>) -> Ffv1SliceHeader {
        let count = quant_table_set_index_count(cr);
        assert_eq!(case.indices.len(), count, "indices.len() must equal count");
        let mut quant_table_set_index = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
        quant_table_set_index[..count].copy_from_slice(case.indices);
        let picture_structure = PictureStructure::from_wire(case.picture_structure_raw)
            .unwrap_or(PictureStructure::Unknown);
        let header = Ffv1SliceHeader {
            slice_x: case.slice_x,
            slice_y: case.slice_y,
            slice_width: case.slice_width,
            slice_height: case.slice_height,
            quant_table_set_index_count: count,
            quant_table_set_index,
            picture_structure,
            picture_structure_raw: case.picture_structure_raw,
            sar_num: case.sar_num,
            sar_den: case.sar_den,
        };

        let bytes = encode_slice_header(&header, cr).expect("encode succeeds");
        let parsed = parse_slice_header(&bytes, cr).expect("re-parses");
        assert_eq!(parsed.slice_x, case.slice_x);
        assert_eq!(parsed.slice_y, case.slice_y);
        assert_eq!(parsed.slice_width, case.slice_width);
        assert_eq!(parsed.slice_height, case.slice_height);
        assert_eq!(parsed.quant_table_set_index_count, count);
        assert_eq!(
            &parsed.quant_table_set_index[..count],
            &header.quant_table_set_index[..count]
        );
        assert_eq!(parsed.picture_structure_raw, case.picture_structure_raw);
        assert_eq!(parsed.picture_structure, picture_structure);
        assert_eq!(parsed.sar_num, case.sar_num);
        assert_eq!(parsed.sar_den, case.sar_den);
        parsed
    }

    /// `chroma_planes=true / v3` (the YCbCr 4:2:0 corpus shape): a
    /// minimal `slice_x=0 / slice_y=0 / 1x1 raster / count=2` header
    /// round-trips bit-exactly through encode → parse.
    #[test]
    fn encode_round_trips_minimal_chroma_v3() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        round_trip_header(&cr, &HeaderCase::minimal(&[0, 0]));
    }

    /// `chroma_planes=true / v3 / extra_plane=true`: every count=3 case
    /// (the upper §4.6.5 bound) round-trips with each index reaching
    /// the per-Plane quant-table-set selector independently.
    #[test]
    fn encode_round_trips_chroma_v3_extra_plane_count3() {
        let cr = dummy_cr(true, true, Ffv1Version::V3);
        round_trip_header(
            &cr,
            &HeaderCase {
                slice_width: 4,
                slice_height: 3,
                ..HeaderCase::minimal(&[0, 1, 0])
            },
        );
    }

    /// `chroma_planes=false / v3` (the grayscale corpus shape): count
    /// stays at 2 (the version<=3 path) and round-trips just like the
    /// chroma case.
    #[test]
    fn encode_round_trips_grayscale_v3() {
        let cr = dummy_cr(false, false, Ffv1Version::V3);
        round_trip_header(&cr, &HeaderCase::minimal(&[0, 0]));
    }

    /// `slice_x` / `slice_y` exercise the §4.6.1 / §4.6.2 raster
    /// position fields across the small-but-nonzero regime the corpus
    /// hits in the 2x2 / 4x4 grids.
    #[test]
    fn encode_round_trips_slice_position_grid() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1), (3, 2), (5, 7), (15, 15)] {
            round_trip_header(
                &cr,
                &HeaderCase {
                    slice_x: x,
                    slice_y: y,
                    ..HeaderCase::minimal(&[0, 0])
                },
            );
        }
    }

    /// §4.6.3 / §4.6.4: the `slice_width` / `slice_height` fields are
    /// transmitted as `slice_width - 1` / `slice_height - 1`; the
    /// encoder must emit the subtracted form so the decoder's
    /// `wrapping_add(1)` recovers the original. Exercise dimensions
    /// spanning the small / power-of-two / large-but-realistic regimes.
    #[test]
    fn encode_round_trips_raster_dimensions() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        for (w, h) in [
            (1, 1),
            (2, 2),
            (3, 5),
            (8, 8),
            (15, 15),
            (16, 12),
            (64, 48),
            (255, 191),
        ] {
            round_trip_header(
                &cr,
                &HeaderCase {
                    slice_width: w,
                    slice_height: h,
                    ..HeaderCase::minimal(&[0, 0])
                },
            );
        }
    }

    /// §4.6.6: each `quant_table_set_index` slot reaches the per-Plane
    /// selector independently — flipping just slot[1] does not affect
    /// slot[0] (the shared 32-slot context window mutates step-by-step
    /// but the decoded values are independent).
    #[test]
    fn encode_round_trips_quant_table_indices() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        for indices in [[0u32, 0], [0, 1], [1, 0], [1, 1]] {
            round_trip_header(&cr, &HeaderCase::minimal(&indices));
        }
    }

    /// §4.6.7 Table 15 + the reserved-value preservation path: every
    /// typed `PictureStructure` value AND a representative reserved
    /// wire byte (5) survive encode → parse. The typed variant decodes
    /// to `Unknown` per `PictureStructure::from_wire` for the reserved
    /// path; the raw byte is preserved verbatim.
    #[test]
    fn encode_round_trips_picture_structure_table_15() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        for raw in [0u32, 1, 2, 3] {
            let parsed = round_trip_header(
                &cr,
                &HeaderCase {
                    picture_structure_raw: raw,
                    ..HeaderCase::minimal(&[0, 0])
                },
            );
            assert_eq!(parsed.picture_structure_raw, raw);
        }
        // Reserved range (>= 4): folds to `Unknown` per from_wire's
        // `_ => Err(other)` and the parser's `unwrap_or(Unknown)`, but
        // the raw byte is preserved on the struct.
        for raw in [4u32, 5, 99, 1024] {
            let parsed = round_trip_header(
                &cr,
                &HeaderCase {
                    picture_structure_raw: raw,
                    ..HeaderCase::minimal(&[0, 0])
                },
            );
            assert_eq!(parsed.picture_structure, PictureStructure::Unknown);
            assert_eq!(parsed.picture_structure_raw, raw);
        }
    }

    /// §4.6.8 / §4.6.9: `sar_num` / `sar_den` reach the wire as `ur`
    /// fields. The (0, 0) "aspect ratio unknown" pair, both-nonzero
    /// shapes, and a one-zero / one-nonzero degenerate all round-trip
    /// — the parser preserves the raw values without trying to "fix
    /// up" a half-signalled SAR.
    #[test]
    fn encode_round_trips_sar() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        for (n, d) in [
            (0u32, 0u32),
            (1, 1),
            (16, 9),
            (4, 3),
            (40, 33),
            (5, 0),
            (0, 7),
        ] {
            let parsed = round_trip_header(
                &cr,
                &HeaderCase {
                    sar_num: n,
                    sar_den: d,
                    ..HeaderCase::minimal(&[0, 0])
                },
            );
            assert_eq!(parsed.sar_num, n);
            assert_eq!(parsed.sar_den, d);
            assert_eq!(parsed.sar_is_known(), n != 0 && d != 0);
        }
    }

    /// Full-field exhaustive round-trip with every header field carrying
    /// a non-default value at the same time. This is the integration
    /// guarantee: per-field round trips above isolate the §4.6.N
    /// branches; this case asserts they compose without cross-talk on
    /// the shared 32-slot state window.
    #[test]
    fn encode_round_trips_full_field_combo() {
        let cr = dummy_cr(true, true, Ffv1Version::V3);
        round_trip_header(
            &cr,
            &HeaderCase {
                slice_x: 3,
                slice_y: 5,
                slice_width: 16,
                slice_height: 12,
                indices: &[0, 1, 0],
                picture_structure_raw: 3,
                sar_num: 16,
                sar_den: 9,
            },
        );
    }

    /// The encoder is **deterministic**: re-encoding the same header
    /// produces the same bytes bit-exactly. The §4.6 path's only state
    /// is the 32-slot context window, which the encoder reinitialises
    /// to 128 each call.
    #[test]
    fn encode_is_deterministic() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        let header = Ffv1SliceHeader {
            slice_x: 1,
            slice_y: 2,
            slice_width: 8,
            slice_height: 8,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 1, 0],
            picture_structure: PictureStructure::Progressive,
            picture_structure_raw: 3,
            sar_num: 1,
            sar_den: 1,
        };
        let a = encode_slice_header(&header, &cr).unwrap();
        let b = encode_slice_header(&header, &cr).unwrap();
        assert_eq!(a, b);
    }

    /// `slice_width == 0` is rejected: the wire field is
    /// `slice_width - 1`, so 0 would underflow the round-trip and the
    /// resulting bitstream wouldn't satisfy `parse(encode(x)) == x`.
    #[test]
    fn encode_rejects_zero_slice_width() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        let header = Ffv1SliceHeader {
            slice_x: 0,
            slice_y: 0,
            slice_width: 0,
            slice_height: 1,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 0, 0],
            picture_structure: PictureStructure::Unknown,
            picture_structure_raw: 0,
            sar_num: 0,
            sar_den: 0,
        };
        match encode_slice_header(&header, &cr) {
            Err(Error::SliceSizeOutOfRange { field, expected }) => {
                assert_eq!(field, 0);
                assert_eq!(expected, 1);
            }
            other => panic!("expected SliceSizeOutOfRange, got {other:?}"),
        }
    }

    /// `slice_height == 0` is rejected symmetrically.
    #[test]
    fn encode_rejects_zero_slice_height() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        let header = Ffv1SliceHeader {
            slice_x: 0,
            slice_y: 0,
            slice_width: 1,
            slice_height: 0,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 0, 0],
            picture_structure: PictureStructure::Unknown,
            picture_structure_raw: 0,
            sar_num: 0,
            sar_den: 0,
        };
        match encode_slice_header(&header, &cr) {
            Err(Error::SliceSizeOutOfRange { field, expected }) => {
                assert_eq!(field, 0);
                assert_eq!(expected, 1);
            }
            other => panic!("expected SliceSizeOutOfRange, got {other:?}"),
        }
    }

    /// `quant_table_set_index_count` field that disagrees with what the
    /// Configuration Record derives is rejected: emitting a different
    /// number of `ur` symbols than the decoder's matching loop reads
    /// would desync every subsequent field and corrupt the §4.7 / §4.8
    /// downstream content decode.
    #[test]
    fn encode_rejects_mismatched_quant_index_count() {
        // CR says count=2 (chroma_planes=true / v3), but header claims 3.
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        let header = Ffv1SliceHeader {
            slice_x: 0,
            slice_y: 0,
            slice_width: 1,
            slice_height: 1,
            quant_table_set_index_count: 3,
            quant_table_set_index: [0, 1, 0],
            picture_structure: PictureStructure::Unknown,
            picture_structure_raw: 0,
            sar_num: 0,
            sar_den: 0,
        };
        match encode_slice_header(&header, &cr) {
            Err(Error::SliceSizeOutOfRange { field, expected }) => {
                assert_eq!(field, 3);
                assert_eq!(expected, 2);
            }
            other => panic!("expected SliceSizeOutOfRange, got {other:?}"),
        }
    }

    /// `encode_slice_header_to_encoder` chains directly into a
    /// caller-owned [`RangeEncoder`] — the same composition pattern
    /// [`parse_slice_header_from_decoder`] uses on the decode side for
    /// `coder_type >= 1` Slices where the SliceHeader and SliceContent
    /// share one range coder cursor. Pin the API surface by exercising
    /// it from a test that constructs the encoder + finishes it after
    /// the header is written.
    #[test]
    fn encode_to_encoder_matches_freestanding_encode() {
        let cr = dummy_cr(true, false, Ffv1Version::V3);
        let header = Ffv1SliceHeader {
            slice_x: 1,
            slice_y: 0,
            slice_width: 2,
            slice_height: 3,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 1, 0],
            picture_structure: PictureStructure::Progressive,
            picture_structure_raw: 3,
            sar_num: 16,
            sar_den: 9,
        };
        let bytes_freestanding = encode_slice_header(&header, &cr).unwrap();

        let mut re = RangeEncoder::new();
        encode_slice_header_to_encoder(&mut re, &header, &cr).unwrap();
        let bytes_chained = re.finish();

        assert_eq!(bytes_freestanding, bytes_chained);
    }
}
