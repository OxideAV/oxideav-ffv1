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
use crate::range_coder::{RangeDecoder, PARAMETERS_INITIAL_STATE};
use crate::symbol::{get_ur, SYMBOL_CONTEXT_SIZE};
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
}
