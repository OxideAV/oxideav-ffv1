//! FFV1 frame-level decode driver (RFC 9043 §4.5, §4.7, §4.8, §4.9).
//!
//! Round 129 wires together every per-stage parser the prior rounds
//! landed (§4.2 Configuration Record, §4.1 Quantization Tables, §4.9.1
//! trailer-pointer chain walker, §4.9 Slice Footer, §4.6 Slice Header,
//! §4.7 Slice Content, §3.1 / §3.3 / §3.5 / §3.7 / §3.8 per-plane
//! reconstruction) into a single end-to-end driver that takes a raw
//! FFV1 frame payload plus the per-stream metadata and emits a typed
//! reconstructed [`DecodedFrame`].
//!
//! ## Driver pipeline
//!
//! ```text
//!   raw frame bytes + Configuration Record + Quant Tables + ec
//!        │
//!        ▼
//!   walk_trailer_chain (§4.9.1)            → Vec<SliceExtent>
//!        │  (one extent per Slice, in slice-index order)
//!        ▼
//!   for each Slice extent:
//!        │
//!        ├── parse_slice_footer (§4.9)     ← validates §4.9.1 size
//!        │                                   field + §4.9.3 CRC
//!        ├── RangeDecoder over body bytes  (§3.8.1)
//!        ├── parse_slice_header_from_decoder
//!        │                                  (§4.6, in-place on `rc`)
//!        ├── compute_slice_content          (§4.7 layout)
//!        ├── route on `coder_type`:
//!        │     0 → byte-align, BitReader, PlaneReconstructor (§3.8.2)
//!        │     1 or 2 → RangePlaneReconstructor on the same `rc`
//!        │              (§3.8.1.2)
//!        └── copy each reconstructed Plane into the frame-level
//!            DecodedFrame at the slice's pixel-space origin
//! ```
//!
//! The driver hands every byte-level decision to the existing per-stage
//! modules — its job is plumbing, not parsing. The §3.1 borders,
//! §3.3 median predictor, §3.5 context table, §3.8 modular add-back,
//! and per-context state windows all live in `crate::reconstruct` and
//! `crate::range_reconstruct`.
//!
//! ## Scope and limitations
//!
//! - **YCbCr / plane-major (`colorspace_type == 0`)** is supported for
//!   both coder paths. Each Plane is reconstructed in one
//!   [`PlaneReconstructor`] / [`RangePlaneReconstructor`] call and the
//!   §4.7 outer-`for p` inner-`for y` traversal falls out naturally.
//! - **RGB / line-major (`colorspace_type == 1`)** is not yet wired up:
//!   the §4.7 pseudocode for RGB interleaves Lines between Planes
//!   row-by-row, which would require a row-by-row driver (the per-plane
//!   entropy state would need to live outside the plane reconstructor).
//!   The driver surfaces this as [`Error::ColorspaceLayoutNotImplemented`]
//!   so callers handle it explicitly. The per-plane reconstructors
//!   themselves are colorspace-agnostic; only the iteration order
//!   differs.
//! - **Frame-level CRC.** §4.5 mentions a per-frame `frame_crc_parity`
//!   when `ec == 1` and not in slice-CRC mode; this driver runs against
//!   the per-Slice CRC mode (`ec == 1 && slicecrc == 1` per the
//!   v3-default fixture), which is what every fixture uses. A whole-
//!   frame CRC would be a separate wiring point.
//! - **`ec` flag.** The Configuration Record's `error_correction` field
//!   is not yet decoded by [`parse_configuration_record`]
//!   (`docs/video/ffv1/spec/...` deferred this with `initial_state_delta`
//!   / `intra` to a later round), so the driver accepts `ec` as an
//!   explicit boolean parameter the caller obtains from a black-box
//!   extractor (e.g. the trace, or a separate ec-only parser).
//!
//! [`parse_configuration_record`]: crate::config::parse_configuration_record

use crate::bit_reader::BitReader;
use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};
use crate::quant_table::QuantizationTableSet;
use crate::range_coder::RangeDecoder;
use crate::range_reconstruct::RangePlaneReconstructor;
use crate::reconstruct::PlaneReconstructor;
use crate::slice_content::{compute_slice_content, FramePixelDimensions, PlaneTraversal};
use crate::slice_footer::{
    parse_slice_footer_with_options, SliceCrcPolicy, SliceErrorStatus, SliceErrorStatusPolicy,
};
use crate::slice_header::parse_slice_header_from_decoder;
use crate::trailer_chain::walk_trailer_chain;
use crate::Error;

/// Frame-decoder runtime options (RFC 9043 §4.9 integrity gates).
///
/// Two policies are gated independently per the two §4.9 fields that
/// signal Slice-level damage:
///
/// * [`slice_crc_policy`][Self::slice_crc_policy] — the §4.9.3
///   whole-Slice CRC residue gate (round 238).
/// * [`slice_error_status_policy`][Self::slice_error_status_policy] —
///   the §4.9.2 `error_status` Table 16 gate (this round).
///
/// The default values match the historical [`decode_frame`] /
/// [`decode_frame_rgb`] behaviour every prior round shipped: any §4.9.3
/// residue must be zero (otherwise [`Error::SliceCrcMismatch`]) AND no
/// Slice may declare the §4.9.2 Table 16 `Uncorrectable` (`2`) status
/// (otherwise [`Error::SliceErrorStatus`]). Use
/// [`DecodeOptions::lenient`] to opt both gates into partial-recovery
/// decode; each gate also has its own field that callers can flip
/// independently when they need a mixed policy
/// (e.g. accept §4.9.3 residue mismatches but still abort on §4.9.2
/// `Uncorrectable`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeOptions {
    /// Per-Slice CRC policy (RFC 9043 §4.9.3). Defaults to
    /// [`SliceCrcPolicy::Reject`].
    pub slice_crc_policy: SliceCrcPolicy,
    /// Per-Slice §4.9.2 `error_status` policy (Table 16). Defaults
    /// to [`SliceErrorStatusPolicy::Reject`] — a Slice whose footer
    /// declares the `Uncorrectable` (`2`) status aborts the frame
    /// decode via [`Error::SliceErrorStatus`].
    pub slice_error_status_policy: SliceErrorStatusPolicy,
}

impl DecodeOptions {
    /// Strict decode (the default): every Slice's §4.9.3 CRC residue
    /// must be zero AND no Slice may declare the §4.9.2 Table 16
    /// `Uncorrectable` status. Either failure aborts the frame
    /// decode. Equivalent to [`DecodeOptions::default`].
    pub fn strict() -> Self {
        Self {
            slice_crc_policy: SliceCrcPolicy::Reject,
            slice_error_status_policy: SliceErrorStatusPolicy::Reject,
        }
    }

    /// Lenient decode: per-Slice §4.9.3 CRC failures AND per-Slice
    /// §4.9.2 `Uncorrectable` status declarations do not abort the
    /// frame decode. Per-Slice structural failures (truncation, size
    /// mismatch) and per-Slice content-level errors (range-coder
    /// truncation, raster-out-of-range slice header, etc.) still
    /// abort. This is the partial-recovery mode — best-effort pixels
    /// on a damaged frame.
    pub fn lenient() -> Self {
        Self {
            slice_crc_policy: SliceCrcPolicy::Accept,
            slice_error_status_policy: SliceErrorStatusPolicy::Accept,
        }
    }
}

/// One Plane of a fully-reconstructed FFV1 frame.
///
/// `samples` is row-major (`samples[y * width + x]`); each entry lands
/// in `0 .. 2^bits_per_raw_sample`. Plane dimensions are the
/// **frame-level** width and height for the plane (accounting for
/// chroma subsampling) — the per-slice plane regions are stitched into
/// this single buffer in pixel-space at the slice's `slice_pixel_x` /
/// `slice_pixel_y` origin.
#[derive(Debug, Clone)]
pub struct DecodedFramePlane {
    /// `p` from RFC 9043 §4.7 — plane index inside the frame
    /// (`0..primary_color_count`).
    pub plane_index: u8,
    /// Frame-level plane width in pixels. Equal to
    /// `frame.width` for plane 0 and the extra plane;
    /// `ceil(frame.width / 2^log2_h_chroma_subsample)` for chroma
    /// planes when `chroma_planes == true`.
    pub width: u32,
    /// Frame-level plane height in pixels. Mirrors
    /// [`Self::width`] over the vertical subsample.
    pub height: u32,
    /// Row-major Plane samples (`width * height` entries, each in
    /// `0 .. 2^bits_per_raw_sample`).
    pub samples: Vec<i32>,
}

/// Reconstructed multi-plane FFV1 frame.
///
/// The `planes` vector length equals `primary_color_count` per
/// RFC 9043 §4.7.1: `1 + (chroma_planes ? 2 : 0) + (extra_plane ? 1 :
/// 0)`. Plane `0` is luma / R / gray; planes `1` / `2` are chroma when
/// `chroma_planes` is set; the trailing extra plane (alpha) is present
/// when `extra_plane` is set.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Reconstructed planes in `primary_color_count` order
    /// (§4.7.1). Index 0 is luma / R / gray; planes 1 / 2 are chroma
    /// when present; the trailing entry is the extra plane (alpha)
    /// when `extra_plane == true`.
    pub planes: Vec<DecodedFramePlane>,
    /// Frame width in pixels (the value the caller supplied as
    /// [`FramePixelDimensions::width`]).
    pub width: u32,
    /// Frame height in pixels (the value the caller supplied as
    /// [`FramePixelDimensions::height`]).
    pub height: u32,
    /// Bits per raw Sample (§4.2.7) — every entry in every plane's
    /// `samples` is in `0 .. 2^bits_per_raw_sample`.
    pub bits_per_raw_sample: u32,
    /// `colorspace_type` (§4.2.5) — informational; the driver always
    /// returns plane-major-ordered planes regardless of colorspace.
    pub colorspace: ColorspaceType,
}

/// Decode one FFV1 v3 frame end-to-end (RFC 9043 §4.5 + §4.7 + §4.8 +
/// §4.9).
///
/// This is the round-129 wiring driver: it walks the §4.9.1
/// trailer-pointer chain, parses each Slice's §4.9 footer + §4.6
/// header, computes the §4.7 plane layout, and routes per-plane
/// reconstruction through [`PlaneReconstructor`] (`coder_type == 0`)
/// or [`RangePlaneReconstructor`] (`coder_type == 1 || coder_type ==
/// 2`). Each per-slice reconstruction is copied into the appropriate
/// pixel-space rectangle inside the frame-level Plane buffers.
///
/// # Parameters
///
/// * `frame_bytes` — raw FFV1 v3 frame payload (the bytes a container
///   demuxer hands the codec, with no container framing). The §4.9.1
///   trailer chain MUST end exactly at `frame_bytes.len()`.
/// * `cr` — the parsed Configuration Record (§4.2). Carries `version`,
///   `coder_type`, `colorspace_type`, `bits_per_raw_sample`,
///   `chroma_planes`, `extra_plane`, the chroma subsampling, and the
///   slice raster (`num_h_slices` / `num_v_slices`).
/// * `quant_table_sets` — the parsed §4.1 Quantization Table Sets,
///   in stream order (index `i` corresponds to
///   `quant_table_set_index[..] == i` in slice headers).
/// * `frame_dims` — the surrounding container's reported pixel
///   dimensions (FFV1's Configuration Record does NOT carry frame
///   width/height; see §4.2 Parameters).
/// * `ec` — the `error_correction` flag the Configuration Record
///   carries in fields not yet decoded by
///   [`parse_configuration_record`]. The caller obtains this from a
///   black-box source (the trace, the container metadata, or a
///   future ec-only parser).
///
/// # Returns
///
/// A [`DecodedFrame`] whose `planes` are fully reconstructed at the
/// frame-level resolution (chroma-subsampled where appropriate).
///
/// # Errors
///
/// * [`Error::SliceRequiresVersion3`] when `cr.version != V3`. v0/v1
///   frames carry the slice grid in the per-keyframe header, not the
///   Configuration Record — not yet supported.
/// * [`Error::ColorspaceLayoutNotImplemented`] for
///   `colorspace_type == 1` (RGB) — the §4.7 row-interleaved
///   traversal is not yet wired (the per-plane reconstructors run a
///   whole Plane at a time; line-major would need a row-by-row
///   variant that keeps per-Plane entropy state external).
/// * [`Error::UnsupportedCoderType`] for `coder_type` values outside
///   `0..=2` (the Configuration Record parser already filters this).
/// * Any error surfaced by the per-stage parsers
///   ([`Error::TruncatedSliceFooter`], [`Error::SliceSizeOutOfRange`],
///   [`Error::SliceCrcMismatch`], [`Error::SliceRasterOutOfRange`],
///   [`Error::TruncatedRangeCoder`], ...).
pub fn decode_frame(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame_dims: FramePixelDimensions,
    ec: bool,
) -> Result<DecodedFrame, Error> {
    decode_frame_with_options(
        frame_bytes,
        cr,
        quant_table_sets,
        frame_dims,
        ec,
        DecodeOptions::default(),
    )
}

/// Decode one FFV1 v3 frame end-to-end with an explicit
/// [`DecodeOptions`] gate (RFC 9043 §4.9.3 per-Slice CRC policy).
///
/// Same parameters and behaviour as [`decode_frame`] except that the
/// per-Slice §4.9.3 CRC residue check is governed by
/// `options.slice_crc_policy`:
///
/// * [`SliceCrcPolicy::Reject`] (default) — aborts the frame decode
///   with [`Error::SliceCrcMismatch`] on a non-zero residue.
/// * [`SliceCrcPolicy::Accept`] — non-zero residues are tolerated and
///   the driver proceeds to reconstruct the per-Slice Planes
///   best-effort. All other per-Slice errors (truncation, header
///   parse failure, range-coder underflow, etc.) still abort.
///
/// This is the partial-recovery entry point — useful for damaged FFV1
/// frames where the §4.9.2 `error_status` field signals a known
/// corruption and the caller wants to render whatever Planes the
/// per-Slice decoders manage to reconstruct (typical of forensic
/// recovery, archival restoration, or stream-replay diagnostics).
///
/// # Errors
///
/// Same as [`decode_frame`], except that
/// [`Error::SliceCrcMismatch`] is suppressed when
/// `options.slice_crc_policy == SliceCrcPolicy::Accept`.
pub fn decode_frame_with_options(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame_dims: FramePixelDimensions,
    ec: bool,
    options: DecodeOptions,
) -> Result<DecodedFrame, Error> {
    if cr.version != Ffv1Version::V3 {
        // v0/v1 frames carry the slice grid in the per-keyframe header,
        // not the Configuration Record; this driver targets v3 only.
        return Err(Error::SliceRequiresVersion3);
    }
    if cr.colorspace_type == ColorspaceType::Rgb {
        // §4.7 RGB path is row-interleaved between Planes; the per-Plane
        // reconstructors run a full Plane at a time, so wiring RGB
        // requires a separate row-by-row driver. Surface explicitly so
        // callers don't get silent wrong output.
        return Err(Error::ColorspaceLayoutNotImplemented);
    }

    // Walk the §4.9.1 trailer-pointer chain. Returns forward-ordered
    // extents (slice 0 first).
    let extents = walk_trailer_chain(frame_bytes, ec)?;

    // Pre-allocate the frame-level Plane buffers. Their dimensions
    // depend on the colorspace + chroma subsampling; we use the same
    // per-plane width/height math the per-slice layout uses, but
    // applied to the frame dimensions rather than the slice dimensions.
    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    let mut planes: Vec<DecodedFramePlane> = (0..primary_color_count)
        .map(|p| {
            let p_u8 = p as u8;
            let (w, h) = frame_plane_dims(frame_dims, p_u8, cr);
            DecodedFramePlane {
                plane_index: p_u8,
                width: w,
                height: h,
                samples: vec![0i32; w as usize * h as usize],
            }
        })
        .collect();

    let footer_len = if ec {
        crate::slice_footer::SLICE_FOOTER_LEN_EC1
    } else {
        crate::slice_footer::SLICE_FOOTER_LEN_EC0
    };

    for (slice_index, ext) in extents.iter().enumerate() {
        let slice_bytes = &frame_bytes[ext.start..ext.end()];
        // §4.9 footer validation: cross-check the §4.9.1 size + (if
        // ec=1) the §4.9.3 whole-Slice CRC. The size cross-check
        // always aborts on mismatch; the §4.9.3 CRC residue check is
        // gated by `options.slice_crc_policy` per RFC 9043 §4.9.3
        // (`Reject` aborts via `Error::SliceCrcMismatch`, `Accept`
        // tolerates a non-zero residue for partial-recovery decode).
        let footer = parse_slice_footer_with_options(slice_bytes, ec, options.slice_crc_policy)?;

        // §4.9.2 `error_status` Table 16 gate: when the encoder
        // declared the Slice `Uncorrectable` (`2`) and the active
        // policy is `Reject`, abort the frame decode before any
        // per-Slice pixel reconstruction touches this slice's body.
        // Other Table 16 values are not rejection targets per the
        // policy doc (`NoError` / `Correctable` / `Reserved`).
        if matches!(
            options.slice_error_status_policy,
            SliceErrorStatusPolicy::Reject
        ) && matches!(footer.error_status, Some(SliceErrorStatus::Uncorrectable))
        {
            let raw = footer.error_status_raw.unwrap_or(2);
            return Err(Error::SliceErrorStatus {
                slice_index: slice_index as u32,
                status: raw,
            });
        }

        // The body (everything before the footer) is where the range
        // coder reads the §4.6 SliceHeader and (for `coder_type >= 1`)
        // the §4.8 SliceContent. For `coder_type == 0` the body's tail
        // is Golomb-Rice bits after a byte-alignment step.
        let body_end = slice_bytes.len() - footer_len;
        let body = &slice_bytes[..body_end];

        // Construct the range coder over the body. Used for the
        // SliceHeader (always) and for SliceContent when
        // `coder_type >= 1`. For `coder_type == 2` the Configuration
        // Record's `state_transition_delta[1..=255]` (RFC 9043 §3.8.1.4
        // Figure 22) is layered onto the default to derive the active
        // `one_state` table per §3.8.1.6; `coder_type == 0` and `1` use
        // the default table directly.
        let mut rc = if cr.coder_type == 2 {
            let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
            RangeDecoder::with_one_state(body, &one_state)?
        } else {
            RangeDecoder::new(body)?
        };

        // RFC 9043 §4.4: a Frame opens with a single range-coded
        // `keyframe` boolean ("has its own initial state, set to 128")
        // that lives at the very start of the FIRST Slice's range-coded
        // region — before that Slice's §4.6 header. Subsequent Slices
        // are independently range-coded and carry no keyframe bit. Read
        // and discard it for slice 0 so the header (and the content that
        // shares this decoder) stay byte-synchronised.
        if slice_index == 0 {
            let mut kf_state = [crate::range_coder::PARAMETERS_INITIAL_STATE; 1];
            let _keyframe = crate::symbol::get_br(&mut rc, &mut kf_state);
        }

        // §4.6 SliceHeader on the same range decoder.
        let header = parse_slice_header_from_decoder(&mut rc, cr)?;
        // RFC 9043 §5 "Restrictions" — on v3 Frames above the §5
        // trigger (`frame_pixel_width * frame_pixel_height > 101376`,
        // i.e. larger than CIF) each Slice's raster footprint must
        // satisfy `slice_width * slice_height <= num_h_slices *
        // num_v_slices / 4`. The §5 cap is the per-thread quota for
        // four-way parallel decoding (the spec text spells out the
        // motive: "to ensure that fast multithreaded decoding is
        // possible"). Validate before the §4.7 layout math so a
        // violation aborts the Frame without pulling the per-Plane
        // reconstructors onto the offending Slice's body.
        crate::slice_content::validate_slice_max_size_restriction(&header, cr, frame_dims)?;
        let sc = compute_slice_content(&header, cr, frame_dims)?;

        debug_assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

        // §4.5 + §4.8: on the `coder_type == 0` path the §4.8
        // SliceContent is a single contiguous Golomb-Rice bit stream
        // starting at the byte boundary after the range-coded
        // SliceHeader. We construct one [`BitReader`] up-front and let
        // each per-Plane [`PlaneReconstructor::reconstruct_plane`] call
        // advance its cursor — the per-Plane §3.8.2.2.1 state reset is
        // handled inside `reconstruct_plane` (fresh
        // [`PlaneEntropyState`] per call), but the bit-stream cursor
        // must persist across Planes so Plane `p+1` reads its bits from
        // where Plane `p` stopped. Prior to round 208 this was a fresh
        // `BitReader` per Plane — correct for the single-Plane
        // grayscale fixtures but wrong for `chroma_planes == true`
        // YCbCr Slices (and for `extra_plane == true`), where Plane 1 /
        // Plane 2 / Plane 3 silently re-read Plane 0's bytes from
        // offset zero.
        let golomb_bit_reader = if cr.coder_type == 0 {
            let consumed = rc.position();
            if consumed > body.len() {
                return Err(Error::TruncatedRangeCoder);
            }
            Some(BitReader::new(&body[consumed..]))
        } else {
            None
        };
        let mut golomb_bit_reader = golomb_bit_reader;

        // RFC 9043 §3.8.1.3 / §3.8.2.5 + §4.6.6: per-context state
        // is keyframe-initialised AND **selected by
        // `quant_table_set_index`** — §4.6.6 reads "indicates the
        // Quantization Table Set index to select the Quantization
        // Table Set **and the initial states** for the Slice
        // Content". §4.6.5 defines
        //
        //     quant_table_set_index_count =
        //         1 + ((chroma_planes || version <= 3) ? 1 : 0)
        //           + (extra_plane ? 1 : 0)
        //
        // so the *slot* (= plane category: luma → 0, both chroma
        // planes → 1, extra plane → 2 if present) determines which
        // state buffer a Plane reads. The chroma slot is shared by Cb
        // and Cr in the §4.7 plane-then-line traversal — they decode
        // back-to-back against the same persistent state, with the
        // §3.8.2.2.1 run-mode triple resetting at the start of each
        // Plane but the per-context VLC / range-coder windows
        // surviving the Plane boundary. The reference trace
        // (`v3-default/trace.txt`) labels both `U` and `V` planes
        // with `plane_index=1`, matching this slot-keyed model.
        //
        // We allocate one entry per slot (0..quant_table_set_index_count),
        // lazily filled on first use of each slot so the storage
        // matches the keyframe-initialisation contract. The slot is
        // the §4.6.6 selector; the *resolved* quant table set (which
        // multiple slots can alias onto a single declared set) drives
        // the table arithmetic and `context_count` sizing, exactly as
        // before.
        let slot_count = header.quant_table_set_index_count;
        let mut per_slot_range_state: Vec<Option<crate::range_reconstruct::RangePlaneState>> =
            (0..slot_count).map(|_| None).collect();
        let mut per_slot_golomb_state: Vec<Option<crate::reconstruct::PlaneEntropyState>> =
            (0..slot_count).map(|_| None).collect();

        // Resolve the per-plane §4.1 quantization tables this slice
        // selected. §4.6.5 says quant_table_set_index_count is bounded
        // by `1 + (chroma||v<=3 ? 1 : 0) + (extra ? 1 : 0)`, i.e. up
        // to 3 — fewer than primary_color_count when chroma planes
        // share a table or when no extra plane is present.
        for (p_idx, plane) in sc.planes.iter().enumerate() {
            // §4.6.6: the i-th quant_table_set_index entry applies to
            // the i-th *category* of planes — luma uses entry 0;
            // chroma planes share entry 1 (so plane 1 and 2 both use
            // it); the extra plane uses entry 2 when present.
            let qts_index_slot = match p_idx {
                0 => 0usize,
                1 | 2 if cr.chroma_planes => 1,
                _ if cr.extra_plane => header.quant_table_set_index_count.saturating_sub(1),
                _ => 0,
            };
            let qts_index = (header.quant_table_set_index[qts_index_slot] as usize)
                .min(quant_table_sets.len().saturating_sub(1));
            let qts = quant_table_sets
                .get(qts_index)
                .ok_or(Error::InvalidQuantTableSetCount(0))?;

            let bits = cr.bits_per_raw_sample;
            let use_16bit_median = cr.colorspace_type == ColorspaceType::YCbCr
                && cr.bits_per_raw_sample == 16
                && (cr.coder_type == 1 || cr.coder_type == 2);

            let reconstructed: Vec<i32> = match cr.coder_type {
                0 => {
                    // Golomb-Rice path: the §4.8 SliceContent is a
                    // single byte-aligned bit stream starting after the
                    // range-coded SliceHeader. The §3.8.2.2.1 per-Plane
                    // reset applies ONLY to the run-mode triple, NOT
                    // to the per-context VLC state (which is §3.8.2.5
                    // keyframe-initialised and persists across Planes
                    // routed through the same §4.6.6 slot). The
                    // bit-stream cursor must also persist across
                    // Planes — see round 208 fix.
                    let br = golomb_bit_reader
                        .as_mut()
                        .expect("golomb_bit_reader is Some when cr.coder_type == 0");
                    let state = per_slot_golomb_state[qts_index_slot].get_or_insert_with(|| {
                        crate::reconstruct::PlaneEntropyState::new(qts.context_count as usize)
                    });
                    PlaneReconstructor::reconstruct_plane_with_state(
                        br,
                        state,
                        &qts.tables,
                        plane.width as usize,
                        plane.height as usize,
                        bits,
                    )
                }
                1 | 2 => {
                    // Range-coder path: the per-slot state buffer is
                    // keyframe-initialised (§3.8.1.3) and evolves
                    // through the Slice. Planes routed through the
                    // same §4.6.6 slot (Cb + Cr on every
                    // `chroma_planes == true` Slice) continue
                    // evolution.
                    let state = per_slot_range_state[qts_index_slot].get_or_insert_with(|| {
                        crate::range_reconstruct::RangePlaneState::new(qts.context_count as usize)
                    });
                    RangePlaneReconstructor::reconstruct_plane_with_state(
                        &mut rc,
                        state,
                        &qts.tables,
                        plane.width as usize,
                        plane.height as usize,
                        bits,
                        use_16bit_median,
                    )
                }
                other => return Err(Error::UnsupportedCoderType(other)),
            };

            // Copy the per-slice reconstruction into the frame Plane
            // at the slice's pixel-space origin (scaled for chroma
            // subsampling).
            let (origin_x, origin_y) =
                plane_origin(sc.slice_pixel_x, sc.slice_pixel_y, plane.plane_index, cr);
            let dst_plane = &mut planes[p_idx];
            let dst_w = dst_plane.width as usize;
            let dst_h = dst_plane.height as usize;
            blit_into(
                &mut dst_plane.samples,
                dst_w,
                dst_h,
                origin_x as usize,
                origin_y as usize,
                &reconstructed,
                plane.width as usize,
                plane.height as usize,
            );
        }
    }

    Ok(DecodedFrame {
        planes,
        width: frame_dims.width,
        height: frame_dims.height,
        bits_per_raw_sample: cr.bits_per_raw_sample,
        colorspace: cr.colorspace_type,
    })
}

/// Frame-level per-plane dimensions: full frame width/height for luma /
/// extra; chroma-subsampled for planes 1 and 2 when `chroma_planes`.
///
/// Mirrors the per-slice `plane_pixel_width` / `plane_pixel_height`
/// math in `slice_content` but applied at frame scale.
fn frame_plane_dims(
    frame: FramePixelDimensions,
    plane_index: u8,
    cr: &Ffv1ConfigurationRecord,
) -> (u32, u32) {
    if cr.chroma_planes && (plane_index == 1 || plane_index == 2) {
        let hshift = cr.log2_h_chroma_subsample;
        let vshift = cr.log2_v_chroma_subsample;
        let hdenom = 1u32 << hshift;
        let vdenom = 1u32 << vshift;
        (
            frame.width.saturating_add(hdenom - 1) / hdenom,
            frame.height.saturating_add(vdenom - 1) / vdenom,
        )
    } else {
        (frame.width, frame.height)
    }
}

/// Per-plane pixel origin: the slice's pixel origin scaled by the
/// per-plane subsample factor for chroma planes.
fn plane_origin(
    slice_pixel_x: u32,
    slice_pixel_y: u32,
    plane_index: u8,
    cr: &Ffv1ConfigurationRecord,
) -> (u32, u32) {
    if cr.chroma_planes && (plane_index == 1 || plane_index == 2) {
        let hshift = cr.log2_h_chroma_subsample;
        let vshift = cr.log2_v_chroma_subsample;
        (slice_pixel_x >> hshift, slice_pixel_y >> vshift)
    } else {
        (slice_pixel_x, slice_pixel_y)
    }
}

/// Copy a `src` rectangle (`src_w * src_h`) into the destination Plane
/// buffer (`dst_w * dst_h`) at origin `(ox, oy)`. Out-of-range writes
/// are silently clipped — a slice that overlaps the right / bottom
/// edge by one pixel (rounding boundary) gets the in-range columns
/// copied and the overshoot ignored, which mirrors how the reference
/// decoder treats the §4.7.3 / §4.8.2 rounding boundary.
#[allow(clippy::too_many_arguments)]
fn blit_into(
    dst: &mut [i32],
    dst_w: usize,
    dst_h: usize,
    ox: usize,
    oy: usize,
    src: &[i32],
    src_w: usize,
    src_h: usize,
) {
    for y in 0..src_h {
        let dy = oy + y;
        if dy >= dst_h {
            break;
        }
        let copy_w = src_w.min(dst_w.saturating_sub(ox));
        if copy_w == 0 {
            break;
        }
        let dst_off = dy * dst_w + ox;
        let src_off = y * src_w;
        dst[dst_off..dst_off + copy_w].copy_from_slice(&src[src_off..src_off + copy_w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_cr_yuv420() -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            version: Ffv1Version::V3,
            micro_version: Some(4),
            coder_type: 1,
            state_transition_delta: [0; crate::config::NUM_TRANSITION_DELTAS],
            colorspace_type: ColorspaceType::YCbCr,
            bits_per_raw_sample: 8,
            chroma_planes: true,
            log2_h_chroma_subsample: 1,
            log2_v_chroma_subsample: 1,
            extra_plane: false,
            num_h_slices: Some(2),
            num_v_slices: Some(2),
            quant_table_set_count: Some(2),
            ec: Some(0),
            intra: Some(false),
            initial_state_delta: None,
        }
    }

    fn dummy_cr_gray() -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            chroma_planes: false,
            extra_plane: false,
            ..dummy_cr_yuv420()
        }
    }

    #[test]
    fn frame_plane_dims_yuv420_64x48() {
        let cr = dummy_cr_yuv420();
        let f = FramePixelDimensions::new(64, 48).unwrap();
        assert_eq!(frame_plane_dims(f, 0, &cr), (64, 48));
        assert_eq!(frame_plane_dims(f, 1, &cr), (32, 24));
        assert_eq!(frame_plane_dims(f, 2, &cr), (32, 24));
    }

    #[test]
    fn frame_plane_dims_gray_keeps_full_size() {
        let cr = dummy_cr_gray();
        let f = FramePixelDimensions::new(64, 48).unwrap();
        assert_eq!(frame_plane_dims(f, 0, &cr), (64, 48));
    }

    #[test]
    fn frame_plane_dims_extra_plane_full_size() {
        let mut cr = dummy_cr_yuv420();
        cr.extra_plane = true;
        let f = FramePixelDimensions::new(64, 48).unwrap();
        // Plane 3 (alpha) runs at full frame resolution.
        assert_eq!(frame_plane_dims(f, 3, &cr), (64, 48));
    }

    #[test]
    fn frame_plane_dims_odd_widths_round_up() {
        let cr = dummy_cr_yuv420();
        let f = FramePixelDimensions::new(63, 47).unwrap();
        assert_eq!(frame_plane_dims(f, 0, &cr), (63, 47));
        // ceil(63/2)=32, ceil(47/2)=24.
        assert_eq!(frame_plane_dims(f, 1, &cr), (32, 24));
    }

    #[test]
    fn plane_origin_chroma_halved() {
        let cr = dummy_cr_yuv420();
        // YUV420: chroma origins are half of luma origins (rounded down
        // by the shift).
        assert_eq!(plane_origin(64, 48, 0, &cr), (64, 48));
        assert_eq!(plane_origin(64, 48, 1, &cr), (32, 24));
        assert_eq!(plane_origin(64, 48, 2, &cr), (32, 24));
    }

    #[test]
    fn blit_into_full_rect() {
        let mut dst = vec![0i32; 16];
        let src = vec![1, 2, 3, 4];
        // 4x4 dst, blit 2x2 src at (1, 1).
        blit_into(&mut dst, 4, 4, 1, 1, &src, 2, 2);
        assert_eq!(dst[5..7], [1, 2]);
        assert_eq!(dst[9..11], [3, 4]);
        assert_eq!(dst[0..4], [0; 4]); // first row untouched
    }

    #[test]
    fn blit_into_clips_right_overshoot() {
        let mut dst = vec![0i32; 9];
        let src = vec![1, 2, 3, 4, 5, 6]; // 3x2
                                          // dst is 3x3; blit src at (2, 0) — only 1 column fits per row.
        blit_into(&mut dst, 3, 3, 2, 0, &src, 3, 2);
        assert_eq!(dst[2], 1);
        assert_eq!(dst[5], 4);
    }

    #[test]
    fn blit_into_clips_bottom_overshoot() {
        let mut dst = vec![0i32; 6]; // 3x2
        let src = vec![1, 2, 3, 4, 5, 6]; // 3x2
                                          // Origin (0, 1) → only the second dst row gets src's first row.
        blit_into(&mut dst, 3, 2, 0, 1, &src, 3, 2);
        assert_eq!(dst[3..6], [1, 2, 3]);
    }

    #[test]
    fn decode_frame_rejects_v0_config_record() {
        let mut cr = dummy_cr_yuv420();
        cr.version = Ffv1Version::V0;
        let result = decode_frame(
            &[0u8; 100],
            &cr,
            &[],
            FramePixelDimensions::new(64, 48).unwrap(),
            true,
        );
        assert!(matches!(result, Err(Error::SliceRequiresVersion3)));
    }

    #[test]
    fn decode_frame_rejects_rgb_with_layout_error() {
        let mut cr = dummy_cr_yuv420();
        cr.colorspace_type = ColorspaceType::Rgb;
        let result = decode_frame(
            &[0u8; 100],
            &cr,
            &[],
            FramePixelDimensions::new(64, 48).unwrap(),
            true,
        );
        assert!(matches!(result, Err(Error::ColorspaceLayoutNotImplemented)));
    }

    #[test]
    fn decode_frame_propagates_truncated_footer() {
        // Frame shorter than a single 8-byte ec=1 footer → trailer chain
        // walk fails.
        let cr = dummy_cr_yuv420();
        let result = decode_frame(
            &[0u8; 4],
            &cr,
            &[],
            FramePixelDimensions::new(64, 48).unwrap(),
            true,
        );
        assert!(matches!(result, Err(Error::TruncatedSliceFooter)));
    }

    #[test]
    fn decoded_frame_planes_have_correct_shape_after_alloc() {
        // We can't drive a real decode without a real fixture (which
        // the integration tests cover), but we can validate the
        // pre-allocation shape via `frame_plane_dims` — every plane
        // buffer must hold exactly `width * height` entries before
        // any per-slice copy.
        let cr = dummy_cr_yuv420();
        let f = FramePixelDimensions::new(64, 48).unwrap();
        // Pre-allocate the planes the same way `decode_frame` does.
        let primary_color_count =
            1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
        for p in 0..primary_color_count {
            let (w, h) = frame_plane_dims(f, p as u8, &cr);
            let len = w as usize * h as usize;
            // Plane 0 = 64*48 = 3072; planes 1, 2 = 32*24 = 768 each.
            match p {
                0 => assert_eq!(len, 3072),
                1 | 2 => assert_eq!(len, 768),
                _ => unreachable!(),
            }
        }
    }
}
