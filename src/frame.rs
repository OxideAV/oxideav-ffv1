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
//! ## Scope
//!
//! - **YCbCr / plane-major (`colorspace_type == 0`)** is supported for
//!   both coder paths. Each Plane is reconstructed in one
//!   [`PlaneReconstructor`] / [`RangePlaneReconstructor`] call and the
//!   §4.7 outer-`for p` inner-`for y` traversal falls out naturally.
//! - **RGB / line-major (`colorspace_type == 1`)** is handled by the
//!   sibling driver [`crate::decode_frame_rgb`]
//!   (`crate::rgb_reconstruct`): the §4.7 pseudocode for RGB interleaves
//!   Lines between Planes row-by-row, so it keeps the per-plane entropy
//!   state external to the plane reconstructor. *This* driver surfaces
//!   RGB input as [`Error::ColorspaceLayoutNotImplemented`] so a caller
//!   routes to the RGB driver explicitly (the registry decoder does this
//!   on §4.2.5 `colorspace_type`).
//! - **CRC coverage.** RFC 9043 defines exactly two CRCs: the §4.3.2
//!   `configuration_record_crc_parity` (validated at record parse) and
//!   the §4.9.3 per-Slice `slice_crc_parity` (validated per §4.9 footer
//!   when `ec == 1`, policy-gated via [`DecodeOptions`]). There is no
//!   frame-level CRC in the specification.
//! - **`ec` flag.** The §4.2.16 `ec` field is parsed onto
//!   [`Ffv1ConfigurationRecord::ec`]; the driver still accepts `ec` as
//!   an explicit boolean parameter (the historical signature) so callers
//!   that obtained it elsewhere keep working — the registry derives it
//!   from the parsed record (`cr.ec != 0`).

use crate::Error;
use crate::bit_reader::BitReader;
use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};
use crate::frame_assembly::{FrameJob, SliceDecoded, SlicePiece, SliceUnit, slice_range_decoder};
use crate::quant_table::QuantizationTableSet;
use crate::range_reconstruct::RangePlaneReconstructor;
use crate::reconstruct::PlaneReconstructor;
use crate::slice_content::{FramePixelDimensions, PlaneTraversal, compute_slice_content};
use crate::slice_footer::{
    SliceCrcPolicy, SliceErrorStatus, SliceErrorStatusPolicy, parse_slice_footer_with_options,
};
use crate::slice_header::parse_slice_header_from_decoder;

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
#[non_exhaustive]
pub struct DecodeOptions {
    /// Per-Slice CRC policy (RFC 9043 §4.9.3). Defaults to
    /// [`SliceCrcPolicy::Reject`].
    pub slice_crc_policy: SliceCrcPolicy,
    /// Per-Slice §4.9.2 `error_status` policy (Table 16). Defaults
    /// to [`SliceErrorStatusPolicy::Reject`] — a Slice whose footer
    /// declares the `Uncorrectable` (`2`) status aborts the frame
    /// decode via [`Error::SliceErrorStatus`].
    pub slice_error_status_policy: SliceErrorStatusPolicy,
    /// §3.8.1.1.1 range-coder termination gate (r411). Defaults to
    /// [`SliceTerminationPolicy::Accept`] (the historical behaviour —
    /// no check). [`SliceTerminationPolicy::Reject`] additionally
    /// requires every v3 range-coded Slice body to end with the
    /// Sentinel-mode terminator at exactly the Slice body length.
    pub termination_policy: SliceTerminationPolicy,
    /// Thread budget for Slice-parallel decode (RFC 9043 §4.5 Slices
    /// are independently coded). `0` and `1` both mean **serial** — the
    /// default; no thread is spawned. A larger value lets the Frame
    /// drivers decode up to that many Slices at once on scoped threads.
    /// The decoded Frame, the carried inter-Frame state and the reported
    /// error are identical at every budget.
    pub threads: usize,
    /// What to do with a Slice the gates above reject or whose content
    /// cannot be decoded. Defaults to
    /// [`SliceConcealment::Off`](crate::SliceConcealment::Off) — the
    /// Frame decode fails. See [`crate::SliceConcealment`].
    pub concealment: crate::frame_assembly::SliceConcealment,
}

/// Policy for the §3.8.1.1.1 range-coder termination check on v3
/// range-coded Slices (RFC 9043: "the end of range-coded Slices, which
/// need to terminate before the CRC at their end").
///
/// Under [`Reject`][Self::Reject] the decoder, after reconstructing a
/// Slice's last Plane, reads the §3.8.1.1.1 sentinel (the discarded
/// state-129 symbol) and requires the recovered end position to equal
/// the Slice body length — the same bookkeeping a conforming decoder
/// uses to flag Slice damage. A conforming stream (including every
/// stream this crate's encoder emits since r411) passes; a truncated /
/// padded body, or one flushed without the terminator, surfaces
/// [`Error::SliceTerminationMismatch`]. The check applies to the
/// range-coded Slice-body case only (`coder_type >= 1`); a Golomb-Rice
/// Slice's byte-aligned bit tail has no §3.8.1.1.1 terminator of its
/// own (its header↔content boundary sentinel is already mandatory for
/// decode).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SliceTerminationPolicy {
    /// Do not check termination (historical default; also tolerates
    /// streams from this crate's pre-r411 encoder, which flushed
    /// without the terminator).
    #[default]
    Accept,
    /// Require the §3.8.1.1.1 Sentinel-mode terminator to land exactly
    /// on the Slice body length.
    Reject,
}

impl DecodeOptions {
    /// Set the per-Slice §4.9.3 CRC policy.
    pub fn with_slice_crc_policy(mut self, v: SliceCrcPolicy) -> Self {
        self.slice_crc_policy = v;
        self
    }

    /// Set the per-Slice §4.9.2 `error_status` policy.
    pub fn with_slice_error_status_policy(mut self, v: SliceErrorStatusPolicy) -> Self {
        self.slice_error_status_policy = v;
        self
    }

    /// Set the §3.8.1.1.1 range-coder termination policy.
    pub fn with_termination_policy(mut self, v: SliceTerminationPolicy) -> Self {
        self.termination_policy = v;
        self
    }

    /// Grant a thread budget for Slice-parallel decode (`0` / `1` =
    /// serial, the default).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Choose how damaged Slices are handled (default: fail the Frame).
    pub fn with_concealment(mut self, v: crate::frame_assembly::SliceConcealment) -> Self {
        self.concealment = v;
        self
    }

    /// Strict decode (the default): every Slice's §4.9.3 CRC residue
    /// must be zero AND no Slice may declare the §4.9.2 Table 16
    /// `Uncorrectable` status. Either failure aborts the frame
    /// decode. Equivalent to [`DecodeOptions::default`]. The
    /// §3.8.1.1.1 termination gate stays
    /// [`SliceTerminationPolicy::Accept`] (opt-in — see
    /// [`DecodeOptions::pedantic`]) so archived streams from this
    /// crate's pre-r411 encoder keep decoding.
    pub fn strict() -> Self {
        Self {
            slice_crc_policy: SliceCrcPolicy::Reject,
            slice_error_status_policy: SliceErrorStatusPolicy::Reject,
            termination_policy: SliceTerminationPolicy::Accept,
            ..Self::default()
        }
    }

    /// Everything [`DecodeOptions::strict`] checks, plus the
    /// §3.8.1.1.1 range-coder termination gate on every v3 range-coded
    /// Slice.
    pub fn pedantic() -> Self {
        Self {
            slice_crc_policy: SliceCrcPolicy::Reject,
            slice_error_status_policy: SliceErrorStatusPolicy::Reject,
            termination_policy: SliceTerminationPolicy::Reject,
            ..Self::default()
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
            termination_policy: SliceTerminationPolicy::Accept,
            ..Self::default()
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
    /// The §4.4 `keyframe` field decoded off the Frame's first Slice.
    ///
    /// RFC 9043 §4.4 opens each Frame with a single range-coded
    /// `keyframe` boolean (its own state, initial value 128). `true`
    /// marks an independently decodable keyframe; `false` a
    /// non-keyframe whose §5 third-paragraph restriction requires its
    /// Slice geometry to match a Slice of the previous Frame. The
    /// frame-level drivers are single-Frame and stateless, so they
    /// surface this value (instead of discarding it) to let a caller
    /// drive the §5 multi-Frame
    /// [`crate::SliceGeometryStabilityTracker`] across a coded Frame
    /// sequence and enforce the §4.2.18 `intra` keyframe constraint.
    pub keyframe: bool,
    /// The Frame's §4.6 Slice Headers in forward Slice-index order
    /// (slice 0 first — the §4.9.1 trailer-chain order both decode
    /// drivers walk).
    ///
    /// These are the headers the §5 second-paragraph raster-coverage
    /// gate already parses (and validates) before any per-Slice pixel
    /// reconstruction starts; surfacing them costs nothing extra. They
    /// are exactly the *other* argument — alongside [`Self::keyframe`]
    /// — that the RFC 9043 §5 third-paragraph multi-Frame rule needs:
    /// a caller walks Frames in coded order and feeds
    /// `tracker.observe_frame(decoded.keyframe, &decoded.slice_headers)`
    /// into a [`crate::SliceGeometryStabilityTracker`] to enforce
    /// "each Slice MUST have the same value of slice_x, slice_y,
    /// slice_width, and slice_height as a Slice in the previous
    /// Frame" on every non-keyframe. A zero-Slice Frame carries no
    /// headers (empty vector). On the encode side this field is
    /// ignored — the encoders derive Slice geometry from the
    /// Configuration Record's slice grid, never from this field.
    pub slice_headers: Vec<crate::slice_header::Ffv1SliceHeader>,
}

/// Per-Slice carried coder state for inter-Frame (non-keyframe) decode
/// (RFC 9043 §3.8.1.3 / §3.8.2.5).
///
/// RFC 9043 §3.8.1.3 (range coder) and §3.8.2.5 (Golomb-Rice VLC) both
/// read: "When the keyframe value (see Section 4.4) is **1**, all
/// [range coder / VLC] state variables are set to their initial state."
/// The keyframe-`0` (non-keyframe) case is the negation: the per-context
/// coder state is **not** re-initialised — it **continues from the value
/// it held at the end of the previous Frame's matching Slice**. (The
/// §3.8.2.2.1 run-mode triple `run_index` / `run_mode` / `run_count` is a
/// separate concern: it "is reset to zero for each Plane and Slice"
/// unconditionally, keyframe or not, and is handled inside the per-Plane
/// reconstructors.)
///
/// RFC 9043 §5 third paragraph guarantees a non-keyframe's Slices have
/// "the same value of slice_x, slice_y, slice_width, and slice_height as
/// a Slice in the previous Frame", so the Slices line up positionally
/// across Frames and the carry is indexed by forward Slice index
/// (slice 0 first — the §4.9.1 trailer-chain order). Per Slice the carry
/// holds one entry per §4.6.6 `quant_table_set_index` slot (luma → 0,
/// the shared chroma slot → 1, the optional extra-plane slot → 2): the
/// same per-slot buffers the single-Frame driver evolves within a Frame,
/// snapshotted at end-of-Frame so the next non-keyframe can resume them.
///
/// A caller does not construct this directly. [`Ffv1DecodeSession`] owns
/// one across a coded Frame sequence; the carry-aware driver entry point
/// [`decode_frame_with_carry`] reads the previous Frame's snapshot (when
/// the current Frame is a non-keyframe) and writes the current Frame's
/// snapshot back out.
///
/// [`Ffv1DecodeSession`]: crate::Ffv1DecodeSession
#[derive(Debug, Clone, Default)]
pub struct Ffv1FrameCarry {
    /// One entry per forward Slice index; each is the per-slot coder
    /// state that Slice held at end-of-Frame.
    slices: Vec<SliceCarry>,
    /// The Frame this carry was produced by, kept only under
    /// [`SliceConcealment::PreviousFrame`](crate::SliceConcealment) as
    /// the concealment source for the next Frame.
    previous_planes: Option<std::sync::Arc<Vec<DecodedFramePlane>>>,
    /// Raster rectangles the producing decode concealed.
    concealed: Vec<crate::frame_assembly::ConcealedRegion>,
}

/// One Slice's end-of-Frame per-slot coder state.
///
/// The two vectors are slot-indexed (`quant_table_set_index` slot);
/// exactly one of the two is populated per slot depending on
/// `coder_type` (range-coder slots fill `range`, Golomb-Rice slots fill
/// `golomb`). A slot a Slice never touched stays `None`.
#[derive(Debug, Clone, Default)]
struct SliceCarry {
    range: Vec<Option<crate::range_reconstruct::RangePlaneState>>,
    golomb: Vec<Option<crate::reconstruct::PlaneEntropyState>>,
    /// The Slice was concealed: it has no end-of-Frame state, so a
    /// following non-keyframe cannot continue it (§3.8.1.3 / §3.8.2.5).
    lost: bool,
}

impl Ffv1FrameCarry {
    /// A carry holding no Slice snapshots — the keyframe / first-Frame
    /// state. Both [`crate::decode_frame_with_carry`] and the RGB /
    /// line-major [`crate::rgb_reconstruct::decode_frame_rgb_with_carry`]
    /// pre-size it per Slice as they snapshot.
    pub(crate) fn with_slice_capacity(slice_count: usize) -> Self {
        Self {
            slices: Vec::with_capacity(slice_count),
            previous_planes: None,
            concealed: Vec::new(),
        }
    }

    /// The raster rectangles (in full-resolution Frame pixels) that the
    /// decode which produced this carry **concealed** instead of
    /// decoding — empty for an intact Frame, and always empty when
    /// [`DecodeOptions::concealment`] is off.
    pub fn concealed_regions(&self) -> &[crate::frame_assembly::ConcealedRegion] {
        &self.concealed
    }

    pub(crate) fn set_concealed_regions(&mut self, v: Vec<crate::frame_assembly::ConcealedRegion>) {
        self.concealed = v;
    }

    /// Number of forward Slice positions this carry describes.
    pub(crate) fn slice_count(&self) -> usize {
        self.slices.len()
    }

    /// Whether Slice `slice_index` was concealed in the producing Frame.
    pub(crate) fn is_lost(&self, slice_index: usize) -> bool {
        self.slices.get(slice_index).is_some_and(|s| s.lost)
    }

    /// Size the carry to `count` Slice positions, all marked lost;
    /// [`Self::set_slice`] then fills in the ones that decoded.
    pub(crate) fn fill_lost(&mut self, count: usize) {
        self.slices = (0..count)
            .map(|_| SliceCarry {
                range: Vec::new(),
                golomb: Vec::new(),
                lost: true,
            })
            .collect();
    }

    /// Record Slice `slice_index`'s end-of-Frame state.
    pub(crate) fn set_slice(
        &mut self,
        slice_index: usize,
        range: Vec<Option<crate::range_reconstruct::RangePlaneState>>,
        golomb: Vec<Option<crate::reconstruct::PlaneEntropyState>>,
    ) {
        if let Some(slot) = self.slices.get_mut(slice_index) {
            *slot = SliceCarry {
                range,
                golomb,
                lost: false,
            };
        }
    }

    pub(crate) fn previous_planes(&self) -> Option<&[DecodedFramePlane]> {
        self.previous_planes.as_deref().map(Vec::as_slice)
    }

    pub(crate) fn set_previous_planes(&mut self, planes: std::sync::Arc<Vec<DecodedFramePlane>>) {
        self.previous_planes = Some(planes);
    }

    /// The per-slot range-coder state carried for forward Slice index
    /// `slice_index`, or an empty slice when no snapshot exists yet
    /// (first Frame, or a Frame with fewer Slices than this one).
    pub(crate) fn range_for(
        &self,
        slice_index: usize,
    ) -> &[Option<crate::range_reconstruct::RangePlaneState>] {
        self.slices
            .get(slice_index)
            .map(|s| s.range.as_slice())
            .unwrap_or(&[])
    }

    /// The per-slot Golomb-Rice VLC state carried for forward Slice
    /// index `slice_index`, or an empty slice when no snapshot exists.
    pub(crate) fn golomb_for(
        &self,
        slice_index: usize,
    ) -> &[Option<crate::reconstruct::PlaneEntropyState>] {
        self.slices
            .get(slice_index)
            .map(|s| s.golomb.as_slice())
            .unwrap_or(&[])
    }

    /// Append one Slice's end-of-Frame per-slot coder state, in forward
    /// Slice-index order. Slices must be pushed in order (slice 0 first)
    /// so [`Self::range_for`] / [`Self::golomb_for`] index correctly.
    pub(crate) fn push_slice(
        &mut self,
        range: Vec<Option<crate::range_reconstruct::RangePlaneState>>,
        golomb: Vec<Option<crate::reconstruct::PlaneEntropyState>>,
    ) {
        self.slices.push(SliceCarry {
            range,
            golomb,
            lost: false,
        });
    }
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
    // No carry: every Slice's per-context coder state is keyframe-
    // initialised (the historical single-Frame contract). Equivalent to
    // a standalone keyframe with no previous Frame.
    let mut carry: Option<Ffv1FrameCarry> = None;
    decode_frame_with_carry(
        frame_bytes,
        cr,
        quant_table_sets,
        frame_dims,
        ec,
        options,
        &mut carry,
    )
}

/// Decode one FFV1 v3 YCbCr / plane-major frame, carrying the §3.8.1.3 /
/// §3.8.2.5 per-context coder state across non-keyframes (RFC 9043
/// §3.8.1.3, §3.8.2.5, §5 third paragraph).
///
/// This is the inter-Frame variant of [`decode_frame_with_options`]. The
/// `carry` argument is the cross-Frame coder-state channel:
///
/// * **On entry**, when the decoded Frame's §4.4 `keyframe` value is `0`
///   (a non-keyframe) and `*carry` is `Some(prev)`, each Slice's
///   per-slot coder state resumes from `prev`'s matching forward Slice
///   index instead of the §3.8.1.3 / §3.8.2.5 `128`-initialised window.
///   This is the negation of "When the keyframe value is 1, all state
///   variables are set to their initial state": on a non-keyframe the
///   state is *not* reset, it continues from the previous Frame.
/// * When the Frame's `keyframe` value is `1` (a keyframe), the carry on
///   entry is ignored — every state buffer is freshly `128`-initialised
///   per §3.8.1.3 / §3.8.2.5, exactly as the single-Frame driver does.
/// * **On return**, `*carry` is overwritten with this Frame's
///   end-of-Frame per-Slice per-slot state snapshot, ready for the next
///   non-keyframe in coded order.
///
/// The §3.8.2.2.1 run-mode triple (`run_index` / `run_mode` /
/// `run_count`) is reset per Plane / Slice unconditionally inside the
/// per-Plane reconstructors — only the per-context VLC / range-coder
/// windows (the "state variables" §3.8.1.3 / §3.8.2.5 name) participate
/// in the carry.
///
/// Callers wanting the historical stateless behaviour use
/// [`decode_frame`] / [`decode_frame_with_options`]; the cross-Frame
/// session [`crate::Ffv1DecodeSession`] owns the `carry` across a coded
/// stream.
pub fn decode_frame_with_carry(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame_dims: FramePixelDimensions,
    ec: bool,
    options: DecodeOptions,
    carry: &mut Option<Ffv1FrameCarry>,
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

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    let plane_dims: Vec<(u32, u32)> = (0..primary_color_count)
        .map(|p| frame_plane_dims(frame_dims, p as u8, cr))
        .collect();
    crate::frame_assembly::decode_frame_slices(
        frame_bytes,
        cr,
        quant_table_sets,
        frame_dims,
        ec,
        options,
        carry,
        &plane_dims,
        decode_plane_major_slice,
    )
}

/// Decode one plane-major (YCbCr) Slice: §4.9 footer gates, §4.6 header,
/// then each Plane's §4.7 content in order (RFC 9043 §4.5).
///
/// The Slice is self-contained: it reads only its own bytes, the
/// Frame-wide `job` inputs, and (on a non-keyframe) its own entry of the
/// previous Frame's carried state — which is what lets the Frame driver
/// run Slices concurrently.
fn decode_plane_major_slice(job: &FrameJob<'_>, unit: SliceUnit) -> Result<SliceDecoded, Error> {
    let cr = job.cr;
    let options = job.options;
    let slice_index = unit.index;
    let slice_bytes = &job.frame_bytes[unit.extent.start..unit.extent.end()];

    // §4.9 footer: the §4.9.1 size cross-check always applies; the
    // §4.9.3 CRC residue is gated by `options.slice_crc_policy`.
    let footer = parse_slice_footer_with_options(slice_bytes, job.ec, options.slice_crc_policy)?;
    // §4.9.2 `error_status` Table 16 gate.
    if matches!(
        options.slice_error_status_policy,
        SliceErrorStatusPolicy::Reject
    ) && matches!(footer.error_status, Some(SliceErrorStatus::Uncorrectable))
    {
        return Err(Error::SliceErrorStatus {
            slice_index: slice_index as u32,
            status: footer.error_status_raw.unwrap_or(2),
        });
    }

    let body = &slice_bytes[..slice_bytes.len() - crate::trailer_chain::slice_footer_len(job.ec)];
    // One range decoder over the body: the §4.6 Slice Header always,
    // and the §4.7 Slice Content when `coder_type >= 1` (§3.8.1.6
    // custom transition table for `coder_type == 2`).
    let mut rc = slice_range_decoder(body, cr)?;
    // RFC 9043 §4.4: the Frame's first Slice opens with the range-coded
    // `keyframe` bit (already read by the Frame planner; skip it here so
    // the header parse starts at the right cursor).
    if unit.opens_frame {
        let mut kf_state = [crate::range_coder::PARAMETERS_INITIAL_STATE; 1];
        let _ = crate::symbol::get_br(&mut rc, &mut kf_state);
    }
    let header = parse_slice_header_from_decoder(&mut rc, cr)?;
    // RFC 9043 §5 per-Slice size cap above the CIF trigger.
    crate::slice_content::validate_slice_max_size_restriction(&header, cr, job.frame_dims)?;
    let sc = compute_slice_content(&header, cr, job.frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

    // §4.5 + §4.8: with `coder_type == 0` the Slice Content is one
    // contiguous Golomb-Rice bit stream for all Planes, starting where
    // the Sentinel-mode terminator of the range-coded header ends
    // (§3.8.1.1.1).
    let mut golomb_bit_reader = if cr.coder_type == 0 {
        let consumed = rc.terminate_sentinel();
        if consumed > body.len() {
            return Err(Error::TruncatedRangeCoder);
        }
        Some(BitReader::new(&body[consumed..]))
    } else {
        None
    };

    // RFC 9043 §3.8.1.3 / §3.8.2.5 + §4.6.6: context state is selected
    // by `quant_table_set_index` slot (luma 0, both chroma Planes 1, the
    // extra Plane last) and is re-initialised only on a keyframe; on a
    // non-keyframe it continues from the previous Frame's same Slice.
    let slot_count = header.quant_table_set_index_count;
    let (mut per_slot_range_state, mut per_slot_golomb_state) = if job.keyframe {
        (
            (0..slot_count).map(|_| None).collect::<Vec<_>>(),
            (0..slot_count).map(|_| None).collect::<Vec<_>>(),
        )
    } else {
        let prev_range = job.prev_carry.range_for(slice_index);
        let prev_golomb = job.prev_carry.golomb_for(slice_index);
        (
            (0..slot_count)
                .map(|s| prev_range.get(s).cloned().flatten())
                .collect::<Vec<_>>(),
            (0..slot_count)
                .map(|s| prev_golomb.get(s).cloned().flatten())
                .collect::<Vec<_>>(),
        )
    };

    let bits = cr.bits_per_raw_sample;
    // §3.3.1: 16-bit YCbCr under the range coder uses the alternate
    // (sign-reinterpreting) median.
    let use_16bit_median = bits == 16 && (cr.coder_type == 1 || cr.coder_type == 2);
    let mut pieces = Vec::with_capacity(sc.planes.len());
    for (p_idx, plane) in sc.planes.iter().enumerate() {
        let qts_index_slot = match p_idx {
            0 => 0usize,
            1 | 2 if cr.chroma_planes => 1,
            _ if cr.extra_plane => header.quant_table_set_index_count.saturating_sub(1),
            _ => 0,
        };
        let qts_index = (header.quant_table_set_index[qts_index_slot] as usize)
            .min(job.quant_table_sets.len().saturating_sub(1));
        let qts = job
            .quant_table_sets
            .get(qts_index)
            .ok_or(Error::InvalidQuantTableSetCount(0))?;

        let samples: Vec<i32> = match cr.coder_type {
            0 => {
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
                let state = per_slot_range_state[qts_index_slot].get_or_insert_with(|| {
                    // §4.2.15 seeds (`states_coded == 1`) replace the
                    // §3.8.1.3 all-128 initialisation.
                    crate::range_reconstruct::RangePlaneState::seeded(
                        qts.context_count as usize,
                        job.initial_states[qts_index].as_deref(),
                    )
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

        let (origin_x, origin_y) =
            plane_origin(sc.slice_pixel_x, sc.slice_pixel_y, plane.plane_index, cr);
        pieces.push(SlicePiece {
            plane: p_idx,
            origin_x: origin_x as usize,
            origin_y: origin_y as usize,
            width: plane.width as usize,
            height: plane.height as usize,
            samples,
        });
    }

    // RFC 9043 §3.8.1.1.1 (opt-in): a conforming range-coded Slice ends
    // with the Sentinel-mode terminator exactly at the body length.
    if options.termination_policy == SliceTerminationPolicy::Reject && cr.coder_type != 0 {
        let recovered_end = rc.terminate_sentinel();
        if recovered_end != body.len() {
            return Err(Error::SliceTerminationMismatch {
                slice_index: slice_index as u32,
                recovered_end: recovered_end as u32,
                body_len: body.len() as u32,
            });
        }
    }

    Ok(SliceDecoded {
        pieces,
        range: per_slot_range_state,
        golomb: per_slot_golomb_state,
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

    #[test]
    fn decoded_frame_carries_keyframe_field() {
        // The §4.4 `keyframe` value is surfaced on `DecodedFrame`
        // (rather than discarded after the §4.4 read) so a caller can
        // drive the §5 multi-Frame stability checks. Lock the field's
        // presence + both polarities at the struct level.
        let mut f = DecodedFrame {
            planes: Vec::new(),
            width: 4,
            height: 4,
            bits_per_raw_sample: 8,
            colorspace: ColorspaceType::YCbCr,
            keyframe: true,
            slice_headers: Vec::new(),
        };
        assert!(f.keyframe);
        f.keyframe = false;
        assert!(!f.keyframe);
    }

    #[test]
    fn decoded_frame_keyframe_drives_section5_stability_tracker() {
        // The intended downstream wiring: a caller pulls `keyframe`
        // AND `slice_headers` off each decoded Frame (both surfaced
        // on `DecodedFrame`) and feeds the pair into the §5
        // third-paragraph `SliceGeometryStabilityTracker`. Here we
        // simulate two decoded Frames: a keyframe that opens the
        // stream and a conforming non-keyframe whose Slice geometry
        // matches the keyframe's. The tracker accepts both, proving
        // the surfaced pair is exactly the `observe_frame` input §5
        // needs — no out-of-band header re-parse required.
        let headers = vec![crate::slice_header::Ffv1SliceHeader {
            slice_x: 0,
            slice_y: 0,
            slice_width: 1,
            slice_height: 1,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 0, 0],
            picture_structure: crate::config::PictureStructure::Progressive,
            picture_structure_raw: 3,
            sar_num: 0,
            sar_den: 0,
        }];

        let keyframe_decoded = DecodedFrame {
            planes: Vec::new(),
            width: 4,
            height: 4,
            bits_per_raw_sample: 8,
            colorspace: ColorspaceType::YCbCr,
            keyframe: true,
            slice_headers: headers,
        };
        let inter_decoded = DecodedFrame {
            keyframe: false,
            ..keyframe_decoded.clone()
        };

        let mut tracker = crate::SliceGeometryStabilityTracker::new();
        tracker
            .observe_frame(keyframe_decoded.keyframe, &keyframe_decoded.slice_headers)
            .expect("keyframe opens the stream without a §5 check");
        tracker
            .observe_frame(inter_decoded.keyframe, &inter_decoded.slice_headers)
            .expect("non-keyframe with matching geometry conforms to §5");
        assert!(tracker.has_previous_frame());
    }
}
