//! FFV1 versions-0/1 single-Slice Frame decode (RFC 9043 §4.4 / §4.5 /
//! §4.7).
//!
//! Versions 0 and 1 differ structurally from version 3 in three ways
//! that make a separate driver cleaner than threading version branches
//! through the v3 [`crate::decode_frame`] path:
//!
//! 1. **Parameters are carried inline in the Frame**, not in a container
//!    Configuration Record (RFC 9043 §4.4: `if (keyframe &&
//!    !ConfigurationRecordIsPresent) Parameters()`). The §4.2 Parameters
//!    plus the single §4.1 Quantization Table Set are range-coded at the
//!    very start of the Frame, right after the §4.4 `keyframe` boolean,
//!    sharing one range-coder pass with the Slice Content (for
//!    `coder_type >= 1`).
//! 2. **There is exactly one implied Slice** covering the whole raster.
//!    §4.5 emits a `SliceHeader()` only for `version >= 3`; §4.2.11 /
//!    §4.2.12 infer `num_h_slices == num_v_slices == 1`; §4.6.1–§4.6.4
//!    infer `slice_x == slice_y == 0`, `slice_width == slice_height ==
//!    1`. The Slice's §4.7 Slice Content begins immediately after the
//!    inline §4.1 cascade — there is **no** §4.6 Slice Header, **no**
//!    §4.9 Slice Footer, and **no** §4.9.1 trailer-pointer chain.
//! 3. **No §4.9.3 per-Slice CRC and no §4.9.2 `error_status`** — the
//!    §4.2.16 `ec` field is `version >= 3`-only (absent here), so the
//!    Slice carries none of the v3 footer machinery.
//!
//! This module wires the §4.4 prologue parse
//! ([`crate::quant_table::parse_v0v1_frame_prologue`]) to the same
//! per-Plane reconstructors the v3 driver uses, over the implied single
//! Slice. The YCbCr / plane-major traversal (`colorspace_type == 0`) is
//! fully supported for the §3.8.2 Golomb-Rice (`coder_type == 0`) and
//! the §3.8.1 default-table range coder (`coder_type == 1`) paths.
//!
//! Deferred (surface explicit errors, see Limitations in the crate
//! README):
//!
//! * **`colorspace_type == 1` (RGB / RCT)** — the §4.7 line-major
//!   traversal needs the per-Plane interleave the v3 RGB driver owns;
//!   v0/v1 RGB is rare and tracked as a follow-up.
//! * **`coder_type == 2` (custom state-transition table)** — for v0/v1
//!   the `state_transition_delta` is read *inside* the same range-coder
//!   pass that then decodes the Slice Content, so the point at which the
//!   custom table takes effect mid-stream is not pinned by the RFC for
//!   the single-stream v0/v1 case (it is unambiguous for v3, where
//!   Parameters live in a separate Configuration Record pass). Surface
//!   explicitly rather than guess.

use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};
use crate::config_encode::encode_v0v1_frame_prologue;
use crate::frame::{DecodedFrame, DecodedFramePlane};
use crate::quant_table::{parse_v0v1_frame_prologue, QuantizationTableSet};
use crate::range_coder::{RangeEncoder, PARAMETERS_INITIAL_STATE};
use crate::range_encode::{RangePlaneEncoder, RangePlaneEncoderState};
use crate::range_reconstruct::RangePlaneReconstructor;
use crate::reconstruct::PlaneReconstructor;
use crate::slice_content::{compute_slice_content, FramePixelDimensions, PlaneTraversal};
use crate::slice_header::{Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES};
use crate::symbol::put_br;
use crate::Error;

/// §4.6.5: `quant_table_set_index_count = 1 + ((chroma_planes ||
/// version <= 3) ? 1 : 0) + (extra_plane ? 1 : 0)`.
///
/// For v0/v1 the `version <= 3` clause always holds, so the chroma slot
/// is always counted (even on a luma-only Frame): the count is `2 +
/// extra_plane`. This is the same value the v3 §4.6 Slice Header parser
/// derives; we compute it directly here because v0/v1 has no Slice
/// Header to read it from.
fn quant_table_set_index_count(cr: &Ffv1ConfigurationRecord) -> usize {
    1 + 1 + usize::from(cr.extra_plane)
}

/// Build the implied single-Slice §4.6 Slice Header for a v0/v1 Frame.
///
/// Every field is the §4.6.1–§4.6.6 "inferred if not present" value:
/// `slice_x == slice_y == 0`, `slice_width == slice_height == 1` (one
/// cell on the inferred 1×1 raster), every `quant_table_set_index ==
/// 0` (the single Quantization Table Set), `picture_structure == 0`
/// (unknown), `sar_num == sar_den == 0` (unknown aspect ratio).
fn implied_v0v1_slice_header(cr: &Ffv1ConfigurationRecord) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x: 0,
        slice_y: 0,
        slice_width: 1,
        slice_height: 1,
        quant_table_set_index_count: quant_table_set_index_count(cr),
        quant_table_set_index: [0u32; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: crate::config::PictureStructure::Unknown,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

/// Decode one FFV1 **version 0 or 1** YCbCr / plane-major keyframe Frame
/// end-to-end (RFC 9043 §4.4 / §4.5 / §4.7).
///
/// `frame_bytes` is the raw FFV1 Frame the container hands the codec (no
/// container framing); `frame_dims` is the container-reported pixel
/// geometry (FFV1 carries no width / height inline — see §4.2). The §4.4
/// in-Frame Parameters + the single §4.1 Quantization Table Set are
/// parsed off the Frame itself, so — unlike the v3
/// [`crate::decode_frame`] — this entry takes **no** separate
/// Configuration Record / quant-table arguments.
///
/// # Returns
///
/// A [`DecodedFrame`] whose `planes` are the reconstructed YCbCr Planes
/// (luma, optional Cb / Cr, optional alpha), `keyframe == true`, and
/// `slice_headers` carrying the single implied §4.6 Slice Header.
///
/// # Errors
///
/// * [`Error::NonKeyframeHasNoInFrameParameters`] when the Frame's §4.4
///   `keyframe` bit is `0`. A v0/v1 non-keyframe inherits the prior
///   keyframe's Parameters and Quantization Table Set; decode it with
///   [`decode_frame_v0v1_inter`], supplying the keyframe's
///   Configuration Record + Quantization Table Set.
/// * [`Error::InFrameParametersForbiddenForVersion`] when the inline
///   `version` decodes to `>= 3`.
/// * [`Error::ColorspaceLayoutNotImplemented`] for `colorspace_type == 1`
///   (RGB / RCT) — the v0/v1 line-major path is a follow-up.
/// * [`Error::UnsupportedCoderType`] for `coder_type == 2` (custom
///   state-transition table) — the single-stream mid-Parameters table
///   ordering is unpinned for v0/v1; the §3.8.1.6 default-table path
///   (`coder_type == 1`) and the §3.8.2 Golomb path (`coder_type == 0`)
///   are supported.
/// * Any error surfaced by the §4.4 prologue parse or the per-Plane
///   reconstructors.
pub fn decode_frame_v0v1(
    frame_bytes: &[u8],
    frame_dims: FramePixelDimensions,
) -> Result<DecodedFrame, Error> {
    let prologue = parse_v0v1_frame_prologue(frame_bytes)?;
    debug_assert!(prologue.keyframe);
    debug_assert!(prologue.record.version != Ffv1Version::V3);
    decode_v0v1_single_slice(
        frame_bytes,
        &prologue.record,
        &prologue.quant_table_set,
        frame_dims,
        prologue.decoder,
        true,
    )
}

/// Decode a v0/v1 **non-keyframe** Frame, reusing a keyframe's inline
/// Parameters + Quantization Table Set (RFC 9043 §4.4: a non-keyframe
/// carries no inline `Parameters()` — it inherits the most recent
/// keyframe's).
///
/// `cr` and `quant_table_set` are the Configuration Record + single
/// Quantization Table Set parsed off the governing keyframe (e.g. via
/// [`crate::quant_table::parse_v0v1_frame_prologue`]). The Frame still
/// opens with the §4.4 `keyframe` boolean (here read and required to be
/// `0`); the Slice Content then begins immediately (no inline
/// Parameters, no Slice Header).
///
/// Unlike v3 inter-Frames, the §3.8.1.3 / §3.8.2.5 per-context coder
/// state on a v0/v1 non-keyframe is **still keyframe-initialised** per
/// Slice — v0/v1 streams are self-contained per Frame at the entropy
/// level (the §3.8.1.3 carry is a v3 multithreading construct keyed off
/// the §5 stable-geometry rule). Only the Parameters / quant tables are
/// inherited, not the coder state.
///
/// # Errors
///
/// Mirrors [`decode_frame_v0v1`] except that a `keyframe == 1` bit
/// surfaces [`Error::UnexpectedKeyframeInInterDecode`] (the caller
/// mis-routed a keyframe Frame to the inter entry).
pub fn decode_frame_v0v1_inter(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    frame_dims: FramePixelDimensions,
) -> Result<DecodedFrame, Error> {
    if cr.version == Ffv1Version::V3 {
        return Err(Error::InFrameParametersForbiddenForVersion(
            cr.version.as_u32(),
        ));
    }
    // §4.4: the Frame opens with the range-coded `keyframe` boolean. For
    // a non-keyframe this is 0 and NO inline Parameters follow — the
    // Slice Content begins right after.
    let mut rc = if cr.coder_type == 2 {
        return Err(Error::UnsupportedCoderType(cr.coder_type));
    } else {
        crate::range_coder::RangeDecoder::new(frame_bytes)?
    };
    let mut kf_state = [crate::range_coder::PARAMETERS_INITIAL_STATE; 1];
    let keyframe = crate::symbol::get_br(&mut rc, &mut kf_state);
    if keyframe {
        return Err(Error::UnexpectedKeyframeInInterDecode);
    }
    decode_v0v1_single_slice(frame_bytes, cr, quant_table_set, frame_dims, rc, false)
}

/// Reconstruct the implied single Slice of a v0/v1 Frame onto fresh
/// frame-level Plane buffers, given a range decoder already positioned
/// at the start of the §4.7 Slice Content (after the §4.4 prologue's
/// `keyframe` + Parameters + §4.1 cascade for a keyframe, or after the
/// `keyframe` boolean alone for a non-keyframe).
fn decode_v0v1_single_slice(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    frame_dims: FramePixelDimensions,
    mut rc: crate::range_coder::RangeDecoder<'_>,
    keyframe: bool,
) -> Result<DecodedFrame, Error> {
    if cr.colorspace_type == ColorspaceType::Rgb {
        // §4.7 line-major (RGB / RCT) for v0/v1 is a follow-up.
        return Err(Error::ColorspaceLayoutNotImplemented);
    }
    if cr.coder_type == 2 {
        // Single-stream mid-Parameters custom-table ordering unpinned
        // for v0/v1 — surface explicitly.
        return Err(Error::UnsupportedCoderType(cr.coder_type));
    }

    let header = implied_v0v1_slice_header(cr);
    let sc = compute_slice_content(&header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

    // The single Slice covers the whole Frame, so the slice-level Plane
    // buffers ARE the frame-level Plane buffers (origin 0, 0). Build the
    // returned planes straight from the reconstruction.
    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);

    // §4.5 + §4.8: on the `coder_type == 0` path the §4.7 Slice Content
    // is a byte-aligned Golomb-Rice bit stream starting at the byte
    // boundary after the range-coded prologue. There is no Slice Header
    // here, so the cursor is exactly the prologue decoder's byte
    // position. The bit reader persists across Planes (Plane p+1 reads
    // where Plane p stopped — same contract as the v3 driver).
    let mut golomb_bit_reader = if cr.coder_type == 0 {
        let consumed = rc.position();
        if consumed > frame_bytes.len() {
            return Err(Error::TruncatedRangeCoder);
        }
        Some(crate::bit_reader::BitReader::new(&frame_bytes[consumed..]))
    } else {
        None
    };

    // §4.6.6: per-context state is keyed by the §4.6.6 slot (luma → 0,
    // both chroma planes → 1, extra plane → last). v0/v1 has a single
    // Quantization Table Set so every slot resolves to it, but the
    // per-slot coder state is still distinct per category (luma vs
    // chroma vs alpha), exactly as the v3 driver allocates.
    let slot_count = header.quant_table_set_index_count;
    let mut per_slot_range_state: Vec<Option<crate::range_reconstruct::RangePlaneState>> =
        (0..slot_count).map(|_| None).collect();
    let mut per_slot_golomb_state: Vec<Option<crate::reconstruct::PlaneEntropyState>> =
        (0..slot_count).map(|_| None).collect();

    let bits = cr.bits_per_raw_sample;
    // §3.3.1 alternate 16-bit median predictor is YCbCr-only and only on
    // the range-coder paths (the v3 driver gates the same way).
    let use_16bit_median =
        cr.bits_per_raw_sample == 16 && (cr.coder_type == 1 || cr.coder_type == 2);

    let mut planes: Vec<DecodedFramePlane> = Vec::with_capacity(primary_color_count);
    for (p_idx, plane) in sc.planes.iter().enumerate() {
        let qts_index_slot = match p_idx {
            0 => 0usize,
            1 | 2 if cr.chroma_planes => 1,
            _ if cr.extra_plane => slot_count.saturating_sub(1),
            _ => 0,
        };

        let reconstructed: Vec<i32> = match cr.coder_type {
            0 => {
                let br = golomb_bit_reader
                    .as_mut()
                    .expect("golomb_bit_reader is Some when cr.coder_type == 0");
                let state = per_slot_golomb_state[qts_index_slot].get_or_insert_with(|| {
                    crate::reconstruct::PlaneEntropyState::new(
                        quant_table_set.context_count as usize,
                    )
                });
                PlaneReconstructor::reconstruct_plane_with_state(
                    br,
                    state,
                    &quant_table_set.tables,
                    plane.width as usize,
                    plane.height as usize,
                    bits,
                )
            }
            1 => {
                let state = per_slot_range_state[qts_index_slot].get_or_insert_with(|| {
                    crate::range_reconstruct::RangePlaneState::new(
                        quant_table_set.context_count as usize,
                    )
                });
                RangePlaneReconstructor::reconstruct_plane_with_state(
                    &mut rc,
                    state,
                    &quant_table_set.tables,
                    plane.width as usize,
                    plane.height as usize,
                    bits,
                    use_16bit_median,
                )
            }
            other => return Err(Error::UnsupportedCoderType(other)),
        };

        planes.push(DecodedFramePlane {
            plane_index: plane.plane_index,
            width: plane.width,
            height: plane.height,
            samples: reconstructed,
        });
    }

    Ok(DecodedFrame {
        planes,
        width: frame_dims.width,
        height: frame_dims.height,
        bits_per_raw_sample: cr.bits_per_raw_sample,
        colorspace: cr.colorspace_type,
        keyframe,
        slice_headers: vec![header],
    })
}

/// Encode one FFV1 **version 0 or 1** YCbCr / plane-major keyframe Frame
/// end-to-end (RFC 9043 §4.4 / §4.5 / §4.7), the symmetric inverse of
/// [`decode_frame_v0v1`].
///
/// `frame` carries the Planes to encode (`primary_color_count` of them,
/// in §4.7.1 order); `cr` is the §4.2 Parameters to embed inline (its
/// `version` MUST be 0 or 1, `colorspace_type` YCbCr, `coder_type` 1 —
/// the §3.8.1 default-table range coder); `quant_table_set` is the single
/// §4.1 Quantization Table Set (`quant_table_set_count == 1` for v0/v1).
///
/// The produced bytes are a complete v0/v1 Frame: the §4.4 `keyframe`
/// boolean (`1`), the inline §4.2 Parameters + §4.1 cascade, then the
/// implied single Slice's §4.7 Slice Content. The range coder is closed
/// once at the end (one continuous Closed-mode pass over the whole
/// Frame). `decode_frame_v0v1` reconstructs `frame.planes` bit-exactly
/// from the output.
///
/// # Errors
///
/// * [`Error::InFrameParametersForbiddenForVersion`] when `cr.version ==
///   V3`.
/// * [`Error::ColorspaceLayoutNotImplemented`] for RGB (`colorspace_type
///   == 1`).
/// * [`Error::UnsupportedCoderType`] for any `coder_type != 1` (the
///   Golomb `coder_type == 0` and custom-table `coder_type == 2` v0/v1
///   encode paths are follow-ups).
/// * Any error surfaced by the prologue encoder
///   ([`Error::MalformedQuantTable`], etc.).
pub fn encode_frame_v0v1(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
) -> Result<Vec<u8>, Error> {
    encode_frame_v0v1_keyframe(frame, cr, quant_table_set, true)
}

/// Encode one FFV1 **version 0 or 1** non-keyframe YCbCr Frame, reusing
/// the governing keyframe's inline §4.2 Parameters + §4.1 Quantization
/// Table Set (RFC 9043 §4.4: a non-keyframe carries no inline
/// `Parameters()`).
///
/// The produced Frame opens with the §4.4 `keyframe` boolean set to `0`,
/// then the implied single Slice's §4.7 Slice Content begins immediately
/// (no inline Parameters, no Slice Header). The §3.8.1.3 per-context
/// coder state is keyframe-initialised per Frame (v0/v1 streams are
/// entropy-self-contained per Frame — see [`decode_frame_v0v1_inter`]).
///
/// [`decode_frame_v0v1_inter`] (supplied the same `cr` + `quant_table_set`)
/// recovers `frame.planes` bit-exactly.
///
/// # Errors
///
/// Mirrors [`encode_frame_v0v1`].
pub fn encode_frame_v0v1_inter(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
) -> Result<Vec<u8>, Error> {
    encode_frame_v0v1_keyframe(frame, cr, quant_table_set, false)
}

fn encode_frame_v0v1_keyframe(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    keyframe: bool,
) -> Result<Vec<u8>, Error> {
    if cr.version == Ffv1Version::V3 {
        return Err(Error::InFrameParametersForbiddenForVersion(
            cr.version.as_u32(),
        ));
    }
    if cr.colorspace_type == ColorspaceType::Rgb {
        return Err(Error::ColorspaceLayoutNotImplemented);
    }
    if cr.coder_type != 1 {
        // Only the §3.8.1 default-table range coder is wired on the
        // v0/v1 encode side this round (the Golomb encode path reuses a
        // row-by-row driver, and the custom-table ordering is unpinned).
        return Err(Error::UnsupportedCoderType(cr.coder_type));
    }

    let frame_dims = FramePixelDimensions::new(frame.width, frame.height)?;
    let header = implied_v0v1_slice_header(cr);
    let sc = compute_slice_content(&header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

    let mut re = RangeEncoder::new();
    // §4.4: the Frame opens with the range-coded `keyframe` boolean (its
    // own 1-slot state window at 128).
    let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
    put_br(&mut re, &mut kf_state, keyframe);

    // §4.4 prologue: inline §4.2 Parameters + the single §4.1 cascade, on
    // the same range-coder pass — keyframe only (a non-keyframe inherits
    // the governing keyframe's Parameters).
    if keyframe {
        encode_v0v1_frame_prologue(&mut re, cr, quant_table_set)?;
    }

    // §4.7 plane-major Slice Content on the same continuous range-coder
    // pass. Per-context state is keyed by the §4.6.6 slot (luma / chroma /
    // extra), shared across Planes routed through the same slot — the
    // exact mirror of `decode_v0v1_single_slice`.
    let slot_count = header.quant_table_set_index_count;
    let mut per_slot_state: Vec<Option<RangePlaneEncoderState>> =
        (0..slot_count).map(|_| None).collect();
    let bits = cr.bits_per_raw_sample;
    let use_16bit_median = cr.bits_per_raw_sample == 16;

    for (p_idx, plane) in sc.planes.iter().enumerate() {
        let qts_index_slot = match p_idx {
            0 => 0usize,
            1 | 2 if cr.chroma_planes => 1,
            _ if cr.extra_plane => slot_count.saturating_sub(1),
            _ => 0,
        };
        let frame_plane = frame
            .planes
            .get(p_idx)
            .ok_or(Error::InvalidQuantTableSetCount(p_idx as u32))?;
        let plane_w = plane.width as usize;
        let plane_h = plane.height as usize;
        if frame_plane.samples.len() != plane_w * plane_h {
            return Err(Error::InvalidFramePixelDimensions {
                width: frame.width,
                height: frame.height,
            });
        }
        let state = per_slot_state[qts_index_slot].get_or_insert_with(|| {
            RangePlaneEncoderState::new(quant_table_set.context_count as usize)
        });
        RangePlaneEncoder::encode_plane_with_state(
            &mut re,
            state,
            &quant_table_set.tables,
            &frame_plane.samples,
            plane_w,
            plane_h,
            bits,
            use_16bit_median,
        );
    }

    Ok(re.finish())
}
