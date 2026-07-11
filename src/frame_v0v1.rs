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
//! Slice. Both colour layouts (`colorspace_type == 0` YCbCr / plane-major
//! and `colorspace_type == 1` RGB / line-major RCT) are supported for all
//! three §4.2.3 coders: §3.8.2 Golomb-Rice (`coder_type == 0`), §3.8.1
//! default-table range coder (`coder_type == 1`), and §3.8.1.6
//! custom-table range coder (`coder_type == 2`).
//!
//! **`coder_type == 2` single-stream table ordering (RFC 9043 §4.4 /
//! §4.2.4 / §3.8.1.6).** Unlike v3 — where the §4.2 Parameters live in a
//! separate §4.3 Configuration Record range-coder pass and each §4.5
//! Slice opens a fresh range coder seeded with the §3.8.1.6 custom table —
//! v0/v1 carries the Parameters inline and shares one continuous
//! range-coder pass with the §4.7 Slice Content. The §4.2.4
//! `state_transition_delta` cannot apply to the very symbols that define
//! them, so the keyframe boolean + §4.2 Parameters + §4.1 cascade are read
//! with the §3.8.1.5 *default* table; once the deltas are known the
//! decoder swaps onto the §3.8.1.6 custom table
//! ([`crate::range_coder::RangeDecoder::set_one_state`]) at the
//! Parameters → Slice-Content boundary. A non-keyframe (no inline
//! Parameters) is seeded with the custom table from the start, exactly as
//! the v3 driver seeds each Slice. The transition table only governs how
//! a context's probability byte evolves, never the `low` / `range`
//! arithmetic-window mechanics, so the mid-pass swap is well-defined and
//! leaves the byte cursor untouched — the encode side
//! ([`encode_frame_v0v1`]) performs the symmetric swap, and self
//! round-trip is bit-exact.

use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};
use crate::config_encode::encode_v0v1_frame_prologue;
use crate::frame::{DecodedFrame, DecodedFramePlane, Ffv1FrameCarry};
use crate::frame_encode::Ffv1EncodeCarry;
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
/// * Any error surfaced by the §4.4 prologue parse or the per-Plane
///   reconstructors.
///
/// All three §4.2.3 coders are supported on both colour layouts: §3.8.2
/// Golomb-Rice (`coder_type == 0`), §3.8.1 default-table range coder
/// (`coder_type == 1`), and §3.8.1.6 custom-table range coder
/// (`coder_type == 2`, with the mid-pass Parameters → Slice-Content table
/// swap described in the module docs).
pub fn decode_frame_v0v1(
    frame_bytes: &[u8],
    frame_dims: FramePixelDimensions,
) -> Result<DecodedFrame, Error> {
    decode_frame_v0v1_with_carry(frame_bytes, frame_dims, &mut None)
}

/// Decode a v0/v1 **keyframe** Frame like [`decode_frame_v0v1`], and
/// additionally snapshot the Frame's end-of-Frame §3.8.1.3 / §3.8.2.5
/// per-context coder state into `carry` for the next (non-keyframe)
/// Frame.
///
/// RFC 9043 §3.8.1.3 / §3.8.2.5: the coder state is re-initialised only
/// "When the keyframe value is 1" — the rule is version-independent, so
/// a v0/v1 non-keyframe continues the previous Frame's per-context
/// state over the implied single Slice, exactly as a v3 non-keyframe
/// Slice does (validated bit-exact against reference-encoded v0/v1
/// keyframe + inter streams, r411). Any incoming `carry` value is
/// ignored (a keyframe re-initialises); on success `carry` holds this
/// Frame's end-of-Frame snapshot for
/// [`decode_frame_v0v1_inter_with_carry`].
///
/// # Errors
///
/// Mirrors [`decode_frame_v0v1`]. Like the v3 carry drivers, the
/// incoming `carry` is consumed up-front, so it is `None` after an
/// error; callers wanting an error-tolerant session decode into a
/// working copy (as the registry decoder does).
pub fn decode_frame_v0v1_with_carry(
    frame_bytes: &[u8],
    frame_dims: FramePixelDimensions,
    carry: &mut Option<Ffv1FrameCarry>,
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
        carry,
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
/// This stateless entry decodes the non-keyframe with **fresh**
/// (keyframe-initialised) per-context coder state — the degenerate
/// no-carry case. Per RFC 9043 §3.8.1.3 / §3.8.2.5 a conforming v0/v1
/// non-keyframe instead **continues** the previous Frame's per-context
/// state ("when the keyframe value is 1, all ... state variables are
/// set to their initial state" — re-initialisation happens *only* on
/// keyframes, on every FFV1 version); use
/// [`decode_frame_v0v1_inter_with_carry`] to decode a conforming
/// stream. This form only round-trips Frames produced by the equally
/// stateless [`encode_frame_v0v1_inter`].
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
    decode_frame_v0v1_inter_with_carry(frame_bytes, cr, quant_table_set, frame_dims, &mut None)
}

/// Decode a v0/v1 **non-keyframe** Frame, resuming the §3.8.1.3 /
/// §3.8.2.5 per-context coder state from `carry` (the previous Frame's
/// end-of-Frame snapshot) — the conforming inter-Frame path.
///
/// RFC 9043 §3.8.1.3 ("Initial Values for the Context Model") and
/// §3.8.2.5 ("Initial Values for the VLC Context State") re-initialise
/// the per-context state only "when the keyframe value is 1"; on a
/// non-keyframe the state continues from the previous Frame over the
/// implied single Slice — the same rule the v3 drivers apply per Slice,
/// with no version qualifier in the RFC. Validated bit-exact against
/// reference-encoded v0/v1 keyframe + inter streams (both coders,
/// r411). A `None` / empty `carry` falls back to fresh
/// keyframe-initialised state (the stateless
/// [`decode_frame_v0v1_inter`] behaviour). On success `carry` holds
/// this Frame's end-of-Frame snapshot.
///
/// # Errors
///
/// Mirrors [`decode_frame_v0v1_inter`]. Like the v3 carry drivers, the
/// incoming `carry` is consumed up-front, so it is `None` after an
/// error; callers wanting an error-tolerant session decode into a
/// working copy (as the registry decoder does).
pub fn decode_frame_v0v1_inter_with_carry(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    frame_dims: FramePixelDimensions,
    carry: &mut Option<Ffv1FrameCarry>,
) -> Result<DecodedFrame, Error> {
    if cr.version == Ffv1Version::V3 {
        return Err(Error::InFrameParametersForbiddenForVersion(
            cr.version.as_u32(),
        ));
    }
    // §4.4: the Frame opens with the range-coded `keyframe` boolean. For
    // a non-keyframe this is 0 and NO inline Parameters follow — the
    // Slice Content begins right after.
    //
    // §3.8.1.6 / §4.2.4: a `coder_type == 2` non-keyframe carries no
    // inline Parameters (those were read on the governing keyframe), so
    // the §4.2.4 `state_transition_delta` are already known from the
    // cached `cr`. Following the v3 driver (`frame.rs`), where the whole
    // Slice — keyframe boolean included — is read with the custom table,
    // seed this decoder with the custom table from the start. The
    // `decode_v0v1_single_slice` swap below then re-derives the same
    // table (a no-op), keeping one code path for keyframe + non-keyframe.
    let mut rc = if cr.coder_type == 2 {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        crate::range_coder::RangeDecoder::with_one_state(frame_bytes, &one_state)?
    } else {
        crate::range_coder::RangeDecoder::new(frame_bytes)?
    };
    let mut kf_state = [crate::range_coder::PARAMETERS_INITIAL_STATE; 1];
    let keyframe = crate::symbol::get_br(&mut rc, &mut kf_state);
    if keyframe {
        return Err(Error::UnexpectedKeyframeInInterDecode);
    }
    decode_v0v1_single_slice(
        frame_bytes,
        cr,
        quant_table_set,
        frame_dims,
        rc,
        false,
        carry,
    )
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
    carry: &mut Option<Ffv1FrameCarry>,
) -> Result<DecodedFrame, Error> {
    // §3.8.1.6 / §4.2.4: for `coder_type == 2` the §4.7 Slice Content
    // uses the custom state-transition table built from
    // `state_transition_delta`. On a keyframe the inline §4.2 Parameters
    // were just read on this same pass with the §3.8.1.5 *default* table
    // (the deltas cannot apply to themselves); now that the deltas are
    // known, swap the live decoder onto the custom table — exactly at the
    // Parameters → Slice-Content boundary — before any Slice Content
    // read. On a non-keyframe the caller already seeded the decoder with
    // the custom table (there are no inline Parameters), so this re-swap
    // is a no-op that re-derives the same table. This mirrors the v3
    // driver (`frame.rs`), where each Slice's range coder is seeded with
    // the custom table for `coder_type == 2`; the only structural
    // difference is that v0/v1 shares one pass with the Parameters and so
    // must switch tables mid-stream instead of opening a fresh decoder.
    if cr.coder_type == 2 {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        rc.set_one_state(&one_state);
    }
    if cr.colorspace_type == ColorspaceType::Rgb {
        // §4.7 RGB / line-major (JPEG 2000 RCT) over the implied single
        // Slice.
        return decode_v0v1_rgb_single_slice(
            frame_bytes,
            cr,
            quant_table_set,
            frame_dims,
            rc,
            keyframe,
            carry,
        );
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
    // boundary after the range-coded prologue. The range-coded prologue
    // is terminated in Sentinel mode (RFC 9043 §3.8.1.1.1: the switch
    // from a range-coded region to a Golomb-coded one): a final state-129
    // symbol is read and discarded, after which the byte position of the
    // end — where the Golomb-Rice bit stream begins — is determined. The
    // bit reader persists across Planes (Plane p+1 reads where Plane p
    // stopped — same contract as the v3 driver).
    let mut golomb_bit_reader = if cr.coder_type == 0 {
        let consumed = rc.terminate_sentinel();
        Some(crate::bit_reader::BitReader::new(&frame_bytes[consumed..]))
    } else {
        None
    };

    // §4.6.6: per-context state is keyed by the §4.6.6 slot (luma → 0,
    // both chroma planes → 1, extra plane → last). v0/v1 has a single
    // Quantization Table Set so every slot resolves to it, but the
    // per-slot coder state is still distinct per category (luma vs
    // chroma vs alpha), exactly as the v3 driver allocates.
    //
    // RFC 9043 §3.8.1.3 / §3.8.2.5: on a non-keyframe the per-slot state
    // resumes from the previous Frame's snapshot (the implied single
    // Slice is forward Slice index 0); on a keyframe every slot starts
    // `None` and is lazily `128`-initialised on first use.
    let slot_count = header.quant_table_set_index_count;
    let prev_carry = carry.take().unwrap_or_default();
    let (mut per_slot_range_state, mut per_slot_golomb_state) = if keyframe {
        (
            (0..slot_count).map(|_| None).collect::<Vec<_>>(),
            (0..slot_count).map(|_| None).collect::<Vec<_>>(),
        )
    } else {
        let prev_range = prev_carry.range_for(0);
        let prev_golomb = prev_carry.golomb_for(0);
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
            // §3.8.1: `coder_type == 1` (default table) and `coder_type
            // == 2` (custom table) both decode the §4.7 Slice Content
            // through the range coder. The custom table was already
            // installed on `rc` above; the per-context Slice-Content
            // state buffers are identical in shape either way.
            1 | 2 => {
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

    // RFC 9043 §3.8.1.3 / §3.8.2.5: snapshot the implied single Slice's
    // end-of-Frame per-slot coder state so the next non-keyframe Frame
    // resumes it — the mirror of the v3 drivers' per-Slice snapshot.
    let mut new_carry = Ffv1FrameCarry::with_slice_capacity(1);
    new_carry.push_slice(per_slot_range_state, per_slot_golomb_state);
    *carry = Some(new_carry);

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

/// Reconstruct the implied single Slice of a v0/v1 **RGB / RCT** Frame
/// (`colorspace_type == 1`) over the §4.7 line-major traversal, given a
/// range decoder positioned at the start of the Slice Content.
///
/// Mirrors the v3 [`crate::decode_frame_rgb`] per-Slice machinery (the
/// §4.7 `for y { for p { Line(p, y) } }` interleave that keeps each
/// Plane's entropy + border state alive across the row interleave, then
/// the §3.7.1 inverse RCT) but over the implied single Slice — there is
/// no Slice Header and no footer. The §3.8.1.3 / §3.8.2.5 per-context
/// coder state is keyframe-initialised on a keyframe and resumes from
/// `carry` on a non-keyframe, exactly as the v3 driver seeds each
/// Slice; the end-of-Frame snapshot is written back into `carry`.
fn decode_v0v1_rgb_single_slice(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    frame_dims: FramePixelDimensions,
    mut rc: crate::range_coder::RangeDecoder<'_>,
    keyframe: bool,
    carry: &mut Option<Ffv1FrameCarry>,
) -> Result<DecodedFrame, Error> {
    // RFC 9043 §4.2.5: RGB always carries the three R / G / B colour
    // Planes (it never subsamples), so a conforming RGB record has
    // `chroma_planes == 1` and `primary_color_count >= 3`. A
    // non-conforming v0/v1 inline-Parameters Record (reachable from
    // untrusted bytes) can clear `chroma_planes`, leaving fewer than
    // three Planes for the §3.7.1 inverse-RCT blit to index. Reject it
    // here with a typed error rather than panicking downstream.
    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    if primary_color_count < 3 {
        return Err(Error::RgbRecordMissingChromaPlanes {
            primary_color_count: primary_color_count as u32,
        });
    }

    let header = implied_v0v1_slice_header(cr);
    let sc = compute_slice_content(&header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::LineMajor);
    let frame_w = frame_dims.width;
    let frame_h = frame_dims.height;
    let mut planes: Vec<DecodedFramePlane> = (0..primary_color_count)
        .map(|p| DecodedFramePlane {
            plane_index: p as u8,
            width: frame_w,
            height: frame_h,
            samples: vec![0i32; frame_w as usize * frame_h as usize],
        })
        .collect();

    // The §3.8 RCT coded Planes use `bits_per_raw_sample + 1`.
    let coded_bits = cr.bits_per_raw_sample + 1;

    // One persistent line-state per Plane; per-§4.6.6-slot entropy state
    // shared across Planes routed to the same slot (luma / chroma / extra).
    let slot_count = header.quant_table_set_index_count;
    let mut plane_states: Vec<crate::rgb_reconstruct::PlaneLineState> =
        Vec::with_capacity(primary_color_count);
    let mut plane_slots: Vec<usize> = Vec::with_capacity(primary_color_count);
    // RFC 9043 §3.8.1.3 / §3.8.2.5: keyframe → fresh (lazy `None`)
    // per-slot windows; non-keyframe → resume from the previous Frame's
    // snapshot (the implied single Slice is forward Slice index 0).
    let prev_carry = carry.take().unwrap_or_default();
    let (mut per_slot_range_state, mut per_slot_golomb_state): (
        Vec<Option<crate::range_reconstruct::RangePlaneState>>,
        Vec<Option<crate::reconstruct::PlaneEntropyState>>,
    ) = if keyframe {
        (
            (0..slot_count).map(|_| None).collect(),
            (0..slot_count).map(|_| None).collect(),
        )
    } else {
        let prev_range = prev_carry.range_for(0);
        let prev_golomb = prev_carry.golomb_for(0);
        (
            (0..slot_count)
                .map(|s| prev_range.get(s).cloned().flatten())
                .collect(),
            (0..slot_count)
                .map(|s| prev_golomb.get(s).cloned().flatten())
                .collect(),
        )
    };
    // §3.8.2.2.1: ONE Slice-scoped run triple shared by every Plane
    // across the §4.7 line-major interleave (`run_index` is "reset to
    // zero for each Plane and Slice"; on the line-major traversal the
    // Slice is the governing scope — mirror of the v3 RGB driver's
    // r411 black-box-pinned resolution).
    let mut slice_run_triple = (0u32, 0u8, 0i32);

    for (p_idx, plane) in sc.planes.iter().enumerate() {
        let qts_slot = match p_idx {
            0 => 0usize,
            1 | 2 if cr.chroma_planes => 1,
            _ if cr.extra_plane => slot_count.saturating_sub(1),
            _ => 0,
        };
        plane_states.push(crate::rgb_reconstruct::PlaneLineState::new(
            plane.width as usize,
            plane.height as usize,
            coded_bits,
            quant_table_set.tables,
        ));
        plane_slots.push(qts_slot);
    }

    // For `coder_type == 0` the Golomb-Rice bits start on a byte boundary
    // right after the range-coded prologue, located by the §3.8.1.1.1
    // Sentinel-mode terminator (state-129 symbol, discarded).
    let mut br_opt = if cr.coder_type == 0 {
        let consumed = rc.terminate_sentinel();
        Some(crate::bit_reader::BitReader::new(&frame_bytes[consumed..]))
    } else {
        None
    };

    // §4.7 line-major traversal: outer y, inner p.
    let slice_h = sc.slice_pixel_height as usize;
    let ctx_count = (quant_table_set.context_count as usize).max(1);
    for y in 0..slice_h {
        for (p_idx, ps) in plane_states.iter_mut().enumerate() {
            if y >= ps.height {
                continue;
            }
            ps.seed_row_border();
            let slot = plane_slots[p_idx];
            match cr.coder_type {
                0 => {
                    let br = br_opt
                        .as_mut()
                        .expect("coder_type == 0 builds a BitReader above");
                    let gr = per_slot_golomb_state[slot].get_or_insert_with(|| {
                        crate::reconstruct::PlaneEntropyState::new(ctx_count)
                    });
                    gr.load_run_state(slice_run_triple);
                    let (prev_prev, prev, cur) = (&ps.prev_prev, &ps.prev, &mut ps.cur);
                    PlaneReconstructor::reconstruct_row(
                        br,
                        gr,
                        &ps.qtable,
                        prev,
                        prev_prev,
                        cur,
                        ps.width,
                        ps.coded_bits,
                    );
                    slice_run_triple = gr.save_run_state();
                }
                _ => {
                    let rcs = per_slot_range_state[slot].get_or_insert_with(|| {
                        crate::range_reconstruct::RangePlaneState::new(ctx_count)
                    });
                    // §3.3.1 alt-median is YCbCr-only — never on RGB.
                    let use_16bit_median = false;
                    let (prev_prev, prev, cur) = (&ps.prev_prev, &ps.prev, &mut ps.cur);
                    RangePlaneReconstructor::reconstruct_row(
                        &mut rc,
                        rcs,
                        &ps.qtable,
                        prev,
                        prev_prev,
                        cur,
                        ps.width,
                        ps.coded_bits,
                        use_16bit_median,
                    );
                }
            }
            ps.commit_and_rotate(y);
        }
    }

    // §3.7.1 inverse RCT + blit into the frame-level R / G / B (+ alpha)
    // Planes. The single Slice covers the whole Frame (origin 0, 0).
    crate::rgb_reconstruct::apply_inverse_rct_and_blit(
        &plane_states,
        &mut planes,
        cr,
        0,
        0,
        sc.slice_pixel_width as usize,
        slice_h,
    );

    // RFC 9043 §3.8.1.3 / §3.8.2.5: snapshot the implied single Slice's
    // end-of-Frame per-slot coder state for the next non-keyframe.
    let mut new_carry = Ffv1FrameCarry::with_slice_capacity(1);
    new_carry.push_slice(per_slot_range_state, per_slot_golomb_state);
    *carry = Some(new_carry);

    Ok(DecodedFrame {
        planes,
        width: frame_dims.width,
        height: frame_dims.height,
        bits_per_raw_sample: cr.bits_per_raw_sample,
        colorspace: ColorspaceType::Rgb,
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
/// `version` MUST be 0 or 1, `colorspace_type` YCbCr, `coder_type` 0 or 1
/// — the §3.8.2 Golomb-Rice or the §3.8.1 default-table range coder);
/// `quant_table_set` is the single §4.1 Quantization Table Set
/// (`quant_table_set_count == 1` for v0/v1).
///
/// The produced bytes are a complete v0/v1 Frame: the §4.4 `keyframe`
/// boolean (`1`), the inline §4.2 Parameters + §4.1 cascade, then the
/// implied single Slice's §4.7 Slice Content. For `coder_type == 1` the
/// whole Frame is one continuous Closed-mode range-coder pass; for
/// `coder_type == 0` the prologue is range-coded, byte-aligned, and the
/// §4.7 Slice Content is appended as a Golomb-Rice bit stream.
/// `decode_frame_v0v1` reconstructs `frame.planes` bit-exactly from the
/// output.
///
/// # Errors
///
/// * [`Error::InFrameParametersForbiddenForVersion`] when `cr.version ==
///   V3`.
/// * Any error surfaced by the prologue encoder
///   ([`Error::MalformedQuantTable`], etc.). The `coder_type == 0`
///   Golomb path carries a non-zero first Sample Difference at an
///   absolute-context-0 run region directly (via a §3.8.2.4.1
///   zero-length short run), so it no longer surfaces a run-mode
///   first-pixel error.
pub fn encode_frame_v0v1(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
) -> Result<Vec<u8>, Error> {
    encode_frame_v0v1_keyframe(frame, cr, quant_table_set, true, &mut None)
}

/// Encode a v0/v1 **keyframe** Frame like [`encode_frame_v0v1`], and
/// additionally snapshot the encoder's end-of-Frame §3.8.1.3 / §3.8.2.5
/// per-context coder state into `carry` so the next Frame can be
/// emitted as a conforming non-keyframe via
/// [`encode_frame_v0v1_inter_with_carry`].
///
/// Any incoming `carry` value is ignored (RFC 9043 §3.8.1.3 / §3.8.2.5:
/// a keyframe re-initialises every state variable); on success `carry`
/// holds this Frame's end-of-Frame snapshot. The write-side mirror of
/// [`decode_frame_v0v1_with_carry`].
///
/// # Errors
///
/// Mirrors [`encode_frame_v0v1`].
pub fn encode_frame_v0v1_with_carry(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    carry: &mut Option<Ffv1EncodeCarry>,
) -> Result<Vec<u8>, Error> {
    encode_frame_v0v1_keyframe(frame, cr, quant_table_set, true, carry)
}

/// Encode one FFV1 **version 0 or 1** non-keyframe YCbCr Frame, reusing
/// the governing keyframe's inline §4.2 Parameters + §4.1 Quantization
/// Table Set (RFC 9043 §4.4: a non-keyframe carries no inline
/// `Parameters()`).
///
/// The produced Frame opens with the §4.4 `keyframe` boolean set to `0`,
/// then the implied single Slice's §4.7 Slice Content begins immediately
/// (no inline Parameters, no Slice Header).
///
/// This stateless entry emits the Frame with **fresh**
/// (keyframe-initialised) per-context coder state — the degenerate
/// no-carry case, readable only by the equally stateless
/// [`decode_frame_v0v1_inter`]. A conforming v0/v1 non-keyframe instead
/// continues the previous Frame's state per RFC 9043 §3.8.1.3 /
/// §3.8.2.5; use [`encode_frame_v0v1_inter_with_carry`] to emit a
/// stream a conforming decoder reads.
///
/// # Errors
///
/// Mirrors [`encode_frame_v0v1`].
pub fn encode_frame_v0v1_inter(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
) -> Result<Vec<u8>, Error> {
    encode_frame_v0v1_keyframe(frame, cr, quant_table_set, false, &mut None)
}

/// Encode a v0/v1 **non-keyframe** Frame whose §3.8.1.3 / §3.8.2.5
/// per-context coder state resumes from `carry` (the previous Frame's
/// end-of-Frame snapshot) — the conforming inter-Frame path, and the
/// write-side mirror of [`decode_frame_v0v1_inter_with_carry`].
///
/// RFC 9043 §3.8.1.3 / §3.8.2.5 re-initialise the per-context state
/// only "when the keyframe value is 1", on every FFV1 version; a
/// conforming decoder therefore reads a v0/v1 non-keyframe with the
/// previous Frame's carried state (validated bit-exact against
/// reference-encoded keyframe + inter streams, r411). On success
/// `carry` holds this Frame's end-of-Frame snapshot, ready for the next
/// non-keyframe.
///
/// # Errors
///
/// Mirrors [`encode_frame_v0v1`].
pub fn encode_frame_v0v1_inter_with_carry(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    carry: &mut Option<Ffv1EncodeCarry>,
) -> Result<Vec<u8>, Error> {
    encode_frame_v0v1_keyframe(frame, cr, quant_table_set, false, carry)
}

fn encode_frame_v0v1_keyframe(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_set: &QuantizationTableSet,
    keyframe: bool,
    carry: &mut Option<Ffv1EncodeCarry>,
) -> Result<Vec<u8>, Error> {
    if cr.version == Ffv1Version::V3 {
        return Err(Error::InFrameParametersForbiddenForVersion(
            cr.version.as_u32(),
        ));
    }

    let frame_dims = FramePixelDimensions::new(frame.width, frame.height)?;
    let header = implied_v0v1_slice_header(cr);

    // RFC 9043 §3.8.1.3 / §3.8.2.5: on a non-keyframe the per-slot coder
    // state resumes from the previous Frame's snapshot (the implied
    // single Slice is forward Slice index 0); a keyframe starts fresh.
    // Either way this Frame's end-of-Frame snapshot is written back into
    // `carry` — the write-side mirror of `decode_v0v1_single_slice`.
    let prev_carry = carry.take().unwrap_or_default();

    if cr.colorspace_type == ColorspaceType::Rgb {
        // §4.7 RGB / line-major (JPEG 2000 RCT) over the implied single
        // Slice. Reuse the v3 RGB per-Slice content encoders with the
        // §4.4 inline-Parameters prologue substituted for the §4.6 Slice
        // Header and the §4.9 footer dropped.
        let mut new_carry = Ffv1EncodeCarry::with_rgb_slice_capacity(1);
        let bytes = if cr.coder_type == 0 {
            let seed: &[Option<crate::sample_diff::LineDecoderState>] = if keyframe {
                &[]
            } else {
                prev_carry.golomb_for(0)
            };
            let (bytes, end_states) = crate::rgb_reconstruct::encode_one_rgb_slice_golomb(
                true,
                keyframe,
                &header,
                cr,
                core::slice::from_ref(quant_table_set),
                frame,
                frame_dims,
                false,
                seed,
                Some(quant_table_set),
            )?;
            new_carry.push_golomb_slice(end_states);
            bytes
        } else {
            let seed: &[Option<RangePlaneEncoderState>] = if keyframe {
                &[]
            } else {
                prev_carry.range_for(0)
            };
            let (bytes, end_states) = crate::rgb_reconstruct::encode_one_rgb_slice_range(
                true,
                keyframe,
                &header,
                cr,
                core::slice::from_ref(quant_table_set),
                frame,
                frame_dims,
                false,
                seed,
                Some(quant_table_set),
            )?;
            new_carry.push_range_slice(end_states);
            bytes
        };
        *carry = Some(new_carry);
        return Ok(bytes);
    }

    let sc = compute_slice_content(&header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

    // §3.8.1.6 / §4.2.4: a `coder_type == 2` non-keyframe carries no
    // inline Parameters, so the §4.7 Slice Content (and the leading §4.4
    // `keyframe` boolean) is emitted with the custom table from the
    // start — the symmetric mirror of `decode_frame_v0v1_inter`, which
    // seeds its decoder with the custom table. A keyframe instead opens
    // with the default table (the §4.2.4 deltas, read on this same pass,
    // cannot apply to themselves) and swaps to the custom table after the
    // prologue, below.
    let mut re = if cr.coder_type == 2 && !keyframe {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        RangeEncoder::with_one_state(&one_state)
    } else {
        RangeEncoder::new()
    };
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

    // §3.8.1.6 / §4.2.4: now that the keyframe's inline §4.2.4 deltas have
    // been emitted (with the default table), the §4.7 Slice Content uses
    // the §3.8.1.6 custom table. Swap the live encoder at the Parameters →
    // Slice-Content boundary — the exact mirror of the
    // `decode_v0v1_single_slice` swap. (Non-keyframes already seeded the
    // custom table above; `keyframe` is the only branch that needs this.)
    if cr.coder_type == 2 && keyframe {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        re.set_one_state(&one_state);
    }

    if cr.coder_type == 0 {
        // §4.5 / §3.8.1.1.1: on the `coder_type == 0` path the prologue is
        // the only range-coded region; it is terminated in Sentinel mode
        // (`re.terminate_sentinel()` writes the discarded state-129 symbol
        // and byte-aligns), and the §4.7 Slice Content is a single
        // Golomb-Rice bit stream appended at that byte boundary — exactly
        // the position the decoder recovers from
        // `rc.terminate_sentinel()`. Reuse the v3 single-Slice Golomb
        // content encoder (the implied single Slice covers the whole
        // Frame, so its pixel rectangle IS the frame). The §3.8.2.5
        // per-slot VLC state is keyframe-initialised on a keyframe and
        // resumes from `carry` on a non-keyframe.
        let mut out = re.terminate_sentinel();
        let quant_table_sets = core::slice::from_ref(quant_table_set);
        let seed: &[Option<crate::sample_diff::LineDecoderState>] = if keyframe {
            &[]
        } else {
            prev_carry.golomb_for(0)
        };
        let (content, end_states) = crate::frame_encode::encode_slice_content_golomb(
            &header,
            cr,
            quant_table_sets,
            frame,
            &sc,
            seed,
        )?;
        out.extend_from_slice(&content);
        let mut new_carry = Ffv1EncodeCarry::with_rgb_slice_capacity(1);
        new_carry.push_golomb_slice(end_states);
        *carry = Some(new_carry);
        return Ok(out);
    }

    // §4.7 plane-major Slice Content on the same continuous range-coder
    // pass (`coder_type == 1`). Per-context state is keyed by the §4.6.6
    // slot (luma / chroma / extra), shared across Planes routed through
    // the same slot — the exact mirror of `decode_v0v1_single_slice` —
    // and per RFC 9043 §3.8.1.3 resumes from `carry` on a non-keyframe.
    let slot_count = header.quant_table_set_index_count;
    let mut per_slot_state: Vec<Option<RangePlaneEncoderState>> = if keyframe {
        (0..slot_count).map(|_| None).collect()
    } else {
        let prev_range = prev_carry.range_for(0);
        (0..slot_count)
            .map(|s| prev_range.get(s).cloned().flatten())
            .collect()
    };
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

    // RFC 9043 §3.8.1.3: snapshot the end-of-Frame per-slot state for
    // the next non-keyframe.
    let mut new_carry = Ffv1EncodeCarry::with_rgb_slice_capacity(1);
    new_carry.push_range_slice(per_slot_state);
    *carry = Some(new_carry);

    Ok(re.finish())
}
