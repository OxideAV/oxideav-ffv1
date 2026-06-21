//! RGB / JPEG 2000 RCT line-major frame reconstruction
//! (RFC 9043 §3.7.2 / §3.7.2.1 / §4.7 `colorspace_type == 1`).
//!
//! The YCbCr driver ([`crate::frame::decode_frame`]) reconstructs each
//! Plane in one call, because §4.7's `colorspace_type == 0` traversal is
//! Plane-major (`for p { for y { Line(p, y) } }`) — a whole Plane's
//! Lines are coded contiguously. The RGB path is **line-major** (§4.7
//! `colorspace_type == 1`):
//!
//! ```text
//!   for (y = 0; y < slice_pixel_height; y++)
//!       for (p = 0; p < primary_color_count; p++)
//!           Line( p, y )
//! ```
//!
//! so each Plane's Lines are interleaved with the other Planes'. Every
//! Plane carries its own §3.1 border neighbours across its
//! (non-contiguous) Lines, and per RFC 9043 §4.6.6 ("indicates the
//! Quantization Table Set index to select the Quantization Table Set
//! **and the initial states** for the Slice Content") the per-context
//! entropy state lives **per §4.6.6 slot**, not per Plane — Planes
//! routed through the same slot (Cb + Cr on every `chroma_planes ==
//! true` Slice) share one persistent per-context state buffer that
//! evolves across both Planes. This module therefore keeps one
//! [`PlaneLineState`] (per-Plane border + qtable) and one entropy
//! state per slot (luma slot, chroma slot, optional extra-plane slot)
//! alive for the whole Slice and steps a single Line per Plane each
//! outer-`y` iteration, reusing the exact per-Line reconstruction the
//! YCbCr path uses ([`PlaneReconstructor::reconstruct_row`] for
//! `coder_type == 0`, [`RangePlaneReconstructor::reconstruct_row`] for
//! `coder_type >= 1`). The slot-keyed model mirrors the YCbCr driver's
//! round-214 fix exactly.
//!
//! ## Coded Sample width (§3.8)
//!
//! RFC 9043 §3.8: "only the n (or n+1, in the case of JPEG 2000 RCT)
//! least significant bits are used". The coded modified-YCbCr Planes
//! therefore use `bits_per_raw_sample + 1` bits, not
//! `bits_per_raw_sample`. The Cb / Cr Planes additionally carry the
//! §3.7.2 positive offset `1 << bits_per_raw_sample`, which the inverse
//! RCT removes before recovering RGB.
//!
//! ## Inverse RCT (§3.7.1 / Figures 7 & 9)
//!
//! The coded Planes are, in order, the modified-YCbCr `Y`, `Cb`, `Cr`
//! (and, when present, a transparency Plane that is **not** transformed).
//! The reversible inverse colour transform recovers `g`, `b`, `r` per
//! Sample. RFC 9043 Figure 7 (the general case) gives:
//!
//! ```text
//!   g = Y - ((Cb + Cr) >> 2)
//!   r = Cr + g
//!   b = Cb + g
//! ```
//!
//! and the §3.7.2.1 exception (Figure 9), used only when
//! `9 <= bits_per_raw_sample <= 15 && extra_plane == 0`:
//!
//! ```text
//!   b = Y - ((Cb + Cr) >> 2)
//!   r = Cr + b
//!   g = Cb + b
//! ```
//!
//! `Cb` / `Cr` are de-offset by `1 << bits_per_raw_sample` before being
//! fed to either formula (§3.7.2: "Cb and Cr are ... negatively offset
//! by the same value before the conversion from the modified YCbCr to
//! RGB").

use crate::bit_reader::{BitReader, BitWriter};
use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};
use crate::frame::{DecodeOptions, DecodedFrame, DecodedFramePlane, Ffv1FrameCarry};
use crate::frame_encode::Ffv1EncodeCarry;
use crate::predictor::{median_predict, QuantTableSet};
use crate::quant_table::QuantizationTableSet;
use crate::range_coder::{RangeDecoder, RangeEncoder, PARAMETERS_INITIAL_STATE};
use crate::range_encode::{RangePlaneEncoder, RangePlaneEncoderState};
use crate::range_reconstruct::{RangePlaneReconstructor, RangePlaneState};
use crate::reconstruct::{PlaneEntropyState, PlaneReconstructor, BORDER_LEFT, BORDER_RIGHT};
use crate::sample_diff::{encode_line, LineDecoderState, LineNeighborBuffers, BORDER_WIDTH};
use crate::slice_content::{compute_slice_content, FramePixelDimensions, PlaneTraversal};
use crate::slice_footer::{
    encode_slice_footer, parse_slice_footer_with_options, SliceErrorStatus, SliceErrorStatusPolicy,
};
use crate::slice_header::{
    encode_slice_header_to_encoder, parse_slice_header_from_decoder, Ffv1SliceHeader,
};
use crate::symbol::put_br;
use crate::trailer_chain::walk_trailer_chain;
use crate::Error;

/// One Plane's persistent line-major reconstruction state.
///
/// Carries the §3.1 border row buffers (`prev_prev`, `prev`, `cur`) and
/// the row-major output samples committed so far for the Plane.
///
/// The per-context entropy state is **not** stored here — it lives at
/// the §4.6.6 slot level and is owned by the driver loop directly
/// (see [`decode_frame_rgb`]). Multiple `PlaneLineState`s whose
/// Planes route through the same §4.6.6 slot share one entropy state
/// buffer so the per-context VLC / range-coder windows evolve
/// continuously across all Planes in that slot — matching the YCbCr
/// driver's round-214 §4.6.6 slot-keyed contract.
///
/// One caveat applies to the Golomb-Rice (`coder_type == 0`) path:
/// §3.8.2.2.1 says the run-mode triple resets at the start of each
/// Plane, but RGB line-major interleaves rows of different Planes that
/// share a slot, so a *shared* run-mode triple would be wrong. The
/// shipped v3 fixtures (and the round-214 milestone fixture) are all
/// range-coded — §3.8.1 has no run mode — so the §4.6.6 slot-sharing
/// applies cleanly to that path. The Golomb-Rice line-major slot-
/// sharing variant additionally needs the run-mode triple to remain
/// per-Plane while the per-context VLC fields share across the slot;
/// see the driver code for the current routing and the Note for
/// future rounds.
pub(crate) struct PlaneLineState {
    /// `plane_pixel_width[p]` (§4.8.1). For valid RGB streams every
    /// Plane has full slice width (RGB never subsamples).
    pub(crate) width: usize,
    /// `plane_pixel_height[p]` (§4.7.2).
    pub(crate) height: usize,
    /// `bits_per_raw_sample + 1` (§3.8 RCT coded-Sample width).
    pub(crate) coded_bits: u32,
    /// The §3.4 Quantization Table Set this Plane selected.
    pub(crate) qtable: QuantTableSet,
    /// Two rows above / one row above / the current row, each padded
    /// with the §3.1 border (`BORDER_LEFT` left, `BORDER_RIGHT` right).
    pub(crate) prev_prev: Vec<i32>,
    pub(crate) prev: Vec<i32>,
    pub(crate) cur: Vec<i32>,
    /// Row-major reconstructed Plane (`width * height`).
    pub(crate) out: Vec<i32>,
}

impl PlaneLineState {
    pub(crate) fn new(width: usize, height: usize, coded_bits: u32, qtable: QuantTableSet) -> Self {
        let stride = BORDER_LEFT + width + BORDER_RIGHT;
        Self {
            width,
            height,
            coded_bits,
            qtable,
            prev_prev: vec![0i32; stride],
            prev: vec![0i32; stride],
            cur: vec![0i32; stride],
            out: vec![0i32; width.saturating_mul(height)],
        }
    }

    /// Seed the §3.1 border cells of `cur` before decoding row `y`.
    pub(crate) fn seed_row_border(&mut self) {
        // Additional left column (x == -2) is always 0.
        self.cur[0] = 0;
        // Left-of-slice column: sample[y][-1] = sample[y-1][0]
        // (`prev[BORDER_LEFT]` is the previous row's first real Sample;
        // 0 for y == 0 since the row above the Slice is the all-zero
        // border).
        self.cur[BORDER_LEFT - 1] = self.prev[BORDER_LEFT];
    }

    /// Commit `cur`'s real Samples to `out[y]` and rotate the border
    /// buffers (prev_prev <- prev <- cur).
    pub(crate) fn commit_and_rotate(&mut self, y: usize) {
        // §3.1 right border: sample[y][W] = sample[y][W-1].
        self.cur[BORDER_LEFT + self.width] = self.cur[BORDER_LEFT + self.width - 1];
        let row_off = y * self.width;
        self.out[row_off..row_off + self.width]
            .copy_from_slice(&self.cur[BORDER_LEFT..BORDER_LEFT + self.width]);
        core::mem::swap(&mut self.prev_prev, &mut self.prev);
        core::mem::swap(&mut self.prev, &mut self.cur);
        self.cur[0] = 0;
    }
}

/// Decode one FFV1 v3 RGB / JPEG 2000 RCT frame end-to-end
/// (RFC 9043 §3.7.2 + §4.7 `colorspace_type == 1`).
///
/// This is the RGB counterpart of [`crate::frame::decode_frame`]: it
/// walks the §4.9.1 trailer chain, parses each Slice's §4.9 footer +
/// §4.6 header, and drives §4.7's **line-major** traversal — decoding
/// one Line per Plane each outer-`y` step while keeping each Plane's
/// entropy + border state alive across the interleave — then applies the
/// §3.7.1 inverse RCT (Figure 7 / Figure 9) to recover the R, G, B (and
/// optional alpha) Planes.
///
/// # Returns
///
/// A [`DecodedFrame`] whose `planes` are the recovered colour Planes in
/// **R, G, B** order (plane 0 = Red, plane 1 = Green, plane 2 = Blue),
/// followed by the transparency Plane (plane 3) when `extra_plane` is
/// set. Every Sample lands in `0 .. 2^bits_per_raw_sample`. `colorspace`
/// is reported as [`ColorspaceType::Rgb`].
///
/// # Errors
///
/// * [`Error::SliceRequiresVersion3`] when `cr.version != V3`.
/// * [`Error::ColorspaceLayoutNotImplemented`] when called on a
///   non-RGB Configuration Record (callers should route YCbCr through
///   [`crate::frame::decode_frame`]).
/// * [`Error::UnsupportedCoderType`] for `coder_type` outside `0..=2`.
/// * Any error surfaced by the per-stage parsers (footer / header /
///   range coder).
pub fn decode_frame_rgb(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame_dims: FramePixelDimensions,
    ec: bool,
) -> Result<DecodedFrame, Error> {
    decode_frame_rgb_with_options(
        frame_bytes,
        cr,
        quant_table_sets,
        frame_dims,
        ec,
        DecodeOptions::default(),
    )
}

/// Decode one FFV1 v3 RGB / JPEG 2000 RCT frame end-to-end with an
/// explicit [`DecodeOptions`] gate (RFC 9043 §4.9.3 per-Slice CRC
/// policy).
///
/// Same parameters and behaviour as [`decode_frame_rgb`] except that
/// the per-Slice §4.9.3 CRC residue check is governed by
/// `options.slice_crc_policy`. Symmetric to
/// [`crate::frame::decode_frame_with_options`] on the YCbCr / plane-
/// major path.
///
/// # Errors
///
/// Same as [`decode_frame_rgb`], except that
/// [`Error::SliceCrcMismatch`] is suppressed when
/// `options.slice_crc_policy == SliceCrcPolicy::Accept`.
pub fn decode_frame_rgb_with_options(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame_dims: FramePixelDimensions,
    ec: bool,
    options: DecodeOptions,
) -> Result<DecodedFrame, Error> {
    // No carry channel: every Slice's §3.8.1.3 / §3.8.2.5 per-context
    // coder state is keyframe-initialised, so the single-Frame decode
    // behaves as a standalone keyframe with no previous Frame.
    let mut carry: Option<Ffv1FrameCarry> = None;
    decode_frame_rgb_with_carry(
        frame_bytes,
        cr,
        quant_table_sets,
        frame_dims,
        ec,
        options,
        &mut carry,
    )
}

/// Decode one FFV1 v3 RGB / JPEG 2000 RCT frame, carrying the §3.8.1.3 /
/// §3.8.2.5 per-context coder state across non-keyframes (RFC 9043
/// §3.8.1.3 / §3.8.2.5 / §4.4 / §5).
///
/// The RGB / line-major analogue of
/// [`crate::frame::decode_frame_with_carry`]. The `carry` argument is
/// the cross-Frame coder-state channel:
///
/// * **On entry**, when the decoded Frame's §4.4 `keyframe` value is `0`
///   (a non-keyframe) and `*carry` is `Some(prev)`, each Slice's
///   per-§4.6.6-slot entropy / range state **resumes** from `prev`'s
///   matching Slice instead of re-initialising. This is the negation of
///   "When the keyframe value is 1, all state variables are set to their
///   initial state": on a non-keyframe the per-context window continues
///   from where the previous Frame's matching Slice left it.
/// * When the Frame's `keyframe` value is `1` (a keyframe), the carry on
///   entry is ignored; every slot is freshly `128`-initialised.
/// * **On return**, `*carry` is overwritten with this Frame's
///   end-of-Frame snapshot, ready to resume from on the next
///   non-keyframe in coded order.
///
/// The §3.8.2.2.1 run-mode triple (`run_index` / `run_mode` /
/// `run_count`) is **not** part of the carry: §3.8.2.2.1 resets it per
/// Plane / Slice unconditionally, keyframe or not. The RGB driver
/// already swaps that triple in / out of the slot window per Plane row,
/// so the snapshot the carry takes is the slot-level per-context VLC /
/// range window only — exactly the state §3.8.2.5 / §3.8.1.3 governs.
///
/// RFC 9043 §5 third paragraph guarantees a non-keyframe's Slices share
/// `slice_x` / `slice_y` / `slice_width` / `slice_height` with a Slice
/// of the previous Frame, so the per-Slice / per-slot carry lines up
/// positionally. The stateful session [`crate::Ffv1DecodeSession`] owns
/// the `carry` across a coded Frame sequence.
///
/// # Errors
///
/// Same as [`decode_frame_rgb_with_options`].
pub fn decode_frame_rgb_with_carry(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame_dims: FramePixelDimensions,
    ec: bool,
    options: DecodeOptions,
    carry: &mut Option<Ffv1FrameCarry>,
) -> Result<DecodedFrame, Error> {
    if cr.version != Ffv1Version::V3 {
        return Err(Error::SliceRequiresVersion3);
    }
    if cr.colorspace_type != ColorspaceType::Rgb {
        // YCbCr / plane-major has its own driver; surface explicitly so
        // a mis-routed call never silently produces wrong output.
        return Err(Error::ColorspaceLayoutNotImplemented);
    }
    if cr.coder_type > 2 {
        return Err(Error::UnsupportedCoderType(cr.coder_type));
    }

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);

    // The recovered colour Planes all run at full frame resolution
    // (RGB never subsamples). Allocate R, G, B (+ alpha) buffers.
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

    let extents = walk_trailer_chain(frame_bytes, ec)?;

    // RFC 9043 §5 second paragraph: "For each Frame, each position in
    // the Slice raster MUST be filled by one and only one Slice of
    // the Frame (no missing Slice position and no Slice
    // overlapping)." Mirror of the YCbCr driver gate — run the
    // round-257 raster-coverage validator before any per-Slice pixel
    // reconstruction starts so an overlap / gap aborts the Frame
    // before the line-major interleave paints into a Plane region
    // claimed by two Slices (or leaves a region unclaimed). Pass-1
    // shares the helper with the YCbCr driver.
    let headers_pass1 =
        crate::frame::collect_slice_headers_for_raster_validation(frame_bytes, &extents, cr, ec)?;
    crate::slice_content::validate_slice_raster_coverage(&headers_pass1, cr)?;

    let footer_len = if ec {
        crate::slice_footer::SLICE_FOOTER_LEN_EC1
    } else {
        crate::slice_footer::SLICE_FOOTER_LEN_EC0
    };

    // RFC 9043 §4.4 `keyframe`, decoded off Slice 0 below and surfaced
    // on the returned `DecodedFrame` (mirror of the YCbCr / plane-major
    // driver in `frame.rs`). An empty Frame is vacuously a keyframe.
    let mut frame_keyframe = true;

    // The previous Frame's end-of-Frame per-Slice per-slot coder state
    // (RFC 9043 §3.8.1.3 / §3.8.2.5), taken out of the caller's `carry`
    // channel. We resume from it only on a non-keyframe; either way this
    // Frame's fresh snapshot is written back at the end. Taking it out
    // up-front lets us read the previous snapshot while building the new
    // one without borrow conflicts — mirror of `decode_frame_with_carry`.
    let prev_carry = carry.take().unwrap_or_default();
    let mut new_carry = Ffv1FrameCarry::with_slice_capacity(extents.len());

    for (slice_index, ext) in extents.iter().enumerate() {
        let slice_bytes = &frame_bytes[ext.start..ext.end()];
        // §4.9 footer validation: the size cross-check always aborts
        // on mismatch; the §4.9.3 CRC residue check is gated by
        // `options.slice_crc_policy` per RFC 9043 §4.9.3.
        let footer = parse_slice_footer_with_options(slice_bytes, ec, options.slice_crc_policy)?;

        // §4.9.2 `error_status` Table 16 gate — mirror of the YCbCr
        // / plane-major driver in `frame.rs`. The `Reject` policy
        // aborts on the `Uncorrectable` (`2`) status; other Table 16
        // values pass through to the per-Slice pixel reconstruction.
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

        let body_end = slice_bytes.len() - footer_len;
        let body = &slice_bytes[..body_end];

        // For `coder_type == 2` the per-Frame `state_transition_delta`
        // is layered onto the default table to derive `one_state` per
        // RFC 9043 §3.8.1.4 Figure 22 / §3.8.1.6.
        let mut rc = if cr.coder_type == 2 {
            let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
            RangeDecoder::with_one_state(body, &one_state)?
        } else {
            RangeDecoder::new(body)?
        };

        // RFC 9043 §4.4: the Frame opens with a single range-coded
        // `keyframe` boolean (initial state 128) at the very start of
        // the FIRST Slice's range-coded region, before that Slice's §4.6
        // header. Capture it for slice 0 so the rest of the decoder
        // stays byte-synchronised and the value reaches the returned
        // `DecodedFrame`; later Slices carry no keyframe bit.
        if slice_index == 0 {
            let mut kf_state = [crate::range_coder::PARAMETERS_INITIAL_STATE; 1];
            frame_keyframe = crate::symbol::get_br(&mut rc, &mut kf_state);
        }

        let header = parse_slice_header_from_decoder(&mut rc, cr)?;
        // RFC 9043 §5 "Restrictions" — mirror of the YCbCr / plane-
        // major driver gate. On v3 Frames above the §5 trigger
        // (`frame_pixel_width * frame_pixel_height > 101376`) each
        // Slice's raster footprint must satisfy `slice_width *
        // slice_height <= num_h_slices * num_v_slices / 4`. Validate
        // before the §4.7 layout math so an offending Slice aborts
        // the Frame without ever reaching the line-major per-Plane
        // reconstructors.
        crate::slice_content::validate_slice_max_size_restriction(&header, cr, frame_dims)?;
        let sc = compute_slice_content(&header, cr, frame_dims)?;
        debug_assert_eq!(sc.traversal, PlaneTraversal::LineMajor);

        // Build one persistent line-state per Plane, plus per-slot
        // entropy state buffers (§4.6.6 — "the Quantization Table Set
        // **and the initial states**"). Planes routed through the
        // same §4.6.6 slot share their entropy state; the slot is the
        // §4.6.6 selector, the resolved set indexes into
        // `quant_table_sets[..]`. For Cb + Cr on `chroma_planes ==
        // true` the §4.7 line-major interleave reads back-to-back per
        // row, so both Planes evolve a single persistent per-context
        // state.
        let mut plane_states: Vec<PlaneLineState> = Vec::with_capacity(primary_color_count);
        let mut plane_slots: Vec<usize> = Vec::with_capacity(primary_color_count);
        let slot_count = header.quant_table_set_index_count;
        // RFC 9043 §3.8.1.3 / §3.8.2.5: the per-context window is
        // re-initialised to its `128` state only "When the keyframe
        // value ... is 1". On a non-keyframe (`frame_keyframe == false`)
        // each slot instead **continues from the previous Frame's
        // matching Slice** (§5 third paragraph pins the per-Slice / per-
        // slot layout across Frames), so we pre-seed each slot from
        // `prev_carry`. On a keyframe every slot starts `None` and the
        // `get_or_insert_with` below lazily allocates a fresh
        // `128`-initialised window on first use — the single-Frame
        // contract, unchanged.
        let mut per_slot_range_state: Vec<Option<RangePlaneState>> = if frame_keyframe {
            (0..slot_count).map(|_| None).collect()
        } else {
            let prev_range = prev_carry.range_for(slice_index);
            (0..slot_count)
                .map(|s| prev_range.get(s).cloned().flatten())
                .collect()
        };
        let mut per_slot_range_ctx_count: Vec<Option<usize>> =
            (0..slot_count).map(|_| None).collect();
        // §3.8.2.2.1 + §4.6.6: for the Golomb-Rice path the per-context
        // VLC window (`drift`, `error_sum`, `bias`, `count` per
        // context) lives at the §4.6.6 *slot* level — two Planes
        // routed to the same slot share one persistent window — but
        // the run-mode triple (`run_index`, `run_mode`, `run_count`)
        // is per-Plane (§3.8.2.2.1 says it resets at the start of each
        // Plane, AND run mode straddles row boundaries within a
        // Plane, so the slot-level window cannot carry it across the
        // §4.7 line-major interleave). The driver holds one
        // [`PlaneEntropyState`] per slot for the VLC window, plus one
        // saved-run-triple snapshot per Plane that is swapped into /
        // out of the slot state around every row decode.
        // Same §3.8.2.5 keyframe / non-keyframe split for the Golomb-Rice
        // slot windows. The §3.8.2.2.1 run-mode triple inside a resumed
        // `PlaneEntropyState` is irrelevant: every Plane row loads its
        // own `per_plane_run_triple` into the slot window before decode
        // and saves it back after, so only the carried per-context VLC
        // window (`drift` / `error_sum` / `bias` / `count`) survives.
        let mut per_slot_golomb_state: Vec<Option<PlaneEntropyState>> = if frame_keyframe {
            (0..slot_count).map(|_| None).collect()
        } else {
            let prev_golomb = prev_carry.golomb_for(slice_index);
            (0..slot_count)
                .map(|s| prev_golomb.get(s).cloned().flatten())
                .collect()
        };
        let mut per_slot_golomb_ctx_count: Vec<Option<usize>> =
            (0..slot_count).map(|_| None).collect();
        let mut per_plane_run_triple: Vec<(u32, u8, i32)> = Vec::with_capacity(primary_color_count);
        for (p_idx, plane) in sc.planes.iter().enumerate() {
            let qts_slot = match p_idx {
                0 => 0usize,
                1 | 2 if cr.chroma_planes => 1,
                _ if cr.extra_plane => header.quant_table_set_index_count.saturating_sub(1),
                _ => 0,
            };
            let qts_index = (header.quant_table_set_index[qts_slot] as usize)
                .min(quant_table_sets.len().saturating_sub(1));
            let qts = quant_table_sets
                .get(qts_index)
                .ok_or(Error::InvalidQuantTableSetCount(0))?;

            // §3.8: the RCT coded Planes use bits_per_raw_sample + 1.
            let coded_bits = cr.bits_per_raw_sample + 1;
            plane_states.push(PlaneLineState::new(
                plane.width as usize,
                plane.height as usize,
                coded_bits,
                qts.tables,
            ));
            plane_slots.push(qts_slot);
            // §3.8.2.2.1: every Plane starts with a fresh run triple
            // (`run_index = run_mode = run_count = 0`). The slot's VLC
            // window evolves across Planes that share the slot.
            per_plane_run_triple.push((0u32, 0u8, 0i32));
            if cr.coder_type == 0 {
                per_slot_golomb_ctx_count[qts_slot]
                    .get_or_insert((qts.context_count as usize).max(1));
            }
            // Pre-pin the per-slot range context_count so two Planes
            // routed through the same slot agree on buffer sizing
            // (they always will, since the slot selects the same
            // resolved set; this is a defensive sanity check via
            // `get_or_insert_with`).
            if cr.coder_type >= 1 {
                per_slot_range_ctx_count[qts_slot].get_or_insert(qts.context_count as usize);
            }
        }

        // For coder_type == 0 the SliceContent's Golomb-Rice bits start
        // on a byte boundary right after the range-coded SliceHeader.
        let mut br_opt = if cr.coder_type == 0 {
            let consumed = rc.position();
            if consumed > body.len() {
                return Err(Error::TruncatedRangeCoder);
            }
            Some(BitReader::new(&body[consumed..]))
        } else {
            None
        };

        // §4.7 line-major traversal: outer y, inner p.
        let slice_h = sc.slice_pixel_height as usize;
        for y in 0..slice_h {
            for (p_idx, ps) in plane_states.iter_mut().enumerate() {
                if y >= ps.height {
                    // Subsampled Planes (none for valid RGB) could be
                    // shorter; guard defensively so the interleave never
                    // over-runs a Plane's row count.
                    continue;
                }
                ps.seed_row_border();
                match cr.coder_type {
                    0 => {
                        let br = br_opt
                            .as_mut()
                            .expect("coder_type == 0 builds a BitReader above");
                        let slot = plane_slots[p_idx];
                        let ctx_count = per_slot_golomb_ctx_count[slot]
                            .expect("Golomb slot context_count was pinned above");
                        let gr = per_slot_golomb_state[slot]
                            .get_or_insert_with(|| PlaneEntropyState::new(ctx_count));
                        // §3.8.2.2.1 + §4.6.6: load the per-Plane
                        // run triple into the slot's VLC window for
                        // this row, decode, then save the triple back
                        // (the slot's VLC fields keep evolving; the
                        // run triple belongs to *this* Plane only).
                        gr.load_run_state(per_plane_run_triple[p_idx]);
                        // Split borrows: copy the row buffers' raw
                        // pointers are not needed — pass disjoint slices.
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
                        per_plane_run_triple[p_idx] = gr.save_run_state();
                    }
                    _ => {
                        let slot = plane_slots[p_idx];
                        let ctx_count = per_slot_range_ctx_count[slot]
                            .expect("range slot context_count was pinned above");
                        let rcs = per_slot_range_state[slot]
                            .get_or_insert_with(|| RangePlaneState::new(ctx_count));
                        let use_16bit_median = false; // §3.3.1 is YCbCr-only.
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

        // §3.7.1 inverse RCT + blit into the frame-level colour Planes.
        apply_inverse_rct_and_blit(
            &plane_states,
            &mut planes,
            cr,
            sc.slice_pixel_x as usize,
            sc.slice_pixel_y as usize,
            sc.slice_pixel_width as usize,
            slice_h,
        );

        // Snapshot this Slice's end-of-Frame per-slot coder state into
        // the new carry (RFC 9043 §3.8.1.3 / §3.8.2.5). The next Frame in
        // coded order, if it is a non-keyframe, resumes exactly these
        // windows for its matching Slice. The §3.8.2.2.1 run-mode triple
        // inside each `PlaneEntropyState` is not part of the carry's
        // contract — it is reset per Plane / Slice unconditionally — but
        // snapshotting the whole state object is harmless because the
        // resuming Frame loads a fresh `per_plane_run_triple` into the
        // slot window before every row decode.
        new_carry.push_slice(per_slot_range_state, per_slot_golomb_state);
    }

    // Publish this Frame's end-of-Frame snapshot so the next coded Frame
    // can resume it on a non-keyframe (RFC 9043 §3.8.1.3 / §3.8.2.5).
    *carry = Some(new_carry);

    Ok(DecodedFrame {
        planes,
        width: frame_dims.width,
        height: frame_dims.height,
        bits_per_raw_sample: cr.bits_per_raw_sample,
        colorspace: ColorspaceType::Rgb,
        keyframe: frame_keyframe,
        slice_headers: headers_pass1,
    })
}

/// Apply the §3.7.1 inverse RCT to each Sample of a Slice and blit the
/// recovered R / G / B (+ alpha) into the frame-level Planes at the
/// Slice's pixel origin.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_inverse_rct_and_blit(
    plane_states: &[PlaneLineState],
    planes: &mut [DecodedFramePlane],
    cr: &Ffv1ConfigurationRecord,
    origin_x: usize,
    origin_y: usize,
    slice_w: usize,
    slice_h: usize,
) {
    let bits = cr.bits_per_raw_sample;
    let mask = (1i64 << bits) - 1;
    let offset = 1i64 << bits; // §3.7.2 Cb/Cr positive offset.

    // §3.7.2.1 exception: 9..=15 bits and no extra plane uses Figure 9.
    let use_exception = (9..=15).contains(&bits) && !cr.extra_plane;

    let y_plane = &plane_states[0];
    let cb_plane = &plane_states[1];
    let cr_plane = &plane_states[2];
    let alpha_plane = if cr.extra_plane {
        plane_states.get(3)
    } else {
        None
    };

    let dst_w = planes[0].width as usize;
    let dst_h = planes[0].height as usize;

    for y in 0..slice_h.min(y_plane.height) {
        let dy = origin_y + y;
        if dy >= dst_h {
            break;
        }
        let copy_w = slice_w
            .min(y_plane.width)
            .min(dst_w.saturating_sub(origin_x));
        for x in 0..copy_w {
            let src = y * y_plane.width + x;
            let y_val = y_plane.out[src] as i64;
            // De-offset Cb / Cr (§3.7.2 negative offset before convert).
            let cb = cb_plane.out[src] as i64 - offset;
            let cr_val = cr_plane.out[src] as i64 - offset;

            // §3.7.1 inverse RCT (Figure 7 general / Figure 9 exception).
            let (r, g, b) = if use_exception {
                // b = Y - ((Cb + Cr) >> 2); r = Cr + b; g = Cb + b
                let b = y_val - ((cb + cr_val) >> 2);
                let r = cr_val + b;
                let g = cb + b;
                (r, g, b)
            } else {
                // g = Y - ((Cb + Cr) >> 2); r = Cr + g; b = Cb + g
                let g = y_val - ((cb + cr_val) >> 2);
                let r = cr_val + g;
                let b = cb + g;
                (r, g, b)
            };

            let dx = origin_x + x;
            let dst = dy * dst_w + dx;
            // Recovered Planes wrap to `0 .. 2^bits` (the transform is
            // exact modulo 2^bits per §3.8's low-bit coding).
            planes[0].samples[dst] = (r & mask) as i32;
            planes[1].samples[dst] = (g & mask) as i32;
            planes[2].samples[dst] = (b & mask) as i32;
            if let Some(ap) = alpha_plane {
                // Transparency Plane is not RCT-transformed; copy it
                // straight (already in 0 .. 2^bits via the coded LSBs).
                planes[3].samples[dst] = ap.out[src] & ((1i32 << bits) - 1);
            }
        }
    }
}

// =====================================================================
// RGB / JPEG 2000 RCT encoder — symmetric inverse of decode_frame_rgb.
// (RFC 9043 §3.7.1 / §3.7.2 / §4.7 `colorspace_type == 1`)
// =====================================================================

/// Per-Plane encoder state for the RGB line-major path.
///
/// Encoder-side mirror of [`PlaneLineState`]. Carries one Plane's §3.1
/// border row buffers and the row-major coded modified-YCbCr Sample
/// buffer the line loop consumes.
///
/// Like the decoder mirror, the per-context entropy state is **not**
/// stored here — it lives at the §4.6.6 slot level and the
/// `encode_one_rgb_slice_range` driver owns the per-slot
/// [`RangePlaneEncoderState`] vector that two Planes sharing a slot
/// (Cb + Cr) thread through their back-to-back encode calls.
struct PlaneLineEncodeState {
    /// `plane_pixel_width[p]` (§4.8.1). RGB never subsamples.
    width: usize,
    /// `plane_pixel_height[p]` (§4.7.2).
    height: usize,
    /// `bits_per_raw_sample + 1` (§3.8 RCT coded-Sample width).
    coded_bits: u32,
    /// The §3.4 Quantization Table Set this Plane selected.
    qtable: QuantTableSet,
    /// Two rows above / one row above / the current row, each padded
    /// with the §3.1 border (`BORDER_LEFT` left, `BORDER_RIGHT` right).
    prev_prev: Vec<i32>,
    prev: Vec<i32>,
    cur: Vec<i32>,
    /// Row-major coded Sample buffer (`width * height`). For each Plane
    /// this is the forward-RCT output (Y / Cb+offset / Cr+offset) or, for
    /// the alpha Plane, the raw input copied straight (§3.7.2: the
    /// transparency Plane is not RCT-transformed).
    coded: Vec<i32>,
}

impl PlaneLineEncodeState {
    fn new(
        width: usize,
        height: usize,
        coded_bits: u32,
        qtable: QuantTableSet,
        coded: Vec<i32>,
    ) -> Self {
        let stride = BORDER_LEFT + width + BORDER_RIGHT;
        debug_assert_eq!(coded.len(), width * height);
        Self {
            width,
            height,
            coded_bits,
            qtable,
            prev_prev: vec![0i32; stride],
            prev: vec![0i32; stride],
            cur: vec![0i32; stride],
            coded,
        }
    }

    /// Seed the §3.1 border cells of `cur` before encoding row `y`.
    /// Mirrors [`PlaneLineState::seed_row_border`].
    fn seed_row_border(&mut self) {
        self.cur[0] = 0;
        self.cur[BORDER_LEFT - 1] = self.prev[BORDER_LEFT];
    }

    /// Right-border mirror, rotate (prev_prev <- prev <- cur), zero the
    /// next `cur`. Mirrors [`PlaneLineState::commit_and_rotate`].
    fn finish_row_and_rotate(&mut self) {
        self.cur[BORDER_LEFT + self.width] = self.cur[BORDER_LEFT + self.width - 1];
        core::mem::swap(&mut self.prev_prev, &mut self.prev);
        core::mem::swap(&mut self.prev, &mut self.cur);
        self.cur.iter_mut().for_each(|s| *s = 0);
    }
}

/// Encode one FFV1 v3 RGB / JPEG 2000 RCT frame end-to-end
/// (RFC 9043 §3.7.1 + §3.7.2 + §4.7 `colorspace_type == 1`,
/// `coder_type == 0 || 1 || 2`).
///
/// Symmetric inverse of [`decode_frame_rgb`]: given a frame's R, G, B
/// (and optional alpha) Planes as a [`DecodedFrame`], the driver runs
/// the §3.7.1 *forward* RCT to produce the coded modified-YCbCr Planes,
/// then walks the §4.7 line-major traversal (`for y { for p { Line(p, y)
/// } }`) emitting per-Sample `sample_difference` values via either
/// [`RangePlaneEncoder::encode_row`] (range-coded `coder_type == 1 || 2`)
/// or [`encode_line`](crate::encode_line) (Golomb-Rice `coder_type == 0`).
/// The range-coded path shares a single per-Slice [`RangeEncoder`]
/// cursor for header + content; the Golomb-Rice path flushes the
/// range-coded SliceHeader to the byte boundary and then writes
/// SliceContent through a [`BitWriter`] tail, mirroring the decoder's
/// `consumed = rc.position()` split. The §4.9 Slice Footer wraps each
/// Slice with the §4.9.3 CRC parity solved by construction.
///
/// # Returns
///
/// The concatenated Slice byte stream — exactly what
/// [`decode_frame_rgb`] reads as `frame_bytes`. Round-tripping the
/// returned buffer back through [`decode_frame_rgb`] recovers the
/// original R, G, B (+ optional alpha) Sample Planes bit-exactly.
///
/// # Parameters
///
/// * `frame` — the [`DecodedFrame`] of R, G, B (+ optional alpha)
///   Planes. `frame.planes[0]` = Red, `[1]` = Green, `[2]` = Blue,
///   `[3]` = alpha when `cr.extra_plane`. Each Sample must lie in
///   `0 .. 2^bits_per_raw_sample`.
/// * `cr` — the per-stream Configuration Record. Must satisfy
///   `version == V3`, `colorspace_type == Rgb`, and
///   `coder_type ∈ {0, 1, 2}`. `coder_type == 0` is the Golomb-Rice
///   RGB path; `coder_type == 1 || 2` is the range-coded path.
/// * `quant_table_sets` — the parsed §4.1 Quantization Table Sets in
///   stream order.
/// * `slice_headers` — one [`Ffv1SliceHeader`] per Slice in slice-index
///   order. The caller supplies the §4.6 raster decomposition.
/// * `ec` — the §4.2.14 `error_correction` flag.
///
/// # Errors
///
/// * [`Error::SliceRequiresVersion3`] when `cr.version != V3`.
/// * [`Error::ColorspaceLayoutNotImplemented`] when
///   `cr.colorspace_type != Rgb`.
/// * [`Error::UnsupportedCoderType`] when `cr.coder_type > 2`.
/// * [`Error::InvalidFramePixelDimensions`] when `frame.width == 0` or
///   `frame.height == 0`.
/// * [`Error::SliceRasterOutOfRange`] when a header addresses an
///   out-of-raster cell.
/// * [`Error::SliceSizeOutOfRange`] when a header / footer constraint
///   fails, or when an assembled body length overflows §4.9.1's `u(24)`.
/// * [`Error::InvalidQuantTableSetCount`] when a slice header selects
///   an out-of-range Quantization Table Set, or `frame.planes` lacks a
///   plane the configuration demands.
pub fn encode_frame_rgb(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    slice_headers: &[Ffv1SliceHeader],
    ec: bool,
) -> Result<Vec<u8>, Error> {
    // The historical single-Frame entry point: a standalone keyframe,
    // no inter-Frame coder-state carry. Mirror of `encode_frame_rgb` ↔
    // `encode_frame_range_coder` on the YCbCr side.
    let mut carry: Option<Ffv1EncodeCarry> = None;
    encode_frame_rgb_with_carry(
        frame,
        cr,
        quant_table_sets,
        slice_headers,
        ec,
        true,
        &mut carry,
    )
}

/// Encode one FFV1 v3 RGB / JPEG 2000 RCT Frame, carrying the §3.8.1.3
/// (range coder) / §3.8.2.5 (Golomb-Rice VLC) per-context coder state
/// across non-keyframes — the write-side mirror of
/// [`decode_frame_rgb_with_carry`].
///
/// This closes the read/write symmetry the single-Frame
/// [`encode_frame_rgb`] left open: with it a synthetic multi-Frame RGB
/// non-keyframe stream can be produced (encode with `keyframe == false`
/// and a live carry) and decoded back bit-exactly with
/// [`decode_frame_rgb_with_carry`] and the matching read-side carry,
/// the per-context windows on both sides staying in lockstep across the
/// Frame boundary exactly as [`encode_frame_range_coder_with_carry`]
/// does for YCbCr.
///
/// * `keyframe` — the §4.4 value written on Slice 0 (`br`). `true` emits
///   an independently decodable keyframe; `false` emits a non-keyframe
///   whose per-context coder state continues from `carry`.
/// * `carry` — on entry, when `keyframe == false` and `*carry` is
///   `Some(prev)`, each Slice's per-slot state resumes from `prev`'s
///   matching Slice instead of the §3.8.1.3 / §3.8.2.5
///   `128`-initialised window; when `keyframe == true` the carry on
///   entry is ignored. On return, `*carry` holds this Frame's
///   end-of-Frame snapshot for the next non-keyframe.
///
/// The §3.8.2.2.1 run-mode triple (`run_index`, `run_mode`,
/// `run_count`) is reset per Plane / Slice unconditionally on the
/// Golomb-Rice path and is therefore NOT part of the carry contract —
/// only the per-context VLC / range-coder windows participate, exactly
/// as on the read side.
///
/// Restricted to `coder_type ∈ {0, 1, 2}` and `colorspace_type == Rgb`
/// (the same surface [`encode_frame_rgb`] covers); `coder_type == 0`
/// uses the Golomb-Rice carry channel, `coder_type ∈ {1, 2}` the
/// range-coder channel.
pub fn encode_frame_rgb_with_carry(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    slice_headers: &[Ffv1SliceHeader],
    ec: bool,
    keyframe: bool,
    carry: &mut Option<Ffv1EncodeCarry>,
) -> Result<Vec<u8>, Error> {
    if cr.version != Ffv1Version::V3 {
        return Err(Error::SliceRequiresVersion3);
    }
    if cr.colorspace_type != ColorspaceType::Rgb {
        return Err(Error::ColorspaceLayoutNotImplemented);
    }
    if cr.coder_type > 2 {
        return Err(Error::UnsupportedCoderType(cr.coder_type));
    }

    let frame_dims = FramePixelDimensions::new(frame.width, frame.height)?;

    let prev_carry = carry.take().unwrap_or_default();
    let mut new_carry = Ffv1EncodeCarry::with_rgb_slice_capacity(slice_headers.len());

    let mut out = Vec::new();
    for (slice_index, header) in slice_headers.iter().enumerate() {
        if cr.coder_type == 0 {
            // RFC 9043 §3.8.2.5: keyframe → fresh `128` VLC windows
            // (lazy `None`); non-keyframe → resume from the previous
            // Frame's matching Slice (per-slot seed).
            let seed: &[Option<LineDecoderState>] = if keyframe {
                &[]
            } else {
                prev_carry.golomb_for(slice_index)
            };
            let (slice_bytes, end_states) = encode_one_rgb_slice_golomb(
                slice_index == 0,
                keyframe,
                header,
                cr,
                quant_table_sets,
                frame,
                frame_dims,
                ec,
                seed,
                None,
            )?;
            out.extend_from_slice(&slice_bytes);
            new_carry.push_golomb_slice(end_states);
        } else {
            // RFC 9043 §3.8.1.3: same keyframe / non-keyframe split for
            // the range-coder per-context state windows.
            let seed: &[Option<RangePlaneEncoderState>] = if keyframe {
                &[]
            } else {
                prev_carry.range_for(slice_index)
            };
            let (slice_bytes, end_states) = encode_one_rgb_slice_range(
                slice_index == 0,
                keyframe,
                header,
                cr,
                quant_table_sets,
                frame,
                frame_dims,
                ec,
                seed,
                None,
            )?;
            out.extend_from_slice(&slice_bytes);
            new_carry.push_range_slice(end_states);
        }
    }

    *carry = Some(new_carry);
    Ok(out)
}

/// Encode one Slice on the RGB / line-major range-coded path.
///
/// Mirrors the per-Slice loop body in [`decode_frame_rgb`]: keyframe
/// bit (slice 0 only) → §4.6 SliceHeader → §4.7 line-major Sample
/// encode → §4.9 SliceFooter, with the SliceHeader and SliceContent
/// sharing a single [`RangeEncoder`] cursor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_one_rgb_slice_range(
    is_first_slice: bool,
    keyframe: bool,
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    frame_dims: FramePixelDimensions,
    ec: bool,
    seed_states: &[Option<RangePlaneEncoderState>],
    v0v1_prologue: Option<&QuantizationTableSet>,
) -> Result<(Vec<u8>, Vec<Option<RangePlaneEncoderState>>), Error> {
    // §3.8.1.4 / §3.8.1.6: pick the active state-transition table for
    // this Slice's range coder.
    //
    // For v3 (and v0/v1 non-keyframes) the whole Slice — keyframe boolean
    // and Slice Header included — is read with the §3.8.1.6 custom table
    // when `coder_type == 2` (the deltas live in the §4.3 Configuration
    // Record / the governing keyframe, already known before this Slice
    // opens). Mirror of `decode_frame_rgb` / `decode_frame_v0v1_inter`.
    //
    // For a v0/v1 **keyframe** the §4.2.4 deltas are emitted on THIS pass
    // (inside `encode_v0v1_frame_prologue`), so the §4.4 keyframe boolean
    // + §4.2 Parameters must be written with the §3.8.1.5 *default*
    // table; the custom table is swapped in only after the prologue,
    // before the §4.7 Slice Content. Exact mirror of
    // `decode_v0v1_single_slice`'s mid-pass `set_one_state` swap.
    let v0v1_keyframe = v0v1_prologue.is_some() && keyframe;
    let mut re = if cr.coder_type == 2 && !v0v1_keyframe {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        RangeEncoder::with_one_state(&one_state)
    } else {
        RangeEncoder::new()
    };

    if is_first_slice {
        // RFC 9043 §4.4: keyframe boolean at the very start of the first
        // Slice's range-coded region. The caller chooses the value
        // (`true` for an independent keyframe, `false` for an
        // inter-Frame non-keyframe whose per-context coder state
        // continues from the previous Frame per §3.8.1.3). Mirrors
        // `decode_frame_rgb_with_carry`'s `get_br` capture.
        let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
        put_br(&mut re, &mut kf_state, keyframe);
    }

    // v3 writes the §4.6 Slice Header; v0/v1 instead carries the §4.4
    // inline §4.2 Parameters + single §4.1 cascade (keyframe only) and
    // emits no Slice Header.
    match v0v1_prologue {
        Some(qts) if keyframe => {
            crate::config_encode::encode_v0v1_frame_prologue(&mut re, cr, qts)?;
            // §3.8.1.6: now that the §4.2.4 deltas are on the wire, swap
            // the live encoder onto the custom table for the §4.7 Slice
            // Content (no-op for `coder_type != 2`).
            if cr.coder_type == 2 {
                let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
                re.set_one_state(&one_state);
            }
        }
        Some(_) => { /* v0/v1 non-keyframe: no inline Parameters */ }
        None => encode_slice_header_to_encoder(&mut re, header, cr)?,
    }

    let sc = compute_slice_content(header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::LineMajor);

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);

    // Build one persistent line-state per Plane, seeded with the
    // forward-RCT output (or the raw alpha for plane 3). The RCT runs
    // over the input R/G/B Planes restricted to this Slice's pixel
    // rectangle and produces row-major coded-Plane buffers
    // (`coded_bits = bits_per_raw_sample + 1`).
    //
    // §4.6.6 + round-214 contract: per-context state is keyed by the
    // §4.6.6 slot, not per Plane. Two Planes sharing a slot (Cb + Cr
    // on `chroma_planes`) thread one [`RangePlaneEncoderState`]
    // through their back-to-back row encode calls; the §3.8.1.3
    // keyframe-init contract holds via lazy `get_or_insert_with` on
    // first touch of each slot.
    let mut plane_states: Vec<PlaneLineEncodeState> = Vec::with_capacity(primary_color_count);
    let mut plane_slots: Vec<usize> = Vec::with_capacity(primary_color_count);
    let slot_count = header.quant_table_set_index_count;
    // RFC 9043 §3.8.1.3: keyframe → fresh `128` windows (lazy `None`);
    // non-keyframe → resume from the previous Frame's matching Slice
    // (the caller-supplied per-slot seed). Mirror of
    // `decode_frame_rgb_with_carry`.
    let mut per_slot_states: Vec<Option<RangePlaneEncoderState>> = (0..slot_count)
        .map(|s| seed_states.get(s).cloned().flatten())
        .collect();
    let mut per_slot_ctx_count: Vec<Option<usize>> = (0..slot_count).map(|_| None).collect();
    let coded_buffers = forward_rct_for_slice(frame, cr, &sc)?;
    for (p_idx, plane) in sc.planes.iter().enumerate() {
        let qts_slot = match p_idx {
            0 => 0usize,
            1 | 2 if cr.chroma_planes => 1,
            _ if cr.extra_plane => header.quant_table_set_index_count.saturating_sub(1),
            _ => 0,
        };
        let qts_index = (header.quant_table_set_index[qts_slot] as usize)
            .min(quant_table_sets.len().saturating_sub(1));
        let qts = quant_table_sets
            .get(qts_index)
            .ok_or(Error::InvalidQuantTableSetCount(0))?;

        let coded_bits = cr.bits_per_raw_sample + 1;
        plane_states.push(PlaneLineEncodeState::new(
            plane.width as usize,
            plane.height as usize,
            coded_bits,
            qts.tables,
            coded_buffers[p_idx].clone(),
        ));
        plane_slots.push(qts_slot);
        per_slot_ctx_count[qts_slot].get_or_insert(qts.context_count as usize);
    }

    // §4.7 line-major traversal: outer y, inner p. Symmetric inverse of
    // the decoder's two-deep loop.
    let slice_h = sc.slice_pixel_height as usize;
    for y in 0..slice_h {
        for (p_idx, ps) in plane_states.iter_mut().enumerate() {
            if y >= ps.height {
                continue;
            }
            ps.seed_row_border();

            let slot = plane_slots[p_idx];
            let ctx_count = per_slot_ctx_count[slot]
                .expect("range encoder slot context_count was pinned above");
            let rcs =
                per_slot_states[slot].get_or_insert_with(|| RangePlaneEncoderState::new(ctx_count));
            // §3.3.1 alt-median is YCbCr-only — never reached on the
            // RGB encode path (decoder gates the same way: see
            // `decode_frame_rgb`).
            let use_16bit_median = false;

            let row_start = y * ps.width;
            let row_samples = &ps.coded[row_start..row_start + ps.width];

            // Split borrow: rebind the row buffers as &/&mut so the
            // borrow checker accepts disjoint access to `prev` / `prev_prev`
            // / `cur` on the same struct.
            let (prev_prev, prev, cur) = (&ps.prev_prev, &ps.prev, &mut ps.cur);
            RangePlaneEncoder::encode_row(
                &mut re,
                rcs,
                &ps.qtable,
                prev,
                prev_prev,
                cur,
                row_samples,
                ps.width,
                ps.coded_bits,
                use_16bit_median,
            );

            ps.finish_row_and_rotate();
        }
    }

    let body = re.finish();

    // v0/v1 carries no §4.9 Slice Footer (`version >= 3`-only); the body
    // IS the Frame's content. v3 wraps each Slice with the §4.9 footer
    // (its §4.9.3 CRC parity solved by construction).
    if v0v1_prologue.is_some() {
        return Ok((body, per_slot_states));
    }
    let slice_bytes = encode_slice_footer(&body, ec, SliceErrorStatus::NoError)?;
    Ok((slice_bytes, per_slot_states))
}

/// Per-Plane Golomb-Rice encoder state for the RGB line-major path.
///
/// Encoder-side mirror of the `coder_type == 0` branch of
/// [`PlaneLineState`]. Each Plane keeps its §3.1 border row buffers
/// alive across the §4.7 line-major interleave, exactly as the decoder
/// keeps each Plane's border buffers alive across the matching
/// `for y { for p { Line(p, y) } }` traversal.
///
/// Per RFC 9043 §4.6.6 the per-context VLC window (`drift`, `error_sum`,
/// `bias`, `count`) is *shared* across Planes routed to the same §4.6.6
/// slot (luma / chroma / extra-plane), and is therefore **not** held
/// per Plane — the driver owns one [`LineDecoderState`] per slot. The
/// run-mode triple (`run_index`, `run_mode`, `run_count`) per
/// §3.8.2.2.1 is per-Plane (it resets at the start of each Plane and
/// the §4.7 line-major interleave reads back-to-back across Planes
/// sharing a slot, so a slot-level triple would be wrong); the driver
/// saves / loads the triple around every per-row encode.
///
/// Row buffers use the [`BORDER_WIDTH`] (=2) convention of
/// [`crate::sample_diff`] / [`encode_line`] (NOT the
/// `BORDER_LEFT`/`BORDER_RIGHT` convention of
/// [`crate::reconstruct::PlaneReconstructor`]) — the per-row Golomb-Rice
/// encoder writes into `cur[BORDER_WIDTH .. BORDER_WIDTH + width]` and
/// reads `prev[BORDER_WIDTH + x]` etc., so the buffer width is
/// `BORDER_WIDTH + width + BORDER_WIDTH`.
struct PlaneLineGolombEncodeState {
    /// `plane_pixel_width[p]` (§4.8.1). RGB never subsamples.
    width: usize,
    /// `plane_pixel_height[p]` (§4.7.2).
    height: usize,
    /// `bits_per_raw_sample + 1` (§3.8 RCT coded-Sample width).
    coded_bits: u32,
    /// The §3.4 Quantization Table Set this Plane selected.
    qtable: QuantTableSet,
    /// `[BORDER_WIDTH | W samples | BORDER_WIDTH]` row buffers — two
    /// rows above, one row above, the current row. The current row's
    /// real-Sample cells hold the just-encoded *Sample* values (so the
    /// next row's median predictor can read them) — `encode_line` writes
    /// diffs into the row while encoding; the outer loop overwrites
    /// those with the actual Sample values before rotating.
    prev_prev: Vec<i32>,
    prev: Vec<i32>,
    cur: Vec<i32>,
    /// Row-major forward-RCT coded Sample buffer (`width * height`).
    coded: Vec<i32>,
}

impl PlaneLineGolombEncodeState {
    fn new(
        width: usize,
        height: usize,
        coded_bits: u32,
        qtable: QuantTableSet,
        coded: Vec<i32>,
    ) -> Self {
        let stride = BORDER_WIDTH + width + BORDER_WIDTH;
        debug_assert_eq!(coded.len(), width * height);
        Self {
            width,
            height,
            coded_bits,
            qtable,
            prev_prev: vec![0i32; stride],
            prev: vec![0i32; stride],
            cur: vec![0i32; stride],
            coded,
        }
    }
}

/// Encode one Slice on the RGB / line-major Golomb-Rice path
/// (`coder_type == 0`).
///
/// Mirrors the `coder_type == 0` branch of [`decode_frame_rgb`]: the
/// keyframe bit (slice 0 only) and the §4.6 SliceHeader are written
/// through a [`RangeEncoder`]; that encoder is then `finish()`-ed,
/// landing on a byte boundary, and a fresh [`BitWriter`] takes over for
/// the §4.7 / §4.8 line-major Golomb-Rice SliceContent. The §4.9
/// SliceFooter wraps the concatenated body (range-coded header bytes ++
/// Golomb-Rice content bytes) with the §4.9.3 CRC parity solved by
/// construction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_one_rgb_slice_golomb(
    is_first_slice: bool,
    keyframe: bool,
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    frame_dims: FramePixelDimensions,
    ec: bool,
    seed_states: &[Option<LineDecoderState>],
    v0v1_prologue: Option<&QuantizationTableSet>,
) -> Result<(Vec<u8>, Vec<Option<LineDecoderState>>), Error> {
    // ---- §4.4 keyframe + §4.6 SliceHeader (range-coded) ----
    //
    // The decoder reads these through a `RangeDecoder::new(body)` cursor
    // and uses `rc.position()` to find the byte boundary where the
    // Golomb-Rice SliceContent begins; `RangeEncoder::finish()` here is
    // the symmetric counterpart — its returned bytes are exactly that
    // prefix.
    let mut re = RangeEncoder::new();
    if is_first_slice {
        // RFC 9043 §4.4: caller-chosen keyframe value (`true` for an
        // independent keyframe, `false` for an inter-Frame non-keyframe
        // whose per-context VLC windows continue from the previous
        // Frame per §3.8.2.5).
        let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
        put_br(&mut re, &mut kf_state, keyframe);
    }
    // v3 writes the §4.6 Slice Header; v0/v1 carries the §4.4 inline §4.2
    // Parameters + single §4.1 cascade (keyframe only) and no Slice Header.
    let is_v0v1 = v0v1_prologue.is_some();
    match v0v1_prologue {
        Some(qts) if keyframe => {
            crate::config_encode::encode_v0v1_frame_prologue(&mut re, cr, qts)?;
        }
        Some(_) => {}
        None => encode_slice_header_to_encoder(&mut re, header, cr)?,
    }
    // The range-coded prologue / Slice Header switches to the Golomb-Rice
    // Slice Content at a byte boundary. The versions-0/1 single-Slice path
    // uses §3.8.1.1.1 Sentinel-mode termination (a discarded state-129
    // symbol marks the boundary the decoder recovers via
    // `terminate_sentinel`); the v3 multi-Slice path uses the plain
    // `finish()` byte-alignment the v3 frame decoder locates with
    // `rc.position()`.
    let mut body = if is_v0v1 {
        re.terminate_sentinel()
    } else {
        re.finish()
    };

    let sc = compute_slice_content(header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::LineMajor);

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    let coded_buffers = forward_rct_for_slice(frame, cr, &sc)?;

    // §3.8.2.2.1 + §4.6.6: per-Plane Golomb-Rice state. The
    // per-context VLC window lives per §4.6.6 *slot* (two Planes
    // routed to the same slot — Cb + Cr on every `chroma_planes ==
    // true` Slice — share one window across the §4.7 line-major
    // interleave). The run-mode triple (`run_index`, `run_mode`,
    // `run_count`) is per-Plane per §3.8.2.2.1 and is swapped into /
    // out of the slot state around every row encode.
    let slot_count = header.quant_table_set_index_count;
    // RFC 9043 §3.8.2.5: keyframe → fresh `128` VLC windows (lazy
    // `None`); non-keyframe → resume each slot's per-context VLC window
    // from the previous Frame's matching Slice (caller-supplied seed).
    // The §3.8.2.2.1 run-mode triple inside a seeded `LineDecoderState`
    // is irrelevant — every per-row encode overwrites `run_index` /
    // `run_mode` / `run_count` from this Plane's own `per_plane_run_triple`
    // before encoding (see below), so only the carried per-context VLC
    // window (`drift` / `error_sum` / `bias` / `count`) survives — mirror
    // of `decode_frame_rgb_with_carry`.
    let mut per_slot_state: Vec<Option<LineDecoderState>> = (0..slot_count)
        .map(|s| seed_states.get(s).cloned().flatten())
        .collect();
    let mut per_slot_ctx_count: Vec<Option<usize>> = (0..slot_count).map(|_| None).collect();
    let mut plane_slots: Vec<usize> = Vec::with_capacity(primary_color_count);
    let mut per_plane_run_triple: Vec<(u32, u8, i32)> = Vec::with_capacity(primary_color_count);
    let mut plane_states: Vec<PlaneLineGolombEncodeState> = Vec::with_capacity(primary_color_count);
    for (p_idx, plane) in sc.planes.iter().enumerate() {
        // §4.6.6 quant_table_set_index mapping (mirrors the YCbCr
        // encoder and the RGB decoder).
        let qts_slot = match p_idx {
            0 => 0usize,
            1 | 2 if cr.chroma_planes => 1,
            _ if cr.extra_plane => header.quant_table_set_index_count.saturating_sub(1),
            _ => 0,
        };
        let qts_index = (header.quant_table_set_index[qts_slot] as usize)
            .min(quant_table_sets.len().saturating_sub(1));
        let qts = quant_table_sets
            .get(qts_index)
            .ok_or(Error::InvalidQuantTableSetCount(0))?;

        let coded_bits = cr.bits_per_raw_sample + 1;
        plane_states.push(PlaneLineGolombEncodeState::new(
            plane.width as usize,
            plane.height as usize,
            coded_bits,
            qts.tables,
            coded_buffers[p_idx].clone(),
        ));
        plane_slots.push(qts_slot);
        per_slot_ctx_count[qts_slot].get_or_insert((qts.context_count as usize).max(1));
        // §3.8.2.2.1: each Plane starts with a fresh run triple.
        per_plane_run_triple.push((0u32, 0u8, 0i32));
    }

    // ---- §4.8 SliceContent (Golomb-Rice, byte-aligned tail) ----
    let mut bw = BitWriter::new();

    // §4.7 line-major traversal: outer y, inner p. Symmetric inverse of
    // the decoder's two-deep loop in `decode_frame_rgb` for
    // `coder_type == 0`.
    let slice_h = sc.slice_pixel_height as usize;
    for y in 0..slice_h {
        for (p_idx, ps) in plane_states.iter_mut().enumerate() {
            if y >= ps.height {
                continue;
            }
            let width = ps.width;
            let bits = ps.coded_bits;

            // ---- Per-row §3.3 / §3.8 sample-difference derivation, in
            // the modified-YCbCr coded space (the forward-RCT output).
            // The `t` neighbour reads `prev[BORDER_WIDTH + x]`, which
            // holds the prior row's Sample values (the outer loop wrote
            // them into `cur` before the rotate). ----
            let row_start = y * width;
            let row_samples = &ps.coded[row_start..row_start + width];
            let prev_samples = &ps.prev[BORDER_WIDTH..BORDER_WIDTH + width];
            let diffs = sample_diffs_for_row_coded(row_samples, prev_samples, bits);

            // ---- §3.1 left-of-slice column: sample[y][-1] =
            // sample[y-1][0]. Mirrors `reconstruct.rs` / the YCbCr
            // Golomb encoder. ----
            ps.cur[0] = 0;
            ps.cur[BORDER_WIDTH - 1] = ps.prev[BORDER_WIDTH];

            // ---- Encode the row through the Golomb-Rice bit engine.
            // `encode_line` writes the reconstructed Samples into
            // `current_row` so the §3.5 context + run predicate match the
            // decoder, and emits Golomb-Rice bits for `diffs`. ----
            {
                let slot = plane_slots[p_idx];
                let ctx_count =
                    per_slot_ctx_count[slot].expect("Golomb slot context_count was pinned above");
                let state =
                    per_slot_state[slot].get_or_insert_with(|| LineDecoderState::new(ctx_count));
                // §3.8.2.2.1 + §4.6.6: load this Plane's run triple
                // into the slot's VLC window for the row, encode, then
                // save the triple back — the slot's VLC fields keep
                // evolving across Planes that share the slot, the run
                // triple belongs to this Plane only.
                let (ri, rm, rc) = per_plane_run_triple[p_idx];
                state.run_index = ri;
                state.run_mode = rm;
                state.run_count = rc;
                let mut neighbours = LineNeighborBuffers {
                    prev_row: &ps.prev,
                    prev_prev_row: &ps.prev_prev,
                    current_row: &mut ps.cur,
                    plane_pixel_width: width as u32,
                };
                encode_line(&mut bw, state, &ps.qtable, &mut neighbours, &diffs, bits)?;
                per_plane_run_triple[p_idx] = (state.run_index, state.run_mode, state.run_count);
            }

            // `encode_line` left `cur` holding this row's reconstructed
            // Sample values (it pre-fills current_row with Samples, not
            // diffs, so the §3.5 context matches the decoder), so the
            // next row's §3.3 median predictor already reads Sample
            // values in the `l` / `t` / `tl` positions.
            // §3.1 right border: sample[y][W] = sample[y][W-1].
            ps.cur[BORDER_WIDTH + width] = ps.cur[BORDER_WIDTH + width - 1];

            // Rotate: prev_prev <- prev <- cur, new cur zeroed.
            core::mem::swap(&mut ps.prev_prev, &mut ps.prev);
            core::mem::swap(&mut ps.prev, &mut ps.cur);
            ps.cur.iter_mut().for_each(|s| *s = 0);
        }
    }

    // §3.8.2 "padded with zeroes": flush the BitWriter, zero-padding
    // the final partial byte so the §4.9 footer sits on a byte boundary.
    let content = bw.finish();
    body.extend_from_slice(&content);

    // v0/v1 carries no §4.9 Slice Footer (`version >= 3`-only).
    if v0v1_prologue.is_some() {
        return Ok((body, per_slot_state));
    }
    // §4.9 SliceFooter — `encode_slice_footer` solves the §4.9.3 CRC
    // parity so the whole-Slice residue is zero by construction.
    let slice_bytes = encode_slice_footer(&body, ec, SliceErrorStatus::NoError)?;
    Ok((slice_bytes, per_slot_state))
}

/// Per-row §3.3 / §3.8 sample-difference derivation for the RGB
/// Golomb-Rice path.
///
/// Mirrors `sample_diffs_for_row` in `frame_encode.rs`, but takes
/// `bits` as the *coded* bit width (`bits_per_raw_sample + 1` for RCT
/// Planes), so the §3.8 modular wrap matches the wider coded-Sample
/// space the modified-YCbCr Planes live in.
///
/// Given a row of *coded* target Samples and the row of coded Sample
/// values immediately above (for the §3.3 `t` neighbour), returns the
/// row of signed `diff` values such that
/// `reconstruct_sample(median(l, t, tl), diff, bits) == sample` for
/// every column. The `l` / `tl` neighbours are taken from
/// `row_samples[x-1]` / `prev_row_samples[x-1]` consistent with
/// `PlaneReconstructor::reconstruct_row`.
fn sample_diffs_for_row_coded(
    row_samples: &[i32],
    prev_row_samples: &[i32],
    bits: u32,
) -> Vec<i32> {
    let w = row_samples.len();
    debug_assert_eq!(
        prev_row_samples.len(),
        w,
        "prev_row sample slice must have the same width as row_samples"
    );
    let half = 1i32 << (bits - 1);
    let modulus = 1i32 << bits;
    let mut diffs = Vec::with_capacity(w);
    for x in 0..w {
        // §3.1 left border: sample[y][-1] = sample[y-1][0]. So the very
        // first column's `l` equals the first column of the previous
        // row; the `tl` border above-the-slice corner is 0.
        let l = if x == 0 {
            prev_row_samples[0]
        } else {
            row_samples[x - 1]
        };
        let t = prev_row_samples[x];
        let tl = if x == 0 { 0 } else { prev_row_samples[x - 1] };
        let pred = median_predict(l, t, tl);

        let raw = row_samples[x] - pred;
        let mut diff = raw % modulus;
        if diff >= half {
            diff -= modulus;
        } else if diff < -half {
            diff += modulus;
        }
        diffs.push(diff);
    }
    diffs
}

/// Apply the §3.7.1 forward RCT to each Slice's pixel rectangle and
/// return the row-major coded modified-YCbCr Plane buffers (and the raw
/// alpha copy, when present) ready for the per-Plane line encoder.
///
/// The §3.7.1 forward transform is (Figure 6 general):
///
/// ```text
///   Cb = b - g
///   Cr = r - g
///   Y  = g + ((Cb + Cr) >> 2)
/// ```
///
/// and the §3.7.2.1 exception (Figure 8, used iff `9 <= bits <= 15 &&
/// !extra_plane`):
///
/// ```text
///   Cb = g - b
///   Cr = r - b
///   Y  = b + ((Cb + Cr) >> 2)
/// ```
///
/// Cb / Cr are stored with the §3.7.2 positive offset `1 <<
/// bits_per_raw_sample` so the coded modified-YCbCr Samples are
/// non-negative on the wire. The transparency Plane is **not** RCT-
/// transformed: the alpha buffer is the input Sample row-major copy
/// masked to `0 .. 2^bits_per_raw_sample`.
fn forward_rct_for_slice(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    sc: &crate::slice_content::SliceContent,
) -> Result<Vec<Vec<i32>>, Error> {
    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    if frame.planes.len() < primary_color_count {
        return Err(Error::InvalidQuantTableSetCount(0));
    }

    let bits = cr.bits_per_raw_sample;
    let offset = 1i64 << bits;
    let use_exception = (9..=15).contains(&bits) && !cr.extra_plane;
    // §3.8 RCT coded Sample mask: coded width is bits + 1, but the
    // modular wrap on Cb / Cr (`b - g + offset`) naturally lands in
    // `0 .. 2^(bits+1)` so an explicit mask is defensive only.
    let coded_mask = if bits + 1 >= 32 {
        !0i32
    } else {
        ((1i64 << (bits + 1)) - 1) as i32
    };

    let r_plane = &frame.planes[0];
    let g_plane = &frame.planes[1];
    let b_plane = &frame.planes[2];
    let alpha_plane = if cr.extra_plane {
        Some(&frame.planes[3])
    } else {
        None
    };

    let dst_w = r_plane.width as usize;
    let dst_h = r_plane.height as usize;
    let origin_x = sc.slice_pixel_x as usize;
    let origin_y = sc.slice_pixel_y as usize;

    let mut coded = Vec::with_capacity(primary_color_count);
    for plane in &sc.planes {
        // RGB never subsamples, so every Plane has the same width / height
        // as Plane 0. Pre-allocate matching `width * height` row-major
        // buffers; per-Plane data is written by the loops below.
        debug_assert_eq!(
            plane.width, sc.planes[0].width,
            "RGB Planes are unsubsampled"
        );
        debug_assert_eq!(
            plane.height, sc.planes[0].height,
            "RGB Planes are unsubsampled"
        );
        coded.push(vec![0i32; plane.width as usize * plane.height as usize]);
    }

    let slice_w = sc.planes[0].width as usize;
    let slice_h = sc.planes[0].height as usize;

    for y in 0..slice_h {
        let sy = origin_y + y;
        if sy >= dst_h {
            break;
        }
        for x in 0..slice_w {
            let sx = origin_x + x;
            if sx >= dst_w {
                break;
            }
            let src = sy * dst_w + sx;
            let r = r_plane.samples[src] as i64;
            let g = g_plane.samples[src] as i64;
            let b = b_plane.samples[src] as i64;

            let (y_val, cb, cr_val) = if use_exception {
                let cb = g - b;
                let cr = r - b;
                let y_v = b + ((cb + cr) >> 2);
                (y_v, cb + offset, cr + offset)
            } else {
                let cb = b - g;
                let cr = r - g;
                let y_v = g + ((cb + cr) >> 2);
                (y_v, cb + offset, cr + offset)
            };

            let dst_idx = y * slice_w + x;
            coded[0][dst_idx] = (y_val as i32) & coded_mask;
            coded[1][dst_idx] = (cb as i32) & coded_mask;
            coded[2][dst_idx] = (cr_val as i32) & coded_mask;
        }
    }

    if let Some(ap) = alpha_plane {
        let mask = if bits >= 32 {
            !0i32
        } else {
            (1i32 << bits) - 1
        };
        for y in 0..slice_h {
            let sy = origin_y + y;
            if sy >= dst_h {
                break;
            }
            for x in 0..slice_w {
                let sx = origin_x + x;
                if sx >= dst_w {
                    break;
                }
                let src = sy * dst_w + sx;
                let dst_idx = y * slice_w + x;
                coded[3][dst_idx] = ap.samples[src] & mask;
            }
        }
    }

    Ok(coded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NUM_TRANSITION_DELTAS;

    fn rgb_cr(coder_type: u32, bits: u32, extra: bool) -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            version: Ffv1Version::V3,
            micro_version: Some(4),
            coder_type,
            state_transition_delta: [0; NUM_TRANSITION_DELTAS],
            colorspace_type: ColorspaceType::Rgb,
            bits_per_raw_sample: bits,
            chroma_planes: true,
            log2_h_chroma_subsample: 0,
            log2_v_chroma_subsample: 0,
            extra_plane: extra,
            num_h_slices: Some(1),
            num_v_slices: Some(1),
            quant_table_set_count: Some(1),
            ec: Some(0),
            intra: Some(false),
            initial_state_delta: None,
        }
    }

    // ---- inverse RCT round-trip (the §3.7.1 math) ----------------------

    /// Forward RCT (Figure 6 general) → coded modified-YCbCr, with the
    /// §3.7.2 positive offset applied to Cb/Cr. Used only to construct
    /// known-good coded samples for the inverse tests below.
    fn forward_rct_general(r: i64, g: i64, b: i64, bits: u32) -> (i64, i64, i64) {
        let offset = 1i64 << bits;
        let cb = b - g;
        let cr = r - g;
        let y = g + ((cb + cr) >> 2);
        (y, cb + offset, cr + offset)
    }

    fn forward_rct_exception(r: i64, g: i64, b: i64, bits: u32) -> (i64, i64, i64) {
        let offset = 1i64 << bits;
        let cb = g - b;
        let cr = r - b;
        let y = b + ((cb + cr) >> 2);
        (y, cb + offset, cr + offset)
    }

    /// Run the inverse-RCT core for a single pixel via the public
    /// driver's helper by stuffing one Sample into a 1x1 Slice's
    /// plane-state and reading back the recovered RGB.
    fn inverse_one(coded: (i64, i64, i64), bits: u32, exception: bool) -> (i32, i32, i32) {
        let cr = rgb_cr(if exception { 2 } else { 1 }, bits, false);
        // Build minimal 1x1 plane states holding the coded samples.
        let mk = |val: i64| {
            let mut ps = PlaneLineState::new(1, 1, bits + 1, [[0i32; 256]; 5]);
            ps.out[0] = val as i32;
            ps
        };
        let _ = cr.coder_type; // unused: 1x1 inverse RCT path needs only the colour Plane samples
        let states = vec![mk(coded.0), mk(coded.1), mk(coded.2)];
        let mut planes = vec![
            DecodedFramePlane {
                plane_index: 0,
                width: 1,
                height: 1,
                samples: vec![0],
            },
            DecodedFramePlane {
                plane_index: 1,
                width: 1,
                height: 1,
                samples: vec![0],
            },
            DecodedFramePlane {
                plane_index: 2,
                width: 1,
                height: 1,
                samples: vec![0],
            },
        ];
        apply_inverse_rct_and_blit(&states, &mut planes, &cr, 0, 0, 1, 1);
        (
            planes[0].samples[0],
            planes[1].samples[0],
            planes[2].samples[0],
        )
    }

    #[test]
    fn inverse_rct_general_roundtrips_8bit() {
        let bits = 8;
        for (r, g, b) in [(0i64, 0, 0), (255, 255, 255), (200, 100, 50), (17, 240, 3)] {
            let coded = forward_rct_general(r, g, b, bits);
            let (rr, gg, bb) = inverse_one(coded, bits, false);
            assert_eq!(
                (rr as i64, gg as i64, bb as i64),
                (r, g, b),
                "rgb=({r},{g},{b})"
            );
        }
    }

    #[test]
    fn inverse_rct_general_roundtrips_16bit() {
        let bits = 16;
        for (r, g, b) in [(0i64, 0, 0), (65535, 65535, 65535), (40000, 1000, 60000)] {
            let coded = forward_rct_general(r, g, b, bits);
            let (rr, gg, bb) = inverse_one(coded, bits, false);
            assert_eq!(
                (rr as i64, gg as i64, bb as i64),
                (r, g, b),
                "rgb=({r},{g},{b})"
            );
        }
    }

    #[test]
    fn inverse_rct_exception_roundtrips_12bit() {
        // 9..=15 bits with extra_plane==0 uses Figure 9.
        let bits = 12;
        for (r, g, b) in [(0i64, 0, 0), (4095, 4095, 4095), (3000, 100, 2000)] {
            let coded = forward_rct_exception(r, g, b, bits);
            let (rr, gg, bb) = inverse_one(coded, bits, true);
            assert_eq!(
                (rr as i64, gg as i64, bb as i64),
                (r, g, b),
                "rgb=({r},{g},{b})"
            );
        }
    }

    #[test]
    fn inverse_rct_is_modular_for_out_of_gamut() {
        // The transform is exact modulo 2^bits even when intermediate
        // values escape the gamut; the recovered Samples stay in range.
        let bits = 8;
        let (r, g, b) = inverse_one((300, 100, 400), bits, false);
        for v in [r, g, b] {
            assert!((0..256).contains(&v), "sample {v} out of 8-bit range");
        }
    }

    // ---- driver-level guards ------------------------------------------

    #[test]
    fn rejects_v0() {
        let mut cr = rgb_cr(1, 8, false);
        cr.version = Ffv1Version::V0;
        let r = decode_frame_rgb(
            &[0u8; 64],
            &cr,
            &[],
            FramePixelDimensions::new(8, 8).unwrap(),
            true,
        );
        assert!(matches!(r, Err(Error::SliceRequiresVersion3)));
    }

    #[test]
    fn rejects_ycbcr_config() {
        let mut cr = rgb_cr(1, 8, false);
        cr.colorspace_type = ColorspaceType::YCbCr;
        let r = decode_frame_rgb(
            &[0u8; 64],
            &cr,
            &[],
            FramePixelDimensions::new(8, 8).unwrap(),
            true,
        );
        assert!(matches!(r, Err(Error::ColorspaceLayoutNotImplemented)));
    }

    #[test]
    fn rejects_unsupported_coder_type() {
        let mut cr = rgb_cr(1, 8, false);
        cr.coder_type = 7;
        let r = decode_frame_rgb(
            &[0u8; 64],
            &cr,
            &[],
            FramePixelDimensions::new(8, 8).unwrap(),
            true,
        );
        assert!(matches!(r, Err(Error::UnsupportedCoderType(7))));
    }

    #[test]
    fn propagates_truncated_footer() {
        let cr = rgb_cr(1, 8, false);
        let r = decode_frame_rgb(
            &[0u8; 4],
            &cr,
            &[],
            FramePixelDimensions::new(8, 8).unwrap(),
            true,
        );
        assert!(matches!(r, Err(Error::TruncatedSliceFooter)));
    }

    #[test]
    fn plane_line_state_allocates_correct_shapes() {
        let ps = PlaneLineState::new(8, 6, 9, [[0i32; 256]; 5]);
        assert_eq!(ps.out.len(), 48);
        assert_eq!(ps.prev.len(), BORDER_LEFT + 8 + BORDER_RIGHT);
        assert_eq!(ps.prev_prev.len(), BORDER_LEFT + 8 + BORDER_RIGHT);
        assert_eq!(ps.cur.len(), BORDER_LEFT + 8 + BORDER_RIGHT);
        assert_eq!(ps.width, 8);
        assert_eq!(ps.height, 6);
        assert_eq!(ps.coded_bits, 9);
    }

    #[test]
    fn plane_line_state_seed_row_border_reads_prev() {
        // After r220 the per-context entropy state is held at slot
        // level by the driver loops, so `PlaneLineState` carries only
        // §3.1 border buffers + output. Seeding the row border copies
        // the prior row's left-edge Sample into `cur[BORDER_LEFT-1]`
        // (the §3.1 "sample[y][-1] = sample[y-1][0]" rule).
        let mut ps = PlaneLineState::new(4, 4, 9, [[0i32; 256]; 5]);
        ps.prev[BORDER_LEFT] = 17;
        ps.seed_row_border();
        assert_eq!(ps.cur[BORDER_LEFT - 1], 17);
        assert_eq!(ps.cur[0], 0);
    }

    // ---- §3.8.1.3 / §3.8.2.5 inter-Frame coder-state carry ------------
    //
    // The RGB / line-major analogue of the YCbCr carry tested in
    // `tests/nonkeyframe_carry.rs`. These unit tests drive the read-side
    // carry mechanics directly: they prove the carry channel is threaded,
    // snapshotted per-Slice / per-§4.6.6-slot, and **ignored on a
    // keyframe** (the §3.8.1.3 "When the keyframe value is 1, all state
    // variables are set to their initial state" branch). The full
    // write-side carry (`encode_frame_rgb_with_carry`) closes the
    // symmetry; an end-to-end multi-Frame RGB non-keyframe round-trip
    // through both halves lives in `tests/rgb_nonkeyframe_carry.rs`.

    /// A multi-context quant table set so a populated carry holds a
    /// non-degenerate per-context window (a 1-context table would make
    /// the snapshot trivially small).
    fn multi_ctx_qts(context_count: u32) -> QuantizationTableSet {
        let mut tables = [[0i32; 256]; 5];
        // A non-flat first sub-table spreads Samples across contexts.
        for (i, slot) in tables[0].iter_mut().enumerate() {
            *slot = (i as i32 % context_count as i32) - (context_count as i32 / 2);
        }
        QuantizationTableSet {
            tables,
            context_count,
        }
    }

    /// Encode one keyframe RGB Frame and return its bytes plus the
    /// inputs the decode drivers need. The pixel pattern is a ramp so
    /// the per-context coder state actually evolves through the Slice.
    fn keyframe_rgb_frame(
        coder_type: u32,
        ec: bool,
    ) -> (
        Vec<u8>,
        Ffv1ConfigurationRecord,
        Vec<QuantizationTableSet>,
        FramePixelDimensions,
    ) {
        let (w, h) = (6u32, 4u32);
        let mut cr = rgb_cr(coder_type, 8, false);
        cr.ec = Some(u32::from(ec));
        let n = (w * h) as usize;
        let r: Vec<i32> = (0..n).map(|i| (i * 3 % 200) as i32).collect();
        let g: Vec<i32> = (0..n).map(|i| (i * 5 % 200) as i32).collect();
        let b: Vec<i32> = (0..n).map(|i| (i * 7 % 200) as i32).collect();
        let mut frame = DecodedFrame {
            planes: vec![
                DecodedFramePlane {
                    plane_index: 0,
                    width: w,
                    height: h,
                    samples: r,
                },
                DecodedFramePlane {
                    plane_index: 1,
                    width: w,
                    height: h,
                    samples: g,
                },
                DecodedFramePlane {
                    plane_index: 2,
                    width: w,
                    height: h,
                    samples: b,
                },
            ],
            width: w,
            height: h,
            bits_per_raw_sample: 8,
            colorspace: ColorspaceType::Rgb,
            keyframe: true,
            slice_headers: Vec::new(),
        };
        let mut idx = [0u32; crate::slice_header::MAX_QUANT_TABLE_SET_INDEXES];
        // quant_table_set_index_count for RGB v3 chroma_planes==true is
        // 1 (luma) + 1 (chroma) = 2; all slots select set 0.
        let header = Ffv1SliceHeader {
            slice_x: 0,
            slice_y: 0,
            slice_width: 1,
            slice_height: 1,
            quant_table_set_index_count: 2,
            quant_table_set_index: {
                idx[0] = 0;
                idx[1] = 0;
                idx
            },
            picture_structure: crate::PictureStructure::Progressive,
            picture_structure_raw: 0,
            sar_num: 0,
            sar_den: 0,
        };
        frame.slice_headers = vec![header.clone()];
        let qts = vec![multi_ctx_qts(8)];
        let bytes = encode_frame_rgb(&frame, &cr, &qts, std::slice::from_ref(&header), ec)
            .expect("encode keyframe RGB frame");
        (bytes, cr, qts, FramePixelDimensions::new(w, h).unwrap())
    }

    #[test]
    fn carry_aware_decode_matches_stateless_on_keyframe() {
        for coder_type in [0u32, 1, 2] {
            let (bytes, cr, qts, dims) = keyframe_rgb_frame(coder_type, true);
            let stateless =
                decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("stateless decode");
            let mut carry: Option<Ffv1FrameCarry> = None;
            let carried = decode_frame_rgb_with_carry(
                &bytes,
                &cr,
                &qts,
                dims,
                true,
                DecodeOptions::default(),
                &mut carry,
            )
            .expect("carry-aware decode");
            assert!(carried.keyframe, "fixture Frame is a keyframe");
            assert_eq!(
                stateless.planes[0].samples, carried.planes[0].samples,
                "coder_type {coder_type}: carry-aware keyframe decode must match the stateless driver"
            );
            // The end-of-Frame snapshot is published for the next Frame.
            assert!(carry.is_some(), "carry channel populated after decode");
        }
    }

    #[test]
    fn keyframe_decode_ignores_a_stale_carry() {
        // §3.8.1.3 / §3.8.2.5: a keyframe re-initialises every per-context
        // window to its 128 state, so feeding a *garbage* carry on entry
        // must produce byte-identical output to a fresh decode — the carry
        // is read only on a non-keyframe.
        for coder_type in [0u32, 1, 2] {
            let (bytes, cr, qts, dims) = keyframe_rgb_frame(coder_type, true);
            let clean = decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("clean decode");

            // First decode to harvest a real (non-empty) carry shape, then
            // re-inject it into a keyframe decode of the same bytes.
            let mut harvest: Option<Ffv1FrameCarry> = None;
            decode_frame_rgb_with_carry(
                &bytes,
                &cr,
                &qts,
                dims,
                true,
                DecodeOptions::default(),
                &mut harvest,
            )
            .expect("harvest carry");
            let mut stale = harvest; // non-None, holds the Frame's slot windows
            let with_stale = decode_frame_rgb_with_carry(
                &bytes,
                &cr,
                &qts,
                dims,
                true,
                DecodeOptions::default(),
                &mut stale,
            )
            .expect("keyframe decode with stale carry");
            assert_eq!(
                clean.planes[0].samples, with_stale.planes[0].samples,
                "coder_type {coder_type}: keyframe must ignore the carried state"
            );
            assert_eq!(clean.planes[1].samples, with_stale.planes[1].samples);
            assert_eq!(clean.planes[2].samples, with_stale.planes[2].samples);
        }
    }

    #[test]
    fn session_threads_rgb_carry_across_keyframes() {
        // The stateful session routes RGB Frames through the carry-aware
        // driver; two keyframe Frames decode identically through the
        // session and the stateless driver, and both count.
        let (bytes, cr, qts, dims) = keyframe_rgb_frame(1, true);
        let mut session = crate::Ffv1DecodeSession::new(cr.clone(), qts.clone(), dims, true);
        let d0 = session.decode_next_frame(&bytes).expect("frame 0");
        let d1 = session.decode_next_frame(&bytes).expect("frame 1");
        let stateless = decode_frame_rgb(&bytes, &cr, &qts, dims, true).expect("stateless");
        assert_eq!(d0.planes[0].samples, stateless.planes[0].samples);
        assert_eq!(d1.planes[0].samples, stateless.planes[0].samples);
        assert_eq!(session.frames_observed(), 2);
    }
}
