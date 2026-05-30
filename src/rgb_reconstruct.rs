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
//! Plane still carries its own §3.5 context state, §3.8.2 run state, and
//! §3.1 border neighbours across its (non-contiguous) Lines. This module
//! therefore keeps one [`PlaneLineState`] per Plane alive for the whole
//! Slice and steps a single Line per Plane each outer-`y` iteration,
//! reusing the exact per-Line reconstruction the YCbCr path uses
//! ([`PlaneReconstructor::reconstruct_row`] for `coder_type == 0`,
//! [`RangePlaneReconstructor::reconstruct_row`] for `coder_type >= 1`).
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
use crate::frame::{DecodedFrame, DecodedFramePlane};
use crate::predictor::{median_predict, QuantTableSet};
use crate::quant_table::QuantizationTableSet;
use crate::range_coder::{RangeDecoder, RangeEncoder, PARAMETERS_INITIAL_STATE};
use crate::range_encode::{RangePlaneEncoder, RangePlaneEncoderState};
use crate::range_reconstruct::{RangePlaneReconstructor, RangePlaneState};
use crate::reconstruct::{PlaneEntropyState, PlaneReconstructor, BORDER_LEFT, BORDER_RIGHT};
use crate::sample_diff::{encode_line, LineDecoderState, LineNeighborBuffers, BORDER_WIDTH};
use crate::slice_content::{compute_slice_content, FramePixelDimensions, PlaneTraversal};
use crate::slice_footer::{encode_slice_footer, parse_slice_footer, SliceErrorStatus};
use crate::slice_header::{
    encode_slice_header_to_encoder, parse_slice_header_from_decoder, Ffv1SliceHeader,
};
use crate::symbol::put_br;
use crate::trailer_chain::walk_trailer_chain;
use crate::Error;

/// One Plane's persistent line-major reconstruction state.
///
/// Carries the §3.1 border row buffers (`prev_prev`, `prev`, `cur`) and
/// the per-coder entropy state across the Plane's non-contiguous Lines,
/// plus the row-major output samples committed so far.
struct PlaneLineState {
    /// `plane_pixel_width[p]` (§4.8.1). For valid RGB streams every
    /// Plane has full slice width (RGB never subsamples).
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
    /// Row-major reconstructed Plane (`width * height`).
    out: Vec<i32>,
    /// Golomb-Rice entropy state (`coder_type == 0` only).
    gr_state: Option<PlaneEntropyState>,
    /// Range entropy state (`coder_type >= 1` only).
    rc_state: Option<RangePlaneState>,
}

impl PlaneLineState {
    fn new(
        width: usize,
        height: usize,
        coded_bits: u32,
        qtable: QuantTableSet,
        context_count: usize,
        coder_type: u32,
    ) -> Self {
        let stride = BORDER_LEFT + width + BORDER_RIGHT;
        let (gr_state, rc_state) = if coder_type == 0 {
            let mut s = PlaneEntropyState::new(context_count.max(1));
            s.reset_run_state();
            (Some(s), None)
        } else {
            (None, Some(RangePlaneState::new(context_count)))
        };
        Self {
            width,
            height,
            coded_bits,
            qtable,
            prev_prev: vec![0i32; stride],
            prev: vec![0i32; stride],
            cur: vec![0i32; stride],
            out: vec![0i32; width.saturating_mul(height)],
            gr_state,
            rc_state,
        }
    }

    /// Seed the §3.1 border cells of `cur` before decoding row `y`.
    fn seed_row_border(&mut self) {
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
    fn commit_and_rotate(&mut self, y: usize) {
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
    let footer_len = if ec {
        crate::slice_footer::SLICE_FOOTER_LEN_EC1
    } else {
        crate::slice_footer::SLICE_FOOTER_LEN_EC0
    };

    for (slice_index, ext) in extents.iter().enumerate() {
        let slice_bytes = &frame_bytes[ext.start..ext.end()];
        let _footer = parse_slice_footer(slice_bytes, ec)?;
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
        // header. Read and discard it for slice 0 so the rest of the
        // decoder stays byte-synchronised; later Slices carry no
        // keyframe bit.
        if slice_index == 0 {
            let mut kf_state = [crate::range_coder::PARAMETERS_INITIAL_STATE; 1];
            let _keyframe = crate::symbol::get_br(&mut rc, &mut kf_state);
        }

        let header = parse_slice_header_from_decoder(&mut rc, cr)?;
        let sc = compute_slice_content(&header, cr, frame_dims)?;
        debug_assert_eq!(sc.traversal, PlaneTraversal::LineMajor);

        // Build one persistent line-state per Plane.
        let mut plane_states: Vec<PlaneLineState> = Vec::with_capacity(primary_color_count);
        for (p_idx, plane) in sc.planes.iter().enumerate() {
            // §4.6.6 quant_table_set_index mapping (mirrors the YCbCr
            // driver): luma/Y uses entry 0; Cb/Cr share entry 1; the
            // extra Plane uses the last entry.
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
                qts.context_count as usize,
                cr.coder_type,
            ));
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
            for ps in plane_states.iter_mut() {
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
                        let gr = ps
                            .gr_state
                            .as_mut()
                            .expect("coder_type == 0 builds a Golomb-Rice state");
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
                    }
                    _ => {
                        let rcs = ps
                            .rc_state
                            .as_mut()
                            .expect("coder_type >= 1 builds a range state");
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
    }

    Ok(DecodedFrame {
        planes,
        width: frame_dims.width,
        height: frame_dims.height,
        bits_per_raw_sample: cr.bits_per_raw_sample,
        colorspace: ColorspaceType::Rgb,
    })
}

/// Apply the §3.7.1 inverse RCT to each Sample of a Slice and blit the
/// recovered R / G / B (+ alpha) into the frame-level Planes at the
/// Slice's pixel origin.
#[allow(clippy::too_many_arguments)]
fn apply_inverse_rct_and_blit(
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
/// border row buffers + the per-coder entropy state + the row-major
/// coded modified-YCbCr Sample buffer the line loop consumes.
///
/// For `coder_type == 0` (Golomb-Rice) the per-Plane state is currently
/// not synthesised here; the Golomb-Rice RGB encode path is a follow-up
/// round (the shipped v3 fixtures all use `coder_type == 1`, so the
/// range-coded path is the priority).
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
    /// Range entropy state. `Some` for `coder_type >= 1`.
    rc_state: Option<RangePlaneEncoderState>,
}

impl PlaneLineEncodeState {
    fn new(
        width: usize,
        height: usize,
        coded_bits: u32,
        qtable: QuantTableSet,
        context_count: usize,
        coder_type: u32,
        coded: Vec<i32>,
    ) -> Self {
        let stride = BORDER_LEFT + width + BORDER_RIGHT;
        let rc_state = if coder_type >= 1 {
            Some(RangePlaneEncoderState::new(context_count.max(1)))
        } else {
            None
        };
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
            rc_state,
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

    let mut out = Vec::new();
    for (slice_index, header) in slice_headers.iter().enumerate() {
        let slice_bytes = if cr.coder_type == 0 {
            encode_one_rgb_slice_golomb(
                slice_index == 0,
                header,
                cr,
                quant_table_sets,
                frame,
                frame_dims,
                ec,
            )?
        } else {
            encode_one_rgb_slice_range(
                slice_index == 0,
                header,
                cr,
                quant_table_sets,
                frame,
                frame_dims,
                ec,
            )?
        };
        out.extend_from_slice(&slice_bytes);
    }
    Ok(out)
}

/// Encode one Slice on the RGB / line-major range-coded path.
///
/// Mirrors the per-Slice loop body in [`decode_frame_rgb`]: keyframe
/// bit (slice 0 only) → §4.6 SliceHeader → §4.7 line-major Sample
/// encode → §4.9 SliceFooter, with the SliceHeader and SliceContent
/// sharing a single [`RangeEncoder`] cursor.
fn encode_one_rgb_slice_range(
    is_first_slice: bool,
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    frame_dims: FramePixelDimensions,
    ec: bool,
) -> Result<Vec<u8>, Error> {
    // §3.8.1.4 / §3.8.1.6: pick the active state-transition table for
    // this Slice's range coder. Same predicate `decode_frame_rgb` uses.
    let mut re = if cr.coder_type == 2 {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        RangeEncoder::with_one_state(&one_state)
    } else {
        RangeEncoder::new()
    };

    if is_first_slice {
        // RFC 9043 §4.4: keyframe boolean at the very start of the first
        // Slice's range-coded region. Mirrors `decode_frame_rgb`'s
        // `let _keyframe = get_br(&mut rc, ...)` consumption.
        let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
        put_br(&mut re, &mut kf_state, true);
    }

    encode_slice_header_to_encoder(&mut re, header, cr)?;

    let sc = compute_slice_content(header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::LineMajor);

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);

    // Build one persistent line-state per Plane, seeded with the
    // forward-RCT output (or the raw alpha for plane 3). The RCT runs
    // over the input R/G/B Planes restricted to this Slice's pixel
    // rectangle and produces row-major coded-Plane buffers
    // (`coded_bits = bits_per_raw_sample + 1`).
    let mut plane_states: Vec<PlaneLineEncodeState> = Vec::with_capacity(primary_color_count);
    let coded_buffers = forward_rct_for_slice(frame, cr, &sc)?;
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
        plane_states.push(PlaneLineEncodeState::new(
            plane.width as usize,
            plane.height as usize,
            coded_bits,
            qts.tables,
            qts.context_count as usize,
            cr.coder_type,
            coded_buffers[p_idx].clone(),
        ));
    }

    // §4.7 line-major traversal: outer y, inner p. Symmetric inverse of
    // the decoder's two-deep loop.
    let slice_h = sc.slice_pixel_height as usize;
    for y in 0..slice_h {
        for ps in plane_states.iter_mut() {
            if y >= ps.height {
                continue;
            }
            ps.seed_row_border();

            let rcs = ps
                .rc_state
                .as_mut()
                .expect("coder_type >= 1 builds a range state");
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

    // §4.9 SliceFooter — `encode_slice_footer` solves the §4.9.3 CRC
    // parity so the whole-Slice residue is zero by construction.
    let slice_bytes = encode_slice_footer(&body, ec, SliceErrorStatus::NoError)?;
    Ok(slice_bytes)
}

/// Per-Plane Golomb-Rice encoder state for the RGB line-major path.
///
/// Encoder-side mirror of the `coder_type == 0` branch of
/// [`PlaneLineState`]. Each Plane keeps its [`LineDecoderState`]
/// (per-context VLC window + run state) and §3.1 border row buffers
/// alive across the §4.7 line-major interleave, exactly as the decoder
/// keeps each Plane's `PlaneEntropyState` + border buffers alive across
/// the matching `for y { for p { Line(p, y) } }` traversal.
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
    /// Per-context VLC window + run state (§3.8.2.4 / §3.8.2.2).
    state: LineDecoderState,
}

impl PlaneLineGolombEncodeState {
    fn new(
        width: usize,
        height: usize,
        coded_bits: u32,
        qtable: QuantTableSet,
        context_count: usize,
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
            state: LineDecoderState::new(context_count.max(1)),
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
fn encode_one_rgb_slice_golomb(
    is_first_slice: bool,
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    frame_dims: FramePixelDimensions,
    ec: bool,
) -> Result<Vec<u8>, Error> {
    // ---- §4.4 keyframe + §4.6 SliceHeader (range-coded) ----
    //
    // The decoder reads these through a `RangeDecoder::new(body)` cursor
    // and uses `rc.position()` to find the byte boundary where the
    // Golomb-Rice SliceContent begins; `RangeEncoder::finish()` here is
    // the symmetric counterpart — its returned bytes are exactly that
    // prefix.
    let mut re = RangeEncoder::new();
    if is_first_slice {
        let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
        put_br(&mut re, &mut kf_state, true);
    }
    encode_slice_header_to_encoder(&mut re, header, cr)?;
    let mut body = re.finish();

    let sc = compute_slice_content(header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::LineMajor);

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    let coded_buffers = forward_rct_for_slice(frame, cr, &sc)?;

    // §3.8.2.2.1: per-Plane Golomb-Rice state — VLC contexts + run mode
    // all fresh at the top of every Plane. Per the §4.7 line-major
    // traversal these states live for the whole Slice and step one row
    // per Plane each outer-`y` iteration.
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
            qts.context_count as usize,
            coded_buffers[p_idx].clone(),
        ));
    }

    // ---- §4.8 SliceContent (Golomb-Rice, byte-aligned tail) ----
    let mut bw = BitWriter::new();

    // §4.7 line-major traversal: outer y, inner p. Symmetric inverse of
    // the decoder's two-deep loop in `decode_frame_rgb` for
    // `coder_type == 0`.
    let slice_h = sc.slice_pixel_height as usize;
    for y in 0..slice_h {
        for ps in plane_states.iter_mut() {
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
            // `encode_line` writes diffs into `current_row` to enable
            // run-mode lookahead. ----
            {
                let mut neighbours = LineNeighborBuffers {
                    prev_row: &ps.prev,
                    prev_prev_row: &ps.prev_prev,
                    current_row: &mut ps.cur,
                    plane_pixel_width: width as u32,
                };
                encode_line(
                    &mut bw,
                    &mut ps.state,
                    &ps.qtable,
                    &mut neighbours,
                    &diffs,
                    bits,
                );
            }

            // ---- Overwrite `cur` with the actual Sample values so the
            // next row's §3.3 median predictor reads Sample, not diff,
            // values in `l` / `t` / `tl` positions. ----
            for (x, &s) in row_samples.iter().enumerate() {
                ps.cur[BORDER_WIDTH + x] = s;
            }
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

    // §4.9 SliceFooter — `encode_slice_footer` solves the §4.9.3 CRC
    // parity so the whole-Slice residue is zero by construction.
    let slice_bytes = encode_slice_footer(&body, ec, SliceErrorStatus::NoError)?;
    Ok(slice_bytes)
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
            let mut ps = PlaneLineState::new(1, 1, bits + 1, [[0i32; 256]; 5], 1, cr.coder_type);
            ps.out[0] = val as i32;
            ps
        };
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
        let ps = PlaneLineState::new(8, 6, 9, [[0i32; 256]; 5], 16, 1);
        assert_eq!(ps.out.len(), 48);
        assert_eq!(ps.prev.len(), BORDER_LEFT + 8 + BORDER_RIGHT);
        assert!(ps.rc_state.is_some());
        assert!(ps.gr_state.is_none());
        assert_eq!(ps.coded_bits, 9);
    }

    #[test]
    fn plane_line_state_golomb_path_picks_gr_state() {
        let ps = PlaneLineState::new(4, 4, 9, [[0i32; 256]; 5], 8, 0);
        assert!(ps.gr_state.is_some());
        assert!(ps.rc_state.is_none());
    }
}
