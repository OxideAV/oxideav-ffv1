//! Frame assembly shared by the two version-3 decode drivers: Slice
//! planning, per-Slice job fan-out, in-order stitching, and (opt-in)
//! concealment of damaged Slices.
//!
//! RFC 9043 §4.5 makes every Slice an independent entropy-coded unit and
//! §4.9 gives each one a size trailer plus, when `ec == 1`, a §4.9.3 CRC
//! and a §4.9.2 `error_status`. That is exactly what both a parallel
//! decoder and an error-resilient decoder need:
//!
//! * **Fan-out.** One job per Slice runs under the caller's thread
//!   budget ([`crate::DecodeOptions::threads`]; serial by default). Jobs
//!   return their reconstructed sample rectangles and end-of-Frame coder
//!   state; the driver stitches them **in Slice order**, so the decoded
//!   Frame and the carried state are identical at every budget, and the
//!   error reported for a damaged Frame is always the lowest-index
//!   Slice's.
//! * **Concealment** ([`SliceConcealment`], off by default). A Slice the
//!   active policies reject — CRC residue, `Uncorrectable` status,
//!   truncated or unparsable content, a raster position already claimed —
//!   no longer aborts the Frame. Its raster area is filled (neutral grey,
//!   or the co-located area of the previous decoded Frame), every other
//!   Slice decodes normally, and the damage is reported through
//!   [`crate::Ffv1FrameCarry::concealed_regions`].
//!
//! # Inter-Frame state under concealment
//!
//! §3.8.1.3 / §3.8.2.5 let a non-keyframe continue each Slice's context
//! state from the previous Frame. A concealed Slice has no end-of-Frame
//! state, so its carry entry is marked *lost*: every following
//! non-keyframe conceals that Slice position again (decoding it against
//! reset state would pass its CRC and still produce wrong samples) until
//! a keyframe re-initialises it. The §4.4 `keyframe` bit itself lives in
//! the first Slice; when that Slice is damaged and the stream is not
//! intra-only (§4.2.18), the Frame's state model is unknown and the whole
//! Frame is concealed.

use std::sync::Arc;

use crate::Error;
use crate::config::Ffv1ConfigurationRecord;
use crate::frame::{DecodeOptions, DecodedFrame, DecodedFramePlane, Ffv1FrameCarry};
use crate::quant_table::QuantizationTableSet;
use crate::range_coder::RangeDecoder;
use crate::range_reconstruct::RangePlaneState;
use crate::reconstruct::PlaneEntropyState;
use crate::slice_content::{FramePixelDimensions, compute_slice_content};
use crate::slice_header::{
    Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES, parse_slice_header_from_decoder,
};
use crate::trailer_chain::{SliceExtent, slice_footer_len, walk_trailer_chain};

/// What the decoder does with a Slice it cannot (or, per the active §4.9
/// policies, must not) decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SliceConcealment {
    /// A damaged Slice fails the Frame decode with its error (the
    /// historical behaviour).
    #[default]
    Off,
    /// Fill a damaged Slice's raster area with neutral mid-grey (every
    /// colour Plane at half scale, the transparency Plane opaque) and
    /// keep decoding the Frame's other Slices.
    Grey,
    /// Fill a damaged Slice's raster area from the co-located samples of
    /// the previous decoded Frame (falling back to
    /// [`Self::Grey`] when there is none), and keep decoding the Frame's
    /// other Slices.
    PreviousFrame,
}

/// A raster rectangle (in full-resolution Frame pixels) the decoder
/// concealed instead of decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConcealedRegion {
    /// Left edge in Frame pixels.
    pub x: u32,
    /// Top edge in Frame pixels.
    pub y: u32,
    /// Width in Frame pixels.
    pub width: u32,
    /// Height in Frame pixels.
    pub height: u32,
}

/// One reconstructed rectangle of one output Plane.
pub(crate) struct SlicePiece {
    pub(crate) plane: usize,
    pub(crate) origin_x: usize,
    pub(crate) origin_y: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) samples: Vec<i32>,
}

/// Everything one Slice job hands back to the stitching pass.
pub(crate) struct SliceDecoded {
    pub(crate) pieces: Vec<SlicePiece>,
    pub(crate) range: Vec<Option<RangePlaneState>>,
    pub(crate) golomb: Vec<Option<PlaneEntropyState>>,
}

/// Frame-wide inputs every Slice job shares.
pub(crate) struct FrameJob<'a> {
    pub(crate) frame_bytes: &'a [u8],
    pub(crate) cr: &'a Ffv1ConfigurationRecord,
    pub(crate) quant_table_sets: &'a [QuantizationTableSet],
    pub(crate) frame_dims: FramePixelDimensions,
    pub(crate) ec: bool,
    pub(crate) options: DecodeOptions,
    /// The §4.4 `keyframe` value governing §3.8.1.3 / §3.8.2.5 state
    /// initialisation for every Slice of this Frame.
    pub(crate) keyframe: bool,
    pub(crate) prev_carry: &'a Ffv1FrameCarry,
    /// §4.2.15 explicit initial states per Quantization Table Set.
    pub(crate) initial_states: &'a [Option<Vec<u8>>],
}

/// One Slice to decode.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliceUnit {
    /// Forward Slice index — the carry key and the index errors report.
    pub(crate) index: usize,
    pub(crate) extent: SliceExtent,
    /// Whether this Slice's range-coded region opens with the §4.4
    /// `keyframe` bit (true for the Frame's first Slice only).
    pub(crate) opens_frame: bool,
}

/// Build the range decoder for one Slice body, honouring the §3.8.1.6
/// custom transition table for `coder_type == 2`.
pub(crate) fn slice_range_decoder<'a>(
    body: &'a [u8],
    cr: &Ffv1ConfigurationRecord,
) -> Result<RangeDecoder<'a>, Error> {
    if cr.coder_type == 2 {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        RangeDecoder::with_one_state(body, &one_state)
    } else {
        RangeDecoder::new(body)
    }
}

/// Parse one Slice's §4.6 header (and, on the Frame's first Slice, the
/// §4.4 `keyframe` bit that precedes it).
fn probe_slice(
    frame_bytes: &[u8],
    extent: SliceExtent,
    opens_frame: bool,
    cr: &Ffv1ConfigurationRecord,
    ec: bool,
) -> Result<(Ffv1SliceHeader, Option<bool>), Error> {
    let footer_len = slice_footer_len(ec);
    let slice_bytes = &frame_bytes[extent.start..extent.end()];
    if slice_bytes.len() < footer_len {
        return Err(Error::TruncatedSliceFooter);
    }
    let body = &slice_bytes[..slice_bytes.len() - footer_len];
    let mut rc = slice_range_decoder(body, cr)?;
    let keyframe = opens_frame.then(|| {
        let mut kf_state = [crate::range_coder::PARAMETERS_INITIAL_STATE; 1];
        crate::symbol::get_br(&mut rc, &mut kf_state)
    });
    let header = parse_slice_header_from_decoder(&mut rc, cr)?;
    Ok((header, keyframe))
}

/// Neutral fill value for output Plane `plane`: half scale for colour
/// Planes, full scale (opaque) for the trailing transparency Plane.
fn grey_value(plane: usize, plane_count: usize, cr: &Ffv1ConfigurationRecord) -> i32 {
    let bits = cr.bits_per_raw_sample.clamp(1, 30);
    if cr.extra_plane && plane + 1 == plane_count {
        (1i32 << bits) - 1
    } else {
        1i32 << (bits - 1)
    }
}

/// Fill one rectangle of one output Plane from the previous Frame (when
/// supplied and shape-compatible) or with the neutral value.
#[allow(clippy::too_many_arguments)]
fn conceal_rect(
    dst: &mut DecodedFramePlane,
    previous: Option<&DecodedFramePlane>,
    grey: i32,
    origin_x: usize,
    origin_y: usize,
    width: usize,
    height: usize,
) {
    let dst_w = dst.width as usize;
    let dst_h = dst.height as usize;
    let previous = previous
        .filter(|p| p.width == dst.width && p.height == dst.height)
        .map(|p| p.samples.as_slice());
    let copy_w = width.min(dst_w.saturating_sub(origin_x));
    if copy_w == 0 {
        return;
    }
    for y in origin_y..(origin_y + height).min(dst_h) {
        let lo = y * dst_w + origin_x;
        let row = &mut dst.samples[lo..lo + copy_w];
        match previous {
            Some(prev) => row.copy_from_slice(&prev[lo..lo + copy_w]),
            None => row.fill(grey),
        }
    }
}

/// Per-Plane origin of a Slice whose luma origin is (`x`, `y`): chroma
/// Planes of a subsampled YCbCr Frame scale down, every other Plane
/// (and every Plane of an RGB Frame, whose shifts are zero) does not.
fn plane_origin(x: u32, y: u32, plane: usize, cr: &Ffv1ConfigurationRecord) -> (usize, usize) {
    if cr.chroma_planes && (plane == 1 || plane == 2) {
        (
            (x >> cr.log2_h_chroma_subsample) as usize,
            (y >> cr.log2_v_chroma_subsample) as usize,
        )
    } else {
        (x as usize, y as usize)
    }
}

/// A 1x1-cell probe header at raster cell (`gx`, `gy`) — used only to
/// derive that cell's pixel rectangle through the §4.6 geometry math.
fn cell_header(gx: u32, gy: u32) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x: gx,
        slice_y: gy,
        slice_width: 1,
        slice_height: 1,
        quant_table_set_index_count: 1,
        quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: crate::config::PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

/// Copy one job's rectangles into the Frame Planes.
fn stitch(planes: &mut [DecodedFramePlane], pieces: &[SlicePiece]) {
    for piece in pieces {
        let Some(dst) = planes.get_mut(piece.plane) else {
            continue;
        };
        let dst_w = dst.width as usize;
        let dst_h = dst.height as usize;
        let copy_w = piece.width.min(dst_w.saturating_sub(piece.origin_x));
        if copy_w == 0 {
            continue;
        }
        for y in 0..piece.height {
            let dy = piece.origin_y + y;
            if dy >= dst_h {
                break;
            }
            let dst_off = dy * dst_w + piece.origin_x;
            let src_off = y * piece.width;
            dst.samples[dst_off..dst_off + copy_w]
                .copy_from_slice(&piece.samples[src_off..src_off + copy_w]);
        }
    }
}

/// Lenient §4.9.1 trailer walk: collect Slices from the end of the Frame
/// backwards until a size field stops making sense. Returns the
/// recovered Slices in forward order and whether the walk reached byte 0
/// (i.e. the chain is complete and the first recovered Slice really is
/// the Frame's first).
fn walk_trailer_chain_lenient(frame: &[u8], ec: bool) -> (Vec<SliceExtent>, bool) {
    let footer_len = slice_footer_len(ec);
    let mut rev: Vec<SliceExtent> = Vec::new();
    let mut cursor = frame.len();
    while cursor > 0 {
        if cursor < footer_len {
            return (reversed(rev), false);
        }
        let off = cursor - footer_len;
        let size = (usize::from(frame[off]) << 16)
            | (usize::from(frame[off + 1]) << 8)
            | usize::from(frame[off + 2]);
        let total_len = size + footer_len;
        if total_len > cursor {
            return (reversed(rev), false);
        }
        let start = cursor - total_len;
        rev.push(SliceExtent { start, total_len });
        cursor = start;
    }
    let complete = !rev.is_empty();
    (reversed(rev), complete)
}

fn reversed(mut v: Vec<SliceExtent>) -> Vec<SliceExtent> {
    v.reverse();
    v
}

/// Decode one version-3 Frame: plan the Slices, run `decode_slice` per
/// Slice under `options.threads`, stitch in Slice order.
///
/// `plane_dims[p]` is output Plane `p`'s Frame-level size. `decode_slice`
/// is the colourspace-specific Slice job (plane-major YCbCr or
/// line-major RGB).
#[allow(clippy::too_many_arguments)] // one Frame's full decode context, passed through unchanged
pub(crate) fn decode_frame_slices<F>(
    frame_bytes: &[u8],
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame_dims: FramePixelDimensions,
    ec: bool,
    options: DecodeOptions,
    carry: &mut Option<Ffv1FrameCarry>,
    plane_dims: &[(u32, u32)],
    decode_slice: F,
) -> Result<DecodedFrame, Error>
where
    F: Fn(&FrameJob<'_>, SliceUnit) -> Result<SliceDecoded, Error> + Sync,
{
    let conceal = options.concealment != SliceConcealment::Off;
    let initial_states = crate::quant_table::reconstruct_initial_states(cr, quant_table_sets);
    let empty_carry = Ffv1FrameCarry::default();
    let prev_carry: &Ffv1FrameCarry = carry.as_ref().unwrap_or(&empty_carry);

    // ----- Plan: which Slices exist, where, and is this a keyframe. -----
    let mut units: Vec<SliceUnit> = Vec::new();
    let mut headers: Vec<Ffv1SliceHeader> = Vec::new();
    let mut frame_keyframe = true;
    // Total forward Slice count (carry length) and the raster cells the
    // planned Slices claim (concealment bookkeeping only).
    let total_slices;
    let grid_w = cr.num_h_slices.unwrap_or(1).max(1) as usize;
    let grid_h = cr.num_v_slices.unwrap_or(1).max(1) as usize;
    let mut claimed: Vec<Option<usize>> = Vec::new();
    let mut whole_frame_lost = false;

    if !conceal {
        // Strict planning: any structural defect fails the Frame.
        let extents = walk_trailer_chain(frame_bytes, ec)?;
        // RFC 9043 §5 second paragraph: "each position in the Slice
        // raster MUST be filled by one and only one Slice of the Frame"
        // — a Frame-level invariant, checked on every header before any
        // Slice content is reconstructed.
        for (index, extent) in extents.iter().enumerate() {
            let (header, keyframe) = probe_slice(frame_bytes, *extent, index == 0, cr, ec)?;
            if let Some(k) = keyframe {
                frame_keyframe = k;
            }
            headers.push(header);
            units.push(SliceUnit {
                index,
                extent: *extent,
                opens_frame: index == 0,
            });
        }
        crate::slice_content::validate_slice_raster_coverage(&headers, cr)?;
        total_slices = units.len();
    } else {
        let (extents, complete) = walk_trailer_chain_lenient(frame_bytes, ec);
        // With a broken chain the recovered Slices are the Frame's last
        // ones; number them from the back of the expected Slice count
        // (the previous Frame's, else one Slice per raster cell).
        let expected = if complete {
            extents.len()
        } else {
            let prev = prev_carry.slice_count();
            let guess = if prev > 0 { prev } else { grid_w * grid_h };
            guess.max(extents.len() + 1)
        };
        total_slices = expected;
        let base = expected - extents.len();
        claimed = vec![None; grid_w.saturating_mul(grid_h)];
        let mut first_slice_ok = false;
        for (k, extent) in extents.iter().enumerate() {
            let index = base + k;
            let opens_frame = complete && k == 0;
            let Ok((header, keyframe)) = probe_slice(frame_bytes, *extent, opens_frame, cr, ec)
            else {
                continue;
            };
            // Claim the Slice's raster cells; a footprint outside the
            // grid or overlapping an earlier Slice is damage.
            let (sx, sy) = (header.slice_x as usize, header.slice_y as usize);
            let (sw, sh) = (header.slice_width as usize, header.slice_height as usize);
            let inside = sw >= 1
                && sh >= 1
                && sx.checked_add(sw).is_some_and(|e| e <= grid_w)
                && sy.checked_add(sh).is_some_and(|e| e <= grid_h);
            if !inside {
                continue;
            }
            let cells = (sy..sy + sh).flat_map(|y| (sx..sx + sw).map(move |x| y * grid_w + x));
            if cells.clone().any(|c| claimed[c].is_some()) {
                continue;
            }
            for c in cells {
                claimed[c] = Some(units.len());
            }
            if let Some(kf) = keyframe {
                frame_keyframe = kf;
                first_slice_ok = true;
            }
            headers.push(header);
            units.push(SliceUnit {
                index,
                extent: *extent,
                opens_frame,
            });
        }
        if !first_slice_ok {
            // The §4.4 keyframe bit is unreadable. An intra-only stream
            // (§4.2.18) has keyframes only; otherwise the state model of
            // this Frame is unknown and nothing in it can be trusted.
            if cr.intra == Some(true) {
                frame_keyframe = true;
            } else {
                frame_keyframe = false;
                whole_frame_lost = true;
            }
        }
    }

    let job = FrameJob {
        frame_bytes,
        cr,
        quant_table_sets,
        frame_dims,
        ec,
        options,
        keyframe: frame_keyframe,
        prev_carry,
        initial_states: &initial_states,
    };

    // ----- Fan out: one job per planned Slice. -----
    let mut results: Vec<Result<SliceDecoded, Error>> = if whole_frame_lost {
        units
            .iter()
            .map(|u| {
                Err(Error::SliceStateLost {
                    slice_index: u.index as u32,
                })
            })
            .collect()
    } else {
        crate::parallel::run_indexed(units.len(), options.threads, |k| {
            let unit = units[k];
            if !job.keyframe && job.prev_carry.is_lost(unit.index) {
                return Err(Error::SliceStateLost {
                    slice_index: unit.index as u32,
                });
            }
            decode_slice(&job, unit)
        })
    };

    // A first Slice that parsed but then failed its own decode (CRC,
    // status, content) makes the §4.4 keyframe bit it carries untrusted:
    // outside an intra-only stream the whole Frame is concealed.
    if conceal
        && !whole_frame_lost
        && cr.intra != Some(true)
        && units
            .iter()
            .zip(&results)
            .any(|(u, r)| u.opens_frame && r.is_err())
    {
        for (unit, result) in units.iter().zip(results.iter_mut()) {
            *result = Err(Error::SliceStateLost {
                slice_index: unit.index as u32,
            });
        }
    }

    // ----- Stitch in Slice order. -----
    let mut planes: Vec<DecodedFramePlane> = plane_dims
        .iter()
        .enumerate()
        .map(|(p, &(w, h))| DecodedFramePlane {
            plane_index: p as u8,
            width: w,
            height: h,
            samples: vec![0i32; w as usize * h as usize],
        })
        .collect();
    let mut new_carry = Ffv1FrameCarry::with_slice_capacity(total_slices);
    let mut concealed: Vec<ConcealedRegion> = Vec::new();

    if !conceal {
        for result in results {
            let decoded = result?;
            stitch(&mut planes, &decoded.pieces);
            new_carry.push_slice(decoded.range, decoded.golomb);
        }
    } else {
        new_carry.fill_lost(total_slices);
        let mut good_headers = Vec::with_capacity(headers.len());
        for (k, (unit, result)) in units.iter().zip(results).enumerate() {
            match result {
                Ok(decoded) => {
                    stitch(&mut planes, &decoded.pieces);
                    new_carry.set_slice(unit.index, decoded.range, decoded.golomb);
                    good_headers.push(headers[k].clone());
                }
                Err(_) => {
                    for cell in claimed.iter_mut().filter(|c| **c == Some(k)) {
                        *cell = None;
                    }
                }
            }
        }
        headers = good_headers;
        // Every raster cell no intact Slice covers is concealed.
        let previous = match options.concealment {
            SliceConcealment::PreviousFrame => prev_carry.previous_planes(),
            _ => None,
        };
        let plane_count = planes.len();
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                if claimed[gy * grid_w + gx].is_some() {
                    continue;
                }
                let Ok(sc) =
                    compute_slice_content(&cell_header(gx as u32, gy as u32), cr, frame_dims)
                else {
                    continue;
                };
                for (p, layout) in sc.planes.iter().enumerate().take(plane_count) {
                    let (ox, oy) = plane_origin(sc.slice_pixel_x, sc.slice_pixel_y, p, cr);
                    conceal_rect(
                        &mut planes[p],
                        previous.and_then(|prev| prev.get(p)),
                        grey_value(p, plane_count, cr),
                        ox,
                        oy,
                        layout.width as usize,
                        layout.height as usize,
                    );
                }
                concealed.push(ConcealedRegion {
                    x: sc.slice_pixel_x,
                    y: sc.slice_pixel_y,
                    width: sc.slice_pixel_width,
                    height: sc.slice_pixel_height,
                });
            }
        }
    }

    if options.concealment == SliceConcealment::PreviousFrame {
        new_carry.set_previous_planes(Arc::new(planes.clone()));
    }
    new_carry.set_concealed_regions(concealed);
    *carry = Some(new_carry);

    Ok(DecodedFrame {
        planes,
        width: frame_dims.width,
        height: frame_dims.height,
        bits_per_raw_sample: cr.bits_per_raw_sample,
        colorspace: cr.colorspace_type,
        keyframe: frame_keyframe,
        slice_headers: headers,
    })
}
