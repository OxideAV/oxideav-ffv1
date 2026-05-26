//! FFV1 frame-level encode driver (RFC 9043 §4.4, §4.5, §4.6, §4.7,
//! §4.8, §4.9) — Golomb-Rice / YCbCr branch.
//!
//! Symmetric inverse of the `coder_type == 0` + `colorspace_type == 0`
//! (plane-major / YCbCr) branch of [`crate::decode_frame`]. Given the
//! reconstructed Sample values the decoder would output, this driver
//! composes the per-stage encoders the earlier rounds landed —
//! [`encode_slice_header_to_encoder`](crate::encode_slice_header_to_encoder)
//! (§4.6), [`encode_line`](crate::encode_line) (§4.8 / §3.8.2), and
//! [`encode_slice_footer`](crate::encode_slice_footer) (§4.9) — into a
//! single end-to-end pipeline that emits a frame payload a matching
//! [`decode_frame`](crate::decode_frame) call reconstructs back to the
//! original pixels.
//!
//! ## Pipeline
//!
//! For each Slice on the §4.6 raster, in slice-index order:
//!
//! ```text
//!   per-Slice Sample rectangle (extracted from the frame Plane)
//!        │
//!        ▼
//!   §3.1 border + §3.3 median + §3.8 modular wrap   →  per-row
//!                                                       sample_difference
//!        │
//!        ▼
//!   §4.4 keyframe bit (slice 0 only, range-coded)
//!        ▼
//!   §4.6 SliceHeader (encode_slice_header_to_encoder; range-coded)
//!        ▼
//!   RangeEncoder::finish()                          →  byte boundary
//!        ▼
//!   §4.8 SliceContent: for each Plane, for each row,
//!        encode_line(...)                           →  Golomb-Rice bits
//!        ▼
//!   §4.9 SliceFooter via encode_slice_footer (ec=1) →  +CRC parity
//! ```
//!
//! The concatenated Slice bytes are the frame payload that
//! [`decode_frame`](crate::decode_frame) consumes.
//!
//! ## Scope
//!
//! Round 159 wires the **`coder_type == 0` + `colorspace_type == 0`**
//! (Golomb-Rice / YCbCr) path. The four shipped v3 fixtures all use
//! the range coder, so this is a synthetic-self-consistent encoder —
//! its primary correctness check is the round-trip through
//! [`decode_frame`](crate::decode_frame), the same way the round-136
//! test-only `tests/frame_assembly_golomb.rs` encoder was validated.
//! Range-coded SliceContent (`coder_type == 1 || 2`) and the
//! RGB / line-major path are follow-ups.
//!
//! The §3.3.1 alternate 16-bit median predictor (range-coder /
//! 16-bit YCbCr) is N/A here — that path is range-coded by spec.

use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};
use crate::frame::DecodedFrame;
use crate::predictor::median_predict;
use crate::range_coder::{RangeEncoder, PARAMETERS_INITIAL_STATE};
use crate::sample_diff::{encode_line, LineDecoderState, LineNeighborBuffers, BORDER_WIDTH};
use crate::slice_content::{compute_slice_content, FramePixelDimensions, PlaneTraversal};
use crate::slice_footer::{encode_slice_footer, SliceErrorStatus};
use crate::slice_header::{encode_slice_header_to_encoder, Ffv1SliceHeader};
use crate::symbol::put_br;
use crate::Error;
use crate::QuantizationTableSet;

/// Encode one FFV1 v3 frame end-to-end on the Golomb-Rice / YCbCr path
/// (RFC 9043 §4.4 + §4.5 + §4.6 + §4.7 + §4.8 + §4.9).
///
/// # Parameters
///
/// * `frame` — the reconstructed [`DecodedFrame`] to encode. The
///   driver reads `frame.planes[p].samples` as the per-Plane Sample
///   rectangle; each Sample must lie in `0 .. 2^bits_per_raw_sample`
///   so the §3.8 modular add-back round-trips. `frame.width` /
///   `frame.height` are the frame pixel dimensions (FFV1's
///   Configuration Record carries no width / height fields, so this
///   value is the surrounding container's reported dimensions).
/// * `cr` — the per-stream Configuration Record (§4.2). Must satisfy
///   `version == V3`, `coder_type == 0`, and
///   `colorspace_type == YCbCr` for this Golomb-Rice / plane-major
///   driver; other combinations return
///   [`Error::ColorspaceLayoutNotImplemented`] /
///   [`Error::UnsupportedCoderType`] /
///   [`Error::SliceRequiresVersion3`].
/// * `quant_table_sets` — the parsed §4.1 Quantization Table Sets in
///   stream order (index `i` corresponds to
///   `quant_table_set_index[..] == i` in the headers).
/// * `slice_headers` — one [`Ffv1SliceHeader`] per Slice in
///   slice-index order. The caller supplies the §4.6 raster (which
///   Slices cover which `num_h_slices × num_v_slices` cells) and the
///   per-plane `quant_table_set_index` selection; the driver does NOT
///   synthesise a slice raster from the Configuration Record because
///   §4.6 admits any decomposition that tiles the
///   `num_h_slices × num_v_slices` grid. Every header must satisfy
///   the constraints [`encode_slice_header_to_encoder`] checks:
///   `slice_width != 0`, `slice_height != 0`, and
///   `quant_table_set_index_count` equal to the value the
///   Configuration Record derives.
/// * `ec` — the §4.2.14 `error_correction` flag governing the §4.9
///   Slice Footer length (3 bytes for `false`, 8 bytes + CRC for
///   `true`).
///
/// # Returns
///
/// The concatenated Slice byte stream — exactly what
/// [`crate::decode_frame`] reads as `frame_bytes`. The returned
/// buffer is the FFV1 v3 *frame payload* a downstream container
/// muxer wraps inside whatever framing the chosen container needs.
///
/// # Errors
///
/// * [`Error::SliceRequiresVersion3`] when `cr.version != V3`.
/// * [`Error::ColorspaceLayoutNotImplemented`] when
///   `cr.colorspace_type != YCbCr` (the RGB / line-major encode path
///   is a follow-up).
/// * [`Error::UnsupportedCoderType`] when `cr.coder_type != 0` (the
///   range-coded encode path is a follow-up).
/// * [`Error::InvalidFramePixelDimensions`] when `frame.width == 0`
///   or `frame.height == 0`.
/// * [`Error::SliceRasterOutOfRange`] propagated from
///   [`compute_slice_content`] when a header addresses a cell
///   outside the configured `num_h_slices × num_v_slices` grid.
/// * [`Error::SliceSizeOutOfRange`] propagated from
///   [`encode_slice_header_to_encoder`] when `slice_width == 0` /
///   `slice_height == 0` or when the header's
///   `quant_table_set_index_count` disagrees with the Configuration
///   Record's derivation, or from [`encode_slice_footer`] when an
///   assembled body length overflows the §4.9.1 `u(24)` size field.
/// * [`Error::InvalidQuantTableSetCount`] when a slice header
///   selects an out-of-range Quantization Table Set.
pub fn encode_frame_golomb_rice(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    slice_headers: &[Ffv1SliceHeader],
    ec: bool,
) -> Result<Vec<u8>, Error> {
    if cr.version != Ffv1Version::V3 {
        return Err(Error::SliceRequiresVersion3);
    }
    if cr.colorspace_type != ColorspaceType::YCbCr {
        // The §4.7 RGB / line-major encode path is a follow-up
        // round (it needs a row-by-row driver that keeps per-Plane
        // VLC state external, mirroring the decode side).
        return Err(Error::ColorspaceLayoutNotImplemented);
    }
    if cr.coder_type != 0 {
        return Err(Error::UnsupportedCoderType(cr.coder_type));
    }

    let frame_dims = FramePixelDimensions::new(frame.width, frame.height)?;

    let mut out = Vec::new();
    for (slice_index, header) in slice_headers.iter().enumerate() {
        let slice_bytes = encode_one_golomb_slice(
            slice_index == 0,
            header,
            cr,
            quant_table_sets,
            frame,
            frame_dims,
            ec,
        )?;
        out.extend_from_slice(&slice_bytes);
    }
    Ok(out)
}

/// Encode one Slice: keyframe bit (slice 0) + §4.6 SliceHeader (range
/// coded) + §4.8 SliceContent (byte-aligned, Golomb-Rice) + §4.9
/// SliceFooter. Returns the whole-Slice byte stream (the buffer
/// `parse_slice_footer` consumes).
fn encode_one_golomb_slice(
    is_first_slice: bool,
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    frame_dims: FramePixelDimensions,
    ec: bool,
) -> Result<Vec<u8>, Error> {
    // ---- §4.4 keyframe + §4.6 SliceHeader (range-coded) ----
    let mut re = RangeEncoder::new();
    if is_first_slice {
        // RFC 9043 §4.4: the Frame's leading `keyframe` boolean lives at
        // the very start of the first Slice's range-coded region (its
        // own initial state 128, separate from the SliceHeader's own
        // state buffer). We encode `keyframe = true` — every FFV1 frame
        // this driver builds is a keyframe (intra-only codec, no
        // inter-frame prediction).
        let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
        put_br(&mut re, &mut kf_state, true);
    }
    encode_slice_header_to_encoder(&mut re, header, cr)?;
    let mut body = re.finish();

    // ---- §4.8 SliceContent (Golomb-Rice, byte-aligned tail) ----
    //
    // Compute the per-slice plane layout: the §4.6 raster cell carries
    // into §4.8 pixel-space, then chroma subsampling shrinks per-plane
    // width/height.
    let sc = compute_slice_content(header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

    // The §4.8 SliceContent for `coder_type == 0` is purely
    // Golomb-Rice bits, written by a separate `BitWriter` and appended
    // to the range-coded header at the byte boundary. The frame
    // decoder reads `body[rc.position()..]` as the Golomb-Rice tail
    // (see decode_frame's `consumed = rc.position()` branch); the
    // `RangeEncoder::finish()` call above guarantees `body.len()` IS
    // that byte boundary because Golomb-Rice writing starts only after
    // the range encoder has flushed.
    let content = encode_slice_content_golomb(header, cr, quant_table_sets, frame, &sc)?;
    body.extend_from_slice(&content);

    // ---- §4.9 SliceFooter (ec selects 3 or 8 bytes) ----
    //
    // `encode_slice_footer` solves the §4.9.3 CRC parity for ec=1 so
    // the whole-Slice residue is zero by construction.
    let slice_bytes = encode_slice_footer(&body, ec, SliceErrorStatus::NoError)?;
    Ok(slice_bytes)
}

/// Emit the §4.8 SliceContent Golomb-Rice tail for one Slice.
///
/// Walks the §4.7 plane-major traversal (`for p { for y { Line(p, y)
/// } }`) — for each Plane: extract the slice's pixel rectangle from
/// the frame buffer, derive per-row `sample_difference` from the §3.3
/// median predictor + §3.8 modular wrap, and call
/// [`encode_line`](crate::encode_line) row-by-row. The per-Plane
/// `LineDecoderState` (VLC contexts + run-mode) is constructed fresh
/// at the top of each Plane per §3.8.2.2.1.
fn encode_slice_content_golomb(
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    sc: &crate::slice_content::SliceContent,
) -> Result<Vec<u8>, Error> {
    let mut bw = crate::bit_reader::BitWriter::new();
    let bits = cr.bits_per_raw_sample;

    for (p_idx, plane) in sc.planes.iter().enumerate() {
        // §4.6.6: which quant_table_set_index entry applies to this
        // plane. Same routing the decoder uses (luma → 0, chroma → 1,
        // extra → tail).
        let qts_index_slot = quant_index_slot(p_idx, header.quant_table_set_index_count, cr);
        let qts_choice = header.quant_table_set_index[qts_index_slot] as usize;
        let qts = quant_table_sets
            .get(qts_choice)
            .ok_or(Error::InvalidQuantTableSetCount(qts_choice as u32))?;

        // The frame-level destination Plane for `p_idx`. Slice's
        // pixel rectangle on this Plane (in Plane coordinates,
        // accounting for chroma subsampling).
        let frame_plane = frame
            .planes
            .get(p_idx)
            .ok_or(Error::InvalidQuantTableSetCount(p_idx as u32))?;
        let (origin_x, origin_y) =
            plane_origin(sc.slice_pixel_x, sc.slice_pixel_y, plane.plane_index, cr);
        let plane_w = plane.width as usize;
        let plane_h = plane.height as usize;

        // §3.8.2.2.1: per-Plane state — VLC contexts + run mode all
        // fresh at the top of every Plane. `context_count` is the
        // §4.1.2 `ceil(scale/2)` from the chosen Quantization Table
        // Set; that's how many VLC slots the per-Sample §3.5 absolute
        // context indexes into.
        let mut state = LineDecoderState::new(qts.context_count as usize);

        // §3.1 border buffers. The Plane reconstruction routine in
        // `reconstruct.rs` uses the same shape: two prev rows (above
        // + above-above) zero-initialised, current row written
        // in-place by `encode_line` so future rows read the
        // already-encoded prefix.
        let stride = BORDER_WIDTH + plane_w + BORDER_WIDTH;
        let mut prev_prev = vec![0i32; stride];
        let mut prev = vec![0i32; stride];
        let mut cur = vec![0i32; stride];

        for y in 0..plane_h {
            // ---- Extract this row's Sample values from the frame
            // Plane at the slice's pixel-space origin. ----
            let dst_w = frame_plane.width as usize;
            let row_start = (origin_y + y as u32) as usize * dst_w + origin_x as usize;
            let row_end = row_start + plane_w;
            let row_samples = &frame_plane.samples[row_start..row_end];

            // ---- Derive `sample_difference` per §3.3 + §3.8. The
            // decoder computes `pred = median(l, t, tl)` then
            // `Sample = (pred + diff) & mask`; to make the decoder
            // arrive at `row_samples[x]` we hand the encoder
            // `diff = (sample - pred)` normalised into the signed
            // half-modulus range so `sign_extend` is the identity in
            // the §3.8.2.4 path. ----
            let diffs = sample_diffs_for_row(
                row_samples,
                &prev[BORDER_WIDTH..BORDER_WIDTH + plane_w],
                bits,
            );

            // ---- §3.1 left-of-slice column: sample[y][-1] =
            // sample[y-1][0]. Encoder mirrors `reconstruct.rs`
            // exactly. ----
            cur[0] = 0;
            cur[BORDER_WIDTH - 1] = prev[BORDER_WIDTH];

            // ---- Encode the row. `encode_line` consumes the diffs
            // and emits Golomb-Rice bits; it also pre-populates
            // current_row with the diffs (decoder-symmetric). After
            // it returns we rewrite current_row with the *Sample*
            // values for the next row's median predictor. ----
            {
                let mut neighbours = LineNeighborBuffers {
                    prev_row: &prev,
                    prev_prev_row: &prev_prev,
                    current_row: &mut cur,
                    plane_pixel_width: plane_w as u32,
                };
                encode_line(
                    &mut bw,
                    &mut state,
                    &qts.tables,
                    &mut neighbours,
                    &diffs,
                    bits,
                );
            }

            // ---- Rewrite cur[] with the actual Sample values for
            // the next iteration's neighbour reads. `encode_line`
            // left cur[] holding the `diff` row (it writes diffs
            // into current_row to enable lookahead); the next row's
            // median predictor needs Sample values in the `l`/`t`/`tl`
            // positions. ----
            for (x, &s) in row_samples.iter().enumerate() {
                cur[BORDER_WIDTH + x] = s;
            }
            // §3.1 right-border mirror.
            cur[BORDER_WIDTH + plane_w] = cur[BORDER_WIDTH + plane_w - 1];

            // Rotate buffers: prev_prev <- prev <- cur; new cur is
            // zeroed for the next row.
            core::mem::swap(&mut prev_prev, &mut prev);
            core::mem::swap(&mut prev, &mut cur);
            cur.iter_mut().for_each(|s| *s = 0);
        }
    }

    // §3.8.2 "padded with zeroes": flush the BitWriter, zero-padding
    // the final partial byte so the §4.9 footer sits on a byte
    // boundary.
    Ok(bw.finish())
}

/// Per-row §3.3 / §3.8 sample-difference derivation.
///
/// Given a row of target Samples and the row of Sample values
/// immediately above (for the §3.3 `t` neighbour), returns the row of
/// signed `diff` values such that
/// `reconstruct_sample(median(l, t, tl), diff, bits) == sample` for
/// every column. The `l` / `tl` neighbours are taken from
/// `prev_row[x.saturating_sub(1)]` / `row_samples[x-1]` consistent
/// with `reconstruct.rs`.
///
/// Each `diff` is normalised to the signed `bits`-wide modulus
/// (`-2^(bits-1) <= diff < 2^(bits-1)`) so the §3.8.2.4 `sign_extend`
/// step in the matching decode is the identity.
fn sample_diffs_for_row(row_samples: &[i32], prev_row_samples: &[i32], bits: u32) -> Vec<i32> {
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
        // §3.1 left border: sample[y][-1] = sample[y-1][0]. So the
        // very first column's `l` equals the first column of the
        // previous row. For the §3.1 tl at x=0 the same border
        // applies (one cell up and one left of the slice top-left
        // corner is `0`, but the median's lone `l` and `tl` both fold
        // into the same border seed for typical reconstructions —
        // matching `residuals_for_plane` in the round-136 test
        // encoder).
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

/// Mirror of the `qts_index` routing in `decode_frame`. Maps a Plane
/// index inside the SliceContent's plane vector to the slot in
/// `header.quant_table_set_index[..]` that selects its Quantization
/// Table Set.
fn quant_index_slot(p_idx: usize, count: usize, cr: &Ffv1ConfigurationRecord) -> usize {
    match p_idx {
        0 => 0,
        1 | 2 if cr.chroma_planes => 1.min(count.saturating_sub(1)),
        _ if cr.extra_plane => count.saturating_sub(1),
        _ => 0,
    }
}

/// Mirror of the `plane_origin` helper in `frame.rs`. Returns the
/// per-Plane pixel origin (the slice's pixel origin shifted by the
/// per-Plane chroma subsample factor when relevant).
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
    use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version, PictureStructure};
    use crate::predictor::QuantTableSet;
    use crate::predictor::NUM_QUANT_SUBTABLES;
    use crate::quant_table::QuantizationTableSet;
    use crate::slice_content::FramePixelDimensions;
    use crate::slice_header::{Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES};
    use crate::{decode_frame, DecodedFrame, DecodedFramePlane};

    // ----- Helpers -------------------------------------------------

    fn grayscale_v3_cr(num_h: u32, num_v: u32, bits: u32) -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            version: Ffv1Version::V3,
            micro_version: Some(4),
            coder_type: 0, // Golomb-Rice
            state_transition_delta: [0; crate::config::NUM_TRANSITION_DELTAS],
            colorspace_type: ColorspaceType::YCbCr,
            bits_per_raw_sample: bits,
            chroma_planes: false,
            log2_h_chroma_subsample: 0,
            log2_v_chroma_subsample: 0,
            extra_plane: false,
            num_h_slices: Some(num_h),
            num_v_slices: Some(num_v),
            quant_table_set_count: Some(1),
        }
    }

    /// A constant-context quantization table set: every §3.5 absolute
    /// context maps to `c` regardless of neighbours. Mirrors the
    /// `constant_context_qts` helper in `tests/frame_assembly_golomb.rs`.
    /// `c != 0` keeps the encoder on the scalar VLC path (run-mode
    /// suppressed).
    fn constant_context_qts(c: i32) -> QuantizationTableSet {
        let mut tables: QuantTableSet = [[0i32; 256]; NUM_QUANT_SUBTABLES];
        tables[0] = [c; 256];
        QuantizationTableSet {
            tables,
            context_count: (c as u32) + 1,
        }
    }

    /// Construct a slice header carrying `quant_index` for every
    /// derived `quant_table_set_index` slot. Width / height are the
    /// raster footprint (in raster cells, post `+1`).
    fn make_header(
        slice_x: u32,
        slice_y: u32,
        slice_width: u32,
        slice_height: u32,
        quant_index_count: usize,
        quant_index: u32,
    ) -> Ffv1SliceHeader {
        let mut idx = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
        for slot in idx.iter_mut().take(quant_index_count) {
            *slot = quant_index;
        }
        Ffv1SliceHeader {
            slice_x,
            slice_y,
            slice_width,
            slice_height,
            quant_table_set_index_count: quant_index_count,
            quant_table_set_index: idx,
            picture_structure: PictureStructure::Progressive,
            picture_structure_raw: 0,
            sar_num: 0,
            sar_den: 0,
        }
    }

    fn make_gray_decoded_frame(samples: Vec<i32>, w: u32, h: u32, bits: u32) -> DecodedFrame {
        DecodedFrame {
            planes: vec![DecodedFramePlane {
                plane_index: 0,
                width: w,
                height: h,
                samples,
            }],
            width: w,
            height: h,
            bits_per_raw_sample: bits,
            colorspace: ColorspaceType::YCbCr,
        }
    }

    // ----- Round trip: single slice, single plane (8-bit) ---------

    #[test]
    fn round_trip_gray_single_slice_8bit() {
        let cr = grayscale_v3_cr(1, 1, 8);
        let qts = vec![constant_context_qts(9)];
        // Slice index count for chroma_planes=false v3 = 1 + 1 + 0 = 2.
        let header = make_header(0, 0, 1, 1, 2, 0);

        #[rustfmt::skip]
        let samples: Vec<i32> = vec![
              0,  10,  20,  40,  80, 160,
            200, 100,  50,  25,  12,   6,
              3, 130, 255,   0, 128,  64,
             77,  88,  99, 111, 222, 233,
        ];
        let frame = make_gray_decoded_frame(samples.clone(), 6, 4, 8);

        let bytes = encode_frame_golomb_rice(&frame, &cr, &qts, &[header], true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(6, 4).unwrap(),
            true,
        )
        .expect("encoded frame must round-trip through decode_frame");
        assert_eq!(decoded.planes.len(), 1);
        assert_eq!(decoded.planes[0].samples, samples);
    }

    // ----- Round trip: 10-bit grayscale ----------------------------

    #[test]
    fn round_trip_gray_single_slice_10bit() {
        let cr = grayscale_v3_cr(1, 1, 10);
        let qts = vec![constant_context_qts(6)];
        let header = make_header(0, 0, 1, 1, 2, 0);

        #[rustfmt::skip]
        let samples: Vec<i32> = vec![
               0,  511, 1023,  256,
             800,  100,  900,   50,
            1000,    1,  512,  300,
        ];
        let frame = make_gray_decoded_frame(samples.clone(), 4, 3, 10);

        let bytes = encode_frame_golomb_rice(&frame, &cr, &qts, &[header], true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(4, 3).unwrap(),
            true,
        )
        .expect("10-bit encoded frame must round-trip through decode_frame");
        assert_eq!(decoded.planes[0].samples, samples);
        assert_eq!(decoded.bits_per_raw_sample, 10);
    }

    // ----- Round trip: 2x2 slice grid (assembly) ------------------

    #[test]
    fn round_trip_2x2_slice_grid_assembles_full_frame() {
        let (fw, fh) = (8u32, 4u32);
        let cr = grayscale_v3_cr(2, 2, 8);
        let qts = vec![constant_context_qts(11)];

        #[rustfmt::skip]
        let pixels: Vec<i32> = vec![
              1,   2,   3,   4,   200, 201, 202, 203,
              5,   6,   7,   8,   204, 205, 206, 207,
             50,  60,  70,  80,   100, 110, 120, 130,
             90, 100, 110, 120,   140, 150, 160, 170,
        ];
        let frame = make_gray_decoded_frame(pixels.clone(), fw, fh, 8);

        let headers = vec![
            make_header(0, 0, 1, 1, 2, 0),
            make_header(1, 0, 1, 1, 2, 0),
            make_header(0, 1, 1, 1, 2, 0),
            make_header(1, 1, 1, 1, 2, 0),
        ];

        let bytes = encode_frame_golomb_rice(&frame, &cr, &qts, &headers, true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(fw, fh).unwrap(),
            true,
        )
        .expect("2x2 grid encoded frame must round-trip");
        assert_eq!(decoded.planes[0].samples, pixels);
    }

    // ----- Round trip: 1x3 vertical stack (row-stride check) -----

    #[test]
    fn round_trip_1x3_vertical_stack() {
        let (fw, fh) = (5u32, 6u32);
        let cr = grayscale_v3_cr(1, 3, 8);
        let qts = vec![constant_context_qts(7)];

        let mut pixels: Vec<i32> = Vec::with_capacity((fw * fh) as usize);
        for y in 0..fh {
            for x in 0..fw {
                let band = y / 2;
                pixels.push((band as i32 * 60 + (x as i32) * 3 + (y as i32)) & 0xFF);
            }
        }
        let frame = make_gray_decoded_frame(pixels.clone(), fw, fh, 8);
        let headers = vec![
            make_header(0, 0, 1, 1, 2, 0),
            make_header(0, 1, 1, 1, 2, 0),
            make_header(0, 2, 1, 1, 2, 0),
        ];

        let bytes = encode_frame_golomb_rice(&frame, &cr, &qts, &headers, true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(fw, fh).unwrap(),
            true,
        )
        .expect("1x3 stack encoded frame must round-trip");
        assert_eq!(decoded.planes[0].samples, pixels);
    }

    // ----- Round trip: ec=0 path (3-byte footer) ------------------

    #[test]
    fn round_trip_gray_single_slice_ec0() {
        let cr = grayscale_v3_cr(1, 1, 8);
        let qts = vec![constant_context_qts(5)];
        let header = make_header(0, 0, 1, 1, 2, 0);

        let samples: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let frame = make_gray_decoded_frame(samples.clone(), 4, 2, 8);

        let bytes = encode_frame_golomb_rice(&frame, &cr, &qts, &[header], false).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(4, 2).unwrap(),
            false,
        )
        .expect("ec=0 round trip");
        assert_eq!(decoded.planes[0].samples, samples);
    }

    // ----- Determinism --------------------------------------------

    #[test]
    fn encoder_is_deterministic() {
        let cr = grayscale_v3_cr(1, 1, 8);
        let qts = vec![constant_context_qts(4)];
        let header = make_header(0, 0, 1, 1, 2, 0);

        let samples: Vec<i32> = (0..(5 * 4)).map(|i| (i * 11 + 17) & 0xFF).collect();
        let frame = make_gray_decoded_frame(samples.clone(), 5, 4, 8);

        let bytes_a =
            encode_frame_golomb_rice(&frame, &cr, &qts, std::slice::from_ref(&header), true)
                .unwrap();
        let bytes_b =
            encode_frame_golomb_rice(&frame, &cr, &qts, std::slice::from_ref(&header), true)
                .unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    // ----- Error paths --------------------------------------------

    #[test]
    fn rejects_non_v3() {
        let mut cr = grayscale_v3_cr(1, 1, 8);
        cr.version = Ffv1Version::V0;
        let qts = vec![constant_context_qts(3)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let frame = make_gray_decoded_frame(vec![0i32; 4], 2, 2, 8);

        assert!(matches!(
            encode_frame_golomb_rice(&frame, &cr, &qts, &[header], true),
            Err(Error::SliceRequiresVersion3)
        ));
    }

    #[test]
    fn rejects_rgb_colorspace() {
        let mut cr = grayscale_v3_cr(1, 1, 8);
        cr.colorspace_type = ColorspaceType::Rgb;
        let qts = vec![constant_context_qts(3)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let frame = make_gray_decoded_frame(vec![0i32; 4], 2, 2, 8);

        assert!(matches!(
            encode_frame_golomb_rice(&frame, &cr, &qts, &[header], true),
            Err(Error::ColorspaceLayoutNotImplemented)
        ));
    }

    #[test]
    fn rejects_non_golomb_coder() {
        let mut cr = grayscale_v3_cr(1, 1, 8);
        cr.coder_type = 1;
        let qts = vec![constant_context_qts(3)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let frame = make_gray_decoded_frame(vec![0i32; 4], 2, 2, 8);

        assert!(matches!(
            encode_frame_golomb_rice(&frame, &cr, &qts, &[header], true),
            Err(Error::UnsupportedCoderType(1))
        ));
    }

    #[test]
    fn rejects_invalid_quant_index() {
        let cr = grayscale_v3_cr(1, 1, 8);
        let qts = vec![constant_context_qts(3)];
        // Slot 0 references quant set index 5, but only one is provided.
        let header = make_header(0, 0, 1, 1, 2, 5);
        let frame = make_gray_decoded_frame(vec![10i32; 4], 2, 2, 8);

        assert!(matches!(
            encode_frame_golomb_rice(&frame, &cr, &qts, &[header], true),
            Err(Error::InvalidQuantTableSetCount(_))
        ));
    }

    #[test]
    fn rejects_bad_header() {
        let cr = grayscale_v3_cr(1, 1, 8);
        let qts = vec![constant_context_qts(3)];
        // slice_width == 0 is rejected by encode_slice_header_to_encoder.
        let header = make_header(0, 0, 0, 1, 2, 0);
        let frame = make_gray_decoded_frame(vec![10i32; 4], 2, 2, 8);

        assert!(matches!(
            encode_frame_golomb_rice(&frame, &cr, &qts, &[header], true),
            Err(Error::SliceSizeOutOfRange { .. })
        ));
    }

    // ----- helper coverage ----------------------------------------

    #[test]
    fn sample_diffs_for_row_zero_input_zero_pred() {
        // First-row, first-column: l = prev_row[0], tl = 0, t = prev_row[0].
        // median(l,t,tl) = median(0,0,0) = 0; diff = sample - 0 = sample.
        let diffs = sample_diffs_for_row(&[5, 6, 7], &[0, 0, 0], 8);
        // After x=0 the next cells take `l = row_samples[x-1]` so they
        // see the *actual* sample value, not zero. Just verify it stays
        // in the signed half-modulus range.
        for &d in &diffs {
            assert!((-128..128).contains(&d));
        }
    }

    #[test]
    fn quant_index_slot_routes_chroma_to_slot_1() {
        let cr = grayscale_v3_cr(1, 1, 8);
        let mut c = cr;
        c.chroma_planes = true;
        // count = 1 + 1 + 0 = 2 (chroma_planes ⇒ chroma slot present).
        assert_eq!(quant_index_slot(0, 2, &c), 0);
        assert_eq!(quant_index_slot(1, 2, &c), 1);
        assert_eq!(quant_index_slot(2, 2, &c), 1);
    }

    #[test]
    fn plane_origin_chroma_halved() {
        let mut cr = grayscale_v3_cr(1, 1, 8);
        cr.chroma_planes = true;
        cr.log2_h_chroma_subsample = 1;
        cr.log2_v_chroma_subsample = 1;
        assert_eq!(plane_origin(64, 48, 0, &cr), (64, 48));
        assert_eq!(plane_origin(64, 48, 1, &cr), (32, 24));
        assert_eq!(plane_origin(64, 48, 2, &cr), (32, 24));
    }
}
