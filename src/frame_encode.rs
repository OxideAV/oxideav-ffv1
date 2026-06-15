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
use crate::range_encode::RangePlaneEncoder;
use crate::rgb_reconstruct::encode_frame_rgb;
use crate::sample_diff::{encode_line, LineDecoderState, LineNeighborBuffers, BORDER_WIDTH};
use crate::slice_content::{compute_slice_content, FramePixelDimensions, PlaneTraversal};
use crate::slice_footer::{encode_slice_footer, SliceErrorStatus};
use crate::slice_header::{encode_slice_header_to_encoder, Ffv1SliceHeader};
use crate::symbol::put_br;
use crate::Error;
use crate::QuantizationTableSet;

/// Encode one FFV1 v3 frame end-to-end, dispatching on the
/// Configuration Record to the matching specialised encoder.
///
/// FFV1 has three encode paths that the earlier rounds landed as
/// separate public entry points — one per `(colorspace_type,
/// coder_type)` combination the spec allows. Callers previously had to
/// replicate the §4.2.3 `coder_type` / §4.2.5 `colorspace_type` switch
/// at every call site to pick the right one. This helper centralises
/// that switch so a caller only needs the parsed
/// [`Ffv1ConfigurationRecord`]; it is the symmetric counterpart to the
/// routing [`crate::decode_frame`] performs on the read side.
///
/// The routing table (RFC 9043 §4.2.3 Table 7 / §4.2.5):
///
/// ```text
///   colorspace_type   coder_type   delegate
///   ───────────────   ──────────   ──────────────────────────
///   Rgb (1)           0 | 1 | 2    encode_frame_rgb           (§4.7 line-major)
///   YCbCr (0)         0            encode_frame_golomb_rice   (§4.8 Golomb-Rice)
///   YCbCr (0)         1 | 2        encode_frame_range_coder   (§3.8.1 range coder)
/// ```
///
/// [`encode_frame_rgb`] performs its own `coder_type == 0` vs
/// `1 | 2` split internally (both RGB sub-paths share the §4.7
/// line-major traversal), so RGB is dispatched on `colorspace_type`
/// alone.
///
/// # Parameters
///
/// Identical to the three delegates — `frame`, `cr`,
/// `quant_table_sets`, `slice_headers`, and the §4.2.14 `ec` flag are
/// forwarded verbatim. See [`encode_frame_golomb_rice`] for the
/// per-argument contract.
///
/// # Errors
///
/// * [`Error::SliceRequiresVersion3`] when `cr.version != V3`
///   (surfaced by the chosen delegate).
/// * [`Error::UnsupportedCoderType`] when `cr.coder_type > 2` — no
///   §4.2.3 Table 7 entry exists. (RGB delegates surface this for
///   `> 2`; the YCbCr arm rejects it directly.)
/// * Every error documented on the chosen delegate
///   ([`encode_frame_rgb`] / [`encode_frame_golomb_rice`] /
///   [`encode_frame_range_coder`]) propagates unchanged.
pub fn encode_frame(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    slice_headers: &[Ffv1SliceHeader],
    ec: bool,
) -> Result<Vec<u8>, Error> {
    match cr.colorspace_type {
        // §4.7 RGB / line-major. `encode_frame_rgb` handles the
        // `coder_type` sub-dispatch (0 → Golomb-Rice, 1 | 2 → range
        // coder) itself, so we route on colorspace alone.
        ColorspaceType::Rgb => encode_frame_rgb(frame, cr, quant_table_sets, slice_headers, ec),
        // §4.2.5 YCbCr / plane-major splits on the §4.2.3 entropy coder.
        ColorspaceType::YCbCr => match cr.coder_type {
            0 => encode_frame_golomb_rice(frame, cr, quant_table_sets, slice_headers, ec),
            1 | 2 => encode_frame_range_coder(frame, cr, quant_table_sets, slice_headers, ec),
            other => Err(Error::UnsupportedCoderType(other)),
        },
    }
}

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
/// [`encode_line`](crate::encode_line) row-by-row.
///
/// Per RFC 9043 §3.8.2.5 the per-context VLC state (`drift`,
/// `error_sum`, `bias`, `count`) is allocated **per Quantization Table
/// Set** (§4.2 Figure 28 `initial_state_delta[i][j][k]`, `i` over
/// `quant_table_set_count`), keyframe-initialised, and evolves through
/// the remainder of the Slice. Planes that share a
/// `quant_table_set_index` (Cb + Cr on every `chroma_planes == true`
/// Slice; an extra Plane aliased onto either Y or chroma's set) must
/// share a single `LineDecoderState` across their per-Plane encode
/// passes — the second Plane to touch a set continues evolving the
/// state the first Plane left it in, exactly as the decoder reads it.
/// Per §3.8.2.2.1 only the `run_index` / `run_mode` / `run_count`
/// triple resets at the top of each Plane.
fn encode_slice_content_golomb(
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    sc: &crate::slice_content::SliceContent,
) -> Result<Vec<u8>, Error> {
    let mut bw = crate::bit_reader::BitWriter::new();
    let bits = cr.bits_per_raw_sample;

    // One per-slot state buffer (§4.6.5 `quant_table_set_index_count`)
    // — luma slot, chroma slot, optional extra-plane slot — lazily
    // allocated on first use so each slot starts at the §3.8.2.5
    // keyframe-init values. Planes that share a slot (Cb + Cr) pick up
    // where the prior Plane left off (mirrors `decode_frame`).
    let slot_count = header.quant_table_set_index_count;
    let mut per_slot_states: Vec<Option<LineDecoderState>> =
        (0..slot_count).map(|_| None).collect();

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

        // §3.8.2.5 + §3.8.2.2.1 + §4.6.6: route this Plane's encode
        // against the per-slot VLC state — fresh on first use of the
        // slot (Y or first chroma plane), continued evolution on
        // subsequent uses (Cr after Cb). `reset_run_state()` resets
        // only the §3.8.2.2.1 run-mode triple at the top of every
        // Plane; the per-context VLC fields survive across Planes
        // sharing the same slot.
        let state = per_slot_states[qts_index_slot]
            .get_or_insert_with(|| LineDecoderState::new(qts.context_count as usize));
        state.reset_run_state();

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
            // and emits Golomb-Rice bits; it pre-populates current_row
            // with the reconstructed *Sample* values so the §3.5 context
            // and §3.8.2.2 run predicate evaluate against the same
            // neighbour Samples the decoder uses. cur[] therefore already
            // holds this row's Samples for the next row's median
            // predictor when this returns. ----
            {
                let mut neighbours = LineNeighborBuffers {
                    prev_row: &prev,
                    prev_prev_row: &prev_prev,
                    current_row: &mut cur,
                    plane_pixel_width: plane_w as u32,
                };
                encode_line(&mut bw, state, &qts.tables, &mut neighbours, &diffs, bits);
            }

            // `encode_line` pre-fills `cur` with this row's reconstructed
            // Sample values (so the §3.5 context + run predicate match the
            // decoder), so the next row's §3.3 median predictor already
            // reads Sample values in the `l` / `t` / `tl` positions.
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

/// Encode one FFV1 v3 frame end-to-end on the **range-coder** YCbCr /
/// plane-major path (RFC 9043 §4.4 + §4.5 + §4.6 + §4.7 + §4.8 + §4.9,
/// `coder_type == 1`).
///
/// Symmetric inverse of the `coder_type == 1` + `colorspace_type == 0`
/// branch of [`crate::decode_frame`]. Where
/// [`encode_frame_golomb_rice`] keeps the §4.6 SliceHeader (range-coded)
/// and the §4.8 SliceContent (Golomb-Rice) on two distinct entropy
/// engines joined at a byte boundary, this driver keeps both on a
/// **single** [`RangeEncoder`] cursor — there is no byte-alignment step
/// between header and content on the range-coded path (§4.5).
///
/// The four shipped v3 fixtures all use `coder_type == 1`, so this is
/// the driver any future fixture-driven encode test will reach for.
/// For round 164 the immediate deliverable is a round-trip through
/// [`crate::decode_frame`]: given a [`DecodedFrame`] of reconstructed
/// Samples, encoding then decoding yields the same Plane bytes.
///
/// # Parameters / errors
///
/// Mirror [`encode_frame_golomb_rice`] except:
///
/// * `cr.coder_type` must equal `1`. Other values return
///   [`Error::UnsupportedCoderType`]. (`coder_type == 2` swaps the
///   per-bit transition table and would otherwise reuse the same
///   per-Sample loop, but the table-swap plumbing is a follow-up
///   round.)
/// * `cr.colorspace_type` must be `YCbCr` (RGB / line-major is a
///   follow-up — same surface as the Golomb-Rice path).
///
/// `use_16bit_median` (the §3.3.1 alternate predictor) auto-activates
/// when `colorspace_type == YCbCr && bits_per_raw_sample == 16` per
/// §3.3.1.
pub fn encode_frame_range_coder(
    frame: &DecodedFrame,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    slice_headers: &[Ffv1SliceHeader],
    ec: bool,
) -> Result<Vec<u8>, Error> {
    // The historical single-Frame entry point: a standalone keyframe,
    // no inter-Frame coder-state carry.
    let mut carry: Option<Ffv1EncodeCarry> = None;
    encode_frame_range_coder_with_carry(
        frame,
        cr,
        quant_table_sets,
        slice_headers,
        ec,
        true,
        &mut carry,
    )
}

/// Inter-Frame per-Slice coder-state carry for the FFV1 encode path
/// (RFC 9043 §3.8.1.3 / §3.8.2.5) — the write-side mirror of
/// [`crate::Ffv1FrameCarry`].
///
/// Holds, per forward Slice index, the per-§4.6.6-slot encoder state
/// each Slice held at end-of-Frame. A caller does not construct this
/// directly; it threads one `&mut Option<Ffv1EncodeCarry>` through a
/// coded Frame sequence so the encoder evolves its per-context state
/// across non-keyframes exactly as the decoder does. This is what makes
/// a genuine multi-Frame non-keyframe round-trip possible: encode with
/// `keyframe == false` and a live carry, decode with
/// [`crate::decode_frame_with_carry`] /
/// [`crate::decode_frame_rgb_with_carry`] and the matching carry, and
/// the per-context windows on both sides stay in lockstep across the
/// Frame boundary.
///
/// The YCbCr range-coder driver
/// ([`encode_frame_range_coder_with_carry`]) only populates the
/// `range_slices` channel; the RGB / line-major driver
/// ([`crate::encode_frame_rgb_with_carry`]) populates exactly one of
/// `range_slices` (`coder_type ∈ {1, 2}`) or `golomb_slices`
/// (`coder_type == 0`) per Frame, mirroring the read-side
/// [`crate::Ffv1FrameCarry`]'s two slot channels.
#[derive(Debug, Clone, Default)]
pub struct Ffv1EncodeCarry {
    range_slices: Vec<Vec<Option<crate::range_encode::RangePlaneEncoderState>>>,
    golomb_slices: Vec<Vec<Option<crate::sample_diff::LineDecoderState>>>,
}

impl Ffv1EncodeCarry {
    /// The per-slot range-coder encoder state carried for forward Slice
    /// index `slice_index`, or an empty slice when no snapshot exists
    /// yet (first Frame, or a Frame with fewer Slices than this one).
    pub(crate) fn range_for(
        &self,
        slice_index: usize,
    ) -> &[Option<crate::range_encode::RangePlaneEncoderState>] {
        self.range_slices
            .get(slice_index)
            .map(|s| s.as_slice())
            .unwrap_or(&[])
    }

    /// The per-slot Golomb-Rice VLC encoder state carried for forward
    /// Slice index `slice_index`, or an empty slice when no snapshot
    /// exists yet.
    pub(crate) fn golomb_for(
        &self,
        slice_index: usize,
    ) -> &[Option<crate::sample_diff::LineDecoderState>] {
        self.golomb_slices
            .get(slice_index)
            .map(|s| s.as_slice())
            .unwrap_or(&[])
    }

    /// An empty carry pre-sized for `slice_count` Slices on the RGB /
    /// line-major write path. The RGB driver fills exactly one of the
    /// two channels per Frame via [`Self::push_range_slice`] /
    /// [`Self::push_golomb_slice`], mirroring the read-side
    /// [`crate::Ffv1FrameCarry`].
    pub(crate) fn with_rgb_slice_capacity(slice_count: usize) -> Self {
        Self {
            range_slices: Vec::with_capacity(slice_count),
            golomb_slices: Vec::with_capacity(slice_count),
        }
    }

    /// Append one Slice's end-of-Frame per-slot range-coder encoder
    /// state, in forward Slice-index order.
    pub(crate) fn push_range_slice(
        &mut self,
        slots: Vec<Option<crate::range_encode::RangePlaneEncoderState>>,
    ) {
        self.range_slices.push(slots);
    }

    /// Append one Slice's end-of-Frame per-slot Golomb-Rice VLC encoder
    /// state, in forward Slice-index order.
    pub(crate) fn push_golomb_slice(
        &mut self,
        slots: Vec<Option<crate::sample_diff::LineDecoderState>>,
    ) {
        self.golomb_slices.push(slots);
    }
}

/// Encode one FFV1 v3 YCbCr range-coded Frame, carrying the §3.8.1.3
/// per-context coder state across non-keyframes (RFC 9043 §3.8.1.3).
///
/// The write-side counterpart of [`crate::decode_frame_with_carry`]:
///
/// * `keyframe` — the §4.4 value written on Slice 0 (`br`). `true`
///   emits an independently decodable keyframe; `false` emits a
///   non-keyframe whose per-context coder state continues from `carry`.
/// * `carry` — on entry, when `keyframe == false` and `*carry` is
///   `Some(prev)`, each Slice's per-slot state resumes from `prev`'s
///   matching Slice instead of the §3.8.1.3 `128`-initialised window;
///   when `keyframe == true` the carry on entry is ignored. On return,
///   `*carry` holds this Frame's end-of-Frame snapshot for the next
///   non-keyframe.
///
/// Restricted to `coder_type ∈ {1, 2}` (the range-coded YCbCr path) —
/// the same surface [`encode_frame_range_coder`] covers.
pub fn encode_frame_range_coder_with_carry(
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
    if cr.colorspace_type != ColorspaceType::YCbCr {
        return Err(Error::ColorspaceLayoutNotImplemented);
    }
    // `coder_type == 1` uses [`DEFAULT_ONE_STATE`]; `coder_type == 2`
    // overlays the Configuration Record's `state_transition_delta[i]`
    // onto the default via [`crate::range_coder::build_one_state`] per
    // RFC 9043 §3.8.1.4 Figure 22 / §3.8.1.6. Any other value is
    // out-of-spec for the range-coded path.
    if cr.coder_type != 1 && cr.coder_type != 2 {
        return Err(Error::UnsupportedCoderType(cr.coder_type));
    }

    let frame_dims = FramePixelDimensions::new(frame.width, frame.height)?;

    let prev_carry = carry.take().unwrap_or_default();
    let mut new_carry = Ffv1EncodeCarry {
        range_slices: Vec::with_capacity(slice_headers.len()),
        golomb_slices: Vec::new(),
    };

    let mut out = Vec::new();
    for (slice_index, header) in slice_headers.iter().enumerate() {
        // RFC 9043 §3.8.1.3: on a non-keyframe the per-slot state
        // resumes from the previous Frame's matching Slice; on a
        // keyframe it is freshly `128`-initialised (seed = empty).
        let seed: &[Option<crate::range_encode::RangePlaneEncoderState>] = if keyframe {
            &[]
        } else {
            prev_carry.range_for(slice_index)
        };
        let (slice_bytes, end_states) = encode_one_range_slice(
            slice_index == 0,
            keyframe,
            header,
            cr,
            quant_table_sets,
            frame,
            frame_dims,
            ec,
            seed,
        )?;
        out.extend_from_slice(&slice_bytes);
        new_carry.range_slices.push(end_states);
    }

    *carry = Some(new_carry);
    Ok(out)
}

/// Encode one Slice on the range-coder path: keyframe bit (slice 0
/// only) + §4.6 SliceHeader + §4.8 SliceContent, all on the **same**
/// `RangeEncoder` cursor, then a §4.9 SliceFooter wrapping the
/// finished byte stream.
#[allow(clippy::too_many_arguments)]
fn encode_one_range_slice(
    is_first_slice: bool,
    keyframe: bool,
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet],
    frame: &DecodedFrame,
    frame_dims: FramePixelDimensions,
    ec: bool,
    seed_states: &[Option<crate::range_encode::RangePlaneEncoderState>],
) -> Result<
    (
        Vec<u8>,
        Vec<Option<crate::range_encode::RangePlaneEncoderState>>,
    ),
    Error,
> {
    // §3.8.1.4 / §3.8.1.6: pick the active state-transition table for
    // this Slice's range coder. `coder_type == 1` keeps the default;
    // `coder_type == 2` layers the Configuration Record's deltas onto
    // it. The encoder and the matching decoder must agree on the table,
    // so the same predicate appears in `decode_frame`.
    let mut re = if cr.coder_type == 2 {
        let one_state = crate::range_coder::build_one_state(&cr.state_transition_delta);
        RangeEncoder::with_one_state(&one_state)
    } else {
        RangeEncoder::new()
    };

    if is_first_slice {
        // RFC 9043 §4.4: keyframe boolean at the very start of the
        // first Slice's range-coded region, before its §4.6 header. The
        // caller chooses the value (`true` for an independent keyframe,
        // `false` for an inter-Frame non-keyframe whose per-context
        // coder state continues from the previous Frame per §3.8.1.3).
        let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
        put_br(&mut re, &mut kf_state, keyframe);
    }

    // §4.6 SliceHeader on the same encoder. The decoder's
    // `parse_slice_header_from_decoder` shares the range coder with
    // `RangePlaneReconstructor::reconstruct_plane`; we mirror that on
    // the encode side by NOT calling `finish()` between header and
    // content — the same cursor carries straight through.
    encode_slice_header_to_encoder(&mut re, header, cr)?;

    let sc = compute_slice_content(header, cr, frame_dims)?;
    debug_assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);

    // The §3.3.1 alt-median predicate matches the decoder's gating in
    // `decode_frame`.
    let use_16bit_median = cr.colorspace_type == ColorspaceType::YCbCr
        && cr.bits_per_raw_sample == 16
        && (cr.coder_type == 1 || cr.coder_type == 2);

    // RFC 9043 §3.8.1.3 + §4.6.6: the range-coder per-context state
    // is keyframe-initialised AND selected by `quant_table_set_index`
    // (§4.6.6 "indicates ... and the initial states"). The *slot* in
    // `quant_table_set_index[..]` (§4.6.5) — i.e. the plane category:
    // luma → 0, both chroma → 1, extra → 2 — keys the state buffer;
    // multiple slots may *alias* onto the same declared
    // Quantization Table Set, but they still own independent state
    // (matching the trace's `plane_index` labelling). Cb + Cr share
    // the chroma slot, so they share one `RangePlaneEncoderState`
    // and Cr continues evolving where Cb left off — exact mirror of
    // `decode_frame`. Lazily allocated per slot so the §3.8.1.3
    // keyframe-init contract holds at first use.
    // RFC 9043 §3.8.1.3: keyframe → fresh `128` windows (lazy `None`);
    // non-keyframe → resume from the previous Frame's matching Slice
    // (the caller supplies the per-slot seed). Mirror of
    // `decode_frame_with_carry`.
    let slot_count = header.quant_table_set_index_count;
    let mut per_slot_states: Vec<Option<crate::range_encode::RangePlaneEncoderState>> = (0
        ..slot_count)
        .map(|s| seed_states.get(s).cloned().flatten())
        .collect();

    for (p_idx, plane) in sc.planes.iter().enumerate() {
        let qts_index_slot = quant_index_slot(p_idx, header.quant_table_set_index_count, cr);
        let qts_choice = header.quant_table_set_index[qts_index_slot] as usize;
        let qts = quant_table_sets
            .get(qts_choice)
            .ok_or(Error::InvalidQuantTableSetCount(qts_choice as u32))?;

        let frame_plane = frame
            .planes
            .get(p_idx)
            .ok_or(Error::InvalidQuantTableSetCount(p_idx as u32))?;
        let (origin_x, origin_y) =
            plane_origin(sc.slice_pixel_x, sc.slice_pixel_y, plane.plane_index, cr);
        let plane_w = plane.width as usize;
        let plane_h = plane.height as usize;

        // Extract this Plane's slice rectangle into a contiguous
        // row-major buffer the per-Plane encoder consumes. The decoder
        // calls `RangePlaneReconstructor::reconstruct_plane_with_state`
        // directly against `rc` and returns a `Vec<i32>`; the encoder
        // takes the same shape and pushes its symbols into `re`.
        let dst_w = frame_plane.width as usize;
        let mut plane_samples = Vec::with_capacity(plane_w * plane_h);
        for y in 0..plane_h {
            let row_start = (origin_y + y as u32) as usize * dst_w + origin_x as usize;
            plane_samples.extend_from_slice(&frame_plane.samples[row_start..row_start + plane_w]);
        }

        let state = per_slot_states[qts_index_slot].get_or_insert_with(|| {
            crate::range_encode::RangePlaneEncoderState::new(qts.context_count as usize)
        });
        RangePlaneEncoder::encode_plane_with_state(
            &mut re,
            state,
            &qts.tables,
            &plane_samples,
            plane_w,
            plane_h,
            cr.bits_per_raw_sample,
            use_16bit_median,
        );
    }

    // §4.8 done; flush the range coder. The resulting byte stream
    // contains keyframe-bit + SliceHeader + SliceContent contiguously
    // and is what the §4.9 footer wraps.
    let body = re.finish();

    // §4.9 SliceFooter. `encode_slice_footer` solves the §4.9.3 CRC
    // parity so the whole-Slice residue is zero by construction.
    let slice_bytes = encode_slice_footer(&body, ec, SliceErrorStatus::NoError)?;
    Ok((slice_bytes, per_slot_states))
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
            ec: Some(0),
            intra: Some(false),
            initial_state_delta: None,
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
            keyframe: true,
            slice_headers: Vec::new(),
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

    // ----- range-coder driver round trips -------------------------

    fn grayscale_v3_range_cr(num_h: u32, num_v: u32, bits: u32) -> Ffv1ConfigurationRecord {
        let mut cr = grayscale_v3_cr(num_h, num_v, bits);
        cr.coder_type = 1; // range-coded SliceContent path
        cr
    }

    #[test]
    fn range_round_trip_gray_single_slice_8bit() {
        let cr = grayscale_v3_range_cr(1, 1, 8);
        let qts = vec![constant_context_qts(9)];
        let header = make_header(0, 0, 1, 1, 2, 0);

        #[rustfmt::skip]
        let samples: Vec<i32> = vec![
              0,  10,  20,  40,  80, 160,
            200, 100,  50,  25,  12,   6,
              3, 130, 255,   0, 128,  64,
             77,  88,  99, 111, 222, 233,
        ];
        let frame = make_gray_decoded_frame(samples.clone(), 6, 4, 8);

        let bytes = encode_frame_range_coder(&frame, &cr, &qts, &[header], true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(6, 4).unwrap(),
            true,
        )
        .expect("range-coded frame must round-trip through decode_frame");
        assert_eq!(decoded.planes.len(), 1);
        assert_eq!(decoded.planes[0].samples, samples);
    }

    #[test]
    fn range_round_trip_gray_single_slice_10bit() {
        let cr = grayscale_v3_range_cr(1, 1, 10);
        let qts = vec![constant_context_qts(6)];
        let header = make_header(0, 0, 1, 1, 2, 0);

        #[rustfmt::skip]
        let samples: Vec<i32> = vec![
               0,  511, 1023,  256,
             800,  100,  900,   50,
            1000,    1,  512,  300,
        ];
        let frame = make_gray_decoded_frame(samples.clone(), 4, 3, 10);

        let bytes = encode_frame_range_coder(&frame, &cr, &qts, &[header], true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(4, 3).unwrap(),
            true,
        )
        .expect("10-bit range-coded round trip");
        assert_eq!(decoded.planes[0].samples, samples);
        assert_eq!(decoded.bits_per_raw_sample, 10);
    }

    #[test]
    fn range_round_trip_2x2_slice_grid_assembles_full_frame() {
        let (fw, fh) = (8u32, 4u32);
        let cr = grayscale_v3_range_cr(2, 2, 8);
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

        let bytes = encode_frame_range_coder(&frame, &cr, &qts, &headers, true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(fw, fh).unwrap(),
            true,
        )
        .expect("2x2 grid range-coded frame must round-trip");
        assert_eq!(decoded.planes[0].samples, pixels);
    }

    #[test]
    fn range_round_trip_gray_single_slice_ec0() {
        // The 3-byte footer (ec=0) holds only the §4.9.1 `slice_size`
        // — no CRC. The encoder must still emit a slice whose body the
        // decoder accepts with `ec=false`.
        let cr = grayscale_v3_range_cr(1, 1, 8);
        let qts = vec![constant_context_qts(5)];
        let header = make_header(0, 0, 1, 1, 2, 0);

        let samples: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let frame = make_gray_decoded_frame(samples.clone(), 4, 2, 8);

        let bytes = encode_frame_range_coder(&frame, &cr, &qts, &[header], false).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(4, 2).unwrap(),
            false,
        )
        .expect("range-coded ec=0 round trip");
        assert_eq!(decoded.planes[0].samples, samples);
    }

    #[test]
    fn range_encoder_is_deterministic() {
        let cr = grayscale_v3_range_cr(1, 1, 8);
        let qts = vec![constant_context_qts(4)];
        let header = make_header(0, 0, 1, 1, 2, 0);

        let samples: Vec<i32> = (0..(5 * 4)).map(|i| (i * 11 + 17) & 0xFF).collect();
        let frame = make_gray_decoded_frame(samples.clone(), 5, 4, 8);

        let bytes_a =
            encode_frame_range_coder(&frame, &cr, &qts, std::slice::from_ref(&header), true)
                .unwrap();
        let bytes_b =
            encode_frame_range_coder(&frame, &cr, &qts, std::slice::from_ref(&header), true)
                .unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn range_rejects_non_v3() {
        let mut cr = grayscale_v3_range_cr(1, 1, 8);
        cr.version = Ffv1Version::V0;
        let qts = vec![constant_context_qts(3)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let frame = make_gray_decoded_frame(vec![0i32; 4], 2, 2, 8);
        assert!(matches!(
            encode_frame_range_coder(&frame, &cr, &qts, &[header], true),
            Err(Error::SliceRequiresVersion3)
        ));
    }

    #[test]
    fn range_rejects_rgb_colorspace() {
        let mut cr = grayscale_v3_range_cr(1, 1, 8);
        cr.colorspace_type = ColorspaceType::Rgb;
        let qts = vec![constant_context_qts(3)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let frame = make_gray_decoded_frame(vec![0i32; 4], 2, 2, 8);
        assert!(matches!(
            encode_frame_range_coder(&frame, &cr, &qts, &[header], true),
            Err(Error::ColorspaceLayoutNotImplemented)
        ));
    }

    #[test]
    fn range_rejects_golomb_rice_coder_type() {
        // The range-coder driver rejects `coder_type == 0` (the
        // Golomb-Rice path uses `encode_frame_golomb_rice` instead) and
        // any out-of-spec value, but `coder_type == 2` is now wired
        // through the §3.8.1.4 / §3.8.1.6 derived transition table —
        // see `range_round_trips_coder_type_2_*` below.
        for coder_type in [0u32, 3, 7, 255] {
            let mut cr = grayscale_v3_range_cr(1, 1, 8);
            cr.coder_type = coder_type;
            let qts = vec![constant_context_qts(3)];
            let header = make_header(0, 0, 1, 1, 2, 0);
            let frame = make_gray_decoded_frame(vec![0i32; 4], 2, 2, 8);
            let err = encode_frame_range_coder(&frame, &cr, &qts, &[header], true);
            assert!(
                matches!(err, Err(Error::UnsupportedCoderType(c)) if c == coder_type),
                "coder_type={coder_type} should reject: got {err:?}"
            );
        }
    }

    /// Build a Configuration Record for `coder_type == 2` with a
    /// non-trivial `state_transition_delta` so the encoder + decoder
    /// run on a derived table rather than the default.
    fn coder_type_2_cr(num_h: u32, num_v: u32, bits: u32) -> Ffv1ConfigurationRecord {
        let mut cr = grayscale_v3_range_cr(num_h, num_v, bits);
        cr.coder_type = 2;
        // Sparse non-zero delta — the very pattern §3.8.1.6 advertises:
        // small per-index nudges that bias the encoder toward shorter
        // outputs. Negative + positive entries exercise both directions
        // of the modular addition `build_one_state` performs.
        let mut delta = [0i32; crate::config::NUM_TRANSITION_DELTAS];
        for (i, slot) in delta.iter_mut().enumerate().skip(1) {
            // Mirror the published Figure 25 alt-table's gentle skew:
            // +1 at one-quarter steps, -1 at three-quarter steps, 0
            // elsewhere. Magnitudes stay well below 256 so no entry
            // wraps in practice.
            *slot = match i % 8 {
                1 => 1,
                5 => -1,
                _ => 0,
            };
        }
        cr.state_transition_delta = delta;
        cr
    }

    #[test]
    fn range_round_trips_coder_type_2_8bit() {
        // `coder_type == 2` round-trip on the range-coded encode path:
        // the encoder picks `build_one_state(&cr.state_transition_delta)`,
        // and the matching `decode_frame` does the same — so the per-bit
        // transitions, the per-Sample state windows, and therefore the
        // recovered Plane samples all match the input exactly.
        let cr = coder_type_2_cr(1, 1, 8);
        let qts = vec![constant_context_qts(9)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let pixels: Vec<i32> = (0..24).map(|i| (i * 13 + 5) & 0xFF).collect();
        let frame = make_gray_decoded_frame(pixels.clone(), 6, 4, 8);
        let bytes = encode_frame_range_coder(&frame, &cr, &qts, &[header], true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(frame.width, frame.height).unwrap(),
            true,
        )
        .unwrap();
        assert_eq!(decoded.planes[0].samples, pixels);
    }

    #[test]
    fn range_round_trips_coder_type_2_10bit() {
        let cr = coder_type_2_cr(1, 1, 10);
        let qts = vec![constant_context_qts(7)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let pixels: Vec<i32> = (0..16).map(|i| (i * 71) % 1024).collect();
        let frame = make_gray_decoded_frame(pixels.clone(), 4, 4, 10);
        let bytes = encode_frame_range_coder(&frame, &cr, &qts, &[header], true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(frame.width, frame.height).unwrap(),
            true,
        )
        .unwrap();
        assert_eq!(decoded.planes[0].samples, pixels);
    }

    #[test]
    fn range_round_trips_coder_type_2_2x2_slice_grid() {
        // Multi-slice round-trip: every Slice picks the same derived
        // table, the keyframe bit (Slice 0 only) honours the same
        // table, and the assembled frame still reconstructs bit-exactly.
        let cr = coder_type_2_cr(2, 2, 8);
        let qts = vec![constant_context_qts(11)];
        let (fw, fh) = (6u32, 4u32);
        let pixels: Vec<i32> = (0..(fw * fh) as usize)
            .map(|i| (((i * 19) ^ 0x5A) & 0xFF) as i32)
            .collect();
        let frame = make_gray_decoded_frame(pixels.clone(), fw, fh, 8);
        let headers = vec![
            make_header(0, 0, 1, 1, 2, 0),
            make_header(1, 0, 1, 1, 2, 0),
            make_header(0, 1, 1, 1, 2, 0),
            make_header(1, 1, 1, 1, 2, 0),
        ];
        let bytes = encode_frame_range_coder(&frame, &cr, &qts, &headers, true).unwrap();
        let decoded = decode_frame(
            &bytes,
            &cr,
            &qts,
            FramePixelDimensions::new(fw, fh).unwrap(),
            true,
        )
        .unwrap();
        assert_eq!(decoded.planes[0].samples, pixels);
    }

    #[test]
    fn coder_type_2_with_zero_delta_matches_coder_type_1() {
        // Sanity: an all-zero delta vector makes `build_one_state`
        // return the default table, so a `coder_type == 2` round-trip
        // with zero deltas must produce exactly the same wire bytes as
        // the same input under `coder_type == 1`. (The §4.4 keyframe
        // bit and §4.6 SliceHeader use the same table too, so this
        // catches any leak of the delta-based table into a place the
        // default should still be used.)
        let cr_one = grayscale_v3_range_cr(1, 1, 8);
        let mut cr_two = cr_one.clone();
        cr_two.coder_type = 2;
        cr_two.state_transition_delta = [0i32; crate::config::NUM_TRANSITION_DELTAS];

        let qts = vec![constant_context_qts(7)];
        let header = make_header(0, 0, 1, 1, 2, 0);
        let pixels: Vec<i32> = (0..16).map(|i| (i * 5 + 1) & 0xFF).collect();
        let frame = make_gray_decoded_frame(pixels, 4, 4, 8);

        let bytes_one =
            encode_frame_range_coder(&frame, &cr_one, &qts, std::slice::from_ref(&header), true)
                .unwrap();
        let bytes_two =
            encode_frame_range_coder(&frame, &cr_two, &qts, std::slice::from_ref(&header), true)
                .unwrap();
        assert_eq!(bytes_one, bytes_two);
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
