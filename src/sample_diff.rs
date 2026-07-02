//! Per-row `sample_difference` decode (RFC 9043 §4.8 / §3.8).
//!
//! Round 4 wires the §4.8 `Line( p, y )` loop body to the §3.8.2
//! Golomb-Rice primitives. For every pixel in a row, the decoder:
//!
//! 1. Computes the §3.5 *absolute* context from the previously
//!    decoded surrounding samples (`L`, `l`, `tl`, `t`, `tr`, `T`)
//!    through a Quantization Table Set (§3.4).
//! 2. Either enters **run mode** (when the absolute `context == 0`, per
//!    §3.8.2.2) or asks the per-context VLC coder for the next
//!    `sample_difference` (§3.8.2.4).
//! 3. If the §3.5 raw context was negative, flips the sign of the
//!    decoded `sample_difference`.
//! 4. Emits the decoded value into the output row.
//!
//! Pixel reconstruction (`pred + diff -> Sample`) is intentionally
//! NOT performed here — round 4's deliverable is the per-row
//! `Vec<i32>` of decoded `sample_difference` values, which a future
//! round will combine with the §3.3 median predictor + the
//! per-`bits_per_raw_sample` modular wrap to recover Sample values.
//!
//! Quantization-table parsing is still pending (round 5 target); this
//! module accepts a caller-supplied [`QuantTableSet`] so the decoder
//! is testable in isolation with synthetic table data.

use crate::bit_reader::{BitReader, BitWriter};
use crate::golomb_rice::{
    get_vlc_symbol, get_vlc_symbol_level, put_vlc_symbol, put_vlc_symbol_level, VlcState, LOG2_RUN,
    VLC_STATE_INITIAL,
};
use crate::predictor::{absolute_context, median_predict, NeighborSamples, QuantTableSet};
use crate::reconstruct::reconstruct_sample;
use crate::Error;

/// Number of samples in the assumed left-of-slice / above-slice
/// border (RFC 9043 §3.1).
///
/// The border is large enough that the `(L, l, tl, t, tr, T)`
/// neighbour stencil never falls off the buffer for any pixel,
/// including the slice's first row and first column. We allocate
/// `WIDTH + 2 * BORDER_WIDTH` columns and two rows of padding
/// before each Line decode.
pub const BORDER_WIDTH: usize = 2;

/// Per-row decode state for one Plane (RFC 9043 §3.8.2).
///
/// `vlc[idx]` is the [`VlcState`] for context index `idx`; the
/// caller supplies the array length, typically `context_count` from
/// the §4.1.2 quant-table cascade. The run-mode state machine
/// fields are reset per Plane and per Slice per §3.8.2.2.1.
#[derive(Debug, Clone)]
pub struct LineDecoderState {
    /// Per-context adaptive VLC coder states. Length is the number of
    /// contexts the active Quantization Table Set produces.
    pub vlc: Vec<VlcState>,
    /// `run_index` per §3.8.2.2.1. Reset to zero at the start of each
    /// Plane and Slice.
    pub run_index: u32,
    /// `run_mode` per §3.8.2.2 — 0 = scalar, 1 = entered run mode,
    /// 2 = run broken (level-coded next sample).
    pub run_mode: u8,
    /// `run_count` per §3.8.2.2.1 — remaining zero samples in the
    /// current run.
    pub run_count: i32,
}

impl LineDecoderState {
    /// Construct a fresh state holding `context_count` VLC slots, all
    /// at the §3.8.2.5 initial values, plus a zeroed run-mode state.
    pub fn new(context_count: usize) -> Self {
        Self {
            vlc: vec![VLC_STATE_INITIAL; context_count],
            run_index: 0,
            run_mode: 0,
            run_count: 0,
        }
    }

    /// Reset the run-mode state machine. Called at the start of each
    /// Plane and Slice per §3.8.2.2.1, but NOT between Lines of the
    /// same Plane (run mode can straddle row boundaries).
    pub fn reset_run_state(&mut self) {
        self.run_index = 0;
        self.run_mode = 0;
        self.run_count = 0;
    }
}

/// Per-Line input view onto the previously decoded sample grid
/// (RFC 9043 §3.1 + §3.2).
///
/// `prev_row` and `prev_prev_row` carry the two rows ABOVE the row
/// being decoded, INCLUDING the §3.1 border. Their layout is:
///
/// ```text
///   [0 .. BORDER_WIDTH)   left border (border samples)
///   [BORDER_WIDTH ..]     plane_pixel_width samples at column 0..W-1
///   [BORDER_WIDTH + W ..] right border (one or more border samples)
/// ```
///
/// `current_row_prefix` is the slice of the row currently being
/// decoded BEFORE the target column — i.e. `current_row_prefix[..x]`
/// is the prefix of the current row, and `current_row_prefix[0]` is
/// the leftmost border sample. The decoder reads from the prefix; it
/// is the caller's job to extend it by the newly decoded value at
/// each step.
#[derive(Debug)]
pub struct LineNeighborBuffers<'a> {
    /// Row immediately above. Length: `BORDER_WIDTH + W + BORDER_WIDTH`.
    pub prev_row: &'a [i32],
    /// Row two above (for the `T` neighbour). Length: same as
    /// `prev_row`.
    pub prev_prev_row: &'a [i32],
    /// Current row buffer including the leading border. Length: same
    /// as `prev_row`. The decoder writes into this buffer as it
    /// proceeds (so future iterations of `decode_line` see the
    /// already-decoded prefix).
    pub current_row: &'a mut [i32],
    /// Plane pixel width — the count of *real* samples per row,
    /// excluding the border padding on either side.
    pub plane_pixel_width: u32,
}

/// Decode one Line's per-pixel `sample_difference` row
/// (RFC 9043 §4.8 + §3.8.2).
///
/// Returns a `Vec<i32>` of length `plane_pixel_width` containing the
/// decoded sample-difference values for the row, *with the §3.5 sign
/// flip already applied*. The `current_row` buffer is also updated
/// in place so subsequent calls (next row, same plane) can read it as
/// `prev_row`. **Pixel reconstruction is not performed**: the values
/// written into `current_row[BORDER_WIDTH + x]` are the decoded
/// `sample_difference` values themselves, NOT the recovered Samples.
/// The caller is expected to apply the median predictor + modular
/// wrap to recover pixel values in a future round.
///
/// `bits` is the per-symbol ESC width used by the §3.8.2.1.1 ESC
/// suffix; pass `bits_per_raw_sample` for native YUV / RGB and
/// `bits_per_raw_sample + 1` for the JPEG 2000 RCT path
/// (RFC 9043 §3.8 Figure 10).
pub fn decode_line(
    br: &mut BitReader<'_>,
    state: &mut LineDecoderState,
    qtable: &QuantTableSet,
    neighbours: &mut LineNeighborBuffers<'_>,
    bits: u32,
) -> Vec<i32> {
    let width = neighbours.plane_pixel_width as usize;
    let mut out = vec![0i32; width];

    // §3.8.2.2 run mode is a per-Line state machine, identical to
    // [`crate::reconstruct::PlaneReconstructor::reconstruct_row`]:
    // `run_index` straddles Lines within a Plane (it lives in `state`),
    // but `run_mode` / `run_count` are local to this Line.
    let mut run_mode: u8 = 0;
    let mut run_count: i32 = 0;

    for (x, slot) in out.iter_mut().enumerate() {
        // Compute neighbour stencil. With BORDER_WIDTH==2:
        //   index into rows is BORDER_WIDTH + x.
        let idx = BORDER_WIDTH + x;

        let n = NeighborSamples {
            tt: neighbours.prev_prev_row[idx],
            ll: neighbours.current_row[idx - 2],
            t: neighbours.prev_row[idx],
            tl: neighbours.prev_row[idx - 1],
            tr: neighbours.prev_row[idx + 1],
            l: neighbours.current_row[idx - 1],
        };

        let abs_ctx = absolute_context(qtable, n);

        // §3.8.2.2: enter run mode at a context-0 Sample when not already
        // running.
        if run_mode == 0 && abs_ctx.index == 0 {
            run_mode = 1;
        }

        let diff_signed: i32 = if run_mode != 0 {
            // §3.8.2.2.1 run-length prefix at the start of a run.
            if run_count == 0 && run_mode == 1 {
                let ri = state.run_index as usize;
                let l2 = LOG2_RUN[ri.min(LOG2_RUN.len() - 1)] as u32;
                if br.get_bit() == 1 {
                    run_count = 1i32 << l2;
                    if x + run_count as usize <= width
                        && (state.run_index as usize) + 1 < LOG2_RUN.len()
                    {
                        state.run_index += 1;
                    }
                } else {
                    run_count = if l2 == 0 { 0 } else { br.get_bits(l2) as i32 };
                    if state.run_index > 0 {
                        state.run_index -= 1;
                    }
                    run_mode = 2;
                }
            }

            run_count -= 1;
            if run_count < 0 {
                // Run ended: level-code the breaking Sample (§3.8.2.4.1).
                run_mode = 0;
                run_count = 0;
                let v = get_vlc_symbol_level(br, &mut state.vlc[abs_ctx.index as usize], bits);
                if abs_ctx.sign_flip {
                    -v
                } else {
                    v
                }
            } else {
                0
            }
        } else {
            // §3.8.2.4 scalar mode (nonzero context).
            let v = get_vlc_symbol(br, &mut state.vlc[abs_ctx.index as usize], bits);
            if abs_ctx.sign_flip {
                -v
            } else {
                v
            }
        };

        *slot = diff_signed;
        // Write the decoded sample_difference into the row buffer so
        // subsequent neighbour lookups see it (the test-only `decode_line`
        // does not reconstruct Samples; the production decoder
        // `reconstruct_row` writes back reconstructed Samples instead).
        neighbours.current_row[idx] = diff_signed;
    }

    out
}

/// Encode one Line's per-pixel `sample_difference` row — the symmetric
/// inverse of [`decode_line`] (RFC 9043 §4.8 + §3.8.2).
///
/// Takes a slice of `diffs` (length = `neighbours.plane_pixel_width`),
/// each entry being the signed `sample_difference` a matching
/// [`decode_line`] call returns at the same position (i.e. with the §3.5
/// sign flip already applied, exactly as [`decode_line`] writes them
/// back into its `current_row` buffer). The encoder walks the same
/// per-pixel state machine the decoder walks — same neighbour stencil,
/// same §3.5 absolute context, same run-mode predicate (`abs_ctx.index
/// == 0 && l == t == tl`), same scalar / level / run-mode dispatch —
/// and emits the bits the decoder would consume to reproduce the input
/// `diffs` row. The `current_row` buffer is updated in place with the
/// `diffs` values (matching what [`decode_line`] writes back), so the
/// caller can chain calls across rows the same way the decoder does.
///
/// The encoder's `state` is mutated in lockstep with the decoder's: the
/// per-context [`VlcState`] entries see the same `vlc_update` walk in
/// the same order, and the run-mode state machine
/// (`run_index` / `run_mode` / `run_count`) evolves identically.
///
/// # Run-mode encoding
///
/// The run-mode encoder uses lookahead within the current row to decide
/// between long-run "1" bits and short-run "0 + l2-bit residual" with a
/// level-coded break. Run state straddles row boundaries on the decoder
/// side (per §3.8.2.2.1, the run-mode state machine resets per-Plane,
/// not per-Line), and this encoder respects that — but it only sees one
/// row of `diffs` at a time, so a run that would span beyond the
/// current row is encoded with long runs only up to the end of the row
/// (the next call to `encode_line` for the next row continues with the
/// same state, again as on the decoder side).
///
/// # Row-end termination
///
/// If the row ends with a zero in run mode and no level-coded break
/// occurred, the encoder leaves `state.run_count` ≥ 0 and the bit
/// stream simply does not contain bits for the unbroken trailing
/// zeros — the matching decoder consumes nothing for those pixels (it
/// returns 0 per the run-count countdown), so the round trip stays
/// bit-exact. The encoder's intra-row lookahead picks long-run chunks
/// greedily and only emits a short-run when a non-zero diff is
/// reachable inside the current row (so the level-coded follow-up has
/// a real value).
///
/// # Bit width
///
/// `bits` is the per-symbol ESC width (`bits_per_raw_sample` for native
/// YUV / RGB, `bits_per_raw_sample + 1` for the JPEG 2000 RCT path) —
/// same contract as [`decode_line`].
///
/// # Run-region first Sample
///
/// A non-zero `sample_difference` on the **first** Sample of a run region
/// (RFC 9043 §3.8.2.2 absolute context 0 with `l == t == tl`) is fully
/// representable: the encoder emits a §3.8.2.4.1 short run of length zero
/// (a `0` run prefix, a zero-width residual, then the level-coded break),
/// so this Sample is carried directly. (Earlier revisions rejected it with
/// `Error::RunModeFirstPixelNonZero`; that guard is retired and the variant
/// is never produced.) The function therefore returns `Ok(())` for every
/// well-formed row; the `Result` is retained for signature stability with
/// the wider encode surface.
pub fn encode_line(
    bw: &mut BitWriter,
    state: &mut LineDecoderState,
    qtable: &QuantTableSet,
    neighbours: &mut LineNeighborBuffers<'_>,
    diffs: &[i32],
    bits: u32,
) -> Result<(), Error> {
    let width = neighbours.plane_pixel_width as usize;
    debug_assert_eq!(
        diffs.len(),
        width,
        "encode_line: diffs length must equal plane_pixel_width"
    );

    // The §3.5 context (`absolute_context`) and the §3.8.2.2 run-region
    // predicate (`l == t == tl`) MUST evaluate against the reconstructed
    // *Sample* neighbours — exactly the ones the production decoder
    // (`PlaneReconstructor::reconstruct_row`) uses, where `l` =
    // `cur[idx-1]` and `ll` = `cur[idx-2]` hold already-reconstructed
    // Samples. We therefore pre-fill `current_row` with this Line's
    // reconstructed Samples (derived left-to-right from the `diffs`,
    // `pred = median(l, t, tl)`, `Sample = reconstruct_sample(pred, diff,
    // bits)`, reading the just-computed Samples for the `l` / `ll`
    // neighbours and the §3.1 border / row-above for the rest). Both the
    // per-pixel context evaluation below and the run-mode lookahead in
    // `encode_run_region_pixel` then read Samples from `current_row`,
    // matching the decoder bit-for-bit.
    //
    // Filling `current_row` with `diff` values instead (the earlier
    // implementation) was a latent bug: it only agreed with the decoder
    // for single-context Quantization Table Sets, where the routed
    // context is constant regardless of the `l` / `ll` neighbour values.
    // Any genuinely multi-context table desynced the §3.5 routing — and
    // the run predicate — between encode and decode.
    for (px, &diff) in diffs.iter().enumerate() {
        let idx = BORDER_WIDTH + px;
        let l = neighbours.current_row[idx - 1];
        let t = neighbours.prev_row[idx];
        let tl = neighbours.prev_row[idx - 1];
        let pred = median_predict(l, t, tl);
        neighbours.current_row[idx] = reconstruct_sample(pred, diff, bits);
    }

    // §3.8.2.2 run mode is a per-Line state machine, mirroring
    // [`crate::reconstruct::PlaneReconstructor::reconstruct_row`]:
    // `run_index` straddles Lines within a Plane (it lives in `state`,
    // reset per Plane / Slice), but `run_mode` / `run_count` are local to
    // this Line and a Line always begins in scalar mode.
    let mut run_mode: u8 = 0;
    let mut run_count: i32 = 0;

    let mut x = 0usize;
    while x < width {
        let idx = BORDER_WIDTH + x;

        let n = NeighborSamples {
            tt: neighbours.prev_prev_row[idx],
            ll: neighbours.current_row[idx - 2],
            t: neighbours.prev_row[idx],
            tl: neighbours.prev_row[idx - 1],
            tr: neighbours.prev_row[idx + 1],
            l: neighbours.current_row[idx - 1],
        };

        let abs_ctx = absolute_context(qtable, n);

        // §3.8.2.2: enter run mode at a context-0 Sample when not already
        // running.
        if run_mode == 0 && abs_ctx.index == 0 {
            run_mode = 1;
        }

        if run_mode != 0 {
            // At a run boundary (`run_count == 0 && run_mode == 1`) the
            // decoder reads a fresh run prefix. The encoder must choose the
            // prefix bits that make the decoder reproduce `diffs` from `x`
            // onward, then advance `x` over every Sample that decision
            // covers — exactly as the decoder's `run_count` countdown does.
            if run_count == 0 && run_mode == 1 {
                let ri = state.run_index as usize;
                let l2 = LOG2_RUN[ri.min(LOG2_RUN.len() - 1)] as u32;
                let long_run_len = 1usize << l2;

                // Count consecutive run-region zero Samples from `x` and
                // classify the Sample that ends the run.
                let (zero_run, level_break) = scan_run(qtable, neighbours, diffs, x, width);

                if level_break && zero_run < long_run_len {
                    // Short run: the decoder will emit `zero_run` zeros and
                    // then level-code the break. `run_count = zero_run`
                    // fits in `l2` bits (`zero_run < 1 << l2`).
                    bw.put_bit(0);
                    if l2 > 0 {
                        bw.put_bits(zero_run as u32, l2);
                    }
                    if state.run_index > 0 {
                        state.run_index -= 1;
                    }
                    // Decoder: `run_count = zero_run; run_count--; ...`.
                    // The `zero_run` zero Samples consume no further bits;
                    // the breaking Sample is level-coded (§3.8.2.4.1).
                    let break_x = x + zero_run;
                    let bidx = BORDER_WIDTH + break_x;
                    let bn = NeighborSamples {
                        tt: neighbours.prev_prev_row[bidx],
                        ll: neighbours.current_row[bidx - 2],
                        t: neighbours.prev_row[bidx],
                        tl: neighbours.prev_row[bidx - 1],
                        tr: neighbours.prev_row[bidx + 1],
                        l: neighbours.current_row[bidx - 1],
                    };
                    let bctx = absolute_context(qtable, bn);
                    let target_v = if bctx.sign_flip {
                        -diffs[break_x]
                    } else {
                        diffs[break_x]
                    };
                    put_vlc_symbol_level(bw, &mut state.vlc[0], bits, target_v);
                    run_mode = 0;
                    run_count = 0;
                    x = break_x + 1;
                    continue;
                }

                // Long run: covers `long_run_len` zero Samples. The
                // §3.8.2.2.1 `x + run_count <= w` guard grows `run_index`.
                bw.put_bit(1);
                if x + long_run_len <= width && (state.run_index as usize) + 1 < LOG2_RUN.len() {
                    state.run_index += 1;
                }
                run_count = long_run_len as i32;
            }

            // Consume one Sample of the active run. The decoder does
            // `run_count--`; here `run_count > 0` always (a long run sets
            // it `>= 1`, and the short-run branch above already advanced
            // past its zeros), so this Sample is a Sample Difference 0.
            run_count -= 1;
            x += 1;
        } else {
            // §3.8.2.4 scalar mode (nonzero context). Feed the decoder the
            // pre-negation magnitude it reads back.
            let target_v = if abs_ctx.sign_flip {
                -diffs[x]
            } else {
                diffs[x]
            };
            put_vlc_symbol(bw, &mut state.vlc[abs_ctx.index as usize], bits, target_v);
            x += 1;
        }
    }
    Ok(())
}

/// Count the consecutive run-region zero Samples starting at `x` and
/// report whether the run is terminated by a level break (a run-region
/// Sample with a nonzero Sample Difference, level-coded per §3.8.2.4.1)
/// rather than a predicate break (a Sample that leaves the run region) or
/// the Line end.
///
/// Returns `(zero_run, level_break)` where `zero_run` is the number of
/// leading zero Samples and `level_break` is `true` iff the Sample at
/// `x + zero_run` is still in the run region with a nonzero difference.
fn scan_run(
    qtable: &QuantTableSet,
    neighbours: &LineNeighborBuffers<'_>,
    diffs: &[i32],
    x: usize,
    width: usize,
) -> (usize, bool) {
    let mut zero_run = 0usize;
    while x + zero_run < width {
        let zx = x + zero_run;
        let zidx = BORDER_WIDTH + zx;
        let zn = NeighborSamples {
            tt: neighbours.prev_prev_row[zidx],
            ll: neighbours.current_row[zidx - 2],
            t: neighbours.prev_row[zidx],
            tl: neighbours.prev_row[zidx - 1],
            tr: neighbours.prev_row[zidx + 1],
            l: neighbours.current_row[zidx - 1],
        };
        let za = absolute_context(qtable, zn);
        // §3.8.2.2: run mode is governed solely by the absolute context
        // being 0 (the decoder enters / stays in run mode on `ctx == 0`).
        if za.index != 0 {
            // Predicate break: scalar mode fires at this Sample.
            return (zero_run, false);
        }
        if diffs[zx] != 0 {
            // Level break.
            return (zero_run, true);
        }
        zero_run += 1;
    }
    // Reached the Line end with only zeros.
    (zero_run, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_reader::BitReader;

    fn zero_qtable() -> QuantTableSet {
        [[0i32; 256]; crate::predictor::NUM_QUANT_SUBTABLES]
    }

    fn make_buffers(width: u32) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let total = BORDER_WIDTH + width as usize + BORDER_WIDTH;
        (vec![0i32; total], vec![0i32; total], vec![0i32; total])
    }

    #[test]
    fn line_decoder_state_new_sizes_vlc_array() {
        let s = LineDecoderState::new(7);
        assert_eq!(s.vlc.len(), 7);
        // All slots initialised to VLC_STATE_INITIAL.
        assert_eq!(s.vlc[0], VLC_STATE_INITIAL);
        assert_eq!(s.run_index, 0);
        assert_eq!(s.run_mode, 0);
        assert_eq!(s.run_count, 0);
    }

    #[test]
    fn line_decoder_state_reset_run_state_zeros_run_fields_only() {
        let mut s = LineDecoderState::new(4);
        s.vlc[2] = VlcState {
            drift: 5,
            error_sum: 9,
            bias: 3,
            count: 4,
        };
        s.run_index = 7;
        s.run_mode = 2;
        s.run_count = 100;
        s.reset_run_state();
        assert_eq!(s.run_index, 0);
        assert_eq!(s.run_mode, 0);
        assert_eq!(s.run_count, 0);
        // VLC slot 2 untouched.
        assert_eq!(s.vlc[2].drift, 5);
    }

    #[test]
    fn decode_line_emits_correct_row_count() {
        // Single-context (zero qtable) → every sample lives in run-mode
        // territory. We feed a bit stream that immediately breaks into
        // level coding so the decoder threads the level path.
        //
        // For a 4-pixel row, with neighbours all zero (so run-mode
        // predicate is true), the run-mode state machine fires:
        // first sample: run_mode=0 → set to 1, read bit. We provide
        // a 0 bit (short-run path); log2_run[0]=0 so rc=0; run_mode=2,
        // run_count=0. Return 0.
        // Second sample: run_count==0, run_mode==2 → level-coded;
        // we provide "1" → get_vlc_symbol returns 0 → level adjusts to 1.
        // Third sample: run_count==0, run_mode==0 → new run. Bit=0 →
        // run_mode=2, run_count=0. Return 0.
        // Fourth sample: same level-coded path; bit pattern "1" → 1.
        //
        // Bit stream (MSB first): 0 1{prefix 1} {k=2 suffix 0 0} 0 1 0 0.
        // get_vlc_symbol with VLC_STATE_INITIAL needs k computed from
        // (count=1, error_sum=4): k=2 (i goes 1→2→4 with k=0→1→2;
        // exit since 4 < 4 is false). sr_golomb k=2 reads "1" prefix
        // then 2-bit suffix; minimum read = 3 bits to get 0.
        //
        // Build a small input that drives this; the exact decode is
        // not the point — we just check that 4 values come out.
        let mut buf = [0u8; 8];
        // Pre-fill with 0x80 bytes so any bit reads return 1-or-0
        // depending on bit position. We don't need exact match for
        // this size sanity test.
        buf[0] = 0b1011_0001;
        buf[1] = 0b0000_0000;

        let qtable = zero_qtable();
        let mut state = LineDecoderState::new(1);
        let (prev_prev, prev, mut current) = make_buffers(4);
        let mut br = BitReader::new(&buf);

        let mut nb = LineNeighborBuffers {
            prev_row: &prev,
            prev_prev_row: &prev_prev,
            current_row: &mut current,
            plane_pixel_width: 4,
        };

        let row = decode_line(&mut br, &mut state, &qtable, &mut nb, 8);
        assert_eq!(row.len(), 4);
    }

    #[test]
    fn decode_line_scalar_path_when_context_nonzero() {
        // Set up a quant table that produces a nonzero context for the
        // first pixel (so we take the scalar-mode path).
        let mut qtable = zero_qtable();
        // Set Q0[0] = 7 so any (l-tl)=0 input produces context 7.
        // But (l-tl) for first pixel is border-border = 0; Q0[0]=7
        // gives raw context = 7 (other Q's are zero). Index 7.
        qtable[0][0] = 7;

        let mut state = LineDecoderState::new(16);
        let (prev_prev, prev, mut current) = make_buffers(2);
        // Bit stream: "1 00" "1 00" → two zero symbols at k=2
        // for context 7 (which sees the fresh VLC_STATE_INITIAL
        // with count=1, error_sum=4).
        // 1 00 1 00 padded → 0b1001_0000 = 0x90.
        let buf = [0x90u8];
        let mut br = BitReader::new(&buf);

        let mut nb = LineNeighborBuffers {
            prev_row: &prev,
            prev_prev_row: &prev_prev,
            current_row: &mut current,
            plane_pixel_width: 2,
        };
        let row = decode_line(&mut br, &mut state, &qtable, &mut nb, 8);
        assert_eq!(row, vec![0, 0]);
        // The VLC state for context 7 should have advanced (count==3
        // after two decodes).
        assert_eq!(state.vlc[7].count, 3);
        // Run-state should NOT have advanced (scalar path).
        assert_eq!(state.run_index, 0);
        assert_eq!(state.run_mode, 0);
    }

    #[test]
    fn decode_line_writes_decoded_diff_into_current_row() {
        // Sanity: after decoding, current_row[BORDER_WIDTH..BORDER_WIDTH+W]
        // contains the same values returned by decode_line.
        let mut qtable = zero_qtable();
        qtable[0][0] = 7;
        let mut state = LineDecoderState::new(16);
        let (prev_prev, prev, mut current) = make_buffers(3);
        let buf = [0x90u8, 0x80u8];
        let mut br = BitReader::new(&buf);

        let row = {
            let mut nb = LineNeighborBuffers {
                prev_row: &prev,
                prev_prev_row: &prev_prev,
                current_row: &mut current,
                plane_pixel_width: 3,
            };
            decode_line(&mut br, &mut state, &qtable, &mut nb, 8)
        };
        for (i, v) in row.iter().enumerate() {
            assert_eq!(current[BORDER_WIDTH + i], *v);
        }
    }

    // --- encode_line round trips --------------------------------------

    /// Encode `diffs` with a fresh state, then decode the produced bytes
    /// with a parallel fresh state and assert the recovered row matches.
    /// Also asserts the post-trip [`LineDecoderState`] of encoder + decoder
    /// are identical (so any state-drift asymmetry surfaces immediately).
    fn round_trip_encode_line_single_row(
        qtable: &QuantTableSet,
        context_count: usize,
        diffs: &[i32],
        bits: u32,
    ) {
        let width = diffs.len() as u32;
        let mut enc_state = LineDecoderState::new(context_count);
        let (prev_prev_e, prev_e, mut current_e) = make_buffers(width);
        let mut bw = BitWriter::new();
        {
            let mut nb = LineNeighborBuffers {
                prev_row: &prev_e,
                prev_prev_row: &prev_prev_e,
                current_row: &mut current_e,
                plane_pixel_width: width,
            };
            encode_line(&mut bw, &mut enc_state, qtable, &mut nb, diffs, bits)
                .expect("encode_line round-trip helper must not hit a run-mode Case A");
        }
        let bytes = bw.finish();

        let mut dec_state = LineDecoderState::new(context_count);
        let (prev_prev_d, prev_d, mut current_d) = make_buffers(width);
        let mut br = BitReader::new(&bytes);
        let row = {
            let mut nb = LineNeighborBuffers {
                prev_row: &prev_d,
                prev_prev_row: &prev_prev_d,
                current_row: &mut current_d,
                plane_pixel_width: width,
            };
            decode_line(&mut br, &mut dec_state, qtable, &mut nb, bits)
        };

        assert_eq!(row, diffs, "encode_line/decode_line row mismatch");
        // The two state windows must match symbol-for-symbol.
        assert_eq!(
            enc_state.run_index, dec_state.run_index,
            "run_index drift after encode/decode"
        );
        assert_eq!(
            enc_state.run_mode, dec_state.run_mode,
            "run_mode drift after encode/decode"
        );
        assert_eq!(
            enc_state.run_count, dec_state.run_count,
            "run_count drift after encode/decode"
        );
        assert_eq!(
            enc_state.vlc, dec_state.vlc,
            "vlc drift after encode/decode"
        );
        // `encode_line` leaves `current_e` holding the *reconstructed
        // Samples* (so the §3.5 context matches the production decoder);
        // the low-level `decode_line` primitive writes raw *diffs* back.
        // Reconstruct the Samples from the decoder's diff row (with the
        // same all-zero prev/border the encoder saw) and compare against
        // the encoder buffer — they must agree pixel-for-pixel.
        let mut expected_samples = vec![0i32; BORDER_WIDTH + width as usize + BORDER_WIDTH];
        for (px, &row_diff) in row.iter().enumerate() {
            let idx = BORDER_WIDTH + px;
            let l = expected_samples[idx - 1];
            let t = prev_d[idx];
            let tl = prev_d[idx - 1];
            let pred = median_predict(l, t, tl);
            expected_samples[idx] = reconstruct_sample(pred, row_diff, bits);
        }
        assert_eq!(
            &current_e[BORDER_WIDTH..BORDER_WIDTH + diffs.len()],
            &expected_samples[BORDER_WIDTH..BORDER_WIDTH + diffs.len()],
            "encoder current_row must hold reconstructed Samples"
        );
    }

    #[test]
    fn encode_line_round_trips_scalar_only_path() {
        // Nonzero qtable → every pixel is on the scalar path. Tests the
        // sign-flip inversion and the bit emission ordering.
        let mut qtable = zero_qtable();
        qtable[0] = [7; 256]; // constant non-zero context → scalar path
        let diffs = [0i32, 1, -1, 2, -2, 5, -5];
        round_trip_encode_line_single_row(&qtable, 32, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_non_zero_first_run_sample() {
        // Zero qtable → context 0 everywhere → run region. The first
        // Sample's neighbours are all the §3.1 border (0), so `l == t ==
        // tl` holds and the Sample enters run mode. A nonzero Sample
        // Difference at this first run Sample is a §3.8.2.2.1 short run of
        // length zero: the prefix bit `0` (with no residual, since
        // `log2_run[0] == 0`) immediately breaks the run and the Sample is
        // level-coded (§3.8.2.4.1). This is a representable, bit-exact
        // round trip — there is no "unencodable first run Sample".
        let qtable = zero_qtable();
        round_trip_encode_line_single_row(&qtable, 4, &[9i32, 0, 0, 0], 8);
    }

    #[test]
    fn encode_line_round_trips_negative_context_sign_flip() {
        // Synthesise a quant table that yields a negative context for the
        // first pixel so the §3.5 sign-flip path is exercised: Q0[0] = -3
        // → raw_context = -3 → sign_flip = true. Subsequent pixels' L-l
        // / l-tl differences will still tap Q0[0] under all-zero diffs
        // (so the negative context persists row-wide).
        let mut qtable = zero_qtable();
        for slot in qtable[0].iter_mut() {
            *slot = -3;
        }
        let diffs = [1i32, -1, 2, -2, 0, 3, -3];
        round_trip_encode_line_single_row(&qtable, 32, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_all_zero_run_mode() {
        // Zero qtable + all-zero diffs → run-region holds for every
        // pixel; encoder emits a sequence of long-run "1" bits.
        let qtable = zero_qtable();
        let diffs = vec![0i32; 16];
        round_trip_encode_line_single_row(&qtable, 1, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_run_then_break() {
        // Zero qtable. Some zeros, then a non-zero. The non-zero must
        // come via the level-coded path after a short-run break.
        let qtable = zero_qtable();
        // log2_run[0] == 0 so the first long-run consumes 1 pixel. With
        // run_index advancing per long-run, log2_run[1..=3] are also 0;
        // log2_run[4] = 1. Lay out a sequence that mixes short and long
        // runs ending with a non-zero break.
        let diffs = vec![0i32, 0, 0, 0, 0, 7];
        round_trip_encode_line_single_row(&qtable, 1, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_short_run_with_level_break() {
        // Zero qtable + one zero followed by a non-zero in run-region.
        // The zero is the current Phase-3 pixel (emits "0" + 0-bit rc,
        // sets run_mode=2); the non-zero at x=1 hits Phase 2 and emits
        // level-coded. This is the canonical short-run + level break
        // pattern.
        let qtable = zero_qtable();
        let diffs = vec![0i32, 7];
        round_trip_encode_line_single_row(&qtable, 1, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_two_zeros_then_break() {
        // 2 zeros + non-zero in run-region. At x=0 Phase 3 emits a
        // long-run "1" (l2=0, consumes 1 pixel; run_index advances to
        // 1). At x=1 Phase 3 emits short-run "0" + 0-bit rc (l2=0 at
        // run_index=1, run_count=0, run_mode=2). At x=2 Phase 2 emits
        // level-coded non-zero.
        let qtable = zero_qtable();
        let diffs = vec![0i32, 0, 5];
        round_trip_encode_line_single_row(&qtable, 1, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_long_then_short_run_split() {
        // 8 zeros + 1 nonzero. Tests run_index progression + transition
        // through long-runs of varying widths into a short-run break.
        let qtable = zero_qtable();
        let mut diffs = vec![0i32; 8];
        diffs.push(11);
        round_trip_encode_line_single_row(&qtable, 1, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_mixed_scalar_run_via_predicate_change() {
        // First pixel out of run region (nonzero Q0[0]), subsequent
        // pixels stay scalar because the constant context table holds.
        // Verifies the encoder doesn't accidentally enter run-mode after
        // a scalar pixel.
        let mut qtable = zero_qtable();
        qtable[0] = [9; 256];
        let diffs = vec![3i32, -2, 1, 0, -1, 4];
        round_trip_encode_line_single_row(&qtable, 32, &diffs, 8);
    }

    #[test]
    fn encode_line_round_trips_higher_bit_depth() {
        // 16-bit symbols force the wider ESC path in put_sr_golomb_esc.
        let mut qtable = zero_qtable();
        qtable[0] = [5; 256];
        let diffs = [0i32, 100, -100, 4096, -4096, 32767, -32768];
        round_trip_encode_line_single_row(&qtable, 32, &diffs, 16);
    }

    #[test]
    fn encode_line_round_trips_multi_row_continuity() {
        // Two consecutive rows sharing state: tests that the per-row
        // encode/decode pair stays bit-exact across the row boundary.
        let mut qtable = zero_qtable();
        qtable[0] = [11; 256];
        let row0 = [1i32, -1, 2, 0];
        let row1 = [0i32, 3, -3, 5];

        // Encode side
        let mut enc_state = LineDecoderState::new(32);
        let (mut prev_prev_e, mut prev_e, mut current_e) = make_buffers(4);
        let mut bw = BitWriter::new();
        for (y, row) in [&row0[..], &row1[..]].into_iter().enumerate() {
            current_e[0] = 0;
            current_e[BORDER_WIDTH - 1] = if y == 0 { 0 } else { prev_e[BORDER_WIDTH] };
            {
                let mut nb = LineNeighborBuffers {
                    prev_row: &prev_e,
                    prev_prev_row: &prev_prev_e,
                    current_row: &mut current_e,
                    plane_pixel_width: 4,
                };
                encode_line(&mut bw, &mut enc_state, &qtable, &mut nb, row, 8)
                    .expect("encode_line two-row test must not hit run-mode Case A");
            }
            // right-border mirror
            current_e[BORDER_WIDTH + 4] = current_e[BORDER_WIDTH + 3];
            core::mem::swap(&mut prev_prev_e, &mut prev_e);
            core::mem::swap(&mut prev_e, &mut current_e);
        }
        let bytes = bw.finish();

        // Decode side
        let mut dec_state = LineDecoderState::new(32);
        let (mut prev_prev_d, mut prev_d, mut current_d) = make_buffers(4);
        let mut br = BitReader::new(&bytes);
        let mut rows_out: Vec<Vec<i32>> = Vec::new();
        for y in 0..2 {
            current_d[0] = 0;
            current_d[BORDER_WIDTH - 1] = if y == 0 { 0 } else { prev_d[BORDER_WIDTH] };
            let r = {
                let mut nb = LineNeighborBuffers {
                    prev_row: &prev_d,
                    prev_prev_row: &prev_prev_d,
                    current_row: &mut current_d,
                    plane_pixel_width: 4,
                };
                decode_line(&mut br, &mut dec_state, &qtable, &mut nb, 8)
            };
            rows_out.push(r);
            current_d[BORDER_WIDTH + 4] = current_d[BORDER_WIDTH + 3];
            core::mem::swap(&mut prev_prev_d, &mut prev_d);
            core::mem::swap(&mut prev_d, &mut current_d);
        }
        assert_eq!(rows_out[0], row0);
        assert_eq!(rows_out[1], row1);
        assert_eq!(enc_state.vlc, dec_state.vlc);
    }

    #[test]
    fn encode_line_empty_row_produces_empty_bytes() {
        // Zero-width row: encoder emits no bits; decoder consumes none.
        let qtable = zero_qtable();
        let mut state = LineDecoderState::new(1);
        let (prev_prev, prev, mut current) = make_buffers(0);
        let mut bw = BitWriter::new();
        {
            let mut nb = LineNeighborBuffers {
                prev_row: &prev,
                prev_prev_row: &prev_prev,
                current_row: &mut current,
                plane_pixel_width: 0,
            };
            encode_line(&mut bw, &mut state, &qtable, &mut nb, &[], 8)
                .expect("empty-row encode_line cannot error");
        }
        let bytes = bw.finish();
        assert!(bytes.is_empty());
    }

    #[test]
    fn encode_line_state_evolves_in_lockstep_with_decode_line() {
        // Encode one row, then decode it through a fresh state, asserting
        // the per-context VlcState matches symbol-for-symbol at the end.
        // The qtable is constructed to keep every pixel scalar (so VLC
        // state mutation drives the test).
        let mut qtable = zero_qtable();
        qtable[0] = [13; 256];
        // 8 mixed-sign diffs touching context 13 each time.
        let diffs = [3i32, -3, 0, 5, -5, 1, -1, 0];

        let mut enc_state = LineDecoderState::new(32);
        let (prev_prev_e, prev_e, mut current_e) = make_buffers(8);
        let mut bw = BitWriter::new();
        {
            let mut nb = LineNeighborBuffers {
                prev_row: &prev_e,
                prev_prev_row: &prev_prev_e,
                current_row: &mut current_e,
                plane_pixel_width: 8,
            };
            encode_line(&mut bw, &mut enc_state, &qtable, &mut nb, &diffs, 8)
                .expect("scalar-path encode_line must not hit run-mode Case A");
        }
        let bytes = bw.finish();

        let mut dec_state = LineDecoderState::new(32);
        let (prev_prev_d, prev_d, mut current_d) = make_buffers(8);
        let mut br = BitReader::new(&bytes);
        let row = {
            let mut nb = LineNeighborBuffers {
                prev_row: &prev_d,
                prev_prev_row: &prev_prev_d,
                current_row: &mut current_d,
                plane_pixel_width: 8,
            };
            decode_line(&mut br, &mut dec_state, &qtable, &mut nb, 8)
        };
        assert_eq!(row, diffs);
        assert_eq!(
            enc_state.vlc[13], dec_state.vlc[13],
            "VLC state for context 13 diverged after encode/decode"
        );
    }
}
