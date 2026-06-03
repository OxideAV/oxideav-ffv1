//! Per-plane pixel reconstruction for the Golomb-Rice path
//! (RFC 9043 §3.1 / §3.3 / §3.5 / §4.7 / §4.8).
//!
//! Round 8 turns the decoded `sample_difference` stream into
//! reconstructed Sample values. Where [`crate::sample_diff::decode_line`]
//! returns the raw per-row `sample_difference` values (and writes those
//! raw values back into its neighbour buffer), reconstruction must fold
//! the §3.3 median predictor back in *inside* the per-pixel loop,
//! because every Sample's context (§3.5) and prediction (§3.3) depend on
//! the **reconstructed** neighbours, not on the raw differences.
//!
//! The reconstruction equation is, for each Sample at `(p, y, x)`:
//!
//! ```text
//!   pred = median( l, t, l + t - tl )                  (§3.3)
//!   diff = sample_difference[ p ][ y ][ x ]            (§3.8.2, sign-
//!                                                       flipped per §3.5)
//!   Sample = ( pred + diff ) mod ( 1 << bits )         (§3.8 — only the
//!                                                       n LSBs are coded)
//! ```
//!
//! where `bits == bits_per_raw_sample` for the native YUV / grayscale
//! path. RFC 9043 §3.8: "only the n ... least significant bits are
//! used, since this is sufficient to recover the original Sample" — so
//! the add-back is performed modulo `2^bits` and the result lands in
//! `0 .. 2^bits`.
//!
//! ## Border (§3.1)
//!
//! The median predictor + context use a one-/two-sample border around
//! the Slice. RFC 9043 §3.1 + Figure 2 specify:
//!
//! * The column **one to the left** of the Slice equals the leftmost
//!   Slice column shifted **down by one row**; the topmost sample of
//!   that left column is `0`. Concretely: `sample[y][-1] =
//!   sample[y-1][0]`, and `sample[0][-1] = 0`.
//! * The column **one to the right** of the Slice equals the rightmost
//!   Slice column: `sample[y][W] = sample[y][W-1]`.
//! * An **additional** column to the left (`x == -2`, the `L`
//!   neighbour at the first real column) and **two rows above**
//!   (`y == -1` and `y == -2`, the `t` / `tl` / `tr` / `T` neighbours
//!   on the first rows) are all `0`.
//!
//! These rules make the `(L, l, tl, t, tr, T)` stencil from §3.2
//! Figure 3 well-defined for every Sample, including the first row and
//! first column. We realize them with a [`BORDER_WIDTH`]-wide left pad
//! and a one-sample right pad on each working row buffer, seeding the
//! left-of-slice column from the previous row's first sample at the
//! start of each row.
//!
//! ## The §3.3.1 16-bit median exception is NOT applied here
//!
//! RFC 9043 §3.3.1 mandates an alternate median for
//! `colorspace_type == 0 && bits_per_raw_sample == 16 && (coder_type ==
//! 1 || coder_type == 2)` — i.e. **range-coder** 16-bit YCbCr only. The
//! Golomb-Rice path (`coder_type == 0`) is explicitly excluded by that
//! predicate ("the 16-bit content with the Golomb Rice coder type ...
//! [was] not implemented"), so this module — which only drives the
//! Golomb-Rice coder — never needs it.

use crate::bit_reader::BitReader;
use crate::golomb_rice::{
    get_vlc_symbol, get_vlc_symbol_level, VlcState, LOG2_RUN, VLC_STATE_INITIAL,
};
use crate::predictor::{absolute_context, median_predict, NeighborSamples, QuantTableSet};

/// Left-pad width of each working row buffer (RFC 9043 §3.1).
///
/// Two samples of left border are needed so the `L` neighbour
/// (two columns left of `X`, §3.2 Figure 3) is always in bounds for
/// `x == 0`. The outer (`x == -2`) column is always `0`; the inner
/// (`x == -1`) column is seeded per §3.1 from the previous row.
pub const BORDER_LEFT: usize = 2;

/// Right-pad width of each working row buffer (RFC 9043 §3.1).
///
/// One sample of right border is needed so the `tr` neighbour (one
/// column right of `X` on the row above, §3.2 Figure 3) is in bounds
/// for `x == W - 1`. It mirrors the rightmost real column.
pub const BORDER_RIGHT: usize = 1;

/// Reconstruct one Sample from its predicted value and decoded
/// difference (RFC 9043 §3.3 + §3.8).
///
/// `pred` is the §3.3 median prediction; `diff` is the §3.8.2 decoded
/// `sample_difference` with the §3.5 sign flip already applied; `bits`
/// is `bits_per_raw_sample`. Only the low `bits` of `pred + diff` are
/// meaningful (§3.8), so the result is reduced modulo `2^bits` and
/// returned in `0 .. 2^bits`.
///
/// `bits` is clamped to `1..=31` here; FFV1 forbids
/// `bits_per_raw_sample == 0` on the wire (the absent-value default is
/// 8) and caps it at 16, so the clamp never triggers on conforming
/// input but keeps the shift defined for adversarial callers.
#[inline]
pub fn reconstruct_sample(pred: i32, diff: i32, bits: u32) -> i32 {
    let bits = bits.clamp(1, 31);
    let modulus = 1i32 << bits;
    let mask = modulus - 1;
    // `pred` is already in `0 .. 2^bits` (it is the median of three
    // in-range reconstructed neighbours). `diff` can be negative; the
    // bitwise AND with `mask` realizes the §3.8 "low n bits" modular
    // reduction for both signs in two's complement.
    (pred.wrapping_add(diff)) & mask
}

/// Per-plane run-mode + VLC state for the Golomb-Rice reconstruction
/// loop (RFC 9043 §3.8.2).
///
/// Mirrors [`crate::sample_diff::LineDecoderState`] but is owned by the
/// reconstructor so a single [`PlaneReconstructor`] carries the full
/// per-plane entropy state across rows (run mode straddles row
/// boundaries per §3.8.2.2.1; the VLC contexts persist for the whole
/// plane).
///
/// Exposed `pub(crate)` so the §3.7.2 RGB line-major driver
/// ([`crate::rgb_reconstruct`]) can keep one entropy state alive per
/// Plane across the row-interleaved traversal (run mode straddles row
/// boundaries within a Plane, so the Lines of a Plane must share state
/// even though they are decoded non-contiguously in RGB).
#[derive(Debug, Clone)]
pub(crate) struct PlaneEntropyState {
    vlc: Vec<VlcState>,
    run_index: u32,
    run_mode: u8,
    run_count: i32,
}

impl PlaneEntropyState {
    pub(crate) fn new(context_count: usize) -> Self {
        Self {
            vlc: vec![VLC_STATE_INITIAL; context_count],
            run_index: 0,
            run_mode: 0,
            run_count: 0,
        }
    }

    /// Reset the run-mode state machine (§3.8.2.2.1 — reset at the
    /// start of each Plane, NOT between Lines of the same Plane).
    pub(crate) fn reset_run_state(&mut self) {
        self.run_index = 0;
        self.run_mode = 0;
        self.run_count = 0;
    }
}

/// Reconstruct one Plane of a Slice from the Golomb-Rice bitstream
/// (RFC 9043 §3.1 / §3.3 / §3.5 / §4.8).
///
/// Decodes `height` Lines of `width` Samples each, applying the §3.3
/// median predictor + §3.8 modular add-back per Sample, and returns
/// the reconstructed Plane as a row-major `Vec<i32>` of length
/// `width * height` (each entry in `0 .. 2^bits`).
///
/// * `br` is the slice's Golomb-Rice bit reader, positioned at the
///   start of this Plane's `sample_difference` stream.
/// * `qtable` is the §3.4 Quantization Table Set selected for this
///   Plane via the §3.6 `quant_table_set_index`.
/// * `context_count` sizes the per-context VLC state array (§4.1.2).
/// * `bits` is `bits_per_raw_sample`.
///
/// The entropy state (run mode + per-context VLC) is created fresh for
/// the Plane and discarded on return — Planes do not share entropy
/// state (each Plane is decoded with its own keyframe-initialised
/// contexts per §3.8.2.5, and run mode resets per Plane per
/// §3.8.2.2.1).
#[derive(Debug)]
pub struct PlaneReconstructor;

impl PlaneReconstructor {
    /// Decode + reconstruct a full Plane.
    ///
    /// Allocates a fresh per-context VLC state buffer
    /// ([`PlaneEntropyState::new`]). Suitable when the caller knows
    /// no other Plane in this Slice will use the same Quantization
    /// Table Set; otherwise route through
    /// [`Self::reconstruct_plane_with_state`] so the VLC state is
    /// shared across Planes that share a `quant_table_set_index`
    /// (RFC 9043 §3.6 / §4.6.6 / §3.8.2.5).
    pub fn reconstruct_plane(
        br: &mut BitReader<'_>,
        qtable: &QuantTableSet,
        context_count: usize,
        width: usize,
        height: usize,
        bits: u32,
    ) -> Vec<i32> {
        let mut state = PlaneEntropyState::new(context_count.max(1));
        Self::reconstruct_plane_with_state(br, &mut state, qtable, width, height, bits)
    }

    /// Decode + reconstruct a full Plane against a caller-supplied
    /// per-context VLC state buffer.
    ///
    /// Per RFC 9043 §3.8.2.5 the VLC state (`drift`, `error_sum`,
    /// `bias`, `count` per context) is initialised at `keyframe == 1`
    /// and evolves through the remainder of the Slice. RFC 9043 §3.6
    /// / §4.6.6 routes multiple Planes to the same Quantization Table
    /// Set, and per §4.2 Figure 28 the state buffer is allocated
    /// **per Quantization Table Set**; Planes that share a
    /// `quant_table_set_index` (Cb + Cr on every `chroma_planes ==
    /// true` Slice, plus any other layout that aliases two Planes
    /// onto one set) must therefore share the same `PlaneEntropyState`
    /// across their reconstruction calls.
    ///
    /// The `run_index` / `run_mode` / `run_count` triple is reset on
    /// entry per RFC 9043 §3.8.2.2.1 ("`run_index` is reset to zero
    /// for each Plane and Slice") — only the per-context VLC state
    /// survives across the per-Plane boundary.
    pub(crate) fn reconstruct_plane_with_state(
        br: &mut BitReader<'_>,
        state: &mut PlaneEntropyState,
        qtable: &QuantTableSet,
        width: usize,
        height: usize,
        bits: u32,
    ) -> Vec<i32> {
        let mut out = vec![0i32; width.saturating_mul(height)];
        if width == 0 || height == 0 {
            return out;
        }

        // §3.8.2.2.1 — reset the run-mode state machine at the start
        // of each Plane (and each Slice). The VLC state (`drift`,
        // `error_sum`, `bias`, `count`) is NOT reset here: it is
        // §3.8.2.5 keyframe-initialised by the caller and evolves
        // across Planes that share this `quant_table_set_index`.
        state.reset_run_state();

        // Two working row buffers carrying the §3.1 border:
        //   [0]            = additional left column (always 0)
        //   [1]            = left-of-slice column (§3.1: prev row's [0])
        //   [2 .. 2+W)     = the W real Samples
        //   [2+W]          = right border (mirror of [2+W-1])
        let stride = BORDER_LEFT + width + BORDER_RIGHT;
        // Row two-above (`T` / initial `prev`) and one-above (`prev`).
        // Both start all-zero per §3.1 (two rows above the Slice are 0).
        let mut prev_prev = vec![0i32; stride];
        let mut prev = vec![0i32; stride];
        let mut cur = vec![0i32; stride];

        for y in 0..height {
            // §3.1 left-of-slice column: sample[y][-1] = sample[y-1][0],
            // with sample[0][-1] = 0. `prev[BORDER_LEFT]` is the
            // previous row's first real Sample (0 for y == 0 since the
            // row above the Slice is the all-zero border).
            cur[0] = 0; // additional left column (x == -2) is always 0.
            cur[BORDER_LEFT - 1] = prev[BORDER_LEFT];

            Self::reconstruct_row(br, state, qtable, &prev, &prev_prev, &mut cur, width, bits);

            // §3.1 right border: sample[y][W] = sample[y][W-1].
            cur[BORDER_LEFT + width] = cur[BORDER_LEFT + width - 1];

            // Commit the reconstructed real Samples to the output.
            let row_off = y * width;
            out[row_off..row_off + width].copy_from_slice(&cur[BORDER_LEFT..BORDER_LEFT + width]);

            // Rotate the row buffers: prev_prev <- prev <- cur, and
            // reuse the old prev_prev as the next `cur` scratch.
            core::mem::swap(&mut prev_prev, &mut prev);
            core::mem::swap(&mut prev, &mut cur);
            // `cur` now holds the old `prev_prev` content; zero only the
            // border cells we re-seed at the top of the loop. The real
            // Sample cells are fully overwritten by reconstruct_row.
            cur[0] = 0;
        }

        out
    }

    /// Reconstruct one Line into `cur[BORDER_LEFT .. BORDER_LEFT + W]`.
    ///
    /// Reads `prev` (row above) and `prev_prev` (two rows above) for
    /// the `t` / `tl` / `tr` / `T` neighbours, and the already-written
    /// prefix of `cur` for the `l` / `L` neighbours. The §3.1 border
    /// cells of `prev` / `cur` are assumed already seeded by the
    /// caller.
    ///
    /// `pub(crate)` so the §3.7.2 RGB line-major driver can step a
    /// single Line per Plane while keeping the per-Plane `state` /
    /// border buffers alive across the interleave.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reconstruct_row(
        br: &mut BitReader<'_>,
        state: &mut PlaneEntropyState,
        qtable: &QuantTableSet,
        prev: &[i32],
        prev_prev: &[i32],
        cur: &mut [i32],
        width: usize,
        bits: u32,
    ) {
        for x in 0..width {
            let idx = BORDER_LEFT + x;

            let n = NeighborSamples {
                // `T` — two rows above, same column.
                tt: prev_prev[idx],
                // `L` — two columns left, same row.
                ll: cur[idx - 2],
                // `t` — one row above, same column.
                t: prev[idx],
                // `tl` — one row above, one column left.
                tl: prev[idx - 1],
                // `tr` — one row above, one column right.
                tr: prev[idx + 1],
                // `l` — same row, one column left.
                l: cur[idx - 1],
            };

            let abs_ctx = absolute_context(qtable, n);

            // §3.8.2.2 run-mode predicate: |context| == 0 and the
            // run-mode predicted Sample equals the median-predicted one
            // (l == t == tl).
            let in_run_region = abs_ctx.index == 0 && n.l == n.t && n.l == n.tl;

            let diff = if in_run_region {
                decode_run_mode_sample(br, state, bits, abs_ctx.sign_flip)
            } else {
                state.reset_run_state();
                let v = get_vlc_symbol(br, &mut state.vlc[abs_ctx.index as usize], bits);
                if abs_ctx.sign_flip {
                    -v
                } else {
                    v
                }
            };

            // §3.3 prediction from reconstructed neighbours + §3.8
            // modular add-back.
            let pred = median_predict(n.l, n.t, n.tl);
            cur[idx] = reconstruct_sample(pred, diff, bits);
        }
    }
}

/// Run-mode per-sample decode helper (RFC 9043 §3.8.2.2.1 +
/// §3.8.2.4.1).
///
/// Returns the decoded `sample_difference` (sign-flipped per §3.5 when
/// `sign_flip` is set). Encapsulates the run-mode state machine via
/// mutations to `state.run_*`.
fn decode_run_mode_sample(
    br: &mut BitReader<'_>,
    state: &mut PlaneEntropyState,
    bits: u32,
    sign_flip: bool,
) -> i32 {
    // Phase 1: still inside a counted run → emit 0, consume nothing.
    if state.run_count > 0 {
        state.run_count -= 1;
        return 0;
    }

    // Phase 2: run just broke → next Sample is level-coded (0 excluded).
    if state.run_mode == 2 {
        let v = get_vlc_symbol_level(br, &mut state.vlc[0], bits);
        state.run_mode = 0;
        return if sign_flip { -v } else { v };
    }

    // Phase 3: start (or continue the unary prefix of) a new run
    // (§3.8.2.2.1).
    if state.run_mode == 0 {
        state.run_mode = 1;
    }

    if br.get_bit() == 1 {
        // Long run: 1 << log2_run[run_index]; the current Sample
        // consumes one slot.
        let l2 = LOG2_RUN[state.run_index as usize % LOG2_RUN.len()] as u32;
        state.run_count = (1i32 << l2) - 1;
        if (state.run_index as usize) + 1 < LOG2_RUN.len() {
            state.run_index += 1;
        }
        0
    } else {
        // Short run: read log2_run[run_index] residual bits; then the
        // next Sample is the level-coded "first different" one.
        let l2 = LOG2_RUN[state.run_index as usize % LOG2_RUN.len()] as u32;
        let rc = if l2 == 0 { 0 } else { br.get_bits(l2) as i32 };
        state.run_count = rc;
        if state.run_index > 0 {
            state.run_index -= 1;
        }
        state.run_mode = 2;
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::NUM_QUANT_SUBTABLES;

    fn zero_qtable() -> QuantTableSet {
        [[0i32; 256]; NUM_QUANT_SUBTABLES]
    }

    fn single_context_qtable(c: i32) -> QuantTableSet {
        let mut q = zero_qtable();
        q[0][0] = c;
        q
    }

    /// Q0 maps every index to `c`, so the §3.5 context is the constant
    /// `c` for every pixel regardless of neighbour values — keeps a
    /// multi-pixel decode on the scalar VLC path (`state.vlc[c]`).
    fn constant_context_qtable(c: i32) -> QuantTableSet {
        let mut q = zero_qtable();
        q[0] = [c; 256];
        q
    }

    // --- reconstruct_sample (the §3.8 modular add-back) ---------------

    #[test]
    fn reconstruct_sample_no_wrap() {
        // 100 + 5 = 105, well within 8-bit range.
        assert_eq!(reconstruct_sample(100, 5, 8), 105);
    }

    #[test]
    fn reconstruct_sample_negative_diff() {
        // 100 + (-30) = 70.
        assert_eq!(reconstruct_sample(100, -30, 8), 70);
    }

    #[test]
    fn reconstruct_sample_wraps_high() {
        // 250 + 10 = 260 → 260 & 0xFF = 4 (§3.8 only low 8 bits coded).
        assert_eq!(reconstruct_sample(250, 10, 8), 4);
    }

    #[test]
    fn reconstruct_sample_wraps_low() {
        // 5 + (-10) = -5 → -5 & 0xFF = 251 (two's-complement modular).
        assert_eq!(reconstruct_sample(5, -10, 8), 251);
    }

    #[test]
    fn reconstruct_sample_10bit_range() {
        // 1000 + 50 = 1050 → 1050 & 0x3FF = 26.
        assert_eq!(reconstruct_sample(1000, 50, 10), 26);
        // 0 + (-1) = -1 → 1023.
        assert_eq!(reconstruct_sample(0, -1, 10), 1023);
    }

    #[test]
    fn reconstruct_sample_16bit_range() {
        assert_eq!(reconstruct_sample(65535, 1, 16), 0);
        assert_eq!(reconstruct_sample(0, -1, 16), 65535);
    }

    // --- median predictor + reconstruction interplay -----------------

    #[test]
    fn reconstruct_sample_uses_predicted_base() {
        // pred = median(l=10, t=20, l+t-tl=10+20-5=25) = 20.
        let pred = median_predict(10, 20, 5);
        assert_eq!(pred, 20);
        // With diff = 3 → Sample = 23.
        assert_eq!(reconstruct_sample(pred, 3, 8), 23);
    }

    // --- whole-plane reconstruction over the Golomb-Rice path --------

    #[test]
    fn reconstruct_flat_plane_run_mode() {
        // A 4x3 plane whose every Sample reconstructs to 0. With an
        // all-zero qtable, the run-mode predicate holds at the first
        // Sample of the first row (all border neighbours 0), and the
        // long-run path covers the rest at zero cost.
        //
        // First Sample (run_mode 0 → 1): bit "1" → long run, l2 =
        // log2_run[0] = 0, run_count = (1<<0) - 1 = 0. Returns 0.
        // run_index advances to 1.
        // Second Sample: run_count == 0, run_mode == 1. bit "1" → long
        // run, l2 = log2_run[1] = 0, run_count = 0. Returns 0. ...
        //
        // Every Sample reads exactly one "1" bit. 12 Samples → 12 ones.
        // 0xFF 0xF0 supplies 12 set bits.
        let qtable = zero_qtable();
        let buf = [0xFFu8, 0xF0u8];
        let mut br = BitReader::new(&buf);
        let plane = PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 1, 4, 3, 8);
        assert_eq!(plane, vec![0; 12]);
    }

    #[test]
    fn reconstruct_single_sample_scalar() {
        // 1x1 plane, nonzero context → scalar path. Fresh VLC state
        // (count=1, error_sum=4) → k=2. Bits "1 10" (sr_golomb k=2):
        // unsigned 2 → even → 1; bias 0 → diff = 1.
        // pred for the single Sample: all neighbours are border 0 →
        // median(0,0,0) = 0. Sample = (0 + 1) & 0xFF = 1.
        // `single_context_qtable` suffices: a 1-pixel plane never reads
        // a nonzero (l-tl) index.
        let qtable = single_context_qtable(7);
        let buf = [0b1100_0000u8];
        let mut br = BitReader::new(&buf);
        let plane = PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 16, 1, 1, 8);
        assert_eq!(plane, vec![1]);
    }

    #[test]
    fn reconstruct_row_predicts_from_left_neighbour() {
        // 2x1 plane on the constant-context-7 scalar path. The second
        // Sample's prediction folds in the first Sample (its `l`
        // neighbour), so the reconstructed values chain.
        //
        // Decoded `sample_difference` sequence is [1, 0]:
        // Sample 0: pred = median(0,0,0) = 0. k=2 (count=1,es=4),
        //   sr_golomb "1 10" → +1 → Sample 0 = 1.
        // Sample 1: l = 1, t = 0 (border above), tl = 0 → pred =
        //   median(1, 0, 1) = 1. State after sample 0: count=2, es=5,
        //   drift nudged to 0, bias 1. k=2; sr_golomb "1 01" → unsigned
        //   1 → -1, + bias 1 → diff 0 → Sample 1 = (1 + 0) & 0xFF = 1.
        //
        // Bits "1 10" + "1 01" = 110101 → 0b1101_0100 = 0xD4.
        let qtable = constant_context_qtable(7);
        let buf = [0xD4u8];
        let mut br = BitReader::new(&buf);
        let plane = PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 16, 2, 1, 8);
        assert_eq!(plane, vec![1, 1]);
    }

    #[test]
    fn reconstruct_empty_plane_dimensions() {
        let qtable = zero_qtable();
        let buf = [0u8; 4];
        let mut br = BitReader::new(&buf);
        assert_eq!(
            PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 1, 0, 5, 8),
            Vec::<i32>::new()
        );
        let mut br = BitReader::new(&buf);
        assert_eq!(
            PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 1, 5, 0, 8),
            Vec::<i32>::new()
        );
    }

    #[test]
    fn reconstruct_samples_stay_in_range() {
        // Drive an 8x8 plane with arbitrary bits and confirm every
        // reconstructed Sample is within 0..256 (the §3.8 modular
        // invariant), regardless of the bit pattern.
        let qtable = single_context_qtable(3);
        let buf = [0xA5u8; 64];
        let mut br = BitReader::new(&buf);
        let plane = PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 16, 8, 8, 8);
        assert_eq!(plane.len(), 64);
        for s in plane {
            assert!((0..256).contains(&s), "sample {s} out of 8-bit range");
        }
    }
}
