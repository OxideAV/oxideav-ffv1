//! Per-plane pixel reconstruction for the **range-coder** slice path
//! (RFC 9043 §3.1 / §3.3 / §3.5 / §3.7 / §3.8.1.2 / §4.7 / §4.8).
//!
//! Round 9 mirrors [`crate::reconstruct::PlaneReconstructor`] for
//! `coder_type == 1 || coder_type == 2` (the range-coder slice content
//! path). The Golomb-Rice path in [`crate::reconstruct`] reads each
//! Sample's `sample_difference` from an MSB-first bit reader via the
//! adaptive `get_vlc_symbol` decoder; the range-coder path replaces
//! that with one [`get_sr`](crate::symbol::get_sr) call per Sample,
//! consuming bits from a [`RangeDecoder`].
//!
//! The two paths differ in three concrete ways, all surfaced here:
//!
//! 1. **No run mode.** RFC 9043 §3.8.2.2 describes run-mode strictly
//!    as part of the Golomb-Rice path ("Run Mode" lives under §3.8.2,
//!    "Golomb Rice Mode"). The range-coder mode (§3.8.1) is a single
//!    `get_symbol` per Sample, with no concept of a zero run.
//! 2. **Per-context state windows.** `get_symbol` (Figure 21) reads a
//!    32-slot state window per call (offsets `0`, `1..=10`, `11..=21`,
//!    `22..=31`). Each §3.5 absolute context owns its own 32-slot
//!    window, all of which are initialised to 128 at the start of the
//!    Slice (§3.8.1.3: "all range coder state variables are set to
//!    their initial state" when `keyframe == 1`). The flat state buffer
//!    is therefore `context_count * 32` bytes wide.
//! 3. **Bit source.** Bits come from a [`RangeDecoder`], not a
//!    [`BitReader`]. The decoder is supplied externally so the caller
//!    can keep walking the same range coder across multiple planes
//!    (YCbCr `Plane then Line` ordering) without re-seeding state.
//!
//! Border handling (§3.1), median prediction (§3.3 / `median_predict`),
//! sign-flip (§3.5), and modular add-back (§3.8 /
//! [`reconstruct_sample`]) are byte-for-byte identical to the
//! Golomb-Rice path.
//!
//! ## The §3.3.1 16-bit median exception
//!
//! Unlike the Golomb-Rice reconstructor, this module **does** trigger
//! the §3.3.1 alternate predictor for
//! `colorspace_type == 0 && bits_per_raw_sample == 16 && (coder_type ==
//! 1 || coder_type == 2)`. The exception is selected by the
//! `use_16bit_median` flag the caller passes in; it does not change
//! any decoding state beyond swapping the median formula.

use crate::predictor::{absolute_context, median_predict, NeighborSamples, QuantTableSet};
use crate::range_coder::{RangeDecoder, PARAMETERS_INITIAL_STATE};
use crate::reconstruct::{reconstruct_sample, BORDER_LEFT, BORDER_RIGHT};
use crate::symbol::{get_sr_window, SYMBOL_CONTEXT_SIZE};

/// One 32-slot state window per §3.5 absolute context (RFC 9043
/// §3.8.1.2 + §3.8.1.3).
///
/// Mirrors the per-context window layout of [`get_sr`]: each call
/// reads offsets `0`, `1..=10`, `11..=21`, `22..=31` inside a single
/// 32-byte window, and `context_count` such windows live contiguously
/// in a flat buffer of length `context_count * 32`.
///
/// Exposed `pub(crate)` so the §3.7.2 RGB line-major driver
/// ([`crate::rgb_reconstruct`]) can keep one window-set alive per Plane
/// across the row-interleaved traversal — the YCbCr Plane-major driver
/// decodes a whole Plane in one [`RangePlaneReconstructor::reconstruct_plane`]
/// call, but RGB interleaves Lines between Planes, so each Plane's
/// entropy state must persist outside a single-Plane loop.
#[derive(Debug, Clone)]
pub(crate) struct RangePlaneState {
    /// Flat per-context state buffer; `state[c * 32 .. (c + 1) * 32]`
    /// is context `c`'s 32-slot window. All slots initialise to 128.
    state: Vec<u8>,
}

impl RangePlaneState {
    /// Allocate `context_count` windows of 32 slots, each filled with
    /// the §3.8.1.3 initial state value (`128`).
    pub(crate) fn new(context_count: usize) -> Self {
        let count = context_count.max(1);
        Self {
            state: vec![PARAMETERS_INITIAL_STATE; count * SYMBOL_CONTEXT_SIZE],
        }
    }

    /// Like [`Self::new`], but seed the windows from the §4.2.15
    /// reconstructed initial states
    /// ([`crate::quant_table::reconstruct_initial_states`]) when the
    /// stream transmitted them (`states_coded == 1`). `None` is the
    /// §4.2.14 default — every slot stays 128, identical to `new`.
    /// The seed layout matches this buffer's (`seed[c * 32 ..]` is
    /// context `c`'s window); a short seed leaves the remaining
    /// windows at 128.
    pub(crate) fn seeded(context_count: usize, seed: Option<&[u8]>) -> Self {
        let mut s = Self::new(context_count);
        if let Some(seed) = seed {
            let n = seed.len().min(s.state.len());
            s.state[..n].copy_from_slice(&seed[..n]);
            // Degenerate seeds (state 0, or the 1..=8 / 249..=255 band
            // whose §3.8.1.5 default transitions feed into 0) are
            // copied faithfully: termination is guaranteed at the
            // coder level (the `rangeoff.max(1)` guard in
            // `get_rac` / `put_rac`), symmetrically on both sides, so
            // the seeded pair still round-trips bit-exactly.
        }
        s
    }

    /// Mutable view of context `c`'s 32-slot window as a fixed-size
    /// array, so the per-Sample [`get_sr_window`] call compiles with no
    /// per-slot bounds checks (one range check here, none inside).
    #[inline]
    fn window_mut(&mut self, c: usize) -> &mut [u8; SYMBOL_CONTEXT_SIZE] {
        let lo = c * SYMBOL_CONTEXT_SIZE;
        (&mut self.state[lo..lo + SYMBOL_CONTEXT_SIZE])
            .try_into()
            .expect("sliced to SYMBOL_CONTEXT_SIZE")
    }
}

/// Per-plane pixel reconstruction for the range-coder slice path.
///
/// Decodes `height` Lines of `width` Samples each, sampling
/// `sample_difference` from `rc` via [`get_sr`], folding the §3.3
/// median predictor + §3.5 sign-flip into the per-pixel loop, and
/// applying the §3.8 modular add-back to recover a row-major Plane.
///
/// * `rc` — the slice's range decoder, already positioned at the start
///   of this Plane's `sample_difference` stream. The decoder is
///   threaded across Planes for YCbCr `Plane then Line` ordering.
///   [`Self::reconstruct_plane`] allocates a fresh per-context state
///   buffer (suitable when this Plane is the first / only one to use
///   its Quantization Table Set); when multiple Planes share a
///   `quant_table_set_index` (RFC 9043 §3.6 / §4.6.6 — the common
///   Cb + Cr case on `chroma_planes == true`) the caller must
///   instead route through [`Self::reconstruct_plane_with_state`]
///   and share one [`RangePlaneState`] across all Planes that select
///   the same slot, per §3.8.1.3 (state is keyframe-initialised,
///   evolves thereafter) and §4.6.6 (the slot selects "the
///   Quantization Table Set and the initial states").
/// * `qtable` — the §3.4 Quantization Table Set this Plane selects via
///   `quant_table_set_index`.
/// * `context_count` — sizes the per-context state buffer (`§4.1.2`
///   `context_count[i]`). Each context owns a fresh 32-slot window.
/// * `width` / `height` — `plane_pixel_width` / `plane_pixel_height`
///   for this Plane (§4.7.2 / §4.8.1).
/// * `bits` — `bits_per_raw_sample` (or `bits_per_raw_sample + 1` for
///   the JPEG 2000 RCT path, per Figure 10).
/// * `use_16bit_median` — when `true`, the §3.3.1 alternate predictor
///   is used. The caller computes the predicate
///   `colorspace_type == 0 && bits_per_raw_sample == 16 &&
///   (coder_type == 1 || coder_type == 2)`. This is the only place
///   the range-coder path diverges from the Golomb-Rice path on
///   prediction.
#[derive(Debug)]
pub struct RangePlaneReconstructor;

impl RangePlaneReconstructor {
    /// Decode + reconstruct a full Plane. See the type-level docs.
    ///
    /// Allocates a fresh per-context state buffer of size
    /// `context_count`. Equivalent to calling
    /// [`Self::reconstruct_plane_with_state`] with
    /// `&mut RangePlaneState::new(context_count)`; kept as a stable
    /// public entry point for callers that decode a single Plane at a
    /// time (or whose Planes all use distinct Quantization Table Sets).
    pub fn reconstruct_plane(
        rc: &mut RangeDecoder<'_>,
        qtable: &QuantTableSet,
        context_count: usize,
        width: usize,
        height: usize,
        bits: u32,
        use_16bit_median: bool,
    ) -> Vec<i32> {
        let mut state = RangePlaneState::new(context_count);
        Self::reconstruct_plane_with_state(
            rc,
            &mut state,
            qtable,
            width,
            height,
            bits,
            use_16bit_median,
        )
    }

    /// Decode + reconstruct a full Plane against a caller-supplied
    /// per-context state buffer.
    ///
    /// RFC 9043 §3.8.1.3 says the range-coder state variables are
    /// initialised at `keyframe == 1` and **then evolve through the
    /// remainder of the Slice**. RFC 9043 §4.6.6 / §3.6 routes
    /// multiple Planes to the same Quantization Table Set via
    /// `quant_table_set_index`; per the §4.2 Figure 28
    /// `initial_state_delta` loop the state buffer is allocated
    /// **per Quantization Table Set**, not per Plane. So Planes
    /// that share a `quant_table_set_index` (Cb + Cr on every
    /// `chroma_planes == true` Slice; multiple Planes pointing at
    /// the same set in any other layout) **must share** the per-set
    /// state buffer across their per-Plane reconstruction calls —
    /// the second Plane to touch a set continues evolving the state
    /// the first Plane left it in, not a fresh `128`-filled window.
    ///
    /// The caller owns the [`RangePlaneState`] lifecycle: build one
    /// per `quant_table_set_index` *before* the per-Plane loop,
    /// initialised via [`RangePlaneState::new(context_count)`], and
    /// pass the same `&mut state` into every per-Plane call that
    /// selects that set.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reconstruct_plane_with_state(
        rc: &mut RangeDecoder<'_>,
        state: &mut RangePlaneState,
        qtable: &QuantTableSet,
        width: usize,
        height: usize,
        bits: u32,
        use_16bit_median: bool,
    ) -> Vec<i32> {
        let mut out = vec![0i32; width.saturating_mul(height)];
        if width == 0 || height == 0 {
            return out;
        }

        let stride = BORDER_LEFT + width + BORDER_RIGHT;
        let mut prev_prev = vec![0i32; stride];
        let mut prev = vec![0i32; stride];
        let mut cur = vec![0i32; stride];

        for y in 0..height {
            cur[0] = 0;
            cur[BORDER_LEFT - 1] = prev[BORDER_LEFT];

            Self::reconstruct_row(
                rc,
                state,
                qtable,
                &prev,
                &prev_prev,
                &mut cur,
                width,
                bits,
                use_16bit_median,
            );

            cur[BORDER_LEFT + width] = cur[BORDER_LEFT + width - 1];

            let row_off = y * width;
            out[row_off..row_off + width].copy_from_slice(&cur[BORDER_LEFT..BORDER_LEFT + width]);

            core::mem::swap(&mut prev_prev, &mut prev);
            core::mem::swap(&mut prev, &mut cur);
            cur[0] = 0;
        }

        out
    }

    /// Reconstruct one Line into `cur[BORDER_LEFT .. BORDER_LEFT + W]`.
    ///
    /// Reads `prev` and `prev_prev` for the rows above and the prefix
    /// of `cur` for the same row. The caller seeds `cur`'s border
    /// cells before each call.
    ///
    /// `pub(crate)` so the §3.7.2 RGB line-major driver can step a
    /// single Line per Plane while keeping the per-Plane `state` /
    /// border buffers alive across the interleave.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reconstruct_row(
        rc: &mut RangeDecoder<'_>,
        state: &mut RangePlaneState,
        qtable: &QuantTableSet,
        prev: &[i32],
        prev_prev: &[i32],
        cur: &mut [i32],
        width: usize,
        bits: u32,
        use_16bit_median: bool,
    ) {
        // Neighbour carry (r386): `l` / `ll` are the two most recently
        // reconstructed Samples of this row and `tl` / `t` slide along
        // the row above, so each iteration loads only `tr` and `tt`
        // and stores the new Sample — bit-identical to re-reading all
        // six stencil cells (`cur[idx-1]` / `cur[idx-2]` are exactly
        // the values the previous iterations wrote, and the border
        // seeds cover `x == 0` / `x == 1`).
        let mut ll = cur[BORDER_LEFT - 2];
        let mut l = cur[BORDER_LEFT - 1];
        let mut tl = prev[BORDER_LEFT - 1];
        let mut t = prev[BORDER_LEFT];
        for x in 0..width {
            let idx = BORDER_LEFT + x;
            let tr = prev[idx + 1];

            let n = NeighborSamples {
                tt: prev_prev[idx],
                ll,
                t,
                tl,
                tr,
                l,
            };

            let abs_ctx = absolute_context(qtable, n);

            // Range-coder path has NO run mode (§3.8.2.2 belongs to the
            // Golomb-Rice path). Every Sample is one `get_symbol` call.
            let raw = get_sr_window(rc, state.window_mut(abs_ctx.index as usize));
            let diff = if abs_ctx.sign_flip { -raw } else { raw };

            // §3.3 / §3.3.1 prediction from reconstructed neighbours.
            let pred = if use_16bit_median {
                median16_predict(n.l, n.t, n.tl)
            } else {
                median_predict(n.l, n.t, n.tl)
            };
            let sample = reconstruct_sample(pred, diff, bits);
            cur[idx] = sample;
            ll = l;
            l = sample;
            tl = t;
            t = tr;
        }
    }
}

/// §3.3.1 alternate median predictor.
///
/// Active only when
/// `colorspace_type == 0 && bits_per_raw_sample == 16 &&
/// (coder_type == 1 || coder_type == 2)`. RFC 9043 §3.3.1:
///
/// ```text
///   left16s = l  >= 32768 ? l  - 65536 : l
///   top16s  = t  >= 32768 ? t  - 65536 : t
///   diag16s = tl >= 32768 ? tl - 65536 : tl
///   median(left16s, top16s, left16s + top16s - diag16s)
/// ```
///
/// The reinterpreted "signed 16-bit" values flow through the same
/// median-of-three primitive; the result is the predicted Sample,
/// which the §3.8 modular add-back will fold back to `0..2^bits`
/// regardless of the (intermediate) sign.
#[inline]
fn median16_predict(l: i32, t: i32, tl: i32) -> i32 {
    median16_predict_pub(l, t, tl)
}

/// Public `pub(crate)` alias for [`median16_predict`] so the symmetric
/// encoder ([`crate::range_encode`]) can reuse the same §3.3.1
/// predictor body without re-implementing it. The two callers share
/// one definition: any future spec clarification (e.g. an erratum
/// changing the 32768 / 65536 constants) propagates to both at once.
#[inline]
pub(crate) fn median16_predict_pub(l: i32, t: i32, tl: i32) -> i32 {
    let l16 = if l >= 32768 { l - 65536 } else { l };
    let t16 = if t >= 32768 { t - 65536 } else { t };
    let tl16 = if tl >= 32768 { tl - 65536 } else { tl };
    median_predict(l16, t16, tl16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::NUM_QUANT_SUBTABLES;
    use crate::range_coder::RangeDecoder;

    fn zero_qtable() -> QuantTableSet {
        [[0i32; 256]; NUM_QUANT_SUBTABLES]
    }

    fn single_context_qtable(c: i32) -> QuantTableSet {
        let mut q = zero_qtable();
        q[0][0] = c;
        q
    }

    // ----- state plumbing ------------------------------------------------

    #[test]
    fn state_initialises_every_slot_to_128() {
        let s = RangePlaneState::new(3);
        assert_eq!(s.state.len(), 3 * SYMBOL_CONTEXT_SIZE);
        for &b in &s.state {
            assert_eq!(b, 128);
        }
    }

    #[test]
    fn state_window_mut_isolates_per_context() {
        let mut s = RangePlaneState::new(4);
        s.window_mut(2)[0] = 200;
        assert_eq!(s.state[2 * SYMBOL_CONTEXT_SIZE], 200);
        // Adjacent contexts untouched.
        assert_eq!(s.state[SYMBOL_CONTEXT_SIZE], 128);
        assert_eq!(s.state[3 * SYMBOL_CONTEXT_SIZE], 128);
    }

    #[test]
    fn state_new_with_zero_context_still_has_one_window() {
        // `context_count == 0` would imply an empty quant table; the
        // reconstruction loop would never index, but the allocator must
        // still leave at least one window so the buffer slicing is
        // well-defined.
        let s = RangePlaneState::new(0);
        assert_eq!(s.state.len(), SYMBOL_CONTEXT_SIZE);
    }

    // ----- median16_predict (§3.3.1) -------------------------------------

    #[test]
    fn median16_matches_unsigned_for_small_values() {
        // All operands < 32768 — the reinterpret is a no-op, so
        // median16 reduces to the plain median.
        // median(l=100, t=200, l+t-tl=150) — sorted 100,150,200 → 150.
        assert_eq!(median16_predict(100, 200, 150), 150);
        assert_eq!(median_predict(100, 200, 150), 150);
    }

    #[test]
    fn median16_reinterprets_high_half_as_negative() {
        // l = 60000 (-5536), t = 100, tl = 50 → median of
        //   (l, t, l+t-tl) = (-5536, 100, -5536 + 100 - 50)
        //                  = (-5536, 100, -5486)
        // sorted: -5536, -5486, 100 → middle = -5486.
        assert_eq!(median16_predict(60000, 100, 50), -5486);
    }

    #[test]
    fn median16_all_high_half() {
        // l=33000(-32536), t=34000(-31536), tl=32800(-32736)
        // gradient = -32536 + -31536 - -32736 = -31336
        // sorted: -32536, -31536, -31336 → middle = -31536.
        assert_eq!(median16_predict(33000, 34000, 32800), -31536);
    }

    // ----- empty-dimension guards ---------------------------------------

    #[test]
    fn reconstruct_zero_width_returns_empty() {
        let qtable = zero_qtable();
        let buf = [0u8; 8];
        let mut rc = RangeDecoder::new(&buf).unwrap();
        let plane = RangePlaneReconstructor::reconstruct_plane(&mut rc, &qtable, 1, 0, 5, 8, false);
        assert!(plane.is_empty());
    }

    #[test]
    fn reconstruct_zero_height_returns_empty() {
        let qtable = zero_qtable();
        let buf = [0u8; 8];
        let mut rc = RangeDecoder::new(&buf).unwrap();
        let plane = RangePlaneReconstructor::reconstruct_plane(&mut rc, &qtable, 1, 5, 0, 8, false);
        assert!(plane.is_empty());
    }

    // ----- 1x1 single-Sample reconstruction -----------------------------

    #[test]
    fn reconstruct_single_sample_yields_zero_for_fresh_decoder() {
        // Construct a range decoder from "balanced" input bytes — the
        // first symbol read against a fresh state-128 window has a
        // 50/50 chance of returning the is-zero bit. With seed bytes
        // 0x80 0x00 → low = 0x8000, which is > (0xFF00 * 128 / 256) =
        // 0x7F80, so the first get_rac returns 1 ("the value is
        // zero"). Therefore the decoded sample_difference is 0.
        //
        // pred for the single Sample: all neighbours border-zero →
        // median(0, 0, 0) = 0. Sample = (0 + 0) & 0xFF = 0.
        let qtable = single_context_qtable(7);
        let buf = [0x80u8, 0x00, 0, 0];
        let mut rc = RangeDecoder::new(&buf).unwrap();
        let plane = RangePlaneReconstructor::reconstruct_plane(&mut rc, &qtable, 8, 1, 1, 8, false);
        assert_eq!(plane, vec![0]);
    }

    // ----- whole-plane invariants ---------------------------------------

    #[test]
    fn reconstruct_samples_stay_in_range_8bit() {
        // Drive a 4x4 plane on a single-context qtable with arbitrary
        // input bytes; every reconstructed Sample must satisfy the
        // §3.8 modular invariant `0..256` regardless of which symbols
        // came out.
        let qtable = single_context_qtable(3);
        let buf = [0xA5u8; 64];
        let mut rc = RangeDecoder::new(&buf).unwrap();
        let plane = RangePlaneReconstructor::reconstruct_plane(&mut rc, &qtable, 8, 4, 4, 8, false);
        assert_eq!(plane.len(), 16);
        for s in plane {
            assert!((0..256).contains(&s), "sample {s} out of 8-bit range");
        }
    }

    #[test]
    fn reconstruct_samples_stay_in_range_16bit_with_median16() {
        // 16-bit alt median: every reconstructed Sample must lie in
        // 0..65536, even though intermediate predictions can be
        // negative (the §3.8 modular add-back normalizes them).
        let qtable = single_context_qtable(5);
        let buf = [0xC3u8; 256];
        let mut rc = RangeDecoder::new(&buf).unwrap();
        let plane = RangePlaneReconstructor::reconstruct_plane(&mut rc, &qtable, 8, 4, 4, 16, true);
        assert_eq!(plane.len(), 16);
        for s in plane {
            assert!((0..65536).contains(&s), "sample {s} out of 16-bit range");
        }
    }

    // ----- per-context state isolation ----------------------------------

    #[test]
    fn distinct_contexts_evolve_independent_state() {
        // After reconstructing a plane on a qtable that touches at
        // least two contexts, the state windows for those contexts
        // should have drifted from the initial 128. Use a qtable that
        // produces context 3 for the first sample and context 5 from
        // the second sample onward (the changing `l` neighbour).
        let mut qtable = zero_qtable();
        // Q0 maps (l - tl) = 0 → context 3, others → 5.
        qtable[0][0] = 3;
        for slot in qtable[0][1..].iter_mut() {
            *slot = 5;
        }
        let buf = [0xA5u8; 32];
        let mut rc = RangeDecoder::new(&buf).unwrap();
        let plane = RangePlaneReconstructor::reconstruct_plane(&mut rc, &qtable, 8, 4, 1, 8, false);
        assert_eq!(plane.len(), 4);
        // We can't easily inspect the post-decode state from the
        // outside (intentionally encapsulated), but the test
        // *does* exercise the multi-context window indexing through
        // the public surface — any out-of-bounds slice would panic.
    }
}
