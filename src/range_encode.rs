//! Per-plane pixel **encoder** for the range-coder slice path (RFC 9043
//! §3.1 / §3.3 / §3.5 / §3.7 / §3.8.1.2 / §4.7 / §4.8 / `coder_type ==
//! 1 || coder_type == 2`).
//!
//! Symmetric inverse of [`crate::range_reconstruct::RangePlaneReconstructor`]:
//! given a row-major Plane of reconstructed Samples, it emits the
//! per-Sample range-coded `sample_difference` stream a matching
//! [`RangePlaneReconstructor::reconstruct_plane`] call decodes back to
//! the same pixels.
//!
//! ## What this module does (mirror table)
//!
//! | RFC §            | Decoder side                                                              | Encoder side (this module)                                                  |
//! | ---------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
//! | §3.1             | Border buffers: zero north + zero west; right-edge mirror                 | Same border buffers, same right-edge mirror                                |
//! | §3.3             | `pred = median(l, t, tl)` from **reconstructed** neighbours              | `pred = median(l, t, tl)` from the **target** Sample's neighbours (same)   |
//! | §3.3.1           | `use_16bit_median` swaps to `median16` for 16-bit YCbCr range-coded path | Same flag, same swap                                                       |
//! | §3.5             | `absolute_context(q, neighbours)` → index + sign-flip                     | Same call (the encoder reads the same neighbour values)                    |
//! | §3.8.1.2         | `raw = get_sr(rc, state_window)`; `diff = sign_flip ? -raw : raw`         | `raw_delta = sample - pred`, normalised; `put_sr(re, state_window, raw)`   |
//! | §3.8             | `Sample = (pred + diff) mod 2^bits`                                       | `diff = (sample - pred) mod 2^bits`, sign-corrected                        |
//! | §3.8.1.3         | `state[..] = 128` at Slice start (`keyframe == 1`)                       | Same initialisation                                                        |
//!
//! ## Why this isn't a one-line shim
//!
//! Three details that bite if dropped:
//!
//! 1. **Neighbour parity.** The decoder reads neighbours from the
//!    *reconstructed* current row prefix (`cur[..idx]`). The encoder
//!    must write Samples into that same prefix slot as soon as it
//!    derives a `diff`, so the next column's `l` / `tl` neighbour reads
//!    return the post-add-back value rather than the original input
//!    Sample. Without this, the encoded `diff` for column `x+1` is
//!    computed against an out-of-sync `l` and the decoder's
//!    reconstruction diverges.
//! 2. **Sign-flip direction.** §3.5 says the *decoder* flips the sign
//!    of the decoded `sample_difference` when the raw context is
//!    negative. The encoder therefore must flip BEFORE entropy coding
//!    so the decoder's flip-back arrives at the right `diff`. The
//!    `raw = sign_flip ? -diff : diff` line below is the symmetric
//!    inverse of `diff = sign_flip ? -raw : raw` in the decoder.
//! 3. **Modular normalisation.** `(sample - pred)` is unbounded in
//!    `i32`, but the decoder reads a §3.8 modular `diff` that the
//!    arithmetic coder represents in `[-2^(bits-1), 2^(bits-1))`.
//!    The `normalise_diff` helper folds the raw difference into that
//!    half-modulus range so the decoder's `reconstruct_sample(pred,
//!    diff, bits)` call yields exactly the input Sample.
//!
//! ## Scope (round 164)
//!
//! Single-Plane encoder for **`coder_type == 1`** only. The Slice-level
//! pipeline (keyframe bit + range-coded SliceHeader on the same encoder
//! cursor, no byte-alignment step between header and content) lives in
//! [`crate::frame_encode`]. The `coder_type == 2` (arithmetic-table
//! variant) path swaps the per-bit transition table but reuses the same
//! per-Sample encode loop; the table swap is plumbed by the caller via
//! [`RangeEncoder::with_one_state`].
//!
//! Run mode is intentionally absent: RFC 9043 §3.8.2.2 ("Run Mode")
//! lives strictly under §3.8.2 Golomb-Rice and does not apply here.

use crate::predictor::{absolute_context, median_predict, NeighborSamples, QuantTableSet};
use crate::range_coder::{RangeEncoder, PARAMETERS_INITIAL_STATE};
use crate::range_reconstruct::median16_predict_pub;
use crate::reconstruct::{BORDER_LEFT, BORDER_RIGHT};
use crate::symbol::{put_sr_window, SYMBOL_CONTEXT_SIZE};

/// One 32-slot state window per §3.5 absolute context (RFC 9043
/// §3.8.1.2 + §3.8.1.3) — encoder-side mirror of
/// [`crate::range_reconstruct::RangePlaneState`].
///
/// Each call to [`put_sr`] reads + writes the same 32 slots the matching
/// decoder-side [`crate::symbol::get_sr`] call would touch, so the
/// encoder's state evolves byte-for-byte alongside the decoder's. Stored
/// as a flat `context_count * 32` byte buffer.
#[derive(Debug, Clone)]
pub(crate) struct RangePlaneEncoderState {
    state: Vec<u8>,
}

impl RangePlaneEncoderState {
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
    /// array (no per-slot bounds checks inside the symbol encoder —
    /// mirror of the decode-side `RangePlaneState::window_mut`).
    #[inline]
    fn window_mut(&mut self, c: usize) -> &mut [u8; SYMBOL_CONTEXT_SIZE] {
        let lo = c * SYMBOL_CONTEXT_SIZE;
        (&mut self.state[lo..lo + SYMBOL_CONTEXT_SIZE])
            .try_into()
            .expect("sliced to SYMBOL_CONTEXT_SIZE")
    }
}

/// Per-plane pixel encoder for the range-coder slice path.
///
/// Encodes `height` Lines of `width` Samples each, emitting one
/// signed-Golomb-of-the-arithmetic-coder symbol per Sample into `re`.
/// Mirrors the per-Plane state lifecycle of
/// [`RangePlaneReconstructor::reconstruct_plane`]:
///
/// * Border buffers initialised to zero (§3.1).
/// * One 32-slot state window per §3.5 absolute context, all 128 at
///   the start of the call (§3.8.1.3 — every Slice in this driver is a
///   keyframe).
/// * Median predictor (§3.3) over the current row's already-encoded
///   prefix and the previous row, with the §3.3.1 alternate predictor
///   gated by `use_16bit_median`.
///
/// # Parameters mirror the decoder exactly
///
/// * `re` — the slice's range encoder, already positioned at the start
///   of this Plane's `sample_difference` stream. For YCbCr (Plane-then-
///   Line traversal) the same encoder is reused across all Planes so the
///   range coder's arithmetic state carries forward.
/// * `qtable` — the §3.4 Quantization Table Set this Plane selects via
///   `quant_table_set_index`.
/// * `context_count` — sizes the per-context state buffer (§4.1.2).
/// * `samples` — row-major Plane of reconstructed Samples to encode;
///   length must equal `width * height`, each Sample in `0 .. 2^bits`.
/// * `width` / `height` — `plane_pixel_width` / `plane_pixel_height`
///   (§4.7.2 / §4.8.1).
/// * `bits` — `bits_per_raw_sample` (or `+1` on the JPEG 2000 RCT
///   path).
/// * `use_16bit_median` — selects the §3.3.1 alternate predictor when
///   the caller's gate (`colorspace_type == 0 && bits == 16 &&
///   coder_type == 1|2`) is true.
#[derive(Debug)]
pub struct RangePlaneEncoder;

impl RangePlaneEncoder {
    /// Encode + emit a full Plane onto `re`. See the type-level docs.
    ///
    /// Allocates a fresh per-context state buffer (suitable when no
    /// other Plane in this Slice selects the same Quantization Table
    /// Set). For multi-Plane Slices where two or more Planes share a
    /// `quant_table_set_index` (Cb + Cr on every `chroma_planes ==
    /// true` Slice) route through
    /// [`Self::encode_plane_with_state`] and share one
    /// [`RangePlaneEncoderState`] across all Planes that select the
    /// set, exactly mirroring the decoder
    /// ([`crate::range_reconstruct::RangePlaneReconstructor::reconstruct_plane_with_state`]).
    ///
    /// # Panics
    ///
    /// Debug-asserts `samples.len() == width * height`; behaviour is
    /// unspecified if violated in release builds (the over- or under-
    /// read would simply truncate the encoded Plane).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_plane(
        re: &mut RangeEncoder,
        qtable: &QuantTableSet,
        context_count: usize,
        samples: &[i32],
        width: usize,
        height: usize,
        bits: u32,
        use_16bit_median: bool,
    ) {
        let mut state = RangePlaneEncoderState::new(context_count);
        Self::encode_plane_with_state(
            re,
            &mut state,
            qtable,
            samples,
            width,
            height,
            bits,
            use_16bit_median,
        );
    }

    /// Encode + emit a full Plane against a caller-supplied per-context
    /// state buffer. Symmetric inverse of
    /// [`crate::range_reconstruct::RangePlaneReconstructor::reconstruct_plane_with_state`].
    ///
    /// The caller owns the [`RangePlaneEncoderState`] lifecycle and
    /// must hand the same `&mut state` to every per-Plane call that
    /// selects the same Quantization Table Set (RFC 9043 §3.6 /
    /// §4.6.6 / §3.8.1.3 / §4.2 Figure 28 — see the decoder doc).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_plane_with_state(
        re: &mut RangeEncoder,
        state: &mut RangePlaneEncoderState,
        qtable: &QuantTableSet,
        samples: &[i32],
        width: usize,
        height: usize,
        bits: u32,
        use_16bit_median: bool,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        debug_assert_eq!(
            samples.len(),
            width.saturating_mul(height),
            "samples.len() must equal width*height for the range-coder Plane encoder",
        );

        // §3.1 border buffers, byte-for-byte symmetric with
        // `RangePlaneReconstructor::reconstruct_plane_with_state`.
        let stride = BORDER_LEFT + width + BORDER_RIGHT;
        let mut prev_prev = vec![0i32; stride];
        let mut prev = vec![0i32; stride];
        let mut cur = vec![0i32; stride];

        for y in 0..height {
            // §3.1 left-of-slice column: sample[y][-1] =
            // sample[y-1][0]. Same border seed as the decoder before
            // its row loop opens.
            cur[0] = 0;
            cur[BORDER_LEFT - 1] = prev[BORDER_LEFT];

            let row_start = y * width;
            let row_samples = &samples[row_start..row_start + width];

            Self::encode_row(
                re,
                state,
                qtable,
                &prev,
                &prev_prev,
                &mut cur,
                row_samples,
                width,
                bits,
                use_16bit_median,
            );

            // §3.1 right-border mirror, identical to the decoder.
            cur[BORDER_LEFT + width] = cur[BORDER_LEFT + width - 1];

            core::mem::swap(&mut prev_prev, &mut prev);
            core::mem::swap(&mut prev, &mut cur);
            // Zero out the next row's working buffer so border cells
            // outside the [BORDER_LEFT, BORDER_LEFT + width] slice
            // start clean.
            cur.iter_mut().for_each(|s| *s = 0);
        }
    }

    /// Encode one Row of `row_samples` into `re`, populating
    /// `cur[BORDER_LEFT .. BORDER_LEFT + width]` with the *Sample*
    /// values as it goes (decoder-symmetric: the next column's
    /// `l` / `tl` neighbour reads see the post-reconstruction value,
    /// not the raw input).
    ///
    /// `pub(crate)` so the §3.7.2 RGB line-major encoder driver can
    /// step a single Line per Plane while keeping per-Plane state +
    /// border buffers alive across the interleave (mirrors the
    /// decoder's `RangePlaneReconstructor::reconstruct_row` exposure).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_row(
        re: &mut RangeEncoder,
        state: &mut RangePlaneEncoderState,
        qtable: &QuantTableSet,
        prev: &[i32],
        prev_prev: &[i32],
        cur: &mut [i32],
        row_samples: &[i32],
        width: usize,
        bits: u32,
        use_16bit_median: bool,
    ) {
        debug_assert_eq!(row_samples.len(), width);

        let mask = if bits >= 32 {
            !0i32
        } else {
            (1i32 << bits) - 1
        };
        let half = if bits == 0 { 1 } else { 1i32 << (bits - 1) };
        let modulus = if bits >= 31 {
            // No safe i32 modulus; the spec never asks for
            // `bits >= 31` on this path (max is 16), but guard anyway.
            i32::MAX
        } else {
            1i32 << bits
        };

        // Neighbour carry (r386): mirror of the decoder's
        // `reconstruct_row` — `l` / `ll` carry this row's two most
        // recent reconstructed Samples, `tl` / `t` slide along the row
        // above; bit-identical to re-reading the stencil cells.
        let mut ll = cur[BORDER_LEFT - 2];
        let mut l = cur[BORDER_LEFT - 1];
        let mut tl = prev[BORDER_LEFT - 1];
        let mut t = prev[BORDER_LEFT];
        for (x, &sample) in row_samples.iter().enumerate().take(width) {
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

            let pred = if use_16bit_median {
                median16_predict_pub(n.l, n.t, n.tl)
            } else {
                median_predict(n.l, n.t, n.tl)
            };

            // §3.8 modular: the decoder reconstructs
            //   Sample = (pred + diff) mod 2^bits
            // so we need a `diff` such that
            //   (pred + diff) mod 2^bits == sample
            // i.e. `diff ≡ (sample - pred) (mod 2^bits)`.
            // Normalise into the signed half-modulus
            // `[-2^(bits-1), 2^(bits-1))` so the §3.8.1.2 `get_sr`
            // path round-trips bit-exactly. The decoder's
            // `reconstruct_sample` masks with `(1 << bits) - 1`, so
            // any representative of the residue class works; we pick
            // the canonical signed one to keep magnitudes minimal and
            // therefore the per-Sample bit cost minimal.
            let diff = normalise_diff(sample - pred, half, modulus);

            // §3.5 sign-flip is performed by the DECODER post-decode;
            // to keep the round-trip bit-exact we invert it here so the
            // decoder's flip-back recovers our `diff`.
            let raw = if abs_ctx.sign_flip { -diff } else { diff };

            put_sr_window(re, state.window_mut(abs_ctx.index as usize), raw);

            // Write the reconstructed Sample (NOT the original input!)
            // into `cur[idx]` so the next column's neighbour reads
            // are decoder-parity. Even though by construction
            // `(pred + diff) & mask == row_samples[x] & mask`, going
            // through the mask is the canonical path the decoder
            // walks; we mirror it byte-for-byte.
            let reconstructed = (pred.wrapping_add(diff)) & mask;
            cur[idx] = reconstructed;
            ll = l;
            l = reconstructed;
            tl = t;
            t = tr;
        }
    }
}

/// Fold `raw` into the signed half-modulus `[-half, half)` for the §3.8
/// modular `sample_difference` representation. Pure i32 arithmetic; no
/// overflow path for `bits <= 30`.
#[inline]
fn normalise_diff(raw: i32, half: i32, modulus: i32) -> i32 {
    let mut diff = raw.rem_euclid(modulus);
    if diff >= half {
        diff -= modulus;
    }
    diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::NUM_QUANT_SUBTABLES;
    use crate::range_coder::{RangeDecoder, RangeEncoder};
    use crate::range_reconstruct::RangePlaneReconstructor;

    fn zero_qtable() -> QuantTableSet {
        [[0i32; 256]; NUM_QUANT_SUBTABLES]
    }

    fn single_context_qtable(c: i32) -> QuantTableSet {
        let mut q = zero_qtable();
        q[0][0] = c;
        q
    }

    // ----- state plumbing ---------------------------------------------

    #[test]
    fn encoder_state_initialises_to_128() {
        let s = RangePlaneEncoderState::new(3);
        assert_eq!(s.state.len(), 3 * SYMBOL_CONTEXT_SIZE);
        for &b in &s.state {
            assert_eq!(b, 128);
        }
    }

    #[test]
    fn encoder_state_window_isolated() {
        let mut s = RangePlaneEncoderState::new(4);
        s.window_mut(2)[0] = 99;
        assert_eq!(s.state[2 * SYMBOL_CONTEXT_SIZE], 99);
        assert_eq!(s.state[SYMBOL_CONTEXT_SIZE], 128);
    }

    #[test]
    fn encoder_state_zero_context_still_has_one_window() {
        let s = RangePlaneEncoderState::new(0);
        assert_eq!(s.state.len(), SYMBOL_CONTEXT_SIZE);
    }

    // ----- normalise_diff invariants ----------------------------------

    #[test]
    fn normalise_diff_8bit_folds_into_signed_half() {
        // 8-bit: half=128, modulus=256. Range is `[-128, 128)`.
        let half = 128i32;
        let modulus = 256i32;
        assert_eq!(normalise_diff(0, half, modulus), 0);
        assert_eq!(normalise_diff(1, half, modulus), 1);
        assert_eq!(normalise_diff(127, half, modulus), 127);
        // 128 wraps to -128.
        assert_eq!(normalise_diff(128, half, modulus), -128);
        // 255 wraps to -1.
        assert_eq!(normalise_diff(255, half, modulus), -1);
        // 256 ≡ 0.
        assert_eq!(normalise_diff(256, half, modulus), 0);
        // -1 ≡ 255 ≡ -1 (rem_euclid normalises negatives upward first).
        assert_eq!(normalise_diff(-1, half, modulus), -1);
        // -128 stays -128.
        assert_eq!(normalise_diff(-128, half, modulus), -128);
        // -129 ≡ 127.
        assert_eq!(normalise_diff(-129, half, modulus), 127);
    }

    // ----- empty Plane guards -----------------------------------------

    #[test]
    fn encode_zero_dimension_is_noop() {
        let mut re = RangeEncoder::new();
        let qt = zero_qtable();
        RangePlaneEncoder::encode_plane(&mut re, &qt, 1, &[], 0, 5, 8, false);
        RangePlaneEncoder::encode_plane(&mut re, &qt, 1, &[], 5, 0, 8, false);
        // No symbols pushed; `finish` still returns a valid (short)
        // range-coded prefix because `RangeEncoder::finish` always
        // flushes the renormalisation state.
        let bytes = re.finish();
        // Two-byte sentinel from `RangeEncoder::finish`; specific
        // length isn't load-bearing — just confirm it didn't panic.
        assert!(!bytes.is_empty());
    }

    // ----- round-trip: 1x1 single Plane (no neighbours) ---------------

    fn round_trip_plane_8bit(width: usize, height: usize, samples: &[i32], context_constant: i32) {
        let qt = single_context_qtable(context_constant);
        let context_count = (context_constant.unsigned_abs() as usize) + 1;

        let mut re = RangeEncoder::new();
        RangePlaneEncoder::encode_plane(
            &mut re,
            &qt,
            context_count,
            samples,
            width,
            height,
            8,
            false,
        );
        let bytes = re.finish();

        let mut rd = RangeDecoder::new(&bytes).expect("range decoder must accept encoder output");
        let decoded = RangePlaneReconstructor::reconstruct_plane(
            &mut rd,
            &qt,
            context_count,
            width,
            height,
            8,
            false,
        );
        assert_eq!(
            decoded, samples,
            "Plane round trip mismatch ({}x{})",
            width, height
        );
    }

    #[test]
    fn round_trip_1x1_zero_sample() {
        round_trip_plane_8bit(1, 1, &[0], 7);
    }

    #[test]
    fn round_trip_1x1_mid_value() {
        round_trip_plane_8bit(1, 1, &[128], 5);
    }

    #[test]
    fn round_trip_1x1_max_value() {
        round_trip_plane_8bit(1, 1, &[255], 3);
    }

    #[test]
    fn round_trip_2x1_gradient_constant_context() {
        // With a single-context Q table the decoder's per-Sample state
        // walks one common 32-slot window; the encoder feeds the same
        // window, so the round trip must be bit-exact regardless of
        // how the arithmetic coder splits the byte stream.
        round_trip_plane_8bit(2, 1, &[10, 200], 11);
    }

    #[test]
    fn round_trip_3x3_arbitrary() {
        round_trip_plane_8bit(3, 3, &[10, 30, 100, 200, 5, 80, 254, 0, 128], 13);
    }

    #[test]
    fn round_trip_4x4_max_magnitude_sweep() {
        // Sample values spanning the full 0..256 range to exercise the
        // negative-magnitude path through `normalise_diff` + the §3.5
        // sign-flip + the §3.8 modular add-back.
        let samples: Vec<i32> = (0..16).map(|i| (i * 17) & 0xFF).collect();
        round_trip_plane_8bit(4, 4, &samples, 9);
    }

    // ----- round-trip: 10-bit Plane -----------------------------------

    #[test]
    fn round_trip_10bit_plane() {
        let qt = single_context_qtable(6);
        let samples: Vec<i32> = vec![0, 511, 1023, 256, 800, 100, 900, 50, 1000, 1, 512, 300];

        let mut re = RangeEncoder::new();
        RangePlaneEncoder::encode_plane(&mut re, &qt, 7, &samples, 4, 3, 10, false);
        let bytes = re.finish();

        let mut rd = RangeDecoder::new(&bytes).unwrap();
        let decoded = RangePlaneReconstructor::reconstruct_plane(&mut rd, &qt, 7, 4, 3, 10, false);
        assert_eq!(decoded, samples);
    }

    // ----- round-trip: 16-bit Plane with alternate median --------------

    #[test]
    fn round_trip_16bit_alt_median_plane() {
        let qt = single_context_qtable(5);
        // Sample values that stress the §3.3.1 alt-median's high-half
        // reinterpret: values in both `< 32768` (normal) and `>= 32768`
        // (high-half) regions.
        let samples: Vec<i32> = vec![
            10, 60000, 5, 33000, 32767, 32768, 65535, 0, 100, 50000, 200, 1,
        ];

        let mut re = RangeEncoder::new();
        RangePlaneEncoder::encode_plane(&mut re, &qt, 6, &samples, 4, 3, 16, true);
        let bytes = re.finish();

        let mut rd = RangeDecoder::new(&bytes).unwrap();
        let decoded = RangePlaneReconstructor::reconstruct_plane(&mut rd, &qt, 6, 4, 3, 16, true);
        assert_eq!(decoded, samples);
    }

    // ----- multi-plane: encoder/decoder cursor shared across planes ---

    #[test]
    fn round_trip_two_planes_on_one_encoder() {
        // Mimics the YCbCr (Plane-then-Line) traversal: encode two
        // Planes back-to-back on a single encoder, then decode both
        // back-to-back on a single decoder. The per-Plane state is
        // freshly initialised inside each `encode_plane` call so the
        // two Planes' entropy state is independent (matching the
        // decoder's per-Plane state lifecycle).
        let qt = single_context_qtable(8);
        let plane_a: Vec<i32> = vec![10, 20, 30, 40];
        let plane_b: Vec<i32> = vec![240, 5, 128, 64];

        let mut re = RangeEncoder::new();
        RangePlaneEncoder::encode_plane(&mut re, &qt, 9, &plane_a, 2, 2, 8, false);
        RangePlaneEncoder::encode_plane(&mut re, &qt, 9, &plane_b, 2, 2, 8, false);
        let bytes = re.finish();

        let mut rd = RangeDecoder::new(&bytes).unwrap();
        let decoded_a = RangePlaneReconstructor::reconstruct_plane(&mut rd, &qt, 9, 2, 2, 8, false);
        let decoded_b = RangePlaneReconstructor::reconstruct_plane(&mut rd, &qt, 9, 2, 2, 8, false);
        assert_eq!(decoded_a, plane_a);
        assert_eq!(decoded_b, plane_b);
    }

    // ----- determinism ------------------------------------------------

    #[test]
    fn encoder_is_deterministic() {
        let qt = single_context_qtable(4);
        let samples: Vec<i32> = (0..20).map(|i| (i * 13 + 7) & 0xFF).collect();

        let mut re1 = RangeEncoder::new();
        RangePlaneEncoder::encode_plane(&mut re1, &qt, 5, &samples, 5, 4, 8, false);
        let b1 = re1.finish();

        let mut re2 = RangeEncoder::new();
        RangePlaneEncoder::encode_plane(&mut re2, &qt, 5, &samples, 5, 4, 8, false);
        let b2 = re2.finish();

        assert_eq!(b1, b2);
    }

    // ----- single-cell extra-context plane round-trip -----------------

    #[test]
    fn round_trip_multi_context_qtable() {
        // A qtable that activates more than one context across the
        // Plane (so the encoder's per-context state windows actually
        // diverge) — verifies the per-context window indexing matches
        // the decoder's.
        let mut qt = zero_qtable();
        qt[0][0] = 3;
        for slot in qt[0][1..].iter_mut() {
            *slot = 5;
        }
        let samples: Vec<i32> = vec![10, 11, 12, 13, 14, 15];

        let mut re = RangeEncoder::new();
        RangePlaneEncoder::encode_plane(&mut re, &qt, 6, &samples, 3, 2, 8, false);
        let bytes = re.finish();

        let mut rd = RangeDecoder::new(&bytes).unwrap();
        let decoded = RangePlaneReconstructor::reconstruct_plane(&mut rd, &qt, 6, 3, 2, 8, false);
        assert_eq!(decoded, samples);
    }
}
