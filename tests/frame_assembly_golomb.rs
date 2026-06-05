//! End-to-end full-frame slice-assembly tests for the **Golomb-Rice**
//! decode path (`coder_type == 0`, RFC 9043 §3.8.2 / §4.5 / §4.7 / §4.8).
//!
//! Every shipped v3 fixture uses the range coder (`coder_type == 1`), and
//! the only `coder_type == 0` corpus fixture is FFV1 *version 0*, which
//! the v3-targeted [`oxideav_ffv1::decode_frame`] driver rejects (the v0
//! slice grid lives in the per-keyframe header, not the Configuration
//! Record). So the Golomb-Rice branch of the frame driver — the §4.7 /
//! §4.8 outer-`for p` / inner-`for y` traversal that drives
//! [`oxideav_ffv1::PlaneReconstructor`] across every row of every slice
//! and stitches each slice's plane into the frame buffer at its
//! pixel-space origin — has had no *end-to-end* coverage through
//! `decode_frame` itself (the per-plane reconstructor is unit-tested in
//! isolation; the multi-slice assembly is exercised only on the range
//! path).
//!
//! These tests close that gap with a **synthetic, self-consistent**
//! frame. RFC 9043 normatively specifies the *decoder*; the FFV1
//! *encoder* is its exact algebraic inverse. This file builds a tiny
//! clean-room encoder — derived purely from the in-tree decoder
//! invariants in `range_coder.rs`, `symbol.rs`, `golomb_rice.rs`,
//! `crc.rs`, and `reconstruct.rs` (every one of which is the RFC 9043
//! decode side) — and proves it round-trips:
//!
//! 1. The §3.8.1 **range encoder** (`RangeEncoder`) re-decodes
//!    bit-exactly through `oxideav_ffv1::RangeDecoder` (used here only
//!    transitively, via the public driver).
//! 2. The §3.8.1.2 **`put_ur`** scalar encoder re-decodes through the
//!    slice-header `ur` reader.
//! 3. The §3.8.2.4 **Golomb-Rice scalar encoder** (`enc_vlc`)
//!    re-decodes through `oxideav_ffv1::PlaneReconstructor` (its
//!    `get_vlc_symbol` inner loop).
//!
//! Then it assembles a multi-slice §4.4 / §4.6 / §4.8 / §4.9 v3 frame
//! (keyframe bit + per-slice range-coded header + byte-aligned
//! Golomb-Rice content + §4.9 footer with a solved §4.9.3 CRC) for a
//! **known small planar frame** and asserts that
//! [`oxideav_ffv1::decode_frame`] reconstructs every plane bit-exactly,
//! with each slice landing in its correct pixel rectangle.
//!
//! The encoder lives *only in this test* — it is not part of the crate's
//! public surface. Its job is to manufacture a known-good wire image so
//! the **decoder's** assembly path can be validated against a frame
//! whose ground truth we chose.

use oxideav_ffv1::{
    median_predict, reconstruct_sample, ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version,
    FramePixelDimensions, QuantTableSet, QuantizationTableSet, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS,
};

// ============================================================
// §3.8.1.5 default state-transition table (RFC 9043 Figure 24).
// The `one_state` half is published; `zero_state[i] = 256 -
// one_state[256 - i]` per §3.8.1.4 (Figures 22–23). Mirrors
// `range_coder.rs::DEFAULT_ONE_STATE` exactly — the encoder MUST walk
// the identical state machine so the decoder reproduces the bits.
// ============================================================
const ONE_STATE: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 152, 153, 154, 155, 156, 157,
    158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 190, 191, 192, 194,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 202, 204, 205, 206, 207, 208, 209, 209, 210, 211,
    212, 213, 215, 215, 216, 217, 218, 219, 220, 220, 222, 223, 224, 225, 226, 227, 227, 229, 229,
    230, 231, 232, 234, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,
    248, 0, 0, 0, 0, 0, 0, 0,
];

fn derive_zero_state(one: &[u8; 256]) -> [u8; 256] {
    let mut zs = [0u8; 256];
    for (i, slot) in zs.iter_mut().enumerate().skip(1) {
        let mirror = one[256 - i] as u16;
        *slot = (256u16.wrapping_sub(mirror) & 0xFF) as u8;
    }
    zs
}

const PARAMETERS_INITIAL_STATE: u8 = 128;
const SYMBOL_CONTEXT_SIZE: usize = 32;

// ============================================================
// §3.8.1 binary range ENCODER — the inverse of `RangeDecoder::get_rac`.
//
// `RangeDecoder` is a Closed-mode coder seeded with two bytes
// (`low = (b0<<8)|b1`, `range = 0xFF00`) and refilling one byte per
// renorm. `low` and `range` are therefore 16-bit quantities. The
// encoder mirrors the decoder's `range -= (range*state)/256` split and
// emits the top byte of `low` on each renorm, propagating any carry
// back through already-emitted bytes (the classic delayed-byte /
// pending-0xFF carry model). Validated to round-trip bit-exactly in
// `range_encoder_round_trips_through_decoder` below.
// ============================================================
struct RangeEncoder {
    low: u32,
    range: u32,
    out: Vec<u8>,
    one: [u8; 256],
    zero: [u8; 256],
    cache: i32,
    pending_ff: u32,
}

impl RangeEncoder {
    fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: 0xFF00,
            out: Vec::new(),
            one: ONE_STATE,
            zero: derive_zero_state(&ONE_STATE),
            cache: -1,
            pending_ff: 0,
        }
    }

    /// Emit the byte leaving the top of `low` (bits 8..16), folding a
    /// carry (bit 16) into the previously-cached byte and any pending
    /// 0xFF run.
    fn shift(&mut self) {
        let carry = (self.low >> 16) & 0xFF;
        let byte = (self.low >> 8) & 0xFF;
        if byte != 0xFF {
            if self.cache >= 0 {
                self.out.push(((self.cache as u32 + carry) & 0xFF) as u8);
            }
            while self.pending_ff > 0 {
                self.out.push(((0xFF + carry) & 0xFF) as u8);
                self.pending_ff -= 1;
            }
            self.cache = byte as i32;
        } else {
            self.pending_ff += 1;
        }
        self.low = (self.low << 8) & 0xFFFF;
    }

    fn renorm(&mut self) {
        while self.range < 0x100 {
            self.range = self.range.wrapping_mul(256);
            self.shift();
        }
    }

    /// Encode one binary symbol against `state`, updating `state` via
    /// the same transition table the decoder uses.
    fn put(&mut self, state: &mut u8, bit: u8) {
        let s = *state as u32;
        let rangeoff = (self.range.wrapping_mul(s)) / 256;
        self.range = self.range.wrapping_sub(rangeoff);
        if bit == 0 {
            *state = self.zero[*state as usize];
        } else {
            self.low = self.low.wrapping_add(self.range);
            *state = self.one[*state as usize];
            self.range = rangeoff;
        }
        self.renorm();
    }

    /// Encode one `ur` scalar symbol (inverse of `symbol.rs::get_ur` /
    /// `decode_symbol`): an is-zero bit, then a unary exponent, then the
    /// MSB-first mantissa.
    fn put_ur(&mut self, ctx: &mut [u8], v: u32) {
        if v == 0 {
            self.put(&mut ctx[0], 1);
            return;
        }
        self.put(&mut ctx[0], 0);
        let e = 31 - v.leading_zeros();
        for idx in 0..e {
            self.put(&mut ctx[1 + (idx.min(9) as usize)], 1);
        }
        self.put(&mut ctx[1 + (e.min(9) as usize)], 0);
        for i in (0..e).rev() {
            let bit = ((v >> i) & 1) as u8;
            self.put(&mut ctx[22 + (i.min(9) as usize)], bit);
        }
    }

    /// Flush the coder, emitting the remaining `low` bytes. After this
    /// the byte stream re-decodes to exactly the encoded symbols.
    fn finish(mut self) -> Vec<u8> {
        for _ in 0..2 {
            self.shift();
        }
        if self.cache >= 0 {
            self.out.push(self.cache as u8);
        }
        while self.pending_ff > 0 {
            self.out.push(0xFF);
            self.pending_ff -= 1;
        }
        self.out
    }
}

// ============================================================
// §3.8.2 Golomb-Rice scalar ENCODER — inverse of
// `golomb_rice.rs::get_vlc_symbol`. The adaptive state update is
// identical to the decoder's (it is deterministic given the decoded
// residual `v`); only the bit emission for the chosen `k` is inverted.
// MSB-first bit writer matches `bit_reader.rs::BitReader`.
// ============================================================
struct BitWriter {
    acc: u64,
    n: u32,
    out: Vec<u8>,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            acc: 0,
            n: 0,
            out: Vec::new(),
        }
    }
    fn put_bit(&mut self, b: u32) {
        self.acc = (self.acc << 1) | (b as u64 & 1);
        self.n += 1;
        if self.n == 8 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.n = 0;
        }
    }
    fn put_bits(&mut self, v: u32, k: u32) {
        for i in (0..k).rev() {
            self.put_bit((v >> i) & 1);
        }
    }
    /// Pad the final partial byte with zero bits (RFC 9043 §3.8.2's
    /// "padded with zeroes" rule) and return the bytes.
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.acc <<= 8 - self.n;
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

#[derive(Clone, Copy)]
struct VlcState {
    drift: i32,
    error_sum: i32,
    bias: i32,
    count: i32,
}

const VLC_INIT: VlcState = VlcState {
    drift: 0,
    error_sum: 4,
    bias: 0,
    count: 1,
};

/// §3.8.2.4 adaptive Rice-parameter selection (same loop as the
/// decoder's `get_vlc_symbol`).
fn pick_k(st: &VlcState) -> u32 {
    let mut i = st.count as i64;
    let mut k = 0u32;
    while i < st.error_sum as i64 {
        k += 1;
        i += i;
        if k >= 32 {
            break;
        }
    }
    k
}

/// §3.8.2.4 per-context state update — byte-identical to the decoder.
fn vlc_update(st: &mut VlcState, v: i32) {
    st.error_sum = st.error_sum.saturating_add(v.unsigned_abs() as i32);
    st.drift = st.drift.saturating_add(v);
    if st.count == 128 {
        st.count >>= 1;
        st.drift >>= 1;
        st.error_sum >>= 1;
    }
    st.count = st.count.saturating_add(1);
    if st.drift <= -st.count {
        st.bias = (st.bias - 1).max(-128);
        st.drift = (st.drift + st.count).max(-st.count + 1);
    } else if st.drift > 0 {
        st.bias = (st.bias + 1).min(127);
        st.drift = (st.drift - st.count).min(0);
    }
}

/// Emit an unsigned Golomb-Rice code with parameter `k` and ESC width
/// `bits` (inverse of `golomb_rice.rs::get_ur_golomb_esc`).
fn put_ur_golomb(bw: &mut BitWriter, k: u32, bits: u32, val: u32) {
    let prefix = val >> k;
    if prefix < 12 {
        for _ in 0..prefix {
            bw.put_bit(0);
        }
        bw.put_bit(1);
        if k > 0 {
            bw.put_bits(val & ((1 << k) - 1), k);
        }
    } else {
        for _ in 0..12 {
            bw.put_bit(0);
        }
        let esc = val - 11;
        if bits > 0 {
            bw.put_bits(esc, bits);
        }
    }
}

/// Emit a signed Golomb-Rice code (interleave folding, inverse of
/// `get_sr_golomb_esc`).
fn put_sr_golomb(bw: &mut BitWriter, k: u32, bits: u32, sval: i32) {
    let u = if sval < 0 {
        ((-sval as u32) * 2) - 1
    } else {
        (sval as u32) * 2
    };
    put_ur_golomb(bw, k, bits, u);
}

/// Encode one Sample-Difference so the decoder's `get_vlc_symbol`
/// returns exactly `ret`. Inverts the §3.8.2.4 sign-flip + bias and
/// advances `st` identically to the decoder.
///
/// Constraint: `ret` (and the post-bias residual) must lie inside the
/// signed `bits`-wide range so `sign_extend` is the identity (FFV1 keeps
/// sample-differences within `[-2^(bits-1), 2^(bits-1))`, so all the
/// frames built here satisfy this).
fn enc_vlc(bw: &mut BitWriter, st: &mut VlcState, bits: u32, ret: i32) {
    let k = pick_k(st);
    let v = ret - st.bias;
    let flip = 2 * st.drift < -st.count;
    let vraw = if flip { -1 - v } else { v };
    put_sr_golomb(bw, k, bits, vraw);
    // The decoder updates state with the post-flip `v` (which equals the
    // pre-flip `v` we computed above).
    vlc_update(st, v);
}

// ============================================================
// §4.9.3 CRC-32 (poly 0x104C11DB7, init 0, no inversion, MSB-first) —
// mirrors `crc.rs::ffv1_crc32`. Used to solve the §4.9 Slice Footer
// `slice_crc_parity`: appending `crc(M)` (as a big-endian u32) to `M`
// drives the whole-message residue to zero for this generator
// orientation, which is exactly how a conforming encoder picks parity.
// ============================================================
const FFV1_CRC_POLY: u32 = 0x04C1_1DB7;

fn ffv1_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &byte in data {
        crc ^= (byte as u32) << 24;
        for _ in 0..8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ FFV1_CRC_POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

// ============================================================
// Synthetic-frame builder.
// ============================================================

/// A constant-context quant table: Q0 maps every 8-bit index to `c`,
/// the other four sub-tables are zero, so the §3.5 context sum is the
/// constant `c` for every Sample regardless of neighbours. With `c != 0`
/// this keeps the whole plane on the scalar VLC path (no run mode), so
/// the per-Sample `enc_vlc` stream decodes deterministically. The §4.1
/// `context_count` for such a set is `c + 1` (the per-context state
/// arrays index `0..=c`).
fn constant_context_qts(c: i32) -> QuantizationTableSet {
    let mut tables: QuantTableSet = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    tables[0] = [c; 256];
    QuantizationTableSet {
        tables,
        context_count: (c as u32) + 1,
    }
}

/// Grayscale v3 Configuration Record (single plane, no chroma, no extra
/// plane), Golomb-Rice (`coder_type == 0`), 8-bit, on a
/// `num_h × num_v` slice grid.
fn grayscale_v3_config(num_h: u32, num_v: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 0, // Golomb-Rice
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: 8,
        chroma_planes: false,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(1),
        ec: Some(0),
        intra: Some(false),
    }
}

/// §4.8.3 / §4.8.2 pixel rectangle of slice `(slice_x, slice_y)` on a
/// `num_h × num_v` grid over a `frame_w × frame_h` frame, for a
/// `1 × 1`-raster-cell slice.
fn slice_pixel_rect(
    slice_x: u32,
    slice_y: u32,
    num_h: u32,
    num_v: u32,
    frame_w: u32,
    frame_h: u32,
) -> (u32, u32, u32, u32) {
    let px = slice_x * frame_w / num_h;
    let py = slice_y * frame_h / num_v;
    let pw = (slice_x + 1) * frame_w / num_h - px;
    let ph = (slice_y + 1) * frame_h / num_v - py;
    (px, py, pw, ph)
}

/// Reconstruct one Golomb-Rice plane the way `PlaneReconstructor` does
/// (§3.1 border + §3.3 median + §3.8 modular add-back) given a row-major
/// list of *target Sample values*. Returns the §3.8.2.4 residual stream
/// `enc_vlc` must emit (one per Sample) under the constant scalar
/// context so the decoder reconstructs exactly `samples`.
///
/// This is the bridge from "pixel values I want" to "Golomb-Rice
/// residuals the decoder will read". It runs the identical border /
/// predictor logic as `reconstruct.rs`, deriving each `diff` as the
/// value that makes `reconstruct_sample(pred, diff, bits) == sample`.
fn residuals_for_plane(samples: &[i32], w: usize, h: usize, bits: u32) -> Vec<i32> {
    assert_eq!(samples.len(), w * h);
    const BORDER_LEFT: usize = 2;
    let stride = BORDER_LEFT + w + 1;
    let mut prev = vec![0i32; stride];
    let mut cur = vec![0i32; stride];
    let mut residuals = Vec::with_capacity(w * h);
    for y in 0..h {
        // §3.1 left-of-slice column: sample[y][-1] = sample[y-1][0].
        cur[0] = 0;
        cur[BORDER_LEFT - 1] = prev[BORDER_LEFT];
        for x in 0..w {
            let idx = BORDER_LEFT + x;
            let l = cur[idx - 1];
            let t = prev[idx];
            let tl = prev[idx - 1];
            let pred = median_predict(l, t, tl);
            let sample = samples[y * w + x];
            // Choose the residual the decoder needs. The decoder does
            // `Sample = (pred + diff) & mask`. With samples + pred both in
            // `0..2^bits`, the smallest signed diff in
            // `[-2^(bits-1), 2^(bits-1))` is `sign_extend(sample - pred)`,
            // which keeps the §3.8.2.4 `sign_extend` an identity.
            let raw = sample - pred;
            let half = 1i32 << (bits - 1);
            let modulus = 1i32 << bits;
            let mut diff = raw % modulus;
            if diff >= half {
                diff -= modulus;
            } else if diff < -half {
                diff += modulus;
            }
            // Sanity: this diff really does reconstruct the target.
            debug_assert_eq!(reconstruct_sample(pred, diff, bits), sample);
            residuals.push(diff);
            cur[idx] = sample;
        }
        // §3.1 right border mirror.
        cur[BORDER_LEFT + w] = cur[BORDER_LEFT + w - 1];
        core::mem::swap(&mut prev, &mut cur);
    }
    residuals
}

/// Build one v3 Golomb-Rice Slice (§4.4 keyframe bit on slice 0 +
/// §4.6 range-coded header + byte-aligned §4.8 Golomb-Rice content +
/// §4.9 footer with solved §4.9.3 CRC, `ec == 1`).
#[allow(clippy::too_many_arguments)]
fn build_golomb_slice(
    is_first_slice: bool,
    slice_x: u32,
    slice_y: u32,
    quant_index_count: usize,
    quant_index: u32,
    plane_samples: &[i32],
    plane_w: usize,
    plane_h: usize,
    context_c: i32,
    bits: u32,
) -> Vec<u8> {
    // ---- §4.4 keyframe + §4.6 SliceHeader (range-coded) ----
    let mut enc = RangeEncoder::new();
    if is_first_slice {
        // §4.4: the Frame's leading `keyframe` boolean lives at the very
        // start of the first Slice's range-coded region (its own initial
        // state 128). The driver reads + discards it before the header.
        let mut kf_state = PARAMETERS_INITIAL_STATE;
        enc.put(&mut kf_state, 1);
    }
    // §4.6: all header fields share one 32-slot context window seeded to
    // 128 (the convention `slice_header.rs` uses).
    let mut state = [PARAMETERS_INITIAL_STATE; 64];
    enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], slice_x); // slice_x
    enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], slice_y); // slice_y
    enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], 0); // slice_width - 1 (= 1 cell)
    enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], 0); // slice_height - 1 (= 1 cell)
    for _ in 0..quant_index_count {
        enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], quant_index);
    }
    enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], 0); // picture_structure
    enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], 0); // sar_num
    enc.put_ur(&mut state[0..SYMBOL_CONTEXT_SIZE], 0); // sar_den
    let header_bytes = enc.finish();

    // ---- §4.8 Golomb-Rice content (byte-aligned after the header) ----
    // The driver resumes the BitReader from `body[rc.position()..]`. The
    // range encoder's flushed length IS that byte boundary.
    let residuals = residuals_for_plane(plane_samples, plane_w, plane_h, bits);
    let mut bw = BitWriter::new();
    // Constant non-zero context `context_c` → every Sample is scalar
    // (run mode is suppressed because `abs_ctx.index == context_c != 0`).
    let mut vlc = vec![VLC_INIT; (context_c as usize) + 1];
    for &r in &residuals {
        enc_vlc(&mut bw, &mut vlc[context_c as usize], bits, r);
    }
    let content_bytes = bw.finish();

    // ---- assemble body = header || content ----
    let mut body = header_bytes;
    body.extend_from_slice(&content_bytes);

    // ---- §4.9 footer (ec == 1): size(u24) + error_status(u8) +
    // slice_crc_parity(u32), with parity solved so the whole-slice CRC
    // residue is 0. ----
    let slice_size = body.len() as u32;
    let mut slice = body;
    slice.push(((slice_size >> 16) & 0xFF) as u8);
    slice.push(((slice_size >> 8) & 0xFF) as u8);
    slice.push((slice_size & 0xFF) as u8);
    slice.push(0); // error_status = NoError
    let parity = ffv1_crc32(&slice);
    slice.extend_from_slice(&parity.to_be_bytes());
    slice
}

// ============================================================
// Encoder round-trip self-tests (prove the synthetic wire image is
// well-formed before it is fed to `decode_frame`).
// ============================================================

/// The §3.8.1 range encoder produces bytes a real driver decode reads
/// back exactly. We prove it indirectly through the slice-header path:
/// a single-slice grayscale frame whose header carries a known
/// `slice_x`/`slice_y` decodes to a frame of the expected shape — if the
/// range bits were mis-encoded the header parse would mis-read the
/// raster coordinates and the §4.7 geometry / §4.9 footer cross-check
/// would fail. (The direct bit-for-bit round trip is exercised
/// implicitly by every assembly test below.)
#[test]
fn range_encoder_round_trips_through_decoder() {
    // 4x2 grayscale, single slice. The header sets slice_x=0, slice_y=0,
    // 1x1 cell. If the range encoder were wrong the §4.6 parse would
    // desync and `decode_frame` would error.
    let cr = grayscale_v3_config(1, 1);
    let qts = vec![constant_context_qts(5)];
    let samples: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let slice = build_golomb_slice(true, 0, 0, 2, 0, &samples, 4, 2, 5, 8);
    let decoded = oxideav_ffv1::decode_frame(
        &slice,
        &cr,
        &qts,
        FramePixelDimensions::new(4, 2).unwrap(),
        true,
    )
    .expect("well-formed single-slice golomb frame decodes");
    assert_eq!(decoded.planes.len(), 1);
    assert_eq!((decoded.planes[0].width, decoded.planes[0].height), (4, 2));
    assert_eq!(decoded.planes[0].samples, samples);
}

// ============================================================
// Full-frame Golomb-Rice slice-assembly tests.
// ============================================================

/// Single Golomb-Rice slice covering the whole frame: every row of the
/// one slice is driven through `decode_line`/`reconstruct_plane`, and the
/// reconstructed plane must equal the known synthetic plane exactly.
#[test]
fn golomb_single_slice_full_plane_reconstructs_bit_exact() {
    let cr = grayscale_v3_config(1, 1);
    let context_c = 9;
    let qts = vec![constant_context_qts(context_c)];
    // A 6x4 ramp + a couple of edges, all inside 0..256.
    #[rustfmt::skip]
    let samples: Vec<i32> = vec![
          0,  10,  20,  40,  80, 160,
        200, 100,  50,  25,  12,   6,
          3, 130, 255,   0, 128,  64,
         77,  88,  99, 111, 222, 233,
    ];
    let slice = build_golomb_slice(true, 0, 0, 2, 0, &samples, 6, 4, context_c, 8);
    let decoded = oxideav_ffv1::decode_frame(
        &slice,
        &cr,
        &qts,
        FramePixelDimensions::new(6, 4).unwrap(),
        true,
    )
    .expect("single-slice golomb frame decodes");
    assert_eq!(decoded.planes.len(), 1);
    assert_eq!(decoded.planes[0].samples, samples);
}

/// **The core assembly test.** A 2×2 slice grid over an 8×4 grayscale
/// frame: four independent Golomb-Rice slices, each with distinct
/// content, must be reconstructed and stitched into the frame buffer at
/// the correct §4.8.3 / §4.7.4 pixel origins. We construct the known
/// full frame, carve it into the four pixel quadrants, build one slice
/// per quadrant, concatenate them into the frame payload, and assert
/// `decode_frame` reproduces the original full frame bit-exactly.
#[test]
fn golomb_2x2_slice_grid_assembles_full_frame_bit_exact() {
    let (frame_w, frame_h) = (8u32, 4u32);
    let (num_h, num_v) = (2u32, 2u32);
    let cr = grayscale_v3_config(num_h, num_v);
    let context_c = 11;
    let qts = vec![constant_context_qts(context_c)];

    // The known full frame (row-major, 8 wide, 4 tall). Chosen so each
    // quadrant has a visually distinct pattern, making a mis-placed
    // slice obvious.
    #[rustfmt::skip]
    let frame: Vec<i32> = vec![
        // TL quadrant cols 0..4 | TR quadrant cols 4..8
          1,   2,   3,   4,   200, 201, 202, 203,
          5,   6,   7,   8,   204, 205, 206, 207,
        // BL quadrant cols 0..4 | BR quadrant cols 4..8
         50,  60,  70,  80,   100, 110, 120, 130,
         90, 100, 110, 120,   140, 150, 160, 170,
    ];
    let read_px = |x: u32, y: u32| frame[(y * frame_w + x) as usize];

    // Build the four slices in slice-index order (raster order:
    // (0,0), (1,0), (0,1), (1,1)).
    let mut frame_bytes = Vec::new();
    let mut slice_idx = 0usize;
    for sy in 0..num_v {
        for sx in 0..num_h {
            let (px, py, pw, ph) = slice_pixel_rect(sx, sy, num_h, num_v, frame_w, frame_h);
            // Extract this slice's pixel rectangle into a row-major plane.
            let mut plane = Vec::with_capacity((pw * ph) as usize);
            for yy in 0..ph {
                for xx in 0..pw {
                    plane.push(read_px(px + xx, py + yy));
                }
            }
            let slice = build_golomb_slice(
                slice_idx == 0,
                sx,
                sy,
                2,
                0,
                &plane,
                pw as usize,
                ph as usize,
                context_c,
                8,
            );
            frame_bytes.extend_from_slice(&slice);
            slice_idx += 1;
        }
    }

    let decoded = oxideav_ffv1::decode_frame(
        &frame_bytes,
        &cr,
        &qts,
        FramePixelDimensions::new(frame_w, frame_h).unwrap(),
        true,
    )
    .expect("2x2 golomb slice grid decodes");

    assert_eq!(decoded.planes.len(), 1);
    assert_eq!(
        (decoded.planes[0].width, decoded.planes[0].height),
        (frame_w, frame_h)
    );
    // The whole reconstructed plane must equal the known frame — this is
    // the assembly assertion: every slice landed in the right rectangle.
    assert_eq!(
        decoded.planes[0].samples, frame,
        "assembled Golomb-Rice frame must be bit-exact"
    );
}

/// A non-square 1×3 vertical slice stack: three stacked slices, each a
/// full-width horizontal band, must reassemble in vertical pixel order.
/// Catches a `slice_pixel_y` origin or row-stride mistake that a square
/// grid would mask.
#[test]
fn golomb_1x3_vertical_stack_assembles_in_row_order() {
    let (frame_w, frame_h) = (5u32, 6u32);
    let (num_h, num_v) = (1u32, 3u32);
    let cr = grayscale_v3_config(num_h, num_v);
    let context_c = 7;
    let qts = vec![constant_context_qts(context_c)];

    // Each of the three 5x2 bands holds a constant offset so a swapped
    // band is immediately visible.
    let mut frame: Vec<i32> = Vec::with_capacity((frame_w * frame_h) as usize);
    for y in 0..frame_h {
        for x in 0..frame_w {
            let band = y / 2;
            frame.push((band as i32 * 60 + (x as i32) * 3 + (y as i32)) & 0xFF);
        }
    }
    let read_px = |x: u32, y: u32| frame[(y * frame_w + x) as usize];

    let mut frame_bytes = Vec::new();
    for sy in 0..num_v {
        let (px, py, pw, ph) = slice_pixel_rect(0, sy, num_h, num_v, frame_w, frame_h);
        let mut plane = Vec::with_capacity((pw * ph) as usize);
        for yy in 0..ph {
            for xx in 0..pw {
                plane.push(read_px(px + xx, py + yy));
            }
        }
        let slice = build_golomb_slice(
            sy == 0,
            0,
            sy,
            2,
            0,
            &plane,
            pw as usize,
            ph as usize,
            context_c,
            8,
        );
        frame_bytes.extend_from_slice(&slice);
    }

    let decoded = oxideav_ffv1::decode_frame(
        &frame_bytes,
        &cr,
        &qts,
        FramePixelDimensions::new(frame_w, frame_h).unwrap(),
        true,
    )
    .expect("1x3 vertical golomb stack decodes");
    assert_eq!(decoded.planes[0].samples, frame);
}

/// 10-bit Golomb-Rice frame: the modular add-back and ESC-mode suffix
/// widths follow `bits_per_raw_sample`, so a bit-depth other than 8 is a
/// distinct path. A single-slice 4×3 10-bit plane must reconstruct
/// bit-exactly (samples span the full 0..1024 range).
#[test]
fn golomb_single_slice_10bit_reconstructs_bit_exact() {
    let mut cr = grayscale_v3_config(1, 1);
    cr.bits_per_raw_sample = 10;
    let context_c = 6;
    let qts = vec![constant_context_qts(context_c)];
    #[rustfmt::skip]
    let samples: Vec<i32> = vec![
           0,  511, 1023,  256,
         800,  100,  900,   50,
        1000,    1,  512,  300,
    ];
    let slice = build_golomb_slice(true, 0, 0, 2, 0, &samples, 4, 3, context_c, 10);
    let decoded = oxideav_ffv1::decode_frame(
        &slice,
        &cr,
        &qts,
        FramePixelDimensions::new(4, 3).unwrap(),
        true,
    )
    .expect("10-bit single-slice golomb frame decodes");
    assert_eq!(decoded.planes[0].samples, samples);
    assert_eq!(decoded.bits_per_raw_sample, 10);
    for &s in &decoded.planes[0].samples {
        assert!((0..1024).contains(&s), "10-bit sample {s} out of range");
    }
}

/// Decoding the same synthetic frame twice yields identical planes — the
/// Golomb-Rice assembly path carries no hidden cross-call state.
#[test]
fn golomb_assembly_is_deterministic() {
    let cr = grayscale_v3_config(2, 1);
    let context_c = 8;
    let qts = vec![constant_context_qts(context_c)];
    let (frame_w, frame_h) = (6u32, 3u32);
    let frame: Vec<i32> = (0..(frame_w * frame_h))
        .map(|i| ((i * 7 + 13) % 256) as i32)
        .collect();
    let read_px = |x: u32, y: u32| frame[(y * frame_w + x) as usize];

    let mut frame_bytes = Vec::new();
    for sx in 0..2u32 {
        let (px, py, pw, ph) = slice_pixel_rect(sx, 0, 2, 1, frame_w, frame_h);
        let mut plane = Vec::new();
        for yy in 0..ph {
            for xx in 0..pw {
                plane.push(read_px(px + xx, py + yy));
            }
        }
        let slice = build_golomb_slice(
            sx == 0,
            sx,
            0,
            2,
            0,
            &plane,
            pw as usize,
            ph as usize,
            context_c,
            8,
        );
        frame_bytes.extend_from_slice(&slice);
    }

    let d1 = oxideav_ffv1::decode_frame(
        &frame_bytes,
        &cr,
        &qts,
        FramePixelDimensions::new(frame_w, frame_h).unwrap(),
        true,
    )
    .unwrap();
    let d2 = oxideav_ffv1::decode_frame(
        &frame_bytes,
        &cr,
        &qts,
        FramePixelDimensions::new(frame_w, frame_h).unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(d1.planes[0].samples, d2.planes[0].samples);
    assert_eq!(d1.planes[0].samples, frame);
}
