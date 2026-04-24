//! FFV1 Golomb-Rice coder (RFC 9043 §3.8.2).
//!
//! When `coder_type == 0` in the configuration record, FFV1 slice *content*
//! (the per-plane sample differences) switches from the range coder to a
//! bit-packed Golomb-Rice / run-length stream. Slice *headers* remain
//! range-coded in all cases — after the header a Sentinel-mode termination
//! (a `get_rac(state=129)` marker that returns 0) marks the byte where the
//! bit-packed stream starts.
//!
//! Per-sample coding pipeline (scalar mode, §3.8.2.4):
//!
//! 1. Compute the six-tap neighbourhood just like the range-coder path.
//! 2. Quantise into a context `c` via the quant tables.
//! 3. If `c == 0` we're in **run mode** (§3.8.2.2) — the predicted sample is
//!    assumed exact until a non-zero run terminator is found, then the
//!    terminating sample difference is coded as a VLC symbol.
//! 4. Otherwise (scalar mode) read a VLC symbol with the per-context `state`
//!    holding `drift / error_sum / bias / count`. The `k` parameter of the
//!    Rice code is derived from the state (`while (count << k) < error_sum
//!    { k++ }`). After decoding, `state.bias/drift/error_sum/count` adapt
//!    so future symbols match the observed variance.
//! 5. Re-add the predicted value; sign-extend to `bits_per_raw_sample`.
//!
//! Both decode and encode paths are now implemented — `coder_type = 0` in
//! the configuration record selects the Golomb-Rice path end-to-end.

use oxideav_core::{Error, Result};

use crate::predictor::predict;
use crate::state::{compute_context, QuantTables};

/// Maximum prefix length before an ESCape value takes over (see Table 1 in
/// RFC 9043 §3.8.2.1.1). 12 zero bits in a row means the value is encoded
/// as a raw `bits`-wide number.
const PREFIX_MAX: u32 = 12;

/// Log2 of the run length held by a given `run_index` (Figure at §3.8.2.2.1).
#[rustfmt::skip]
const LOG2_RUN: [u32; 41] = [
     0, 0, 0, 0, 1, 1, 1, 1,
     2, 2, 2, 2, 3, 3, 3, 3,
     4, 4, 5, 5, 6, 6, 7, 7,
     8, 9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,
    24,
];

/// Big-endian bit reader over a byte slice. Bits are consumed MSB-first from
/// each byte — the order FFV1 uses for its VLC stream.
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Byte index of the next byte to fetch (current byte = `byte_pos - 1`
    /// when `bit_pos < 8`).
    byte_pos: usize,
    /// Number of valid bits still held in `current` (0..=8). When 0 we must
    /// load the next byte from `buf` to serve the next `get_bits` call.
    bit_buf: u64,
    bits_in_buf: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            byte_pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    fn fill(&mut self) {
        while self.bits_in_buf <= 56 && self.byte_pos < self.buf.len() {
            self.bit_buf = (self.bit_buf << 8) | (self.buf[self.byte_pos] as u64);
            self.bits_in_buf += 8;
            self.byte_pos += 1;
        }
        // Pad with zero bytes if we run out — FFV1 slices are byte-padded so
        // any bits past the last meaningful one are zero.
        while self.bits_in_buf <= 56 {
            self.bit_buf <<= 8;
            self.bits_in_buf += 8;
        }
    }

    /// Read `n` bits as an unsigned integer, MSB first. `n` MUST be `<= 32`.
    pub fn get_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        if self.bits_in_buf < n {
            self.fill();
        }
        let shift = self.bits_in_buf - n;
        let mask: u64 = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let v = (self.bit_buf >> shift) & mask;
        self.bits_in_buf -= n;
        self.bit_buf &= (1u64 << self.bits_in_buf).wrapping_sub(1);
        v as u32
    }

    /// Read one bit (`true` = 1).
    pub fn get_bit(&mut self) -> bool {
        self.get_bits(1) != 0
    }
}

/// Big-endian bit writer that appends MSB-first to an output byte buffer.
/// Mirror of [`BitReader`]: bits emitted here are recovered in the same order
/// by the reader, with a final flush that byte-aligns by padding the tail
/// with zero bits (§3.8.2 "The end of the bitstream of the Frame is padded
/// with zeroes until the bitstream contains a multiple of eight bits.").
pub struct BitWriter {
    out: Vec<u8>,
    /// Bit accumulator holding the bits that haven't been committed yet, in
    /// MSB-first order within the accumulator (high bits = most-recently
    /// written in time). Kept <= 56 bits so we can always shift in up to
    /// 8 more without overflow.
    bit_buf: u64,
    bits_in_buf: u32,
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    pub fn with_initial(out: Vec<u8>) -> Self {
        Self {
            out,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Append `n` bits (low-order `n` bits of `value`) to the stream
    /// MSB-first. `n` MUST be `<= 32`.
    pub fn put_bits(&mut self, n: u32, value: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let masked: u64 = if n == 64 {
            value as u64
        } else {
            (value as u64) & ((1u64 << n) - 1)
        };
        self.bit_buf = (self.bit_buf << n) | masked;
        self.bits_in_buf += n;
        while self.bits_in_buf >= 8 {
            let shift = self.bits_in_buf - 8;
            let byte = ((self.bit_buf >> shift) & 0xFF) as u8;
            self.out.push(byte);
            self.bits_in_buf -= 8;
            self.bit_buf &= (1u64 << self.bits_in_buf).wrapping_sub(1);
        }
    }

    /// Append a single bit (`true` = 1).
    pub fn put_bit(&mut self, v: bool) {
        self.put_bits(1, v as u32);
    }

    /// Flush any pending bits, padding the last byte with zeros on the low
    /// side (which end up being the *trailing* bits in MSB-first order, i.e.
    /// they match what a matching [`BitReader`] would read as zero padding).
    /// Returns the accumulated byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_buf > 0 {
            let shift = 8 - self.bits_in_buf;
            let byte = ((self.bit_buf << shift) & 0xFF) as u8;
            self.out.push(byte);
            self.bits_in_buf = 0;
            self.bit_buf = 0;
        }
        self.out
    }
}

/// Encode an unsigned Golomb-Rice code with parameter `k` and `bits` wide
/// ESC fallback — inverse of [`get_ur_golomb`]. Values in
/// `[0, (PREFIX_MAX - 1) << k]` use the Rice prefix/suffix form; larger
/// values fall back to the ESC path (12 zero bits + `bits`-wide raw).
pub fn put_ur_golomb(w: &mut BitWriter, value: u32, k: u32, bits: u32) {
    // Rice-mode: the decoder loops `for prefix = 0 .. PREFIX_MAX` reading one
    // bit each iteration and returns on the first 1-bit. So `prefix` values
    // `0..=PREFIX_MAX - 1` (= 0..=11) are non-ESC. Only when all 12 bits are
    // zero does the ESC branch fire. Hence the encoder emits ESC iff
    // `value >> k >= PREFIX_MAX`.
    let prefix = value >> k;
    if prefix < PREFIX_MAX {
        // Non-ESC: `prefix` zero bits, then a 1 bit, then k LSBs of `value`.
        for _ in 0..prefix {
            w.put_bit(false);
        }
        w.put_bit(true);
        if k > 0 {
            let suffix = value & ((1u32 << k) - 1);
            w.put_bits(k, suffix);
        }
    } else {
        // ESC: 12 zero bits, then `value - 11` in `bits` MSB-first.
        for _ in 0..PREFIX_MAX {
            w.put_bit(false);
        }
        w.put_bits(bits, value - 11);
    }
}

/// Encode a signed Golomb-Rice code — inverse of [`get_sr_golomb`]: zig-zag
/// encode `v` to unsigned and delegate to `put_ur_golomb`.
pub fn put_sr_golomb(w: &mut BitWriter, v: i32, k: u32, bits: u32) {
    let u: u32 = if v < 0 {
        ((-v as u32) << 1).wrapping_sub(1)
    } else {
        (v as u32) << 1
    };
    put_ur_golomb(w, u, k, bits);
}

/// Encode one signed integer using the per-context VLC state machine —
/// symmetric twin of [`get_vlc_symbol`]. The same adaptive `k` is chosen,
/// the same bias inversion applied, and the state update rules are identical
/// so encoder and decoder walk the same `(k, state)` trajectory.
pub fn put_vlc_symbol(w: &mut BitWriter, st: &mut VlcState, bits: u32, value: i32) {
    // Same k derivation as the decoder.
    let mut i = st.count;
    let mut k: u32 = 0;
    while i < st.error_sum && k < 31 {
        k += 1;
        i += i;
    }

    // Inverse of the decoder's `ret = sign_extend(v + bias, bits)`. On decode
    // we recovered `ret` from the wire and reconstructed the *sample*; here
    // we have the sample's residual (already sign-extended to `bits` bits
    // from the caller) and need to find the `v` that would round-trip.
    //
    // The decoder computed `sum = (v + bias) & mask` and sign-extended. So
    // `v = (ret - bias)` sign-extended back into `bits` bits. Bias inversion
    // happens after: if `2 * drift < -count`, the decoder does `v = -1 - v`,
    // so the encoder must emit `v_wire = -1 - v_natural` in that case.
    let mask = (1i32 << bits) - 1;
    let sign_bit = 1i32 << (bits - 1);
    let diff = (value - st.bias) & mask;
    // `v_used` is the decoder's v *after* the optional bias-inversion flip —
    // the same value used for the state update. Compute it first.
    let v_used = if diff & sign_bit != 0 {
        diff - (1i32 << bits)
    } else {
        diff
    };
    // `v_wire` is what we emit on the wire: the decoder will un-flip it to
    // recover `v_used`. When the bias-inversion path isn't active, they are
    // equal.
    let v_wire = if 2 * st.drift < -st.count {
        -1 - v_used
    } else {
        v_used
    };

    put_sr_golomb(w, v_wire, k, bits);

    // State update: decoder uses the *post-flip* `v_used` value; mirror
    // that here so our trajectory stays in lock-step.
    st.error_sum += v_used.abs();
    st.drift += v_used;

    if st.count == 128 {
        st.count >>= 1;
        st.drift >>= 1;
        st.error_sum >>= 1;
    }
    st.count += 1;
    if st.drift <= -st.count {
        st.bias = (st.bias - 1).max(-128);
        st.drift = (st.drift + st.count).max(-st.count + 1);
    } else if st.drift > 0 {
        st.bias = (st.bias + 1).min(127);
        st.drift = (st.drift - st.count).min(0);
    }
}

/// Read an unsigned Golomb-Rice code with parameter `k` and `bits` wide ESC
/// fallback, matching Figure 26 of RFC 9043.
pub fn get_ur_golomb(r: &mut BitReader<'_>, k: u32, bits: u32) -> u32 {
    let mut prefix: u32 = 0;
    while prefix < PREFIX_MAX {
        if r.get_bit() {
            // Non-ESC: read k LSBs.
            let suffix = if k == 0 { 0 } else { r.get_bits(k) };
            return (prefix << k) | suffix;
        }
        prefix += 1;
    }
    // ESC: the raw value is stored as `bits` bits and `+ 11` is added so
    // smaller values that happen to consume 12 zero bits still decode
    // uniquely (see §3.8.2.1.2).
    r.get_bits(bits) + 11
}

/// Signed Golomb-Rice wrapper from §3.8.2.1 (Figure 27): zig-zag-decodes the
/// unsigned value into a signed one.
pub fn get_sr_golomb(r: &mut BitReader<'_>, k: u32, bits: u32) -> i32 {
    let v = get_ur_golomb(r, k, bits) as i32;
    if v & 1 != 0 {
        -(v >> 1) - 1
    } else {
        v >> 1
    }
}

/// Per-context VLC state (§3.8.2.5). One entry per quantised context.
#[derive(Clone, Copy, Debug)]
pub struct VlcState {
    pub drift: i32,
    pub error_sum: i32,
    pub bias: i32,
    pub count: i32,
}

impl Default for VlcState {
    fn default() -> Self {
        Self {
            drift: 0,
            error_sum: 4,
            bias: 0,
            count: 1,
        }
    }
}

/// Per-plane container of VLC states: one entry per context index.
pub struct VlcPlaneState {
    pub states: Vec<VlcState>,
}

impl VlcPlaneState {
    pub fn new(context_count: usize) -> Self {
        Self {
            states: vec![VlcState::default(); context_count],
        }
    }
}

/// Decode one signed integer using the per-context VLC state machine
/// (`get_vlc_symbol` in §3.8.2.4). `bits` is the width of the ESC fallback
/// (FFV1 uses `bits_per_raw_sample + 1` to give room for the sign-extended
/// residual, e.g. 9 for 8-bit samples).
pub fn get_vlc_symbol(r: &mut BitReader<'_>, st: &mut VlcState, bits: u32) -> i32 {
    // Derive the Rice k: smallest k such that `count << k >= error_sum`.
    let mut i = st.count;
    let mut k: u32 = 0;
    while i < st.error_sum && k < 31 {
        k += 1;
        i += i;
    }

    let mut v = get_sr_golomb(r, k, bits);
    // Bias inversion: if the running bias signals we've been over-shooting
    // (`2 * drift < -count`), the coded sign is inverted.
    if 2 * st.drift < -st.count {
        v = -1 - v;
    }

    // `ret = sign_extend(v + bias, bits)` — but the value returned to the
    // caller is a *residual* in the `bits_per_raw_sample`-wide range, which
    // the caller will fold back into the reconstruction. We compute that
    // fold here using the same two's-complement sign-extension trick as the
    // range-coder path.
    let mask = (1i32 << bits) - 1;
    let sum = (v + st.bias) & mask;
    let sign_bit = 1i32 << (bits - 1);
    let ret = if sum & sign_bit != 0 {
        sum - (1i32 << bits)
    } else {
        sum
    };

    // Update the adaptive state.
    st.error_sum += v.abs();
    st.drift += v;

    if st.count == 128 {
        st.count >>= 1;
        st.drift >>= 1;
        st.error_sum >>= 1;
    }
    st.count += 1;
    if st.drift <= -st.count {
        st.bias = (st.bias - 1).max(-128);
        st.drift = (st.drift + st.count).max(-st.count + 1);
    } else if st.drift > 0 {
        st.bias = (st.bias + 1).min(127);
        st.drift = (st.drift - st.count).min(0);
    }

    ret
}

/// Decode one plane of Golomb-coded sample differences into a byte buffer.
/// Runs the per-line mixture of run mode + scalar mode described in §3.8.2.
pub fn decode_plane_u8(
    r: &mut BitReader<'_>,
    samples: &mut [u8],
    width: u32,
    height: u32,
    tables: &QuantTables,
    plane_state: &mut VlcPlaneState,
) -> Result<()> {
    decode_plane_generic(
        r,
        SampleViewMut::U8(samples),
        width,
        height,
        8,
        tables,
        plane_state,
    )
}

/// Decode one plane of Golomb-coded sample differences into a `u16` buffer.
/// Used for `coder_type == 0` with `bits_per_raw_sample > 8` — RFC §3.1.3
/// labels this SHOULD NOT, but FFmpeg has historically emitted it for
/// `-coder 0 -pix_fmt yuv420p10le` so we accept it.
pub fn decode_plane_u16(
    r: &mut BitReader<'_>,
    samples: &mut [u16],
    width: u32,
    height: u32,
    bit_depth: u32,
    tables: &QuantTables,
    plane_state: &mut VlcPlaneState,
) -> Result<()> {
    decode_plane_generic(
        r,
        SampleViewMut::U16(samples),
        width,
        height,
        bit_depth,
        tables,
        plane_state,
    )
}

/// Encode one plane of 8-bit samples into the Golomb bit stream. Mirror of
/// [`decode_plane_u8`].
pub fn encode_plane_u8(
    w: &mut BitWriter,
    samples: &[u8],
    width: u32,
    height: u32,
    tables: &QuantTables,
    plane_state: &mut VlcPlaneState,
) -> Result<()> {
    encode_plane_generic(
        w,
        SampleView::U8(samples),
        width,
        height,
        8,
        tables,
        plane_state,
    )
}

/// Encode one plane of >8-bit samples into the Golomb bit stream. Mirror of
/// [`decode_plane_u16`].
pub fn encode_plane_u16(
    w: &mut BitWriter,
    samples: &[u16],
    width: u32,
    height: u32,
    bit_depth: u32,
    tables: &QuantTables,
    plane_state: &mut VlcPlaneState,
) -> Result<()> {
    encode_plane_generic(
        w,
        SampleView::U16(samples),
        width,
        height,
        bit_depth,
        tables,
        plane_state,
    )
}

/// Immutable sample view — read-only twin of [`SampleViewMut`].
enum SampleView<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

impl SampleView<'_> {
    fn len(&self) -> usize {
        match self {
            SampleView::U8(s) => s.len(),
            SampleView::U16(s) => s.len(),
        }
    }

    #[inline]
    fn get(&self, idx: usize) -> i32 {
        match self {
            SampleView::U8(s) => s[idx] as i32,
            SampleView::U16(s) => s[idx] as i32,
        }
    }
}

#[inline]
fn fetch_neighbours_view(
    samples: &SampleView<'_>,
    w: usize,
    x: usize,
    y: usize,
) -> (i32, i32, i32, i32, i32, i32) {
    let get = |i: usize| samples.get(i);
    fetch_neighbours_impl(&get, w, x, y)
}

fn encode_plane_generic(
    w: &mut BitWriter,
    samples: SampleView<'_>,
    width: u32,
    height: u32,
    bit_depth: u32,
    tables: &QuantTables,
    plane_state: &mut VlcPlaneState,
) -> Result<()> {
    let ww = width as usize;
    let hh = height as usize;
    if samples.len() != ww * hh {
        return Err(Error::invalid("golomb encode_plane: bad buffer length"));
    }
    let bits = bit_depth;
    let mask: i32 = if bits >= 32 { -1 } else { (1i32 << bits) - 1 };
    let shift = 32 - bits as i32;

    // Run-mode state machine, driven by what the DECODER reads at each
    // pixel position. The decoder pulls a single "run-start" bit whenever
    // `run_count == 0 && run_mode == 1`; we emit that bit at the same
    // stream position. Two encodings are possible:
    //
    // * Big run (bit = 1): the next `big_size = 1 << log2_run[run_index]`
    //   pixels all have diff = 0. No terminator follows.
    // * Small run (bit = 0): a `log2_run[run_index]`-bit count gives the
    //   number of zero pixels to skip, then a non-zero VLC terminator
    //   follows. If the row ends mid-small-run, no terminator is written
    //   (the decoder's row loop exits before reading one).
    //
    // To decide which, we look ahead at x_start and count consecutive
    // zero-diff pixels up to `big_size`. Exactly `big_size` zeros → big
    // run. Anything less → small run.
    let mut run_index: i32 = 0;
    for y in 0..hh {
        let mut x = 0usize;
        let mut run_mode: i32 = 0;

        while x < ww {
            let (big_l, l, t, tl, big_t, tr) = fetch_neighbours_view(&samples, ww, x, y);
            let mut ctx = compute_context(tables, big_l, l, t, tl, big_t, tr);
            let sign_flip = ctx < 0;
            if sign_flip {
                ctx = -ctx;
            }
            let pred = predict(l, t, tl);
            let s = samples.get(y * ww + x);
            let raw_diff = s - pred;
            let wrapped: i32 = (raw_diff << shift) >> shift;
            let diff = if sign_flip { -wrapped } else { wrapped };

            // Enter run mode when the decoder would.
            if ctx == 0 && run_mode == 0 {
                run_mode = 1;
            }

            if run_mode == 1 {
                // Count consecutive zero-diff pixels starting at x, capped
                // at big_size.
                let big_size = 1i32 << LOG2_RUN[run_index as usize];
                let mut zeros = 0i32;
                let mut xk = x;
                while xk < ww && zeros < big_size {
                    let (bl, l2, t2, tl2, bt2, tr2) = fetch_neighbours_view(&samples, ww, xk, y);
                    let mut cx = compute_context(tables, bl, l2, t2, tl2, bt2, tr2);
                    let flip = cx < 0;
                    if flip {
                        cx = -cx;
                    }
                    let _ = cx;
                    let p2 = predict(l2, t2, tl2);
                    let s2 = samples.get(y * ww + xk);
                    let rd = s2 - p2;
                    let w2: i32 = (rd << shift) >> shift;
                    let d = if flip { -w2 } else { w2 };
                    if d != 0 {
                        break;
                    }
                    zeros += 1;
                    xk += 1;
                }

                if zeros == big_size {
                    // Big run. Emit `1`, mirror the decoder's run_index++
                    // gate: `x_start + run_count <= w`.
                    w.put_bit(true);
                    if (x as i32 + big_size) <= ww as i32 {
                        run_index += 1;
                    }
                    x += big_size as usize;
                    // Stay in run_mode = 1 for the next pixel.
                    continue;
                }

                // Small run: `0` bit, count, (optional) terminator.
                w.put_bit(false);
                let log2 = LOG2_RUN[run_index as usize];
                if log2 != 0 {
                    w.put_bits(log2, zeros as u32);
                }
                if run_index != 0 {
                    run_index -= 1;
                }
                x += zeros as usize;
                if x >= ww {
                    // Row ended before the terminator. The decoder's row
                    // loop exits before run_count reaches 0, so no
                    // terminator VLC is read.
                    break;
                }

                // Emit the terminator at pixel x.
                let (bl, l2, t2, tl2, bt2, tr2) = fetch_neighbours_view(&samples, ww, x, y);
                let mut cx = compute_context(tables, bl, l2, t2, tl2, bt2, tr2);
                let flip = cx < 0;
                if flip {
                    cx = -cx;
                }
                let p2 = predict(l2, t2, tl2);
                let s2 = samples.get(y * ww + x);
                let rd = s2 - p2;
                let w3: i32 = (rd << shift) >> shift;
                let d2 = if flip { -w3 } else { w3 };

                // Undo the sign flip for the state's view of the residual —
                // the decoder applies the flip AFTER the `+= 1` shift, so
                // the wire terminator is the pre-flip value.
                let mut term = d2;
                if flip {
                    term = -term;
                }
                // Reverse decoder's `if terminator >= 0 { terminator += 1 }`.
                if term > 0 {
                    term -= 1;
                }
                let state_idx = cx as usize;
                let masked = term & mask;
                let sign_bit = 1i32 << (bits - 1);
                let ext = if masked & sign_bit != 0 {
                    masked - (1i32 << bits)
                } else {
                    masked
                };
                put_vlc_symbol(w, &mut plane_state.states[state_idx], bits, ext);
                x += 1;
                run_mode = 0;
                continue;
            }

            // Scalar mode: emit the VLC symbol directly.
            let state_idx = ctx as usize;
            let masked = diff & mask;
            let sign_bit = 1i32 << (bits - 1);
            let ext = if masked & sign_bit != 0 {
                masked - (1i32 << bits)
            } else {
                masked
            };
            put_vlc_symbol(w, &mut plane_state.states[state_idx], bits, ext);
            x += 1;
        }
    }
    Ok(())
}

#[inline]
fn fetch_neighbours_impl<F: Fn(usize) -> i32>(
    get: &F,
    w: usize,
    x: usize,
    y: usize,
) -> (i32, i32, i32, i32, i32, i32) {
    let prev_row_exists = y >= 1;
    let prev_row_base = if prev_row_exists { (y - 1) * w } else { 0 };
    let prev_row_sample = |col: isize| -> i32 {
        if !prev_row_exists {
            return 0;
        }
        if col < 0 {
            if y >= 2 {
                return get((y - 2) * w);
            }
            return 0;
        }
        if (col as usize) >= w {
            return get(prev_row_base + w - 1);
        }
        get(prev_row_base + col as usize)
    };
    let cur_row_sample = |col: isize| -> i32 {
        if col < 0 {
            return if y >= 1 { get(prev_row_base) } else { 0 };
        }
        get(y * w + col as usize)
    };
    let l = cur_row_sample(x as isize - 1);
    let big_l = if x >= 2 { get(y * w + x - 2) } else { 0 };
    let t = prev_row_sample(x as isize);
    let tl = prev_row_sample(x as isize - 1);
    let tr = prev_row_sample(x as isize + 1);
    let big_t = if y >= 2 { get((y - 2) * w + x) } else { 0 };
    (big_l, l, t, tl, big_t, tr)
}

/// Mutable view over a Golomb plane's sample buffer, supporting 8 or 9..=16
/// bits per sample.
enum SampleViewMut<'a> {
    U8(&'a mut [u8]),
    U16(&'a mut [u16]),
}

impl SampleViewMut<'_> {
    fn len(&self) -> usize {
        match self {
            SampleViewMut::U8(s) => s.len(),
            SampleViewMut::U16(s) => s.len(),
        }
    }

    #[inline]
    fn get(&self, idx: usize) -> i32 {
        match self {
            SampleViewMut::U8(s) => s[idx] as i32,
            SampleViewMut::U16(s) => s[idx] as i32,
        }
    }

    #[inline]
    fn set(&mut self, idx: usize, v: u32) {
        match self {
            SampleViewMut::U8(s) => s[idx] = v as u8,
            SampleViewMut::U16(s) => s[idx] = v as u16,
        }
    }
}

fn decode_plane_generic(
    r: &mut BitReader<'_>,
    mut samples: SampleViewMut<'_>,
    width: u32,
    height: u32,
    bit_depth: u32,
    tables: &QuantTables,
    plane_state: &mut VlcPlaneState,
) -> Result<()> {
    let w = width as usize;
    let h = height as usize;
    if samples.len() != w * h {
        return Err(Error::invalid("golomb decode_plane: bad buffer length"));
    }
    // Per RFC §3.8, `bits` used by both the VLC sign_extend and the
    // get_ur_golomb ESC fallback equals `bits_per_raw_sample` (or
    // `bits_per_raw_sample + 1` for JPEG 2000 RCT — not supported here).
    let bits = bit_depth;
    let mask: u32 = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    // Run mode is per plane — reset these for each plane per §3.8.2.2.1.
    let mut run_index: i32 = 0;

    for y in 0..h {
        // FFmpeg resets run mode at the start of each row (this matches the
        // RFC's "run_index is reset to zero for each Plane and Slice" but
        // the actual run state machine only lives within a row — run_count
        // / run_mode are line-local).
        let mut run_count: i32 = 0;
        let mut run_mode: i32 = 0; // 0 = not in run, 1 = scanning zeros, 2 = found terminator

        for x in 0..w {
            let (big_l, l, t, tl, big_t, tr) = fetch_neighbours(&samples, w, x, y);
            let mut ctx = compute_context(tables, big_l, l, t, tl, big_t, tr);
            let sign_flip = ctx < 0;
            if sign_flip {
                ctx = -ctx;
            }
            let pred = predict(l, t, tl);

            let diff: i32;
            if ctx == 0 && run_mode == 0 {
                run_mode = 1;
            }

            if run_mode != 0 {
                if run_count == 0 && run_mode == 1 {
                    if r.get_bit() {
                        // "Big" run of `1 << log2_run[run_index]` zeros.
                        run_count = 1i32 << LOG2_RUN[run_index as usize];
                        if (x as i32 + run_count) <= w as i32 {
                            run_index += 1;
                        }
                    } else {
                        // Small run: next log2_run[run_index] bits give the
                        // remaining count, then a terminator follows.
                        if LOG2_RUN[run_index as usize] != 0 {
                            run_count = r.get_bits(LOG2_RUN[run_index as usize]) as i32;
                        } else {
                            run_count = 0;
                        }
                        if run_index != 0 {
                            run_index -= 1;
                        }
                        run_mode = 2;
                    }
                }

                if run_count > 0 {
                    // Still inside a zero run → sample difference is 0.
                    run_count -= 1;
                    diff = 0;
                } else {
                    // End of run: read the terminating non-zero VLC symbol.
                    // The RFC's run mode uses the current pixel's context
                    // state for the terminator (context 0 if the run is
                    // still ongoing, or the new non-zero context the sample
                    // pattern has moved into). Context 0 lives at state
                    // index 0; `sign_flip` applies only to non-zero
                    // contexts so leave the terminator untouched in that
                    // case.
                    let state_idx = ctx as usize;
                    let mut terminator =
                        get_vlc_symbol(r, &mut plane_state.states[state_idx], bits);
                    // Per §3.8.2.4.1: the zero value cannot occur inside a
                    // run, so shift non-negative values up by one.
                    if terminator >= 0 {
                        terminator += 1;
                    }
                    if sign_flip {
                        terminator = -terminator;
                    }
                    diff = terminator;
                    run_mode = 0;
                    run_count = 0;
                }
            } else {
                // Scalar mode.
                let state_idx = ctx as usize;
                let signed_val = get_vlc_symbol(r, &mut plane_state.states[state_idx], bits);
                diff = if sign_flip { -signed_val } else { signed_val };
            }

            // Reconstruct and mask to `bits` width.
            let recon = ((pred + diff) as u32) & mask;
            samples.set(y * w + x, recon);
        }
    }
    Ok(())
}

#[inline]
fn fetch_neighbours(
    samples: &SampleViewMut<'_>,
    w: usize,
    x: usize,
    y: usize,
) -> (i32, i32, i32, i32, i32, i32) {
    // Mirror the range-coder decode path's neighbour convention (see the
    // extended commentary in `slice.rs::neighbours_impl`).
    let prev_row_exists = y >= 1;
    let prev_row_base = if prev_row_exists { (y - 1) * w } else { 0 };
    let prev_row_sample = |col: isize| -> i32 {
        if !prev_row_exists {
            return 0;
        }
        if col < 0 {
            if y >= 2 {
                return samples.get((y - 2) * w);
            }
            return 0;
        }
        if (col as usize) >= w {
            return samples.get(prev_row_base + w - 1);
        }
        samples.get(prev_row_base + col as usize)
    };
    let cur_row_sample = |col: isize| -> i32 {
        if col < 0 {
            return if y >= 1 {
                samples.get(prev_row_base)
            } else {
                0
            };
        }
        samples.get(y * w + col as usize)
    };
    let l = cur_row_sample(x as isize - 1);
    let big_l = if x >= 2 {
        samples.get(y * w + x - 2)
    } else {
        0
    };
    let t = prev_row_sample(x as isize);
    let tl = prev_row_sample(x as isize - 1);
    let tr = prev_row_sample(x as isize + 1);
    let big_t = if y >= 2 {
        samples.get((y - 2) * w + x)
    } else {
        0
    };
    (big_l, l, t, tl, big_t, tr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_msb_first() {
        // 0xA5 = 1010 0101
        let buf = [0xA5, 0x3C];
        let mut r = BitReader::new(&buf);
        assert!(r.get_bit());
        assert!(!r.get_bit());
        assert_eq!(r.get_bits(2), 0b10);
        assert_eq!(r.get_bits(4), 0b0101);
        assert_eq!(r.get_bits(8), 0x3C);
    }

    #[test]
    fn unsigned_golomb_k0() {
        // Build a bitstream for k=0: values 0, 2, 1 → bits: 1, 001, 01.
        // 1 001 01 00 = 0x94
        let buf = [0x94, 0x00];
        let mut r = BitReader::new(&buf);
        assert_eq!(get_ur_golomb(&mut r, 0, 9), 0);
        assert_eq!(get_ur_golomb(&mut r, 0, 9), 2);
        assert_eq!(get_ur_golomb(&mut r, 0, 9), 1);
    }

    #[test]
    fn bit_writer_msb_first() {
        let mut w = BitWriter::new();
        // 1 bit, 1; 2 bits, 0b10; 3 bits, 0b101; 2 bits, 0b11; total 8 bits.
        // Expect 0b11010111 = 0xD7.
        w.put_bit(true);
        w.put_bits(2, 0b10);
        w.put_bits(3, 0b101);
        w.put_bits(2, 0b11);
        let out = w.finish();
        assert_eq!(out, vec![0xD7]);
    }

    #[test]
    fn bit_writer_reader_roundtrip() {
        let mut w = BitWriter::new();
        for v in 0u32..200 {
            w.put_bits((v % 16) + 1, v);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for v in 0u32..200 {
            let n = (v % 16) + 1;
            let got = r.get_bits(n);
            let mask: u32 = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
            assert_eq!(got, v & mask);
        }
    }

    #[test]
    fn ur_golomb_roundtrip() {
        for k in 0..=3u32 {
            let mut w = BitWriter::new();
            let values: Vec<u32> = (0u32..500).collect();
            for &v in &values {
                put_ur_golomb(&mut w, v, k, 9);
            }
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            for &v in &values {
                let got = get_ur_golomb(&mut r, k, 9);
                assert_eq!(got, v, "k={k} v={v}");
            }
        }
    }

    #[test]
    fn sr_golomb_roundtrip() {
        for k in 0..=3u32 {
            let mut w = BitWriter::new();
            let values: Vec<i32> = (-200i32..=200).collect();
            for &v in &values {
                put_sr_golomb(&mut w, v, k, 9);
            }
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            for &v in &values {
                let got = get_sr_golomb(&mut r, k, 9);
                assert_eq!(got, v, "k={k} v={v}");
            }
        }
    }

    #[test]
    fn plane_roundtrip_flat() {
        // A uniform grey plane should round-trip through the Golomb encoder
        // and decoder losslessly — the run mode handles this efficiently.
        let tables = crate::state::default_quant_tables();
        let ctx_count = crate::state::context_count(&tables);
        let width = 16u32;
        let height = 8u32;
        let pixels = vec![128u8; (width * height) as usize];
        let mut enc_state = VlcPlaneState::new(ctx_count);
        let mut w = BitWriter::new();
        encode_plane_u8(&mut w, &pixels, width, height, &tables, &mut enc_state).unwrap();
        let bytes = w.finish();
        let mut dec_state = VlcPlaneState::new(ctx_count);
        let mut r = BitReader::new(&bytes);
        let mut out = vec![0u8; pixels.len()];
        decode_plane_u8(&mut r, &mut out, width, height, &tables, &mut dec_state).unwrap();
        assert_eq!(out, pixels);
    }

    #[test]
    fn plane_roundtrip_first_row_scalar() {
        // First row of the gradient: (x*7) & 0xFF for x=0..16.
        let tables = crate::state::default_quant_tables();
        let ctx_count = crate::state::context_count(&tables);
        let width = 16u32;
        let height = 1u32;
        let pixels: Vec<u8> = (0u32..width).map(|x| ((x * 7) & 0xFF) as u8).collect();
        let mut enc_state = VlcPlaneState::new(ctx_count);
        let mut w = BitWriter::new();
        encode_plane_u8(&mut w, &pixels, width, height, &tables, &mut enc_state).unwrap();
        let bytes = w.finish();
        let mut dec_state = VlcPlaneState::new(ctx_count);
        let mut r = BitReader::new(&bytes);
        let mut out = vec![0u8; pixels.len()];
        decode_plane_u8(&mut r, &mut out, width, height, &tables, &mut dec_state).unwrap();
        assert_eq!(out, pixels);
    }

    #[test]
    fn plane_roundtrip_gradient() {
        // A horizontal gradient forces the scalar path (non-zero contexts).
        let tables = crate::state::default_quant_tables();
        let ctx_count = crate::state::context_count(&tables);
        let width = 32u32;
        let height = 8u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                pixels[y * width as usize + x] = ((x * 7 + y * 3) & 0xFF) as u8;
            }
        }
        let mut enc_state = VlcPlaneState::new(ctx_count);
        let mut w = BitWriter::new();
        encode_plane_u8(&mut w, &pixels, width, height, &tables, &mut enc_state).unwrap();
        let bytes = w.finish();
        let mut dec_state = VlcPlaneState::new(ctx_count);
        let mut r = BitReader::new(&bytes);
        let mut out = vec![0u8; pixels.len()];
        decode_plane_u8(&mut r, &mut out, width, height, &tables, &mut dec_state).unwrap();
        assert_eq!(out, pixels);
    }

    #[test]
    fn signed_golomb_zigzag_basic() {
        // Signed Golomb maps unsigned N to signed: 0→0, 1→-1, 2→1, 3→-2, 4→2.
        // Build a stream of unsigned 0,1,2,3,4 at k=0:
        //   0 = "1"       (prefix 0)
        //   1 = "01"      (prefix 1, suffix 0)
        //   2 = "001"     (prefix 2)
        //   3 = "0001"    (prefix 3)
        //   4 = "00001"   (prefix 4)
        // Concat MSB-first: 1 01 001 0001 00001 = 15 bits → pad with zero.
        //   bits: 10100100 01000010 → 0xA4, 0x42 (trailing bit is padding)
        let buf = [0xA4, 0x42];
        let mut r = BitReader::new(&buf);
        assert_eq!(get_sr_golomb(&mut r, 0, 9), 0);
        assert_eq!(get_sr_golomb(&mut r, 0, 9), -1);
        assert_eq!(get_sr_golomb(&mut r, 0, 9), 1);
        assert_eq!(get_sr_golomb(&mut r, 0, 9), -2);
        assert_eq!(get_sr_golomb(&mut r, 0, 9), 2);
    }
}
