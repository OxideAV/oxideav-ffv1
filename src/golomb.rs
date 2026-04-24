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
//! Only the decode path is implemented — the encoder continues to emit the
//! range-coded form.

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
