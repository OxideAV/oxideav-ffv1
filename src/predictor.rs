//! Median predictor and context computation (RFC 9043 §3.3 + §3.5).
//!
//! The §3.3 median predictor produces a per-Sample predicted value
//! from the three already-decoded neighbours `l` (left), `t` (top),
//! and `tl` (top-left). The decoded `sample_difference` is added to
//! that predicted value to recover the Sample.
//!
//! The §3.5 context computation produces an *absolute* context value
//! by passing five quantized neighbour differences through five
//! quantization tables and summing. If the raw context is negative,
//! the implementation flips the sign of the context AND the
//! `sample_difference` symbol — so the per-context VLC / range state
//! arrays only ever index by `|context|`.

/// Median predictor (RFC 9043 §3.3).
///
/// `median(l, t, l + t - tl)` — the middle of the three values is
/// returned. The third operand is the "gradient" estimate (top-left's
/// position projected from `t` and `l`).
///
/// All inputs and outputs are `i32`. FFV1 samples are bounded by
/// `bits_per_raw_sample <= 16`, so intermediate `l + t - tl` cannot
/// overflow `i32` (worst case 3 * 2^16 = 196608).
#[inline]
pub fn median_predict(l: i32, t: i32, tl: i32) -> i32 {
    let gradient = l + t - tl;
    median_of_three(l, t, gradient)
}

/// Numerical median of three values per RFC 9043 §2.2.5
/// (`median(a, b, c) = a + b + c - min(a,b,c) - max(a,b,c)`).
fn median_of_three(a: i32, b: i32, c: i32) -> i32 {
    let lo = a.min(b).min(c);
    let hi = a.max(b).max(c);
    a.wrapping_add(b)
        .wrapping_add(c)
        .wrapping_sub(lo)
        .wrapping_sub(hi)
}

/// Number of quantization sub-tables in one Quantization Table Set
/// (RFC 9043 §4.1 / §3.5).
///
/// A Quantization Table Set is exactly 5 sub-tables; the §3.5 context
/// computation indexes one quantized neighbour-difference through
/// each.
pub const NUM_QUANT_SUBTABLES: usize = 5;

/// One Quantization Table Set: 5 sub-tables of 256 entries each
/// (RFC 9043 §3.4).
///
/// Per §3.4 the eight least significant bits of the quantized sample
/// difference are used as the index, i.e. `Q[(diff) & 0xFF]`. The
/// table values are signed (the table is symmetric around zero with
/// the high half being the negative reflection of the low half per
/// §4.1's "first half" / "second half" storage).
pub type QuantTableSet = [[i32; 256]; NUM_QUANT_SUBTABLES];

/// Neighbouring samples around a target sample `X` (RFC 9043 §3.2
/// Figure 3) used by the context formula.
///
/// All values are `i32` post-border-padding (§3.1): the border samples
/// north and west of the slice's top-left are zero, plus the
/// repetition rules for the leftmost / rightmost column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborSamples {
    /// `T` — sample two rows above `X`.
    pub tt: i32,
    /// `L` — sample two columns left of `X`.
    pub ll: i32,
    /// `t` — sample directly above `X`.
    pub t: i32,
    /// `tl` — sample one above and one left of `X`.
    pub tl: i32,
    /// `tr` — sample one above and one right of `X`.
    pub tr: i32,
    /// `l` — sample directly left of `X`.
    pub l: i32,
}

/// Compute the §3.5 raw context for a target sample with neighbours
/// `n` through quantization table set `q`.
///
/// Returns the **signed** context value
/// `Q0[l-tl] + Q1[tl-t] + Q2[t-tr] + Q3[L-l] + Q4[T-t]`.
///
/// To use this as a per-context state-array index, the caller must
/// take `context.unsigned_abs()`; if the raw context was negative the
/// caller MUST also flip the sign of the decoded `sample_difference`
/// (per §3.5 "the difference between the Sample and its predicted
/// value is encoded with a flipped sign"). Use [`absolute_context`]
/// for that combined behaviour.
pub fn raw_context(q: &QuantTableSet, n: NeighborSamples) -> i32 {
    let i0 = n.l - n.tl;
    let i1 = n.tl - n.t;
    let i2 = n.t - n.tr;
    let i3 = n.ll - n.l;
    let i4 = n.tt - n.t;

    // §3.4: the eight least significant bits of the quantized sample
    // difference are used as the table index.
    let q0 = q[0][(i0 as u32 & 0xFF) as usize];
    let q1 = q[1][(i1 as u32 & 0xFF) as usize];
    let q2 = q[2][(i2 as u32 & 0xFF) as usize];
    let q3 = q[3][(i3 as u32 & 0xFF) as usize];
    let q4 = q[4][(i4 as u32 & 0xFF) as usize];

    q0.wrapping_add(q1)
        .wrapping_add(q2)
        .wrapping_add(q3)
        .wrapping_add(q4)
}

/// Result of [`absolute_context`]: the non-negative context index used
/// for state lookup plus a flag indicating whether the decoded
/// `sample_difference` needs its sign flipped after decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsoluteContext {
    /// `|context|` — index into the per-context VLC / range state
    /// arrays.
    pub index: u32,
    /// True when the §3.5 raw context was negative; the decoded
    /// `sample_difference` MUST have its sign flipped after entropy
    /// decoding.
    pub sign_flip: bool,
}

/// Compute the absolute context index + sign-flip flag from neighbours
/// `n` and the quant table set `q` (RFC 9043 §3.5).
pub fn absolute_context(q: &QuantTableSet, n: NeighborSamples) -> AbsoluteContext {
    let ctx = raw_context(q, n);
    if ctx < 0 {
        AbsoluteContext {
            index: ctx.unsigned_abs(),
            sign_flip: true,
        }
    } else {
        AbsoluteContext {
            index: ctx as u32,
            sign_flip: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_qtable() -> QuantTableSet {
        [[0i32; 256]; NUM_QUANT_SUBTABLES]
    }

    // --- median predictor ---------------------------------------------

    #[test]
    fn median_predict_all_equal() {
        // l == t == tl → predictor returns the common value.
        assert_eq!(median_predict(50, 50, 50), 50);
    }

    #[test]
    fn median_predict_gradient_chosen() {
        // Pure gradient: tl=10, l=20, t=30 → grad=20+30-10=40; median(20,30,40)=30.
        assert_eq!(median_predict(20, 30, 10), 30);
    }

    #[test]
    fn median_predict_edge_high() {
        // Sharp edge: tl=0, l=100, t=0 → grad=100; median(100, 0, 100)=100.
        assert_eq!(median_predict(100, 0, 0), 100);
    }

    #[test]
    fn median_predict_edge_low() {
        // Mirror: tl=100, l=0, t=100 → grad=0; median(0, 100, 0)=0.
        assert_eq!(median_predict(0, 100, 100), 0);
    }

    #[test]
    fn median_of_three_sorted_inputs() {
        assert_eq!(median_of_three(1, 2, 3), 2);
        assert_eq!(median_of_three(3, 1, 2), 2);
        assert_eq!(median_of_three(-5, 0, 5), 0);
    }

    // --- context computation ------------------------------------------

    #[test]
    fn raw_context_all_zero_table_yields_zero() {
        // Any neighbour configuration through a zero-everywhere quant
        // table gives a zero context. Useful as a sanity check that
        // the sum-of-five plumbing doesn't accidentally inject a
        // constant.
        let q = zero_qtable();
        let n = NeighborSamples {
            tt: 12,
            ll: 34,
            t: 56,
            tl: 78,
            tr: 90,
            l: 11,
        };
        assert_eq!(raw_context(&q, n), 0);
    }

    #[test]
    fn raw_context_diagonal_qtables() {
        // Construct a "Q_i[v] = v" identity table for each subtable and
        // verify the formula directly:
        //   ctx = (l-tl) + (tl-t) + (t-tr) + (L-l) + (T-t)
        let mut q = zero_qtable();
        for table in q.iter_mut() {
            for (idx, slot) in table.iter_mut().enumerate() {
                // Treat low 8 bits as signed
                *slot = if idx < 128 {
                    idx as i32
                } else {
                    (idx as i32) - 256
                };
            }
        }
        // Choose differences that fit in [-128, 127] so the &0xFF lookup
        // exactly recovers the value through the signed interpretation.
        let n = NeighborSamples {
            tt: 30,
            ll: 70,
            t: 10,
            tl: 20,
            tr: 5,
            l: 40,
        };
        // i0 = l-tl = 20; i1 = tl-t = 10; i2 = t-tr = 5; i3 = L-l = 30; i4 = T-t = 20.
        // sum = 20 + 10 + 5 + 30 + 20 = 85.
        assert_eq!(raw_context(&q, n), 85);
    }

    #[test]
    fn absolute_context_handles_sign_flip() {
        // Synthesize a context that goes negative: only Q3 nonzero with
        // sign reversal.
        let mut q = zero_qtable();
        // i3 = L - l. Let L=10, l=20 → diff = -10. Set Q3[-10 & 0xFF]
        // = Q3[246] = -42.
        q[3][246] = -42;
        let n = NeighborSamples {
            tt: 0,
            ll: 10,
            t: 0,
            tl: 0,
            tr: 0,
            l: 20,
        };
        let raw = raw_context(&q, n);
        assert_eq!(raw, -42);
        let abs = absolute_context(&q, n);
        assert_eq!(abs.index, 42);
        assert!(abs.sign_flip);
    }

    #[test]
    fn absolute_context_no_flip_when_positive() {
        let mut q = zero_qtable();
        q[0][10] = 7;
        let n = NeighborSamples {
            tt: 0,
            ll: 0,
            t: 0,
            tl: 0,
            tr: 0,
            l: 10,
        };
        let abs = absolute_context(&q, n);
        assert_eq!(abs.index, 7);
        assert!(!abs.sign_flip);
    }
}
