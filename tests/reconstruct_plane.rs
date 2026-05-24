//! End-to-end integration tests for the round-8 per-plane pixel
//! reconstruction path (RFC 9043 §3.1 / §3.3 / §3.5 / §3.8 / §4.8).
//!
//! These tests drive [`oxideav_ffv1::PlaneReconstructor::reconstruct_plane`]
//! over synthetic Golomb-Rice bitstreams with hand-crafted quantization
//! table sets, verifying the reconstructed Plane byte-exactly. The key
//! property under test is that the §3.3 median predictor + §3.1 border
//! are folded into the per-pixel decode loop, so a Sample's prediction
//! depends on the **reconstructed** neighbours of the previous row, not
//! on the raw `sample_difference` values.

use oxideav_ffv1::{
    median_predict, reconstruct_sample, BitReader, PlaneReconstructor, QuantTableSet,
    NUM_QUANT_SUBTABLES,
};

fn zero_qtable() -> QuantTableSet {
    [[0i32; 256]; NUM_QUANT_SUBTABLES]
}

/// A quant table forcing context `c` for **every** stencil regardless
/// of neighbour values: Q0 maps every index to `c`, the other four
/// sub-tables are zero, so the §3.5 context sum is the constant `c` for
/// all pixels. This keeps the whole Plane on the scalar VLC path
/// (`state.vlc[c]`) so a byte-exact decode is hand-traceable.
fn constant_context_qtable(c: i32) -> QuantTableSet {
    let mut q = zero_qtable();
    q[0] = [c; 256];
    q
}

#[test]
fn reconstruct_2x2_scalar_plane_byte_exact() {
    // Hand-traced 2x2 plane, 8-bit, context 7 (scalar path).
    //
    // The decoded `sample_difference` sequence (with the adaptive VLC
    // k-selection + bias/drift nudge of §3.8.2.4, and the §3.8.2.4
    // `2*drift < -count` sign-flip honoured) is [3, 1, 2, 0] for the
    // bitstream 0x69 0x90.
    //
    // Reconstruction (§3.3 median + §3.1 border + §3.8 mod-256):
    //   row 0:
    //     (0,0): pred=median(0,0,0)=0 → 0+3 = 3
    //     (1,0): l=3,t=0,tl=0 → pred=median(3,0,3)=3 → 3+1 = 4
    //   row 1 (left-of-slice column seeded from prev row's [0]=3):
    //     (0,1): l=3,t=3,tl=3 → pred=3 → 3+2 = 5
    //     (1,1): l=5,t=4,tl=3 → pred=median(5,4,6)=5 → 5+0 = 5
    //   => [3, 4, 5, 5]
    let qtable = constant_context_qtable(7);
    let buf = [0x69u8, 0x90u8];
    let mut br = BitReader::new(&buf);
    let plane = PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 16, 2, 2, 8);
    assert_eq!(plane, vec![3, 4, 5, 5]);
}

#[test]
fn reconstruct_flat_zero_plane_run_mode_byte_exact() {
    // All-zero qtable → run mode at the first Sample, long-run path
    // covers the whole 5x4 plane at one "1" bit per Sample. 20 Samples
    // → 20 set bits = 0xFF 0xFF 0xF0 supplies 20 ones (24 bits, last 4
    // are padding and unread).
    let qtable = zero_qtable();
    let buf = [0xFFu8, 0xFFu8, 0xF0u8];
    let mut br = BitReader::new(&buf);
    let plane = PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 1, 5, 4, 8);
    assert_eq!(plane, vec![0i32; 20]);
}

#[test]
fn reconstruct_sample_matches_spec_modular_addback() {
    // §3.8 modular add-back is byte-exact across the sign boundary.
    // pred=median(200, 100, 150)=150 (gradient=200+100-150=150).
    let pred = median_predict(200, 100, 150);
    assert_eq!(pred, 150);
    // +120 = 270 → 270 & 0xFF = 14.
    assert_eq!(reconstruct_sample(pred, 120, 8), 14);
    // -160 = -10 → -10 & 0xFF = 246.
    assert_eq!(reconstruct_sample(pred, -160, 8), 246);
}

#[test]
fn reconstruct_higher_bit_depth_keeps_full_range() {
    // 10-bit plane: a single scalar Sample exercises the wider modular
    // reduction. With qtable context 7 and bits=10, the fresh state
    // selects k=2; bits "1 10" decode sr_golomb to unsigned 2 → +1.
    // pred=0 → Sample = 1.
    let qtable = constant_context_qtable(7);
    let buf = [0b1100_0000u8];
    let mut br = BitReader::new(&buf);
    let plane = PlaneReconstructor::reconstruct_plane(&mut br, &qtable, 16, 1, 1, 10);
    assert_eq!(plane, vec![1]);
}
