#![no_main]
//! `decode_v0v1` fuzz target — versions 0 / 1 inline-Parameters decode
//! panic-freedom.
//!
//! Drives attacker-controlled bytes through the crate's
//! [`oxideav_ffv1::decode_frame_v0v1`] entry. Versions 0 / 1 carry NO
//! Configuration Record (RFC 9043 §4.3.3 / §4.4): the §4.2 Parameters and
//! the single §4.1 Quantization Table Set are inline in the keyframe
//! Frame, parsed off one resumed §3.8.1 range-coder pass
//! (`parse_v0v1_frame_prologue`) before the implied single §4.7 Slice
//! Content is reconstructed. Every byte of the prologue and the content
//! is attacker-chosen.
//!
//! Layout of the attacker buffer:
//!   byte 0 — width  (1..=MAX_DIM, after `% MAX_DIM + 1`)
//!   byte 1 — height (1..=MAX_DIM, after `% MAX_DIM + 1`)
//!   then   — the rest is the coded v0/v1 Frame payload
//!
//! The contract under test: no input shape may panic. A malformed Frame
//! must surface a typed [`oxideav_ffv1::Error`], never an out-of-bounds
//! index, an arithmetic overflow, or an `unwrap` on a value the attacker
//! forced to `None` / `Err`.

use libfuzzer_sys::fuzz_target;
use oxideav_ffv1::{decode_frame_v0v1, FramePixelDimensions};

/// Cap on each frame dimension; the single-Slice reconstruction
/// allocates plane buffers proportional to `width * height`.
const MAX_DIM: u32 = 96;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let width = (u32::from(data[0]) % MAX_DIM) + 1;
    let height = (u32::from(data[1]) % MAX_DIM) + 1;
    let frame_bytes = &data[2..];

    let Ok(dims) = FramePixelDimensions::new(width, height) else {
        return;
    };

    // Success and every typed error are acceptable; only a panic is a
    // finding.
    let _ = decode_frame_v0v1(frame_bytes, dims);
});
