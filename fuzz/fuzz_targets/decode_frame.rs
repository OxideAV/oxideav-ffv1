#![no_main]
//! `decode_frame` fuzz target — v3 YCbCr + RGB decode-pipeline
//! panic-freedom.
//!
//! Drives attacker-controlled bytes through the crate's two top-level v3
//! frame drivers, [`oxideav_ffv1::decode_frame`] (YCbCr / plane-major,
//! `colorspace_type == 0`) and [`oxideav_ffv1::decode_frame_rgb`] (RGB /
//! line-major JPEG 2000 RCT, `colorspace_type == 1`). Each walks the
//! §4.9.1 trailer chain, validates §4.9 footers, parses §4.6 Slice
//! Headers, lays out §4.7 planes, and runs §3.3 / §3.5 / §3.7 / §3.8
//! per-plane reconstruction — all over attacker bytes.
//!
//! The attacker also controls the §4.2 Configuration Record (parsed via
//! [`oxideav_ffv1::parse_quantization_table_sets`], which carries the
//! §4.1 cascade) and the frame dimensions. Dimensions are clamped to a
//! small bound so a malformed record can never request an unbounded
//! allocation that would mask real findings behind an OOM.
//!
//! Layout of the attacker buffer:
//!   byte 0      — width  (1..=MAX_DIM, after `% MAX_DIM + 1`)
//!   byte 1      — height (1..=MAX_DIM, after `% MAX_DIM + 1`)
//!   bytes 2..4  — little-endian record-blob length, modulo remaining
//!   then        — `record_len` bytes of Configuration Record
//!   then        — the rest is the coded Frame payload
//!
//! The contract under test: no input shape may panic. A malformed stream
//! must surface a typed [`oxideav_ffv1::Error`] (or decode), never an
//! out-of-bounds index, an arithmetic overflow, or an `unwrap` on a
//! value the attacker forced to `None` / `Err`.

use libfuzzer_sys::fuzz_target;
use oxideav_ffv1::{
    decode_frame, decode_frame_rgb, parse_quantization_table_sets, FramePixelDimensions,
};

/// Cap on each frame dimension. The decoders allocate plane buffers
/// proportional to `width * height`; bounding both keeps a fuzz input's
/// peak allocation finite while still exercising multi-Slice grids,
/// chroma subsampling, and the trailer-chain walk.
const MAX_DIM: u32 = 96;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let width = (u32::from(data[0]) % MAX_DIM) + 1;
    let height = (u32::from(data[1]) % MAX_DIM) + 1;
    let rest = &data[2..];

    // Two-byte length prefix selects how many of the remaining bytes form
    // the Configuration Record; the rest is the coded Frame.
    let raw_len = usize::from(u16::from_le_bytes([rest[0], rest[1]]));
    let body = &rest[2..];
    let record_len = if body.is_empty() {
        0
    } else {
        raw_len % (body.len() + 1)
    };
    let (record_bytes, frame_bytes) = body.split_at(record_len.min(body.len()));

    let Ok(dims) = FramePixelDimensions::new(width, height) else {
        return;
    };

    // A record that doesn't parse is a valid (typed-error) outcome; bail
    // before reaching the decoders, which need a parsed record.
    let Ok(parsed) = parse_quantization_table_sets(record_bytes) else {
        return;
    };
    let ec = parsed.record.ec.is_some();

    // Drive both v3 pipelines with the same attacker-chosen record +
    // frame bytes. Success and every typed error are acceptable; only a
    // panic is a finding.
    let _ = decode_frame(
        frame_bytes,
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        ec,
    );
    let _ = decode_frame_rgb(
        frame_bytes,
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        ec,
    );
});
