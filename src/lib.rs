//! Pure-Rust FFV1 (RFC 9043) lossless intra-only video codec.
//!
//! This crate implements version 3 of the FFV1 bitstream for 8-bit and
//! 10-bit YUV 4:2:0 / 4:2:2 / 4:4:4 sources. FFV1 is lossless: decoding
//! must reproduce the encoder's input samples exactly. The decoder can
//! consume data produced by a conforming encoder (including libavcodec);
//! the encoder produces single-slice frames which FFmpeg's decoder accepts
//! bit-exactly.
//!
//! On reading a foreign configuration record, this crate materialises the
//! shipped quantisation tables and refuses streams whose table 0 diverges
//! from FFmpeg's default `quant11` — most notably FFmpeg's 10-bit
//! `quant9_10bit` set, which would otherwise silently produce wrong
//! samples. A decoder-side override that materialises the extradata's own
//! tables remains future work.
//!
//! **Supported**:
//! - Version 3 bitstream only (v0/v1/v2 rejected at parse time).
//! - Range coder with the RFC default state transition table
//!   (`coder_type = 1`): encode and decode.
//! - Golomb-Rice coder (`coder_type = 0`): **decode only**, 8-bit samples.
//!   Custom transition tables (`coder_type = 2`) are rejected.
//! - 8-bit and 10-bit samples; YUV 4:2:0, 4:2:2 and 4:4:4.
//! - Decoder reads any `num_h_slices × num_v_slices` grid; slice CRC-32
//!   parity is verified when `ec != 0`. Encoder always emits a single
//!   slice covering the whole frame.
//! - Configuration record CRC is verified on parse and appended on emit.
//! - Decoder accepts FFmpeg's default 2-set extradata; per-slice
//!   `qt_idx != 0` (FFmpeg's `-context 1`) is rejected.
//! - Our encoder's output decodes bit-exactly in FFmpeg (verified via
//!   `ffmpeg_decodes_our_encoder_output` in `tests/ffmpeg_interop.rs`).
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - Golomb-Rice **encode** (we still emit range-coded on the encoder path).
//! - Golomb-Rice decode with `bits_per_raw_sample > 8`.
//! - Cross-frame state retention for `intra=0` streams with non-keyframes
//!   (our decoder resets VLC state per packet).
//! - 9/12/14/16-bit sample depths.
//! - RGB / JPEG 2000 RCT colorspace and alpha (`extra_plane`) channel.
//! - Multi-slice encoding (the decoder still accepts multi-slice input).
//! - `-context 1` / `initial_state_delta` quant-table-set overrides.
//! - Non-default quantisation tables in the first table set.
//! - Bayer / packed pixel formats.

#![allow(clippy::needless_range_loop)]

pub mod config;
pub mod crc;
pub mod decoder;
pub mod encoder;
pub mod golomb;
pub mod predictor;
pub mod range_coder;
pub mod slice;
pub mod state;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

pub const CODEC_ID_STR: &str = "ffv1";

pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("ffv1_sw")
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_size(65535, 65535);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tag(CodecTag::fourcc(b"FFV1")),
    );
}
