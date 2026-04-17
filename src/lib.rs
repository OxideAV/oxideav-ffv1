//! Pure-Rust FFV1 (RFC 9043) lossless intra-only video codec.
//!
//! This crate implements version 3 of the FFV1 bitstream for 8-bit and
//! 10-bit YUV 4:2:0 / 4:2:2 / 4:4:4 sources. FFV1 is lossless: decoding
//! must reproduce the encoder's input samples exactly. The decoder can
//! consume data produced by a conforming encoder (including libavcodec);
//! the encoder produces single-slice frames which are understood by
//! conforming FFV1 decoders.
//!
//! For self-consistency, this crate reuses FFmpeg's 8-bit `quant11`-based
//! default quantisation tables at all supported bit depths. FFmpeg's own
//! encoder switches to a `quant9_10bit` table set for depths above 8, so
//! our 10-bit output is byte-identical to our own decoder but not to
//! FFmpeg-encoded 10-bit streams. Decoding FFmpeg-produced 10-bit streams
//! therefore requires the foreign extradata's quant tables to match ours;
//! a future change will read the tables from extradata rather than relying
//! on compile-time defaults.
//!
//! **Supported**:
//! - Version 3 bitstream only (v0/v1/v2 rejected at parse time).
//! - Range coder with the RFC default state transition table
//!   (`coder_type = 1`). Golomb-Rice (`coder_type = 0`) and custom
//!   transition tables (`coder_type = 2`) are rejected.
//! - 8-bit and 10-bit samples; YUV 4:2:0, 4:2:2 and 4:4:4.
//! - Decoder reads any `num_h_slices × num_v_slices` grid; slice CRC-32
//!   parity is verified when `ec != 0`. Encoder always emits a single
//!   slice covering the whole frame.
//! - Configuration record CRC is verified on parse and appended on emit.
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - 9/12/14/16-bit sample depths.
//! - RGB / JPEG 2000 RCT colorspace and alpha (`extra_plane`) channel.
//! - Multi-slice encoding (the decoder still accepts multi-slice input).
//! - Custom `initial_state_delta` quant-table-set overrides.
//! - Bayer / packed pixel formats.

#![allow(clippy::needless_range_loop)]

pub mod config;
pub mod crc;
pub mod decoder;
pub mod encoder;
pub mod predictor;
pub mod range_coder;
pub mod slice;
pub mod state;

use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecCapabilities, CodecId};

pub const CODEC_ID_STR: &str = "ffv1";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("ffv1_sw")
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_size(65535, 65535);
    reg.register_both(cid, caps, decoder::make_decoder, encoder::make_encoder);
}
