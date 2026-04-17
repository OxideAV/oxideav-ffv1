//! Pure-Rust FFV1 (RFC 9043) lossless intra-only video codec.
//!
//! This crate implements version 3 of the FFV1 bitstream for 8-bit and
//! 10-bit YUV 4:2:0 / 4:2:2 / 4:4:4 sources (the 4:2:2 variant currently
//! requires the 10-bit pixel format). FFV1 is lossless: decoding must
//! reproduce the encoder's input samples exactly. The decoder can consume
//! data produced by a conforming encoder (including libavcodec); the
//! encoder produces single-slice frames which are understood by conforming
//! FFV1 decoders.
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
//! **Not supported** (will return `Error::Unsupported`):
//! - 12/14/16-bit sample depths
//! - RGB / JPEG 2000 RCT colorspace
//! - Multi-threaded / multi-slice encoding (decoder accepts any slice count)
//! - Error-correction CRC verification (the ec flag is parsed and slice
//!   footers are skipped, but CRC parity bytes are ignored)
//! - Bayer / packed pixel formats

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
