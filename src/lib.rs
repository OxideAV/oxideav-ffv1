//! Pure-Rust FFV1 (RFC 9043) lossless intra-only video codec.
//!
//! This crate implements version 3 of the FFV1 bitstream. FFV1 is lossless:
//! decoding must reproduce the encoder's input samples exactly. The
//! decoder consumes data produced by a conforming encoder (including
//! libavcodec); the encoder produces single-slice frames which FFmpeg's
//! decoder accepts bit-exactly.
//!
//! The configuration-record parser materialises every shipped quantisation
//! table set; per-slice `qt_idx[p]` values drive which set each plane uses
//! (this matches FFmpeg's default 10-bit YUV stream shape, where both
//! luma and chroma point at set 0 even though set 1 is coded).
//!
//! **Supported**:
//! - Version 3 bitstream only (v0/v1/v2 rejected at parse time).
//! - Range coder with the RFC default state transition table
//!   (`coder_type = 1`): encode and decode.
//! - Range coder with custom state transition table (`coder_type = 2`):
//!   decode on both YCbCr and RCT paths (ffmpeg emits this shape by
//!   default for 10-bit YUV and 8-bit RGB).
//! - Golomb-Rice coder (`coder_type = 0`): **decode only**, 8-bit samples,
//!   no alpha.
//! - 8..=16 bit samples on the range-coder path; YUV 4:2:0, 4:2:2 and
//!   4:4:4.
//! - `extra_plane` alpha channel on YCbCr (8-bit `Yuva420P`) and RCT
//!   (packed `Rgba` for 8-bit, `Rgba64Le` for 9..=16-bit).
//! - 8-bit RGB decode via the JPEG 2000 Reversible Colour Transform
//!   (`colorspace_type = 1`, wire plane order Y/Cb/Cr = G/B/R; decoded to
//!   packed `Rgb24`).
//! - 9..=16 bit RGB decode via RCT with the RFC §3.7.2.1 "BGR Exception"
//!   (9..=15 bit `extra_plane == 0` streams have Y = blue, Cb = g − b,
//!   Cr = r − b). Decoded to packed `Rgb48Le`.
//! - Decoder reads any `num_h_slices × num_v_slices` grid; slice CRC-32
//!   parity is verified when `ec != 0`. Encoder emits a single slice by
//!   default and accepts a `slices=N` option (see
//!   [`encoder::Ffv1EncoderOptions`]) to split the frame across a grid.
//! - 8-bit and 10-bit YUV 4:2:0 / 4:2:2 / 4:4:4 encode.
//! - Configuration record CRC is verified on parse and appended on emit.
//! - Per-slice `qt_idx` values are consulted to pick quant-table sets.
//! - Our encoder's output (single- and multi-slice, 8-bit and 10-bit)
//!   decodes bit-exactly in FFmpeg (verified via the `ffmpeg_decodes_*`
//!   tests in `tests/ffmpeg_interop.rs`).
//!
//! **Not supported** (will return `Error::Unsupported`):
//! - Golomb-Rice **encode** (we still emit range-coded on the encoder path).
//! - Golomb-Rice decode with `bits_per_raw_sample > 8` or with alpha.
//! - Cross-frame state retention for `intra=0` streams with non-keyframes
//!   (our decoder resets VLC state per packet).
//! - `initial_state_delta` (a.k.a. FFmpeg `-context 1`): the config-record
//!   parser materialises the per-context initial state matrix (RFC §4.2.15)
//!   and the slice decoders accept an override via
//!   [`slice::decode_frame_ex_with_states`] / `_u16_with_states` /
//!   `_rct_with_states`; end-to-end decode still diverges mid-plane against
//!   a real FFmpeg `-context 1` file — the remaining gap is tracked by the
//!   `#[ignore]`d `our_decoder_accepts_ffmpeg_context1` interop test.
//! - RGB **encode**; YUV encode beyond 10-bit (decode supports 9..=16 bit).
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
            .encoder_options::<encoder::Ffv1EncoderOptions>()
            .tag(CodecTag::fourcc(b"FFV1")),
    );
}
