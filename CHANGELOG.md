# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.8](https://github.com/OxideAV/oxideav-ffv1/compare/v0.0.7...v0.0.8) - 2026-05-17

### Other

- 12-bit YUV (Yuv{420,422,444}P12Le), range + Golomb, ffmpeg bit-exact
- range-coded YUVA (extra_plane alpha) on coder_type=1

### Added

- 12-bit YUV encode (`Yuv420P12Le`, `Yuv422P12Le`, `Yuv444P12Le`) on both
  the range-coder and Golomb-Rice paths, single-slice and multi-slice.
  FFmpeg's reference decoder accepts the output bit-exactly across all
  three chroma subsamplings (verified by the new
  `ffmpeg_decodes_our_yuv420p12le_output`,
  `ffmpeg_decodes_our_yuv420p12le_multi_slice_output` and
  `ffmpeg_decodes_our_yuv444p12le_output` interop tests). The
  config-record parser now accepts `bits_per_raw_sample ∈ 8..=16` per
  RFC 9043 §3.8 (was: 8 or 10 only), and the decoder accepts the same
  range for YUV input — the new `our_decoder_accepts_ffmpeg_yuv420p12le`
  test reads an FFmpeg-produced `yuv420p12le` stream (which ships
  FFmpeg's bespoke 12-bit quant table set) and reproduces every plane
  byte-for-byte. On a 720x480 gradient fixture the new path compresses
  the 1,036,800-byte raw frame down to 17,970 bytes (1.7% of raw).
  Closes the "YUV encode beyond 10-bit" gap documented in the README.

- Range-coded YUVA encode (`coder_type = 1`, `Yuva420P`, `extra_plane`):
  both single-slice and multi-slice. The Golomb-Rice path was the only
  alpha encoder before this round; on a 720p photographic+alpha fixture
  the new range-coded path is ~6.7% smaller than Golomb-Rice. FFmpeg's
  reference decoder accepts the output bit-exactly (verified by the new
  `ffmpeg_decodes_our_range_coded_yuva420p[_multislice]` interop tests).
  Closes a documented "Not supported" gap.

## [0.0.7](https://github.com/OxideAV/oxideav-ffv1/compare/v0.0.6...v0.0.7) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-ffv1/pull/502))

## [0.0.6](https://github.com/OxideAV/oxideav-ffv1/compare/v0.0.5...v0.0.6) - 2026-05-03

### Other

- cargo fmt the import block
- silence rust-1.95 clippy lints
- rustfmt docs_corpus.rs
- integrate docs/video/ffv1 fixture corpus

## [0.0.5](https://github.com/OxideAV/oxideav-ffv1/compare/v0.0.4...v0.0.5) - 2026-05-03

### Other

- silence too_many_arguments on encode_frame
- cargo fmt: pending rustfmt cleanup
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

## [0.0.4](https://github.com/OxideAV/oxideav-ffv1/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- Golomb-Rice 10-bit + alpha encode/decode, ffmpeg bit-exact
- Golomb-Rice encode path (coder_type=0), ffmpeg bit-exact
- cross-frame state retention, 10-bit Golomb-Rice decode, RGB encode
- cargo fmt
- multi-slice encode + initial_state_delta parsing
- 10-bit YUV/RGB decode, extra_plane alpha, BGR exception
- decode RGB via JPEG 2000 RCT (colorspace_type=1)
- decode Golomb-Rice coder_type=0 (8-bit YCbCr)
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
- claim AVI FourCC via oxideav-codec CodecTag registry
