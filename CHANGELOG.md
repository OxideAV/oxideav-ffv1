# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
