# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Configuration Record parser per RFC 9043 §4.2 / §4.3:
  - `Ffv1ConfigurationRecord` struct exposing `version`,
    `micro_version`, `coder_type`, `state_transition_delta`,
    `colorspace_type`, `bits_per_raw_sample`, `chroma_planes`,
    `log2_h_chroma_subsample`, `log2_v_chroma_subsample`,
    `extra_plane`, and (for v3) `num_h_slices` / `num_v_slices` /
    `quant_table_set_count`.
  - Binary range decoder (RFC 9043 §3.8.1.1, Closed mode) with the
    default state-transition table (§3.8.1.5).
  - Scalar symbol decoder (`get_ur` / `get_sr` / `get_br`) per
    Figure 21.
  - `Ffv1Version`, `ColorspaceType`, and `PictureStructure` enums
    (the last is exported for future slice-header parsers; not
    decoded in this round).
- Unit tests (10 total): 5 range-coder primitives + 5 fixture-based
  Configuration Record tests using the v3-default / v3-rgb-bgr0 /
  v3-grayscale extradata blobs from `docs/video/ffv1/fixtures/`.

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`. The crates.io version
  (`0.0.7`) is preserved on the new master to avoid breaking
  downstream version pins; the published versions on crates.io will
  be yanked by the maintainer.
- The `oxideav-avi` / `oxideav-mkv` dev-dependencies (used by the
  prior crate's roundtrip integration test) are dropped from the
  scaffold and will be re-introduced in a future round if needed.

### Next

- Quantization-table cascade decode (§4.1).
- Configuration Record CRC validation (§4.3.2).
- Slice header / content decode + Golomb-Rice mode (§3.8.2).
