# oxideav-ffv1

A pure-Rust FFV1 (RFC 9043) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Orphan-rebuild scaffold (2026-05-18).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
the range-coder termination path's source comments reproduced an
external library's internal statement sequence verbatim (including
internal variable names) — clean-room provenance for that path
could not be defended. Master history was fully erased per the
Hat-3 cold-enforcement procedure.

The implementation will be re-built against RFC 9043 (FFV1 Video
Coding Format Version 0, 1, and 3) in a future clean-room round.

## License

MIT — see [LICENSE](./LICENSE).
