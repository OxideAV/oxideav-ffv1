# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-row `sample_difference` decode (RFC 9043 §4.8 + §3.8.2) — the
  round-4 deliverable wires the §4.8 `Line( p, y )` body to the
  Golomb-Rice path:
  - `BitReader` — MSB-first `get_bits(n)` over a byte slice
    (RFC 9043 §2.2.9.4). Reads past the buffer end inject zero bits
    per §3.8.2's "padded with zeroes" rule.
  - `get_ur_golomb_esc(k, bits)` + `get_sr_golomb_esc(k, bits)` —
    unsigned/signed Golomb-Rice VLC per Figures 26 / 27 with ESC mode
    per §3.8.2.1.1 (12 zero prefix → flat `bits`-wide suffix + 11).
    The `get_ur_golomb(k)` form (no ESC width) is kept for stand-alone
    prefix-table tests.
  - `sign_extend(input_number, input_bits)` per §3.8.2.3.
  - `VlcState { drift, error_sum, bias, count }` + `VLC_STATE_INITIAL`
    (`{0, 4, 0, 1}`) per §3.8.2.5; `get_vlc_symbol` implements the
    full §3.8.2.4 scalar-mode pseudocode including adaptive `k`
    estimation, error/drift accumulation, count rescale at 128, and
    bias nudging.
  - `get_vlc_symbol_level` — the §3.8.2.4.1 level-coding variant that
    skips the zero value (used for the first post-run-break sample).
  - `LOG2_RUN` — the 41-entry run-mode log2-of-suffix-width table
    (§3.8.2.2.1).
  - `median_predict(l, t, tl)` per §3.3.
  - `NeighborSamples { tt, ll, t, tl, tr, l }` + `QuantTableSet` (5
    sub-tables of 256 i32 entries each per §3.4) + `raw_context` /
    `absolute_context` per §3.5. `AbsoluteContext.sign_flip` flags
    the §3.5 negative-context case so the caller knows to flip the
    decoded `sample_difference`'s sign.
  - `LineDecoderState { vlc: Vec<VlcState>, run_index, run_mode,
    run_count }` carries the per-Plane decoder state; `new(N)`
    initialises N VLC slots from `VLC_STATE_INITIAL`.
  - `LineNeighborBuffers { prev_row, prev_prev_row, current_row,
    plane_pixel_width }` is the view onto the §3.1 bordered sample
    grid the per-row decoder reads from / writes back into.
  - `decode_line(br, state, qtable, neighbours, bits) -> Vec<i32>`
    decodes one Line of sample_difference values: per pixel, computes
    §3.5 absolute context, dispatches to run-mode or scalar VLC per
    §3.8.2.2 / §3.8.2.4, applies the §3.5 sign-flip, writes the
    decoded value back into `current_row` so subsequent neighbour
    lookups see it. **Pixel reconstruction is NOT performed** — the
    returned values are decoded sample_difference symbols, not
    Sample values. The median-predict + modular-wrap reconstruction
    is queued for a later round.
  - 42 new tests (80 total, was 38):
    - 6 `bit_reader::tests` (MSB-first read, cross-byte, past-end
      zero, `bits_left`, 32-bit wide read).
    - 16 `golomb_rice::tests` (`sign_extend` at 0/8-bit, `ur_golomb`
      at k=0/k=2 against RFC §3.8.2.1.3 Table 3 entries, `sr_golomb`
      interleave mapping, ESC mode reading value 139 from the
      `0000_0000_0000 1000_0000` table-3 bit pattern, VLC state
      initialisation, `get_vlc_symbol` zero-decode, `get_vlc_symbol_level`
      zero-skip, `LOG2_RUN` length / monotonicity / last-entry).
    - 9 `predictor::tests` (median predictor on flat / gradient /
      sharp-edge configurations, `median_of_three` precondition,
      zero-quant-table context, identity-quant-table context, signed
      context with §3.5 sign-flip).
    - 6 `sample_diff::tests` (state init, run-state reset preserves
      VLC, decode emits correct row count, scalar path on nonzero
      context, row writeback persists).
    - 5 `tests/golomb_rice_decode.rs` integration tests driving
      `decode_line` end-to-end through hand-crafted bit streams
      against synthetic quant tables.

- Slice Content scaffold per RFC 9043 §4.7 / §4.8:
  - `FramePixelDimensions` value type for caller-supplied frame
    width / height (FFV1's Configuration Record carries neither
    field; the container reports them).
  - `SliceContent { planes: Vec<Plane>, slice_pixel_x/y/width/height,
    traversal }` materializing §4.7.1 `primary_color_count` planes,
    each pre-sized with `Plane.lines: Vec<Line>` identity-only
    placeholders so a future pixel-decode round can fill them.
  - `Plane` carries `plane_index` + `width` / `height` per §4.7.2 /
    §4.8.1 (chroma planes subsampled via
    `log2_h_chroma_subsample` / `log2_v_chroma_subsample`).
  - `PlaneTraversal::{PlaneMajor, LineMajor}` typed mirror of the
    §4.7 `colorspace_type == 0` vs `colorspace_type == 1` interleave;
    `SliceContent::line_visits()` enumerates `(plane_index, y)`
    pairs in the order the §4.7 pseudocode mandates.
  - `compute_slice_content(header, cr, frame)` does raster-bounds
    checks (`slice_x + slice_width <= num_h_slices` etc.), rejects
    zero frame dimensions, and rejects v0/v1 records (which have no
    Configuration-Record-level slice grid).
  - New `Error` variants: `SliceRequiresVersion3`,
    `InvalidFramePixelDimensions { width, height }`,
    `SliceRasterOutOfRange { slice_x, slice_y, slice_width,
    slice_height, num_h_slices, num_v_slices }`.
- 14 new tests (40 total, was 22):
  - 10 unit tests in `slice_content::tests` covering
    `primary_color_count` for {Y, YA, YUV, YUVA, RGB, RGBA},
    `plane_pixel_width` / `plane_pixel_height` at no-subsample / 4:2:0
    / odd-width-rounds-up, the v3-default 2×2 raster pixel rectangle,
    `for_colorspace` traversal mapping, zero-dimension rejection,
    off-raster rejection, v0/v1 rejection, YUV 4:2:0 plane shape +
    `line_visits()` output, RGB line-major interleave, grayscale
    single-plane shape, YUVA extra-plane append, and per-line identity
    invariant.
  - 4 fixture-based integration tests in
    `tests/fixture_slice_content.rs` against extracted slice bytes
    + the trace ground truth from `docs/video/ffv1/fixtures/v3-default/`,
    `/v3-grayscale/`, and `/v3-rgb-bgr0/` — per-plane pixel dimensions
    match the reference decoder's PLANE events bit-exactly, and the
    4-slice v3-default tiling test confirms the §5 "every position
    filled by exactly one Slice" restriction is honoured (12288 luma
    pixels + 6144 chroma pixels = the full 128×96 frame).
- Spec gap noted: RFC 9043 §4.7.3 reads
  `slice_pixel_height = floor((slice_y + slice_height) * slice_pixel_height / num_v_slices) - slice_pixel_y`,
  with `slice_pixel_height` on its own RHS. This is a documentation
  typo (the §4.8.2 sibling formula uses `frame_pixel_width`); we use
  the unambiguous `frame_pixel_height` reading and three fixture
  trace files confirm the per-plane pixel dimensions match bit-exactly.

- Slice Header parser per RFC 9043 §4.6:
  - `Ffv1SliceHeader` struct exposing `slice_x`, `slice_y`,
    `slice_width` (raster), `slice_height` (raster),
    `quant_table_set_index_count` + `quant_table_set_index[..]`,
    `picture_structure` (typed) + `picture_structure_raw`,
    `sar_num`, `sar_den`, plus `sar_is_known()` / `quant_table_indices()`
    helpers and a `MAX_QUANT_TABLE_SET_INDEXES` constant.
  - `parse_slice_header(slice_bytes, &Ffv1ConfigurationRecord)`
    composes with the round-1 range coder + `ur` symbol decoder.
  - Confirms the §4.6 "Slice Header has its own initial states"
    ambiguity resolves the same way as §4.2 Parameters: a single
    shared 32-slot context window. All 6 slice-header fixtures
    decode bit-correctly under that hypothesis.
- 7 fixture-based integration tests against extracted slice bytes
  from `docs/video/ffv1/fixtures/v3-default/` (all 4 slices),
  `/v3-grayscale/` (slice 0, validates `chroma_planes=0` with
  `quant_table_set_index_count=2` via §4.6.5 `version<=3`), and
  `/v3-rgb-bgr0/` (slice 0, RGB/RCT path). Slice bytes were extracted
  via a black-box `ffmpeg -c copy -f rawvideo` invocation + the
  trailer-pointer chain walk per §4.9.1.
- 5 unit tests covering `quant_table_set_index_count` arithmetic and
  truncated-slice rejection.

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
- Per-line `sample_difference` decode (§4.8) — round 3 scaffolds the
  per-line container; round 4 fills each `Plane.lines[y]` with the
  decoded sample-difference row.
- Slice Footer parsing (§4.9) + Golomb-Rice mode (§3.8.2).
