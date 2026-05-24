# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Per-plane pixel reconstruction for the Golomb-Rice path
  (RFC 9043 §3.1 / §3.3 / §3.5 / §3.8 / §4.8) — the round-8 deliverable:
  - `PlaneReconstructor::reconstruct_plane(br, qtable, context_count,
    width, height, bits)` decodes a full Plane's `sample_difference`
    stream and reconstructs Samples into a row-major `Vec<i32>` of
    length `width * height` (each entry in `0 .. 2^bits`). Unlike
    `decode_line` (which returns the raw `sample_difference` row), the
    reconstructor folds the §3.3 median predictor + §3.5 sign-flip into
    the per-pixel loop, because every Sample's §3.5 context and §3.3
    prediction depend on the **reconstructed** neighbours of the
    surrounding Samples, not on the raw differences.
  - `reconstruct_sample(pred, diff, bits)` implements the §3.8 modular
    add-back `(pred + diff) mod 2^bits` standalone — only the `n`
    (= `bits_per_raw_sample`) least-significant bits are coded, so the
    sum is reduced modulo `2^bits` and lands in `0 .. 2^bits`.
  - The §3.1 Slice border is maintained across the per-row decode:
    the column one left of the Slice is seeded from the previous row's
    first Sample (`sample[y][-1] = sample[y-1][0]`, `sample[0][-1] = 0`),
    the additional left column and the two rows above are zero, and the
    right border mirrors the rightmost Sample. `BORDER_LEFT` /
    `BORDER_RIGHT` expose the working-buffer pad widths.
  - Run mode (§3.8.2.2.1) and the per-context adaptive VLC state
    (§3.8.2.4 / §3.8.2.5) persist for the whole Plane; run mode resets
    per Plane, and the VLC contexts are keyframe-initialised.
  - The §3.3.1 16-bit median exception is deliberately NOT applied: its
    predicate requires the **range** coder (`coder_type == 1 || 2`),
    which the Golomb-Rice path excludes.
  - 16 new tests (12 module unit + 4 `reconstruct_plane.rs`
    integration), including a hand-traced byte-exact 2x2 scalar plane
    (`0x69 0x90` → `[3, 4, 5, 5]`) that exercises cross-row prediction
    chaining and the §3.1 left-of-slice border seed, plus a run-mode
    flat-zero plane and modular-wrap boundary cases at 8/10/16-bit.

- Slice Footer parser (RFC 9043 §4.9) — the round-7 deliverable:
  - `parse_slice_footer(full_slice_bytes, ec)` reads the §4.9
    `SliceFooter()`: `slice_size` (§4.9.1, `u(24)`), and — when `ec` is
    set — `error_status` (§4.9.2, `u(8)`) + `slice_crc_parity`
    (§4.9.3, `u(32)`). The footer is the trailing 8 bytes for `ec=1`
    or 3 bytes for `ec=0` (it is always byte-aligned per §4.9). The
    caller passes the *whole* Slice byte range (SliceHeader +
    SliceContent + Golomb-Rice padding + footer), typically obtained
    by walking the §4.9.1 trailer-pointer chain backwards from the end
    of the FFV1 frame.
  - `Ffv1SliceFooter { slice_size, total_size, error_status,
    error_status_raw, slice_crc_parity }` plus a `footer_len()` helper.
    `slice_size` is the footer-excluded body length; `total_size` is
    the on-wire length the caller supplied.
  - `SliceErrorStatus::{NoError, Correctable, Uncorrectable, Reserved}`
    typed mirror of §4.9.2 Table 16, with `from_wire` + an
    `is_uncorrectable()` helper. The raw wire byte is preserved on
    `error_status_raw` for `Reserved` diagnostics.
  - For `ec=1` the parser validates the §4.9.3 whole-Slice CRC: the
    same IEEE generator (poly `0x104C11DB7`, init 0, no inversion,
    MSB-first) as the §4.3.2 Configuration Record CRC — so it reuses
    the internal `crc::ffv1_crc32` — must leave a residue of zero over
    the entire Slice (footer included). RFC 9043 §4.9.3: "the Slice as
    a whole has a CRC remainder of 0."
  - Structural cross-check: the on-wire `slice_size` must equal
    `buffer_len - footer_len`; a mismatch (`SliceSizeOutOfRange`) is
    surfaced *before* the CRC check, so a mis-walked trailer chain or a
    wrong `ec` flag is diagnosed structurally rather than as a
    downstream CRC failure.
  - New `Error` variants: `TruncatedSliceFooter` (buffer shorter than
    the footer), `SliceSizeOutOfRange { field, expected }`, and
    `SliceCrcMismatch { residue, stored_parity }`.
  - `SLICE_FOOTER_LEN_EC0` (3) / `SLICE_FOOTER_LEN_EC1` (8) constants.
  - 21 new tests (123 total, was 102): 9 unit tests in
    `slice_footer::tests` (ec=0 size-zero / one-byte body, ec=0 + ec=1
    size-mismatch rejection, ec=0 + ec=1 truncated rejection, a
    solved-parity ec=1 round trip with residue 0, a corrupted-body
    rejection surfacing residue + stored parity, the §4.9.2 Table 16
    mapping) + 12 fixture tests in `tests/fixture_slice_footer.rs`
    reproducing the `trace.txt` `SLICE` `header_crc` parity bit-exactly
    for all four `v3-default` slices and slice 0 of `v3-grayscale` /
    `v3-rgb-bgr0` / `v3-yuv444p16` (residue 0 over each whole Slice),
    plus corrupted-body / corrupted-parity / truncated / wrong-`ec`
    rejection. The whole-Slice byte ranges were extracted black-box via
    `ffmpeg -c copy -f rawvideo` + the §4.9.1 trailer-pointer chain
    walk (`tests/data/slice_footer_fixtures.rs`).

- Configuration Record CRC validation (RFC 9043 §4.3.2) — the round-6
  deliverable:
  - `validate_configuration_record_crc(extradata)` runs the §4.9.3
    generator (IEEE polynomial `0x104C11DB7`, initial value 0, no
    pre-inversion, no post-inversion, MSB-first) over the *entire*
    extradata blob and asserts the residue is `0`. RFC 9043 §4.3.2:
    "configuration_record_crc_parity is 32 bits chosen so that the
    Configuration Record as a whole has a CRC remainder of zero." No
    table is used — the byte-at-a-time bit loop is exact and the record
    is tiny.
  - New internal `crc::ffv1_crc32` (crate-private, reused by the future
    §4.9.3 Slice Footer CRC) and the public
    `validate_configuration_record_crc` entry point.
  - New `Error` variant `ConfigurationRecordCrcMismatch(u32)` carrying
    the non-zero residue on failure; a too-short blob (< 4 parity
    bytes) returns `TruncatedRangeCoder`.
  - 12 new tests (102 total, was 90): 6 unit tests in `crc::tests`
    (all-zero input → 0, valid-record residue 0, corrupted-payload +
    corrupted-parity detection, too-short rejection, single-byte
    known-answers `0x80 → 0x690CE0EE` / `0xFF → 0xB1F740B4`) + 6
    fixture tests in `tests/fixture_config_crc.rs` confirming the four
    v3 fixtures (`v3-default` / `v3-grayscale` / `v3-rgb-bgr0` /
    `v3-yuv444p16`) all CRC to `0`, matching their `trace.txt`
    `GLOBAL_HEADER` `crcref=0x00000000`, plus flipped-byte and
    truncated-parity rejection.

- Quantization Table Set cascade decode (RFC 9043 §4.1) — the round-5
  deliverable, decoded from the same §4.2 Parameters range-coder
  stream:
  - `parse_quantization_table_sets(extradata)` walks the §4.2
    Parameters block (via the new internal `config::parse_parameters`
    helper, factored out of `parse_configuration_record`) and then
    continues into `for (i = 0; i < quant_table_set_count; i++)
    QuantizationTableSet( i )` (Figure 28). Returns a
    `ParametersWithQuantTables { record, quant_table_sets }`.
  - `QuantizationTableSet { tables: QuantTableSet, context_count }` —
    one parsed set: the five 256-entry signed tables (directly the
    `QuantTableSet` the §3.5 context computation indexes) plus
    `context_count = ceil(scale / 2)` (§4.1.2, bounded `<= 32768`).
  - `QuantizationTable(i, j, scale)` (§4.1): run-length `len - 1`
    (`ur`, §3.8.1.2 method) fills the first half with `scale * v`; the
    second half is the sign-flipped reflection
    (`table[256 - k] = -table[k]`, plus the dedicated
    `table[128] = -table[127]`). `scale *= 2 * len_count - 1` after
    each of the five sub-tables. `MAX_CONTEXT_INPUTS = 5`.
  - Reset granularity (the #904 context-buffer-width ambiguity): the
    per-context state window resets to 128 at the start of **each**
    `QuantizationTable`; the arithmetic coder (low / range / byte
    position) continues in the Parameters bitstream. This is the only
    interpretation reproducing the fixture `QUANT_TABLE`
    `context_count` trace values.
  - New `Error` variants: `InvalidQuantTableSetCount(u32)` (§4.2.13,
    `1..=8`), `QuantContextCountOutOfRange(u32)` (§4.1.2), and
    `MalformedQuantTable` (a run overruns the buffer or > 128 levels).
  - 10 new tests (90 total, was 80): 3 unit tests (sign-reflection
    mirror invariant, `ceil(scale/2)`, `MAX_CONTEXT_INPUTS == 5`) +
    7 fixture tests in `tests/fixture_quant_table.rs` reproducing the
    `QUANT_TABLE` `context_count` ground truth bit-exactly —
    `v3-default` / `v3-grayscale` / `v3-rgb-bgr0` → 666 / 7563 (8-bit;
    `len_count` cascades `{6,6,6,1,1}` / `{6,6,3,3,3}`) and
    `v3-yuv444p16` → 365 / 5063 (16-bit; `{5,5,5,1,1}` /
    `{5,5,3,3,3}`), plus the §4.1 second-half sign-flip invariant,
    the entry-0-is-zero invariant, and truncated-input rejection. The
    `v3-yuv444p16` extradata was extracted black-box via
    `ffprobe -show_data` on `input.mkv`.

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

- `initial_state_delta` / `ec` / `intra` — the v3 tail of Parameters
  (§4.2.14 / §4.2.15), decoded from the same stream right after the
  §4.1 cascade.
- Configuration Record CRC validation (§4.3.2).
- Wire the §4.1 parsed `QuantTableSet`s into `decode_line` (drop the
  caller-supplied table parameter) + the §3.6 plane-to-set selection.
- Pixel reconstruction (median predict + modular wrap recovery of the
  Sample from the decoded sample_difference).
- Slice Footer parsing (§4.9) + range non-binary slice-data mode.
