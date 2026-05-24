# oxideav-ffv1

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

Clean-room rebuild, round 6 (2026-05-24). The prior implementation was
retired on 2026-05-18 under the workspace clean-room policy.

Round 1 landed the **Configuration Record parser** plus its
range-coder dependencies; round 2 added the **Slice Header parser**
(RFC 9043 §4.6); round 3 added the **Slice Content scaffold**
(§4.7 / §4.8); round 4 wired the §4.8 `Line( p, y )` body to the
**§3.8.2 Golomb-Rice decode path** (MSB-first bit reader,
unsigned/signed Golomb-Rice VLC + ESC mode, per-context adaptive
VLC state, `log2_run` run-mode primitives, the §3.3 median predictor,
and the §3.5 context computation including the negative-context
sign-flip); round 5 adds the **§4.1 Quantization Table Set cascade
decode** — the run-length `len - 1` decode (§3.8.1.2 method), the
`scale *= 2*len_count - 1` accumulation, the
`context_count = ceil(scale/2)` derivation, and the symmetric
sign-flipped second-half reflection. The new
`parse_quantization_table_sets` API decodes the whole §4.2 Parameters
block *and* its embedded §4.1 cascade from one extradata blob,
producing `quant_table_set_count` ready-to-use `QuantTableSet`s — so
the round-4 `decode_line` no longer needs a caller-supplied table.
Round 6 adds the **§4.3.2 Configuration Record CRC** check
(`validate_configuration_record_crc`): the §4.9.3 generator (poly
`0x104C11DB7`, init 0, no inversion) run over the whole extradata blob
must leave a remainder of zero, matching every fixture's
`trace.txt` `crcref=0x00000000`. Round 7 adds the **§4.9 Slice Footer
parser** (`parse_slice_footer`): it reads `slice_size` (§4.9.1),
`error_status` (§4.9.2), and `slice_crc_parity` (§4.9.3) from the
trailing 8 bytes (`ec=1`) or 3 bytes (`ec=0`) of a Slice, cross-checks
the size field against the buffer length, and validates the §4.9.3
whole-Slice CRC residue is zero — reproducing every fixture's
`trace.txt` `SLICE` `header_crc` parity bit-exactly.

Implemented (RFC 9043 §3.3 / §3.5 / §3.8.1.1 / §3.8.1.2 / §3.8.2 /
§4.1 / §4.2 / §4.3 / §4.3.2 / §4.6 / §4.7 / §4.8 / §4.9 / §4.9.3):

- Binary range decoder (Closed mode), default state-transition table.
- Scalar symbol decoder (`ur` / `sr` / `br`) per Figure 21.
- MSB-first bit reader for the Golomb-Rice path (§2.2.9.4 /
  §3.8.2).
- Golomb-Rice VLC: `get_ur_golomb_esc(k, bits)` and
  `get_sr_golomb_esc(k, bits)` per Figures 26 / 27 + §3.8.2.1.1 ESC
  mode; `sign_extend` per §3.8.2.3.
- Adaptive VLC state (`VlcState { drift, error_sum, bias, count }`)
  + `get_vlc_symbol` / `get_vlc_symbol_level` per §3.8.2.4 / §3.8.2.4.1.
- `LOG2_RUN` run-mode table per §3.8.2.2.1.
- Median predictor `median(l, t, l+t-tl)` per §3.3.
- §3.5 context computation: 5-subtable quant-table lookup, sum, and
  absolute-value mapping with sign-flip flag.
- Per-Line `sample_difference` decode (`decode_line`) — drives the
  §4.8 `Line( p, y )` pseudocode over a caller-supplied
  `QuantTableSet`, emitting a `Vec<i32>` of decoded sample
  differences per row.
- Configuration Record fields: `version`, `micro_version`,
  `coder_type`, `state_transition_delta`, `colorspace_type`,
  `bits_per_raw_sample`, `chroma_planes`, `log2_h_chroma_subsample`,
  `log2_v_chroma_subsample`, `extra_plane`, `num_h_slices`,
  `num_v_slices`, `quant_table_set_count`.
- Slice Header fields: `slice_x`, `slice_y`, `slice_width` (raster),
  `slice_height` (raster), `quant_table_set_index[..]`,
  `picture_structure`, `sar_num`, `sar_den`.
- Slice Content scaffold (§4.7): `primary_color_count` (§4.7.1),
  `slice_pixel_x` / `slice_pixel_y` / `slice_pixel_width` /
  `slice_pixel_height`, per-plane `plane_pixel_width` /
  `plane_pixel_height` with 4:2:0 / 4:2:2 / 4:4:4 chroma subsampling,
  and the §4.7 plane-then-line (YCbCr) vs. line-then-plane (RGB)
  traversal order as a typed `PlaneTraversal` + `line_visits()`
  enumerator.
- Quantization Table Set cascade (§4.1): `parse_quantization_table_sets`
  decodes the §4.2 Parameters block plus its embedded
  `QuantizationTableSet( i )` cascade from one extradata blob,
  emitting `quant_table_set_count` `QuantizationTableSet`s. Each is a
  ready-to-use `QuantTableSet` (5 × 256 signed entries) plus its
  `context_count` (`ceil(scale/2)`, §4.1.2). Per-context state resets
  to 128 at the start of each of the five `QuantizationTable`s; the
  arithmetic coder continues in the Parameters bitstream.
- Configuration Record CRC validation (§4.3.2):
  `validate_configuration_record_crc(extradata)` runs the §4.9.3 CRC
  (poly `0x104C11DB7`, init 0, MSB-first, no pre/post-inversion) over
  the whole extradata blob and requires a zero residue. Reuses an
  internal `ffv1_crc32` that the §4.9.3 Slice Footer CRC shares.
  Returns `ConfigurationRecordCrcMismatch(residue)` on a non-zero
  residue.
- Slice Footer parsing (§4.9): `parse_slice_footer(full_slice, ec)`
  reads `slice_size` (§4.9.1), `error_status` (§4.9.2, typed
  `SliceErrorStatus` per Table 16), and `slice_crc_parity` (§4.9.3)
  from the trailing 8 bytes (`ec=1`) / 3 bytes (`ec=0`) of a Slice.
  Cross-checks `slice_size == buffer_len - footer_len`
  (`SliceSizeOutOfRange`) and, for `ec=1`, validates the §4.9.3
  whole-Slice CRC residue is zero via the shared `ffv1_crc32`
  (`SliceCrcMismatch { residue, stored_parity }`). The whole-Slice
  byte range is what the §4.9.1 trailer-pointer chain walk yields.

Not yet implemented:

- `states_coded` / `initial_state_delta` / `ec` / `intra` (the v3 tail
  of Parameters) — **blocked** on a §4.2.14 loop-count discrepancy; see
  Notes for future rounds (#904 DOCS-GAP).
- Range non-binary mode for slice data (the *Range Coding* alternative
  to the round-4 Golomb-Rice path; uses the same context model but
  routes through `get_symbol` in `symbol.rs`).
- Pixel reconstruction (median predict + modular wrap recovery of the
  Sample from the decoded sample_difference).
- RCT colorspace post-transform.
- Encoder.

Until those land, the public `Decoder` / `Encoder` traits return
`Error::NotImplemented` and no codec is registered into the runtime.

## Verification

The parsers are validated against the workspace's black-box FFV1
fixture corpus under `docs/video/ffv1/fixtures/`:

- Configuration Records are the Matroska CodecPrivate bytes of each
  fixture's `input.mkv`, checked against the `trace.txt`
  `GLOBAL_HEADER` event.
- Slice Headers are the leading bytes of each slice's range-coded
  region — located by walking the trailer-pointer chain backwards
  from the end of the raw FFV1 frame (extracted via a black-box
  `ffmpeg -c copy -f rawvideo` invocation). Expected `slice_x` /
  `slice_y` come straight from the `trace.txt` `SLICE` events.
- Quantization Table Set cascades are checked against the `trace.txt`
  `QUANT_TABLE` events' `context_count` field. `context_count` is a
  function of every `len_count` across all five sub-tables, so a
  single off-by-one anywhere in the run-length stream desynchronises
  it — making it a tight bit-exactness check on the whole cascade.
- Configuration Record CRCs are checked against the `trace.txt`
  `GLOBAL_HEADER` event's `crcref` field (`0x00000000` for every
  fixture): the §4.9.3 generator run over the whole extradata blob
  reproduces the reference decoder's zero residue. A clean-room CRC
  that hits the same `0` over the same bytes has the polynomial
  orientation and the no-inversion convention exactly right.
- Slice Footers are checked against each slice's `trace.txt` `SLICE`
  event: the parsed `slice_size` matches the trailer-chain-walked
  body length (and the trace's `len` minus the 8-byte footer), and the
  §4.9.3 whole-Slice CRC residue is zero — which is equivalent to the
  parsed `slice_crc_parity` reproducing the trace's `header_crc`
  bit-exactly (the encoder solved the parity word for the zero
  residue). The whole-Slice byte ranges are extracted black-box via
  `ffmpeg -c copy -f rawvideo` + the §4.9.1 trailer-pointer chain
  walk.

| Fixture | Round 1 (cfg record) | Round 2 (slice header) | Round 3 (slice content) |
| --- | --- | --- | --- |
| `v3-default` | v3 / 8-bit YUV 4:2:0 / range coder default / 2x2 slices | all 4 slices: raster cells (0,0)(1,0)(0,1)(1,1) | slice 0: 64x48 Y + 32x24 U + 32x24 V (matches trace PLANE), 4-slice tiling exhausts 128x96 |
| `v3-rgb-bgr0` | v3 / RGB (RCT) / chroma_planes=1 / no subsample | slice 0 (chroma_planes=1 / RCT path) | slice 0: 3 planes × 32x24, line-major traversal per §4.7 |
| `v3-grayscale` | v3 / single-plane / chroma_planes=0 | slice 0 (chroma_planes=0, count=2 via version<=3) | slice 0: single 32x24 plane |

Round 4's Golomb-Rice decode primitives are validated by 42 in-tree
tests including the §3.8.2.1.3 Table 3 examples (k=0/2 unary / suffix
decode, ESC mode reading the `0000_0000_0000 1000_0000` byte pattern
as value 139), the §3.3 / §3.5 predictor and context calculations,
and 5 integration tests driving `decode_line` end-to-end with
synthetic quant tables and hand-crafted bit streams.

Round 5's §4.1 quant-table cascade adds 10 tests (90 total, was 80):
3 unit tests (sign-reflection mirror, `ceil(scale/2)`,
`MAX_CONTEXT_INPUTS == 5`) + 7 fixture tests reproducing the
`QUANT_TABLE` `context_count` ground truth bit-exactly —
`v3-default` / `v3-grayscale` / `v3-rgb-bgr0` decode to 666 / 7563
(8-bit, sub-table `len_count` cascades `{6,6,6,1,1}` / `{6,6,3,3,3}`)
and `v3-yuv444p16` to 365 / 5063 (16-bit, `{5,5,5,1,1}` /
`{5,5,3,3,3}`), plus the §4.1 second-half sign-flip invariant and
truncated-input rejection.

Round 6's §4.3.2 Configuration Record CRC adds 12 tests (102 total,
was 90): 6 unit tests in `crc::tests` (all-zero input → 0, valid-record
residue 0, corrupted-payload + corrupted-parity rejection, too-short
rejection, single-byte known-answers `0x80 → 0x690CE0EE` /
`0xFF → 0xB1F740B4`) and 6 fixture tests reproducing `crcref=0x00000000`
for `v3-default` / `v3-grayscale` / `v3-rgb-bgr0` / `v3-yuv444p16` plus
flipped-byte and truncated-parity rejection.

Round 7's §4.9 Slice Footer parser adds 21 tests (123 total, was 102):
9 unit tests in `slice_footer::tests` (ec=0 size-zero / one-byte body
round trips, ec=0 + ec=1 size-mismatch rejection, ec=0 + ec=1
truncated rejection, a solved-parity ec=1 round trip with whole-Slice
residue 0, a corrupted-body rejection surfacing residue + stored
parity, and the §4.9.2 Table 16 `error_status` mapping) + 12 fixture
tests in `tests/fixture_slice_footer.rs` reproducing the `trace.txt`
`SLICE` `header_crc` parity bit-exactly for all four `v3-default`
slices (`0xCB530827` / `0xC93079C7` / `0xB8923B4F` / `0x42C8841D`) and
slice 0 of `v3-grayscale` (`0x44C7D58E`) / `v3-rgb-bgr0` (`0x3BBFE098`)
/ `v3-yuv444p16` (`0x0AD980DC`) — each whole-Slice CRC residue is 0 —
plus corrupted-body, corrupted-parity, truncated-footer, and
wrong-`ec`-flag rejection.

## Notes for future rounds

- **DOCS-GAP (#904, §4.2.14 `states_coded` loop count).** Round 6
  attempted the §4.2.14 / §4.2.15 / §4.2.16 / §4.2.17 Parameters tail
  (`states_coded` / `initial_state_delta` / `ec` / `intra`) but found a
  contradiction between Figure 28 and the corpus. Figure 28 reads
  `states_coded` once **per Quantization Table Set** — two `br` symbols
  for these `quant_table_set_count == 2` fixtures. Yet the trailing
  `ec` / `intra` (`ec=1 intra=0` in every `GLOBAL_HEADER`) are
  reproducible bit-exactly across **all four** v3 fixtures (8-bit and
  16-bit, three colorspaces) only when **exactly one** `states_coded`
  `br` is consumed before `ec` (after resetting the Parameters state
  window to 128 post-cascade). Every two-`br` model — residual state,
  whole-buffer reset, per-`br` reset, distinct per-set slots — desyncs
  the range coder and corrupts `ec` / `intra` for at least one fixture.
  This is the §4.2/§4.3 Parameters context-buffer-width ambiguity (the
  per-set loop semantics + reset granularity). Figures 29 / 30 are
  otherwise clear (`pred = j ? initial_states[i][j-1][k] : 128`;
  `initial_state[i][j][k] = (pred + initial_state_delta[i][j][k]) & 255`;
  `CONTEXT_SIZE = 32`; "k as context index"), but no fixture exercises
  `states_coded == 1` to validate the per-`k` window placement. Recommend
  a §4.2.14 clarification of how many `states_coded` symbols are coded
  for `quant_table_set_count > 1` and the exact state-window/reset model
  for the §4.2 tail, plus (ideally) one fixture with `states_coded == 1`.
- RFC 9043 §4.2 says "Parameters has its own initial states, all set
  to 128" without specifying the state-buffer width. Empirically, **all
  Parameters symbols share a single 32-slot context window**: the test
  fixtures decode correctly only with that interpretation. §4.6 has
  the same wording for the Slice Header — round 2 confirmed the
  shared-window hypothesis holds there too (all 6 slice-header
  fixtures decode bit-correctly with a single 32-slot window).
- §4.1's QuantizationTableSet carries the SAME "has its own initial
  states, all set to 128" wording (the #904 context-buffer-width
  ambiguity). Round 5 resolved its reset granularity empirically: the
  per-context state window resets to 128 at the start of **each of the
  five `QuantizationTable`s** (not once per Set, not shared with the
  Parameters prefix), while the arithmetic coder (low / range / byte
  position) continues uninterrupted. This is the *only* interpretation
  under which the `v3-default` (666 / 7563) and `v3-yuv444p16`
  (365 / 5063) `QUANT_TABLE` `context_count` trace values reproduce
  bit-exactly.
- The Configuration Record's last 4 bytes are
  `configuration_record_crc_parity`; the range decoder is in Closed
  mode and reads past-end as zero, so passing the full extradata blob
  (including those 4 bytes) is safe — the early Parameters symbols
  never reach them. Round 6's `validate_configuration_record_crc`
  consumes those bytes explicitly: §4.3.2 makes the whole-blob CRC
  residue (including the parity word) the canonical fixity check, so the
  validator never re-derives the stored parity — it just asserts the
  §4.9.3 generator leaves a remainder of zero.
- `parse_slice_header` takes the slice's range-coded byte region
  excluding its 8-byte SliceFooter (`ec=1`) or 3-byte footer
  (`ec=0`). It returns a typed [`Ffv1SliceHeader`] with the raster
  position and size (post-`+1`), the per-plane
  `quant_table_set_index` array, the picture structure, and SAR.
  The range coder's residual state is *not* exposed — slice content
  decode is a later round.
- `compute_slice_content` consumes a parsed `Ffv1SliceHeader` + the
  Configuration Record + a caller-supplied `FramePixelDimensions`
  (the container's reported frame width / height — FFV1's
  Configuration Record carries no width / height fields per §4.2)
  and returns a `SliceContent` with one `Plane` per
  `primary_color_count` entry. RFC 9043 §4.7.3 has a documentation
  typo: its right-hand side reads `slice_pixel_height` where it
  should read `frame_pixel_height` (the §4.8.2 sibling formula uses
  `frame_pixel_width`). We use the unambiguous reading and the three
  fixture trace files confirm the per-plane pixel dimensions match
  the reference decoder bit-exactly.

## License

MIT — see [LICENSE](./LICENSE).

[RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043.html
