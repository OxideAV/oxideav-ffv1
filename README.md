# oxideav-ffv1

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

Clean-room rebuild, round 3 (2026-05-22). The prior implementation was
retired on 2026-05-18 under the workspace clean-room policy.

Round 1 landed the **Configuration Record parser** plus its
range-coder dependencies; round 2 added the **Slice Header parser**
(RFC 9043 §4.6); round 3 adds the **Slice Content scaffold**
(RFC 9043 §4.7 / §4.8) — typed `Plane` / `Line` / `SliceContent`
structs plus the `(plane, line)` iteration-order primitive, so a
downstream pixel-decode round can fill each `Plane.lines[y]` without
recomputing geometry or traversal.

Implemented (RFC 9043 §3.8.1.1 / §3.8.1.2 / §4.2 / §4.3 / §4.6 / §4.7 / §4.8):

- Binary range decoder (Closed mode), default state-transition table.
- Scalar symbol decoder (`ur` / `sr` / `br`) per Figure 21.
- Configuration Record fields: `version`, `micro_version`,
  `coder_type`, `state_transition_delta`, `colorspace_type`,
  `bits_per_raw_sample`, `chroma_planes`, `log2_h_chroma_subsample`,
  `log2_v_chroma_subsample`, `extra_plane`, `num_h_slices`,
  `num_v_slices`, `quant_table_set_count`.
- Slice Header fields: `slice_x`, `slice_y`, `slice_width` (raster),
  `slice_height` (raster), `quant_table_set_index[..]`,
  `picture_structure`, `sar_num`, `sar_den`. The
  `quant_table_set_index_count` (§4.6.5) is derived from
  `chroma_planes` / `extra_plane` / `version` on the configuration
  record handed in by the caller.
- Slice Content scaffold (§4.7): `primary_color_count` (§4.7.1),
  `slice_pixel_x` / `slice_pixel_y` (§4.7.4 / §4.8.3),
  `slice_pixel_width` / `slice_pixel_height` (§4.7.3 / §4.8.2),
  per-plane `plane_pixel_width` / `plane_pixel_height`
  (§4.7.2 / §4.8.1) with 4:2:0 / 4:2:2 / 4:4:4 chroma subsampling,
  and the §4.7 plane-then-line (YCbCr) vs. line-then-plane (RGB)
  traversal order as a typed `PlaneTraversal` + `line_visits()`
  enumerator.

Not yet implemented:

- Quantization-table cascade decode (§4.1).
- `initial_state_delta` / `ec` / `intra` (the v3 tail of Parameters).
- `configuration_record_crc_parity` validation (§4.3.2).
- `sample_difference` per-row decode (§4.8) — round 3 builds the
  per-line container, round 4 fills it.
- Slice Footer parsing (§4.9) — the trailer-pointer chain walk is
  trivial (last 3 bytes per slice) but no `Decoder` consumer exists
  yet to need it. Tests do the walk inline.
- Range non-binary mode for slice data, Golomb-Rice mode, RCT.
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

| Fixture | Round 1 (cfg record) | Round 2 (slice header) | Round 3 (slice content) |
| --- | --- | --- | --- |
| `v3-default` | v3 / 8-bit YUV 4:2:0 / range coder default / 2x2 slices | all 4 slices: raster cells (0,0)(1,0)(0,1)(1,1) | slice 0: 64x48 Y + 32x24 U + 32x24 V (matches trace PLANE), 4-slice tiling exhausts 128x96 |
| `v3-rgb-bgr0` | v3 / RGB (RCT) / chroma_planes=1 / no subsample | slice 0 (chroma_planes=1 / RCT path) | slice 0: 3 planes × 32x24, line-major traversal per §4.7 |
| `v3-grayscale` | v3 / single-plane / chroma_planes=0 | slice 0 (chroma_planes=0, count=2 via version<=3) | slice 0: single 32x24 plane |

## Notes for future rounds

- RFC 9043 §4.2 says "Parameters has its own initial states, all set
  to 128" without specifying the state-buffer width. Empirically, **all
  Parameters symbols share a single 32-slot context window**: the test
  fixtures decode correctly only with that interpretation. §4.6 has
  the same wording for the Slice Header — round 2 confirmed the
  shared-window hypothesis holds there too (all 6 slice-header
  fixtures decode bit-correctly with a single 32-slot window).
  QuantizationTableSet has the same ambiguity and is still untested.
- The Configuration Record's last 4 bytes are
  `configuration_record_crc_parity`; the range decoder is in Closed
  mode and reads past-end as zero, so passing the full extradata blob
  (including those 4 bytes) is safe — the early Parameters symbols
  never reach them.
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
