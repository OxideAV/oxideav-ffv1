# oxideav-ffv1

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

Clean-room rebuild, round 2 (2026-05-22). The prior implementation was
retired on 2026-05-18 under the workspace clean-room policy.

Round 1 landed the **Configuration Record parser** plus its
range-coder dependencies; round 2 adds the **Slice Header parser**
(RFC 9043 §4.6), so downstream callers can now walk a v3 frame's
per-slice raster geometry, quant-table-set selection, picture
structure, and SAR — still without any pixel decoding.

Implemented (RFC 9043 §3.8.1.1 / §3.8.1.2 / §4.2 / §4.3 / §4.6):

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

Not yet implemented:

- Quantization-table cascade decode (§4.1).
- `initial_state_delta` / `ec` / `intra` (the v3 tail of Parameters).
- `configuration_record_crc_parity` validation (§4.3.2).
- Slice Content (sample-difference decoding, §4.7 / §4.8).
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

| Fixture | Round 1 (cfg record) | Round 2 (slice header) |
| --- | --- | --- |
| `v3-default` | v3 / 8-bit YUV 4:2:0 / range coder default / 2x2 slices | all 4 slices: raster cells (0,0)(1,0)(0,1)(1,1) |
| `v3-rgb-bgr0` | v3 / RGB (RCT) / chroma_planes=1 / no subsample | slice 0 (chroma_planes=1 / RCT path) |
| `v3-grayscale` | v3 / single-plane / chroma_planes=0 | slice 0 (chroma_planes=0, count=2 via version<=3) |

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

## License

MIT — see [LICENSE](./LICENSE).

[RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043.html
