# oxideav-ffv1

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

Clean-room rebuild, round 1 (2026-05-20). The prior implementation was
retired on 2026-05-18 under the workspace clean-room policy. The
current round lands only the **Configuration Record parser** — version
detection, coder selection, chroma layout, slice grid — so downstream
containers can answer "is this an FFV1 v3 stream?" without pulling in
any decode logic.

Implemented (RFC 9043 §3.8.1.1 / §4.2 / §4.3):

- Binary range decoder (Closed mode), default state-transition table.
- Scalar symbol decoder (`ur` / `sr` / `br`) per Figure 21.
- Configuration Record fields: `version`, `micro_version`,
  `coder_type`, `state_transition_delta`, `colorspace_type`,
  `bits_per_raw_sample`, `chroma_planes`, `log2_h_chroma_subsample`,
  `log2_v_chroma_subsample`, `extra_plane`, `num_h_slices`,
  `num_v_slices`, `quant_table_set_count`.

Not yet implemented:

- Quantization-table cascade decode (§4.1).
- `initial_state_delta` / `ec` / `intra` (the v3 tail of Parameters).
- `configuration_record_crc_parity` validation (§4.3.2).
- Slice header / content / footer.
- Range non-binary mode for slice data, Golomb-Rice mode, RCT.
- Encoder.

Until those land, the public `Decoder` / `Encoder` traits return
`Error::NotImplemented` and no codec is registered into the runtime.

## Verification

The parser is validated against the workspace's black-box FFV1 fixture
corpus under `docs/video/ffv1/fixtures/` — Configuration Records
extracted from real FFmpeg-encoded `input.mkv` files (Matroska
CodecPrivate elements) and compared against the `trace.txt`
`GLOBAL_HEADER` events. Three fixtures are exercised in unit tests:

| Fixture | What it pins down |
| --- | --- |
| `v3-default` | v3 / 8-bit YUV 4:2:0 / range coder default / 2x2 slices |
| `v3-rgb-bgr0` | v3 / RGB (RCT) / chroma_planes=1 / no subsample |
| `v3-grayscale` | v3 / single-plane / chroma_planes=0 |

## Notes for future rounds

- RFC 9043 §4.2 says "Parameters has its own initial states, all set
  to 128" without specifying the state-buffer width. Empirically, **all
  Parameters symbols share a single 32-slot context window**: the test
  fixtures decode correctly only with that interpretation. The same
  ambiguity applies to QuantizationTableSet and SliceHeader; future
  rounds should re-test the shared-window hypothesis for each section.
- The Configuration Record's last 4 bytes are
  `configuration_record_crc_parity`; the range decoder is in Closed
  mode and reads past-end as zero, so passing the full extradata blob
  (including those 4 bytes) is safe — the early Parameters symbols
  never reach them.

## License

MIT — see [LICENSE](./LICENSE).

[RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043.html
