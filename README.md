# oxideav-ffv1

[![CI](https://github.com/OxideAV/oxideav-ffv1/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-ffv1/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-ffv1.svg)](https://crates.io/crates/oxideav-ffv1) [![docs.rs](https://docs.rs/oxideav-ffv1/badge.svg)](https://docs.rs/oxideav-ffv1) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Clean-room
rebuild (the prior implementation was retired on 2026-05-18 under the
workspace clean-room policy).

[RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043

## Status

A working FFV1 **decoder and encoder** for all three RFC 9043 versions
(0 / 1 / 3), both colour layouts (YCbCr plane-major and RGB / RCT
line-major), and all three entropy-coder modes (§3.8.2 Golomb-Rice,
§3.8.1 default-table range coder, §3.8.1.6 custom-table range coder) —
both through the direct `decode_frame*` / `encode_frame*` API and
end-to-end through the `oxideav_core::Decoder` / `Encoder` traits.

### Decode

- **Configuration Record** (§4.2 / §4.3) parse + §4.3.2 CRC validation,
  and the §4.1 Quantization Table Set cascade
  (`parse_quantization_table_sets`).
- **§4.2.14 / §4.2.15 explicit initial states** — the `states_coded ==
  1` triple-loop parses under the wire layout the hand-authored
  `states-coded-1` fixture pins byte-exactly (one dedicated fresh
  32-slot window per coded set; the FFmpeg-interop row count of
  `QuantizationTableSet::initial_state_row_count`, 942/645 for the
  pinned `[6,6,6,1,1]` / `[5,5,5,1,1]` shapes vs the RFC §4.1 counts
  of 666/365), and `reconstruct_initial_states` APPLIES the Figures
  29/30 predictor chain + modular fold to seed every
  keyframe-initialised per-§4.6.6-slot range window in both frame
  drivers. The fixture (64×48 gray, 30 144 transmitted deltas) decodes
  bit-exactly as a conformance gate; interop caveats for non-zero
  deltas are documented on `reconstruct_initial_states`.
- **Versions 0 / 1 single-Slice YCbCr decode** — `decode_frame_v0v1`
  reconstructs a v0/v1 keyframe Frame end-to-end: the §4.4 inline §4.2
  Parameters + the single §4.1 Quantization Table Set
  (`parse_v0v1_frame_prologue`, on one resumed range-coder pass) + the
  implied single §4.7 Slice Content, with **no** §4.6 Slice Header,
  **no** §4.9 Slice Footer, and **no** §4.9.1 trailer chain (all
  `version >= 3`-only). `decode_frame_v0v1_inter` decodes a v0/v1
  non-keyframe, which inherits the keyframe's inline Parameters +
  Quantization Table Set. Covers both colour layouts —
  `colorspace_type == 0` (YCbCr / plane-major, gray / YUV 4:2:0 / 4:4:4 /
  YUVA, 8/10/16-bit including the §3.3.1 16-bit alternate predictor) and
  `colorspace_type == 1` (RGB / line-major JPEG 2000 RCT, reusing the v3
  RGB line-major + inverse-RCT machinery over the implied single Slice) —
  for all three §4.2.3 coders: the §3.8.2 Golomb-Rice (`coder_type == 0`),
  the §3.8.1 default-table range coder (`coder_type == 1`), and the
  §3.8.1.6 custom-table range coder (`coder_type == 2`, whose
  single-stream Parameters → Slice-Content table swap is resolved against
  the v3 driver's behaviour).
- **Frame drivers** — `decode_frame` (YCbCr / plane-major,
  `colorspace_type == 0`) and `decode_frame_rgb` (RGB / line-major
  JPEG 2000 RCT, `colorspace_type == 1`, including the §3.7.2.1
  9..15-bit exception). Both walk the §4.9.1 trailer chain, §4.9 footer
  validate, §4.6 header parse, §4.7 plane layout, and per-plane
  reconstruction.
- **Entropy coders** — the §3.8.2 Golomb-Rice path (`coder_type == 0`,
  with run mode), the §3.8.1 range-coder path (`coder_type == 1`), and
  the §3.8.1.6 alternative state-transition table (`coder_type == 2`).
- **Predictor / transform** — §3.3 median predictor (incl. §3.3.1
  16-bit alternate), §3.5 context computation with sign-flip, §3.8
  modular reconstruction, and §3.7.1 inverse RCT.
- **Conformance gates** — `DecodeOptions` selects the §4.9.3 per-Slice
  CRC policy (`Reject` default / `Accept` best-effort), the §4.9.2
  `error_status` policy, and the opt-in §3.8.1.1.1 range-coder
  termination gate (`DecodeOptions::pedantic()` /
  `SliceTerminationPolicy` — every v3 range-coded Slice must end with
  the Sentinel-mode terminator at exactly its body length); the §5
  raster-coverage partition gate runs in both drivers; the §5 non-keyframe geometry-stability gate, the §4.2.17
  `intra` gate, and §3.8.1.3 / §3.8.2.5 inter-Frame coder-state carry
  are driven by the stateful `Ffv1DecodeSession`.
- `DecodedFrame` surfaces the decoded planes, the §4.4 `keyframe` bit,
  and the §4.6 Slice Headers.
- **Framework integration** — `register` installs an `ffv1`
  [`oxideav_core::Decoder`] behind the registry, reading the §4.2
  Configuration Record from `CodecParameters::extradata` (RFC 9043
  §4.3.3) and claiming the two §4.3.3 container tags: the AVI / VfW
  FourCC `FFV1` (§4.3.3.1) and the Matroska Codec ID `V_FFV1`
  (§4.3.3.4). A packet decodes through the trait to an
  `oxideav_core::VideoFrame` (one byte per Sample at ≤ 8-bit depth,
  two little-endian bytes otherwise); the §3.8.1.3 / §3.8.2.5
  per-context coder state is carried across non-keyframes. **Versions
  0 / 1** route through the same trait: those streams carry no
  Configuration Record (RFC 9043 §4.3.3 / §4.4 — their §4.2 Parameters
  are inline in each keyframe Frame), so the registry decoder accepts
  `CodecParameters` with dimensions but empty `extradata`, parses the
  §4.4 prologue off the first keyframe packet, and reuses its cached
  record + Quantization Table Set for later non-keyframes. The
  historical direct API (`decode_frame*` / `encode_frame*` /
  `decode_frame_v0v1*`) is retained unchanged.

### Encode

- **Configuration Record + quant-table cascade encoder**
  (`encode_configuration_record_with_quant_tables`), the symmetric
  inverse of the parser, with §4.3.2 CRC parity solved by construction
  — including the §4.2.14 / §4.2.15 `states_coded == 1` triple-loop
  under the same fixture-pinned wire layout (the parsed fixture record
  re-encodes and re-parses to the identical tail), and frame encoders
  seed their per-slot windows from the same reconstructed initial
  states so seeded streams round-trip bit-exactly.
- **Frame encoders** — `encode_frame` (YCbCr) and `encode_frame_rgb`
  (RGB / RCT), each covering `coder_type ∈ {0, 1, 2}`. Forward RCT,
  §4.6 Slice Headers, §4.9 footers (CRC parity by construction), and
  multi-slice grids are all emitted.
- **Versions 0 / 1 encoder** — `encode_frame_v0v1` /
  `encode_frame_v0v1_inter` emit a complete v0/v1 Frame: the §4.4
  `keyframe` boolean, the inline §4.2 Parameters + single §4.1 cascade
  (keyframe only), then the implied single §4.7 Slice Content — the
  symmetric inverse of `decode_frame_v0v1`, for both YCbCr and RGB / RCT.
  `coder_type == 1` (one continuous range-coder pass) and `coder_type ==
  0` (range-coded prologue, byte-aligned, then a Golomb-Rice content tail)
  are both emitted, for YCbCr and RGB / RCT alike; the §3.8.2 run-mode
  encoder carries a non-zero first Sample Difference at a run-region start
  via a §3.8.2.4.1 zero-length short run (no first-pixel restriction).
  The `_with_carry` variants (`encode_frame_v0v1_with_carry` /
  `encode_frame_v0v1_inter_with_carry`, plus the matching decode pair)
  carry the §3.8.1.3 / §3.8.2.5 per-context coder state across
  non-keyframes over the implied single Slice — RFC 9043 re-initialises
  the state only "when the keyframe value is 1", on every version — so
  the emitted inter Frames are what a conforming decoder expects
  (validated bit-exact both directions against the external reference
  implementation, r411). The stateless `encode_frame_v0v1_inter` /
  `decode_frame_v0v1_inter` remain as the degenerate no-carry pair.
- **Inter-Frame carry** — `encode_frame_with_carry` dispatches on §4.2.5
  `colorspace_type` + §4.2.3 `coder_type` to
  `encode_frame_golomb_rice_with_carry` (YCbCr Golomb-Rice),
  `encode_frame_range_coder_with_carry` (YCbCr range), or
  `encode_frame_rgb_with_carry` (RGB), each carrying the §3.8.1.3 /
  §3.8.2.5 per-context coder state across non-keyframes — the symmetric
  write-side mirror of the decode side, for **all three coders** on both
  colorspaces. Versions 0/1 carry the same state over their implied
  single Slice via `encode_frame_v0v1_with_carry` /
  `encode_frame_v0v1_inter_with_carry` (and the decode-side
  `decode_frame_v0v1_with_carry` / `decode_frame_v0v1_inter_with_carry`),
  used by the registry's v0/v1 routes.
- **Framework integration** — `register` installs an `ffv1`
  [`oxideav_core::Encoder`] alongside the decoder (the registry
  advertises both directions). It reuses the same `CodecParameters` the
  decoder consumes (§4.2 Configuration Record in `extradata`, frame
  `width` / `height`) and derives the §4.6 Slice Header grid from the
  Configuration Record's §4.2.11 / §4.2.12 `num_h_slices × num_v_slices`
  (one Slice per raster cell). `send_frame` converts an
  `oxideav_core::VideoFrame` to the internal `DecodedFrame` (the inverse
  of the decode-side plane packing) and emits the stream's **first Frame
  as a keyframe and every later Frame as a §4.4 non-keyframe** whose
  §3.8.1.3 / §3.8.2.5 per-context coder state continues from the previous
  Frame (via `encode_frame_with_carry`), unless the §4.2.17 `intra` flag
  forces keyframe-only output; it propagates the input PTS and sets the
  `Packet` keyframe flag to the actual §4.4 value. The framework
  `Decoder` carries the matching state across packets, so a multi-Frame
  inter stream round-trips end-to-end through the trait surface.
  `output_params` carries the Configuration Record back out for a muxer
  along with the §4.2-derived `PixelFormat` (see `pixel_format_for`
  below). The historical direct `encode_frame*` API is retained
  unchanged. **Versions 0 / 1 encode through the same trait**: since
  those streams carry no Configuration Record (RFC 9043 §4.3.3 / §4.4),
  the encoder accepts `CodecParameters` with **empty `extradata`** plus a
  `pixel_format` and dimensions — the same shape the registry decoder
  accepts — synthesises a version-1 record from the pixel format (the
  exact inverse of `pixel_format_for`), installs a §4.1-constructed
  default Quantization Table Set (11 symmetric levels on the three §3.5
  Figure 5 primary differences, `context_count == 666`), and emits the
  first Frame as a §4.4 keyframe (inline Parameters + Set) with later
  Frames as non-keyframes, so a v0/v1 stream encodes → decodes end-to-end
  through the trait with no out-of-band configuration at all.

- **§4.2 pixel-format mapping** — `pixel_format_for(&Ffv1Configuration
  Record)` maps the §4.2 Parameters (`colorspace_type` §4.2.5,
  `chroma_planes` §4.2.6, `bits_per_raw_sample` §4.2.7,
  `log2_*_chroma_subsample` §4.2.8 / §4.2.9, `extra_plane` §4.2.10) to
  the exact `oxideav_core::PixelFormat` the decoder's plane packing
  yields: `Gray8` / `Gray10Le` / `Gray12Le` / `Gray16Le` for luma-only
  YCbCr; `Yuv420P` / `Yuv422P` / `Yuv444P` / `Yuv411P` (plus 10/12-bit
  `*Le` siblings) keyed on the subsample shift pair; `Yuva420P` for
  8-bit 4:2:0 + alpha; and — for `colorspace_type == 1` (RGB / RCT, §4.2.5
  fixes it at 4:4:4) — the planar `Gbrp10Le` / `Gbrp12Le` / `Gbrp14Le`
  (and `Gbrap*Le` with the §4.2.10 alpha plane) at 10 / 12 / 14 bits.
  Because RFC 9043 §3.7 recovers Planes in **R, G, B (, A)** order while
  the framework's `Gbr*` formats store **G, B, R (, A)**, the registry's
  `Decoder` / `Encoder` reorder Planes at the trait boundary
  (`gbr_plane_order` / its inverse `gbr_input_order`) so the advertised
  format and the emitted / consumed plane order agree by construction —
  an RGB stream round-trips bit-exact **through the framework trait** in
  `Gbr` order, not just via the direct API. `pixel_format_for` still
  returns `None` for layouts with no exact planar framework variant —
  8-bit and 16-bit planar RGB (the framework's 8/16-bit RGB formats are
  packed), 16-bit YUV, subsampled-plus-alpha YUV, planar gray + alpha,
  and reserved subsample shifts — so a caller never advertises a
  misleading format. The framework `Encoder` populates
  `output_params.pixel_format` from it when an exact variant exists.

Round-trip and bit-exact tests cover both colorspaces, all three coder
types, every chroma subsampling × extra-plane shape, **8/9/10/12/15/16-bit
depths** (range-coded YCbCr chroma-Frame round-trips at 9 / 12 / 16 bits
including the §3.3.1 16-bit exception predictor; RGB / RCT round-trips at
the §3.7.2.1 9-bit and 15-bit exception boundaries plus the 16-bit general
path; the same depth ladder on the v0/v1 inline-Parameters path),
multi-slice grids (**including non-uniform §4.8 floor-division grids whose
Slices differ in size**), multi-context Quantization Table Sets, and a
**reference-fixture decode corpus** that decodes each fixture's coded
Frame bit-exactly against the reference decoder's `expected.raw`:

- v3 range-coder, single-Slice: `v3-flat-color` (low-entropy run/zero
  range paths), `v3-grayscale` (single-Plane luma-only,
  `chroma_planes == 0`), `v3-yuv422p10` (10-bit 4:2:2,
  `log2_h_chroma_subsample == 1` / `log2_v_chroma_subsample == 0`),
  `v3-yuv420p12` (12-bit 4:2:0), `v3-yuv444p16` (16-bit 4:4:4
  full-precision), `v3-rgba` (`transparency == 1`, four-Plane RGB + alpha
  over the JPEG 2000 RCT), `v3-rgb-bgr0` (RGB, **no** alpha, three
  Planes), and `v3-context-1` (the large `-context 1` Quantization Table
  Set, ~7563 contexts).
- v3 range-coder, multi-Slice: `v3-default` (128×96, 2×2 = 4 Slices,
  per-Slice CRC), `v3-multislice-4x4` (128×96, 4×4 = 16 Slices), and
  `v3-frame-mt` (256×192, 4×4 = 16 Slices, each luma Slice 64×48) — the
  §5 slice-grid raster partition + §4.9.1 trailer chain across multiple
  Slices.
- v0/v1 single-stream: `v0-yuv420-rangecoder` (FFV1 version 0, inline
  Parameters, range coder), `v1-single-slice` (version 1, 128×96, range
  coder), `v0-yuv420-golomb-rice` (version 0, **Golomb-Rice**
  `coder_type == 0`) — the §3.8.2 adaptive run-length / level-coding
  decode loop driven directly by a reference-encoded stream — and
  `v1-golomb` (version 1, 64×48, **Golomb-Rice** `coder_type == 0`),
  which pins the §3.8.2 residual path on a version-1 header
  (`bits_per_raw_sample` present in the inline Parameters). The real
  `v1-golomb` Planes additionally re-encode → decode bit-exactly through
  both entropy back-ends (`tests/reference_content_roundtrip.rs`, the
  §1 lossless identity over real `testsrc2` residuals) and decode
  end-to-end through the `oxideav_core::Decoder` trait's empty-extradata
  v0/v1 route (`tests/registry_v1_golomb_fixture.rs`).

Every fixture under `docs/video/ffv1/fixtures/` is covered. The
version-2 stream (`v2-multislice-2x2`), which FFV1 reserves as
experimental and never emits in conforming bitstreams, is exercised as a
**negative** gate: its real Configuration Record must be rejected with
the typed `Error::UnsupportedVersion(2)`, not mis-decoded
(`tests/v2_reserved_rejected.rs`, RFC 9043 §4.2.1 Table 5).

Fixture Frames are extracted black-box from each `input.mkv` / `input.avi`
(Matroska / AVI container parsing is independent of the FFV1 bitstream)
and inlined alongside the reference `expected.raw` in
`tests/data/reference_fixtures.rs`.

### Inter-frame reference corpus (r416)

A second, **multi-frame** reference corpus
(`docs/video/ffv1/fixtures/inter-*`, 16 streams; inlined packets +
per-frame SHA-256 pins in `tests/data/reference_inter_fixtures.rs`)
drives the §3.8.1.3 / §3.8.2.5 inter-Frame coder-state carry against
reference-**encoded** bytes — the decode-side mirror of the
`external_conformance` self-encoded corpus. Every stream is one §4.4
keyframe plus carried non-keyframes; `tests/reference_inter_decode.rs`
decodes each stream bit-exactly under `DecodeOptions::pedantic()` (the
§3.8.1.1.1 termination gate against reference bytes) and requires the
§4.4 keyframe flag of every Frame to match the reference toolchain's
report. Coverage: versions 0 / 1 (inline Parameters, range +
Golomb-Rice) and version 3 across 8/10/12/16-bit, 4:2:0 / 4:2:2 /
4:4:4, gray, RGB / RGBA, a 2×2 slice grid, the large `-context 1`
table, Golomb-Rice, a reference-encoded **`coder_type == 2` custom
state-transition-table** stream (previously the custom-table decode
path was validated only against this crate's own encoder), a
**mid-stream keyframe** stream (`-g 2`: the carry re-initialises on a
later keyframe, not just Frame 0), and a `-slicecrc 0` stream.

The `-slicecrc 0` stream pins an interop finding (full write-up in
`tests/reference_inter_decode.rs`): the current reference encoder's
Configuration Record **tail** (§4.2.14 `states_coded` / §4.2.16 `ec` /
§4.2.17 `intra`) does not read back under the RFC 9043 Figure 28 layout
that the same build's parser accepts — records this crate writes per
Figure 28 are honoured bit-exactly (including the `ec` gate), while the
reference writer's own two-set records parse to non-physical tail
values (`ec` up to 7, `intra == 1` on streams with non-keyframes).
Because a `-slicecrc 0` stream can therefore misdeclare `ec != 0`, the
registry `Decoder` treats the record-derived `ec` as a hypothesis until
the first Frame decodes: on failure it retries the packet once with the
opposite §4.9 footer shape and locks in whichever hypothesis yields a
fully-validated Frame (`tests/registry_inter_ec_resilience.rs`).
Truthful records decode on the first attempt and never retry; the
direct `decode_frame*` API is unchanged (the caller still supplies
`ec`).

### External encoder conformance (r411, completed r416)

The encoder axis is validated **against the external reference decoder
run black-box** (an opaque process; no library or source access):
`tests/external_conformance.rs` pins a 27-stream self-encoded corpus —
versions 0/1/3 × all three §4.2.3 coders × gray / YUV
4:2:0/4:2:2/4:4:4 / YUVA / RGB / RGBA × 8/10/12/14/16-bit ×
single-slice / 2×2 / non-uniform 3×2-on-odd-dimensions grids × ec 0/1,
every stream a keyframe **plus carried non-keyframes** (up to a
4-frame chain), plus a §4.2.17 `intra` stream — by SHA-256 per packet.
**All 27 decode bit-exactly in the reference decoder with zero
warnings.** The last cell (`v0` + `coder_type == 2`, recorded in r411
as a validator limitation) was root-caused in r416 by black-box delta
probes: the validator's v0/v1 inline-Parameters path rejects a
transmitted custom table containing any zero transition (even one
equal to the §3.8.1.5 default, whose entries `1..=8` / `249..=255` are
zero), while its v3 record path accepts the same bytes. Both shapes
are RFC-conforming (the zeroed states are unreachable), so the corpus
now transmits the interoperable fully-live table — every zero-default
entry lifted to the self-loop `i` (procedure, wrap details, probe
matrix, and per-stream results:
`tests/external_conformance_notes.md`). Reference-encoded
keyframe+inter probe streams across the same matrix decode bit-exactly
through the carry drivers in the opposite direction. Three conformance
fixes came out of this campaign (r411): §3.8.1.1.1 Sentinel-mode
termination of every v3 range-coded Slice region (bare `finish()`
previously left each Slice one byte long — tolerated on keyframes,
*concealed as damage* on every inter Frame), the Slice-scoped
§3.8.2.2.1 run triple on the §4.7 line-major RGB Golomb interleave, and
the v0/v1 inter-Frame coder-state carry above.

### §3.8.2 run-mode decode loop + Sentinel-mode handoff

The §3.8.2.2 Golomb-Rice run mode is a per-Line state machine governed
solely by the absolute context being 0 (§3.8.2.2): a context-0 Sample
enters run mode; the §3.8.2.2.1 run-length prefix selects a long run
(`1 << log2_run[run_index]` zero Samples, `run_index` grows when the run
fits in the remaining Line width) or a short run (a residual zero count
followed by a level-coded break, §3.8.2.4.1, zero excluded). A short run
of length zero level-codes the very first run Sample — so a nonzero
Sample Difference at the start of a run region **is** representable (the
former `Error::RunModeFirstPixelNonZero` restriction is retired; that
variant is now never produced). The encoder is the exact bit-for-bit
inverse of this loop.

The switch from the range-coded inline Parameters (versions 0/1) to the
byte-aligned Golomb-Rice Slice Content uses **Sentinel mode** (RFC 9043
§3.8.1.1.1): the encoder writes a discarded state-129 terminator and the
decoder recovers the byte boundary one byte before the Closed-mode
look-ahead cursor. This is what lets `v0-yuv420-golomb-rice` decode
bit-exact against a reference-produced stream. The encoder rounds the
final `low` register down to a zero low byte before flushing, so the
decoder's mandatory one-byte over-read past the boundary (RFC 9043
§3.8.1.1.1) lands on a true **don't-care** — any value in `[low, low +
range)` decodes identically, and the first appended Golomb-Rice byte that
physically occupies the boundary can never change the last §4.1 sub-table
symbol's `low < range` decision. Without this rounding, certain prologue
byte-alignments (first observed on 16-bit RGB, where the §3.7 RCT coded
width `bits + 1 == 17` shifts the prologue length) let the recovered
Quantization Table Set's `context_count` come out wrong and corrupted the
self round-trip; the fix is alignment-agnostic and is covered by a
`RangeEncoder::terminate_sentinel` symbol-count sweep plus a
depth × dimension v0/v1 Golomb round-trip matrix.
- The framework `Encoder` derives one Slice per `num_h_slices ×
  num_v_slices` raster cell and selects Quantization Table Set 0 for
  every plane slot. A stream that needs a non-trivial slice
  decomposition (Slices spanning multiple cells) or per-plane
  multi-Quantization-Table-Set selection must use the direct
  `encode_frame*` API with bespoke `Ffv1SliceHeader`s. The framework
  encoder emits the stream's first Frame as a keyframe and every later
  Frame as an inter-Frame non-keyframe (carrying §3.8.1.3 / §3.8.2.5
  coder state), unless the §4.2.17 `intra` flag forces keyframe-only
  output; finer keyframe-cadence control (e.g. periodic keyframes) is
  available through the direct `encode_frame_with_carry` /
  `encode_frame_*_with_carry` functions with a caller-chosen `keyframe`
  value per Frame.

- **Versions 0 / 1 support all three coders on both colour layouts.**
  YCbCr / plane-major (`colorspace_type == 0`) and RGB / line-major RCT
  (`colorspace_type == 1`) both decode and encode end-to-end with
  bit-exact lossless self round-trip (`decode_frame_v0v1` /
  `encode_frame_v0v1` and their `_inter` non-keyframe siblings) for all of
  `coder_type == 0` (Golomb-Rice), `coder_type == 1` (range default), and
  `coder_type == 2` (custom state-transition table). The
  single-stream `coder_type == 2` table ordering is resolved against the
  v3 driver's own behaviour (RFC 9043 §4.4 / §4.2.4 / §3.8.1.6): v0/v1
  shares one continuous range-coder pass between the inline §4.2
  Parameters and the §4.7 Slice Content, so the §4.2.4
  `state_transition_delta` (and the keyframe boolean + Parameters that
  precede them) are read with the §3.8.1.5 *default* table — they cannot
  define themselves — and the live coder swaps onto the §3.8.1.6 custom
  table (`RangeDecoder::set_one_state` / `RangeEncoder::set_one_state`) at
  the Parameters → Slice-Content boundary; a non-keyframe (no inline
  Parameters) is seeded with the custom table from the start, exactly as
  the v3 driver seeds each Slice. The RGB Golomb (`coder_type == 0`)
  encode round-trips bit-exact even though the forward RCT lifts the
  Cb / Cr corner to the §3.7.2 offset (a non-zero run-region first Sample):
  the §3.8.2 encoder represents it with a §3.8.2.4.1 zero-length short run,
  covered by the `v0v1_roundtrip` RGB Golomb depth × dimension matrix.
  Version 3 supports all colour layouts and all three coders.

## Usage

```rust
use oxideav_ffv1::{parse_quantization_table_sets, decode_frame, FramePixelDimensions};

// `extradata` is the FFV1 Configuration Record; `frame` is one coded Frame.
let extradata: &[u8] = b"";
let frame: &[u8] = b"";

let parsed = parse_quantization_table_sets(extradata)?;
let dims = FramePixelDimensions::new(128, 96)?;
let decoded = decode_frame(
    frame,
    &parsed.record,
    &parsed.quant_table_sets,
    dims,
    parsed.record.ec.is_some(),
)?;
let _ = decoded.planes;
# Ok::<(), oxideav_ffv1::Error>(())
```

## Benchmarks

A Criterion harness (`benches/decode.rs` / `benches/encode.rs`) covers
the coder (Golomb-Rice / range default / range custom) × depth
(8/10/16-bit) × colorspace (YCbCr 4:2:0 / RGB-RCT) × slice-grid
(1/4/16) matrix on synthesised 320×240 frames with a realistic
666-context Quantization Table Set. Results, `sample`-profiler
breakdowns, and the round-386 optimization log (decode −10…−20%,
encode −4…−29%, byte-identical outputs guarded by
`tests/optimization_pins.rs`) live in [BENCHMARKS.md](BENCHMARKS.md).

```
cargo bench -p oxideav-ffv1 --bench decode
cargo bench -p oxideav-ffv1 --bench encode
```

## Fuzzing

A `fuzz/` [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html)
package drives attacker-controlled bytes through the decoder's public
parse / decode surface; a scheduled `Fuzz` workflow runs all targets
daily under libFuzzer + AddressSanitizer. The contract under test is
**panic-freedom on every input shape** — no out-of-bounds index, no
debug-build arithmetic overflow, no `unwrap` on an attacker-forced
`None` / `Err`; a malformed stream must surface a typed `Error`, never a
panic. The decode targets are:

- `config_record` — the §4.2 Configuration Record parse
  (`parse_configuration_record`) + the §4.1 Quantization Table Set
  cascade (`parse_quantization_table_sets`). Every Parameter field
  (§4.2.1 version, §4.2.3 coder_type, §4.2.5 colorspace, §4.2.6 / §4.2.7 /
  §4.2.8 / §4.2.9 plane + depth + subsample shifts) plus the §4.3.2 CRC
  trailer and the per-context quant-table deltas are the §3.8.1 range
  coder reading attacker bytes.
- `decode_frame` — the v3 YCbCr (`decode_frame`) and RGB
  (`decode_frame_rgb`) drivers, with the attacker controlling the
  Configuration Record bytes, the coded Frame bytes, and the frame
  dimensions (bounded so a malformed record cannot request an unbounded
  allocation). Reaches the §4.6 / §4.7 / §4.9 header / content / footer
  walk, the §4.9.1 trailer chain, and the §3.3 / §3.5 / §3.7 / §3.8
  reconstruction.
- `decode_v0v1` — the versions-0/1 inline-Parameters decode
  (`decode_frame_v0v1`), parsing the §4.4 prologue off one resumed
  range-coder pass.
- `registry_decode` — the realistic container surface:
  `CodecParameters` (§4.3.3 extradata + dims) plus a coded `Packet`
  through the registry-installed `oxideav_core::Decoder` trait
  (`send_packet` / `receive_frame`), covering both the v3 and the
  empty-extradata v0/v1 routing, with two Packets per input to reach the
  §3.8.1.3 / §3.8.2.5 cross-Frame coder-state carry.

Two further targets invert the surface to test the **lossless identity**
(FFV1 is lossless, RFC 9043 §1: `decode(encode(x)) == x`):

- `roundtrip` — builds a *well-formed* `DecodedFrame` + matching
  `Ffv1ConfigurationRecord` / §4.1 Quantization Table Set / §4.6 Slice
  Header from the attacker bytes, encodes it with the crate's encoder,
  decodes the result with the crate's decoder, and asserts the recovered
  Planes are bit-exact. It sweeps the version (0/1/3) × `coder_type`
  (0/1/2) × colorspace (YCbCr/RGB) × bit-depth (8/9/10/12/16) cross
  product so an encoder/decoder asymmetry on an off-grid shape — like the
  §3.8.1.1.1 Sentinel-mode boundary corruption fixed this round — surfaces
  as a finding rather than silently shipping a stream the decoder
  mis-reads.
- `registry_roundtrip` — the same contract lifted onto the **framework
  trait surface**: a well-formed `VideoFrame` (one of the 21 mapped
  `PixelFormat`s, bounded dims, depth-masked samples) is encoded through
  the registry `oxideav_core::Encoder` and decoded back through the
  registry `oxideav_core::Decoder`, asserting bit-exact plane bytes over
  a keyframe + non-keyframe pair. Everything the trait wiring adds is
  under the identity: the §4.2 pixel-format mapping and its inverse (the
  empty-extradata v0/v1 encoder synthesising inline §4.4 Parameters), the
  §4.1 default Quantization Table Set's wire round-trip, the
  little-endian plane packing, the planar-`Gbr*` plane reorder, and the
  keyframe → non-keyframe sequencing.

Each target links only this crate's public API plus `oxideav-core`'s
public surface — no external decoder, library, or oracle.

## Clean-room provenance

Implemented entirely from RFC 9043; all clause / equation / figure
numbers cite the RFC directly. No external decoder or library source was
consulted.

## License

MIT — see [LICENSE](LICENSE).
