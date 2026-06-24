# oxideav-ffv1

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Clean-room
rebuild (the prior implementation was retired on 2026-05-18 under the
workspace clean-room policy).

[RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043

## Status

A working FFV1 v3 **decoder and encoder** for both colour layouts and
all three entropy-coder modes.

### Decode

- **Configuration Record** (§4.2 / §4.3) parse + §4.3.2 CRC validation,
  and the §4.1 Quantization Table Set cascade
  (`parse_quantization_table_sets`).
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
  CRC policy (`Reject` default / `Accept` best-effort) and the §4.9.2
  `error_status` policy; the §5 raster-coverage partition gate runs in
  both drivers; the §5 non-keyframe geometry-stability gate, the §4.2.17
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
  inverse of the parser, with §4.3.2 CRC parity solved by construction.
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
  are emitted; the Golomb path inherits the §3.8.2.2
  `RunModeFirstPixelNonZero` limitation shared with the v3 Golomb encoder.
- **Inter-Frame carry** — `encode_frame_with_carry` dispatches on §4.2.5
  `colorspace_type` + §4.2.3 `coder_type` to
  `encode_frame_golomb_rice_with_carry` (YCbCr Golomb-Rice),
  `encode_frame_range_coder_with_carry` (YCbCr range), or
  `encode_frame_rgb_with_carry` (RGB), each carrying the §3.8.1.3 /
  §3.8.2.5 per-context coder state across non-keyframes — the symmetric
  write-side mirror of the decode side, for **all three coders** on both
  colorspaces.
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
  unchanged.

- **§4.2 pixel-format mapping** — `pixel_format_for(&Ffv1Configuration
  Record)` maps the §4.2 Parameters (`colorspace_type` §4.2.5,
  `chroma_planes` §4.2.6, `bits_per_raw_sample` §4.2.7,
  `log2_*_chroma_subsample` §4.2.8 / §4.2.9, `extra_plane` §4.2.10) to
  the exact `oxideav_core::PixelFormat` the decoder's plane packing
  yields: `Gray8` / `Gray10Le` / `Gray12Le` / `Gray16Le` for luma-only
  YCbCr; `Yuv420P` / `Yuv422P` / `Yuv444P` / `Yuv411P` (plus 10/12-bit
  `*Le` siblings) keyed on the subsample shift pair; `Yuva420P` for
  8-bit 4:2:0 + alpha. It returns `None` for layouts with no exact
  framework variant — RGB / RCT (the decoder's R, G, B plane order has
  no planar match; §4.2.5 fixes RGB at 4:4:4), 16-bit YUV,
  subsampled-plus-alpha YUV, planar gray + alpha, and reserved subsample
  shifts — so a caller never advertises a misleading format. The
  framework `Encoder` populates `output_params.pixel_format` from it
  when an exact variant exists.

Round-trip and bit-exact tests cover both colorspaces, all three coder
types, every chroma subsampling × extra-plane shape, 8/10/12/16-bit
depths, multi-slice grids, multi-context Quantization Table Sets, and a
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
  coder), and `v0-yuv420-golomb-rice` (version 0, **Golomb-Rice**
  `coder_type == 0`) — the §3.8.2 adaptive run-length / level-coding
  decode loop driven directly by a reference-encoded stream.

Every fixture under `docs/video/ffv1/fixtures/` is covered except the
version-2 stream (`v2-multislice-2x2`), which FFV1 reserves as
experimental and never emits in conforming bitstreams.

Fixture Frames are extracted black-box from each `input.mkv` / `input.avi`
(Matroska / AVI container parsing is independent of the FFV1 bitstream)
and inlined alongside the reference `expected.raw` in
`tests/data/reference_fixtures.rs`.

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
bit-exact against a reference-produced stream.
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
  encode is wired but its §3.8.2.2 `RunModeFirstPixelNonZero` constraint
  (the forward RCT lifts the Cb / Cr corner to the §3.7.2 offset) makes a
  synthetic round-trip fixture hard to build. Version 3 supports all
  colour layouts and all three coders.

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

## Fuzzing

A `fuzz/` [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html)
package drives attacker-controlled bytes through the decoder's public
parse / decode surface; a scheduled `Fuzz` workflow runs all targets
daily under libFuzzer + AddressSanitizer. The contract under test is
**panic-freedom on every input shape** — no out-of-bounds index, no
debug-build arithmetic overflow, no `unwrap` on an attacker-forced
`None` / `Err`; a malformed stream must surface a typed `Error`, never a
panic. The four targets are:

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

Each target links only this crate's public API plus `oxideav-core`'s
public surface — no external decoder, library, or oracle.

## Clean-room provenance

Implemented entirely from RFC 9043; all clause / equation / figure
numbers cite the RFC directly. No external decoder or library source was
consulted.

## License

MIT — see [LICENSE](LICENSE).
