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
  Quantization Table Set. Supports the §3.8.1 default-table range coder
  (`coder_type == 1`) over gray / YUV (4:2:0 / 4:4:4) / YUVA layouts and
  8/10/16-bit depths (the §3.3.1 16-bit alternate predictor included).
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
  per-context coder state is carried across non-keyframes. The
  historical direct API (`decode_frame*` / `encode_frame*`) is
  retained unchanged.

### Encode

- **Configuration Record + quant-table cascade encoder**
  (`encode_configuration_record_with_quant_tables`), the symmetric
  inverse of the parser, with §4.3.2 CRC parity solved by construction.
- **Frame encoders** — `encode_frame` (YCbCr) and `encode_frame_rgb`
  (RGB / RCT), each covering `coder_type ∈ {0, 1, 2}`. Forward RCT,
  §4.6 Slice Headers, §4.9 footers (CRC parity by construction), and
  multi-slice grids are all emitted.
- **Versions 0 / 1 encoder** — `encode_frame_v0v1` /
  `encode_frame_v0v1_inter` emit a complete v0/v1 YCbCr Frame: the §4.4
  `keyframe` boolean, the inline §4.2 Parameters + single §4.1 cascade
  (keyframe only), then the implied single §4.7 Slice Content — one
  continuous Closed-mode range-coder pass, the symmetric inverse of
  `decode_frame_v0v1`. `coder_type == 1` (range default) for now.
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
types, every chroma subsampling × extra-plane shape, 8/10/16-bit
depths, multi-slice grids, multi-context Quantization Table Sets, and
the four v3 reference fixtures (`v3-default`, `v3-grayscale`,
`v3-rgb-bgr0`, `v3-yuv444p16`), which decode bit-exactly against their
`expected.raw`.

### Limitations

- A non-zero Sample Difference at the *first* Sample of a Golomb-Rice
  run region (absolute context 0, immediately after a run-state reset)
  has no §3.8.2.2 encoding; the encoder rejects it with
  `Error::RunModeFirstPixelNonZero` (the range coder carries such pixels
  without restriction — the recommended escape). This never arises in a
  stream a conforming FFV1 encoder produced.
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

- **Versions 0 / 1 cover YCbCr + range-coder only.** The v0/v1 YCbCr /
  plane-major path (`colorspace_type == 0`, `coder_type == 1`) decodes and
  encodes end-to-end with bit-exact lossless self round-trip
  (`decode_frame_v0v1` / `encode_frame_v0v1` and their `_inter`
  non-keyframe siblings). Still to wire for v0/v1: the §4.7 RGB /
  line-major path (`colorspace_type == 1`); the §3.8.2 Golomb-Rice encode
  path (`coder_type == 0` — the decode side already accepts it); and
  `coder_type == 2` (custom state-transition table), whose mid-Parameters
  table-ordering for the single-stream v0/v1 case is not pinned by RFC 9043
  (it is unambiguous for v3, where Parameters live in a separate
  Configuration Record pass). Version 3 supports all colour layouts and
  all three coders.

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

## Clean-room provenance

Implemented entirely from RFC 9043; all clause / equation / figure
numbers cite the RFC directly. No external decoder or library source was
consulted.

## License

MIT — see [LICENSE](LICENSE).
