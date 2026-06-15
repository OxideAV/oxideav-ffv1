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

### Encode

- **Configuration Record + quant-table cascade encoder**
  (`encode_configuration_record_with_quant_tables`), the symmetric
  inverse of the parser, with §4.3.2 CRC parity solved by construction.
- **Frame encoders** — `encode_frame` (YCbCr) and `encode_frame_rgb`
  (RGB / RCT), each covering `coder_type ∈ {0, 1, 2}`. Forward RCT,
  §4.6 Slice Headers, §4.9 footers (CRC parity by construction), and
  multi-slice grids are all emitted.
- **Inter-Frame carry** — `encode_frame_range_coder_with_carry` /
  `encode_frame_rgb_with_carry` carry the §3.8.1.3 / §3.8.2.5 per-context
  coder state across non-keyframes, mirroring the decode side.

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
- The framework registry exposes the **decoder** behind the
  `oxideav_core::Decoder` trait (`register` installs the `ffv1` codec id
  and claims the AVI FourCC `FFV1` / Matroska Codec ID `V_FFV1`, RFC 9043
  §4.3.3). The **encoder** trait is not yet wired — the §4.6 Slice-Header
  synthesis the trait factory would need is currently the caller's
  responsibility on the direct `encode_frame*` entry points. Encode via
  those functions directly for now.

### Framework registration

`oxideav_core::register!("ffv1", register)` installs a `Decoder`
implementation. A container demuxer routes an FFV1 stream by setting
`CodecParameters::extradata` to the §4.2 Configuration Record (AVI `strf`
tail / Matroska `CodecPrivate`) and `width` / `height` to the §4 frame
dimensions; the registry-built decoder then drives `decode_frame` /
`decode_frame_rgb` per the §4.2.5 `colorspace_type`. Decoded planes map
onto `VideoFrame` as one byte per Sample for ≤8-bit depths and
little-endian `u16` per Sample for >8-bit depths.

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
