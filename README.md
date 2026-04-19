# oxideav-ffv1

Pure-Rust **FFV1** (RFC 9043) lossless intra-only video codec — version 3
bitstream, range coder with the default state transition table, 8-bit and
10-bit planar YUV. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-ffv1  = "0.0"
```

## What works today

| Area                                | State                                                         |
|-------------------------------------|---------------------------------------------------------------|
| Bitstream version                   | v3 only (v0/v1/v2 rejected on parse)                          |
| Coder type                          | Range coder, default state transition (`coder_type = 1`)      |
| Pixel formats                       | `Yuv420P`, `Yuv422P`, `Yuv444P` (8-bit)                       |
|                                     | `Yuv420P10Le`, `Yuv422P10Le`, `Yuv444P10Le` (10-bit LE)       |
| Lossless                            | Yes — encode then decode reproduces the source bit-for-bit    |
| Config record                       | Parse + emit, CRC-32 verified / appended                      |
| Quant tables                        | Read from extradata and verified against defaults             |
| Slice layout (decode)               | Any `num_h_slices × num_v_slices` grid                        |
| Slice layout (encode)               | One slice covering the whole frame                            |
| Slice-footer CRC (`ec != 0`)        | Verified on decode; can be emitted on encode                  |
| Interop (FFmpeg → us)               | Decoder accepts FFmpeg's `ffv1 -level 3 -coder range_def`     |
| Interop (us → FFmpeg)               | FFmpeg decodes our output bit-exactly                         |

## Not supported

- Golomb-Rice coder (`coder_type = 0`) or custom state tables (`coder_type = 2`).
- 9/12/14/16-bit sample depths.
- RGB / JPEG 2000 RCT colorspace and alpha (`extra_plane`) channel.
- Multi-slice encoding (the decoder handles multi-slice input; the encoder
  always writes a single slice).
- `-context 1` / `initial_state_delta` quant-table-set overrides.
- Non-default quantisation tables in the first table set, including
  FFmpeg's 10-bit `quant9_10bit`. Foreign streams that ship alternate tables
  are rejected at parse time rather than silently decoded to garbage.
- Bayer and packed pixel formats.

## Quick use

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, TimeBase, VideoFrame,
    frame::VideoPlane,
};

let mut codecs = CodecRegistry::new();
oxideav_ffv1::register(&mut codecs);

// Build a source frame. FFV1 is intra-only, so every packet is a keyframe.
let width = 64;
let height = 48;
let y = vec![128u8; (width * height) as usize];
let cw = (width / 2) as usize;
let ch = (height / 2) as usize;
let frame = VideoFrame {
    format: PixelFormat::Yuv420P,
    width,
    height,
    pts: Some(0),
    time_base: TimeBase::new(1, 30),
    planes: vec![
        VideoPlane { stride: width as usize, data: y },
        VideoPlane { stride: cw, data: vec![128u8; cw * ch] },
        VideoPlane { stride: cw, data: vec![128u8; cw * ch] },
    ],
};

let mut params = CodecParameters::video(CodecId::new("ffv1"));
params.width = Some(width);
params.height = Some(height);
params.pixel_format = Some(PixelFormat::Yuv420P);
params.frame_rate = Some(Rational::new(30, 1));

let mut enc = codecs.make_encoder(&params)?;
enc.send_frame(&Frame::Video(frame.clone()))?;
let pkt = enc.receive_packet()?;

// The encoder fills in `output_params().extradata` with the config record
// — hand that to the decoder so the pixel format and slice grid match.
let dec_params = enc.output_params().clone();
let mut dec = codecs.make_decoder(&dec_params)?;
dec.send_packet(&pkt)?;
let Frame::Video(out) = dec.receive_frame()? else { unreachable!() };
assert_eq!(out.width, width);
assert_eq!(out.planes[0].data, frame.planes[0].data);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Codec ID

- `"ffv1"`. Register with `oxideav_ffv1::register(&mut codecs)` or build
  the decoder / encoder directly via
  [`oxideav_ffv1::decoder::make_decoder`] and
  [`oxideav_ffv1::encoder::make_encoder`].

The output stream's configuration record lives in
`Encoder::output_params().extradata`. Muxers such as Matroska pick it up
from there.

## License

MIT — see [LICENSE](LICENSE).
