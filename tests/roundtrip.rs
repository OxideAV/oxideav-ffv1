//! End-to-end FFV1 bit-exact round-trip tests via the `Decoder` / `Encoder`
//! trait surfaces.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, Rational, VideoFrame};
use oxideav_ffv1::decoder::make_decoder;
use oxideav_ffv1::encoder::make_encoder;

fn make_params(pix: PixelFormat, width: u32, height: u32) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("ffv1"));
    p.width = Some(width);
    p.height = Some(height);
    p.pixel_format = Some(pix);
    p.frame_rate = Some(Rational::new(30, 1));
    p
}

fn synth_yuv420(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    // Deterministic pattern, no RNG dependency.
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 7 + j * 11 + 16) & 0xFF) as u8;
        }
    }
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = ((i * 19 + j * 3 + 64) & 0xFF) as u8;
            v[j * cw + i] = ((i * 5 + j * 23 + 128) & 0xFF) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: u,
            },
            VideoPlane {
                stride: cw,
                data: v,
            },
        ],
    }
}

fn synth_yuv422(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 5 + j * 13 + 32) & 0xFF) as u8;
        }
        for i in 0..cw {
            u[j * cw + i] = ((i * 19 + j * 3 + 64) & 0xFF) as u8;
            v[j * cw + i] = ((i * 7 + j * 11 + 200) & 0xFF) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: u,
            },
            VideoPlane {
                stride: cw,
                data: v,
            },
        ],
    }
}

fn synth_yuv444(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; w * h];
    let mut v = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let idx = j * w + i;
            y[idx] = ((i * 3 + j * 7) & 0xFF) as u8;
            u[idx] = ((i * 11 + j * 5 + 50) & 0xFF) as u8;
            v[idx] = ((i * 13 + j * 17 + 100) & 0xFF) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane { stride: w, data: u },
            VideoPlane { stride: w, data: v },
        ],
    }
}

fn assert_frames_equal(
    a: &VideoFrame,
    b: &VideoFrame,
    format: PixelFormat,
    width: u32,
    height: u32,
) {
    assert_eq!(a.planes.len(), b.planes.len(), "plane count");
    // Bytes-per-sample: 2 for LE 10/12-bit variants, 1 otherwise.
    let bps = match format {
        PixelFormat::Yuv420P10Le
        | PixelFormat::Yuv422P10Le
        | PixelFormat::Yuv444P10Le
        | PixelFormat::Yuv420P12Le
        | PixelFormat::Yuv422P12Le
        | PixelFormat::Yuv444P12Le => 2,
        _ => 1,
    };
    for (i, (pa, pb)) in a.planes.iter().zip(b.planes.iter()).enumerate() {
        // Compare the `width × height` active region, not the raw data array
        // — strides may differ if encoder and decoder disagree on padding.
        let (w, h) = match (i, format) {
            (0, PixelFormat::Rgb24) => (width as usize * 3, height as usize),
            (0, _) => (width as usize, height as usize),
            (_, PixelFormat::Yuv420P | PixelFormat::Yuv420P10Le | PixelFormat::Yuv420P12Le) => {
                ((width as usize).div_ceil(2), (height as usize).div_ceil(2))
            }
            (_, PixelFormat::Yuv422P | PixelFormat::Yuv422P10Le | PixelFormat::Yuv422P12Le) => {
                ((width as usize).div_ceil(2), height as usize)
            }
            (_, PixelFormat::Yuv444P | PixelFormat::Yuv444P10Le | PixelFormat::Yuv444P12Le) => {
                (width as usize, height as usize)
            }
            _ => panic!("unhandled format/plane combo"),
        };
        let row_bytes = w * bps;
        for y in 0..h {
            let row_a = &pa.data[y * pa.stride..y * pa.stride + row_bytes];
            let row_b = &pb.data[y * pb.stride..y * pb.stride + row_bytes];
            assert_eq!(row_a, row_b, "plane {} row {} mismatch", i, y);
        }
    }
}

fn roundtrip_one(frame: VideoFrame, pix: PixelFormat, width: u32, height: u32) {
    let params = make_params(pix, width, height);

    let mut enc = make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    assert!(pkt.flags.keyframe);

    // Hand off the encoder's output CodecParameters (with extradata) to the
    // decoder so the configuration record matches.
    let dec_params = enc.output_params().clone();
    let mut dec = make_decoder(&dec_params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    match out {
        Frame::Video(v) => assert_frames_equal(&v, &frame, pix, width, height),
        _ => panic!("decoder returned non-video frame"),
    }
}

#[test]
fn yuv420_16x16_roundtrip() {
    roundtrip_one(synth_yuv420(16, 16), PixelFormat::Yuv420P, 16, 16);
}

#[test]
fn yuv420_64x48_roundtrip() {
    roundtrip_one(synth_yuv420(64, 48), PixelFormat::Yuv420P, 64, 48);
}

#[test]
fn yuv420_odd_dimensions_roundtrip() {
    roundtrip_one(synth_yuv420(17, 11), PixelFormat::Yuv420P, 17, 11);
}

#[test]
fn yuv422_32x16_roundtrip() {
    roundtrip_one(synth_yuv422(32, 16), PixelFormat::Yuv422P, 32, 16);
}

#[test]
fn yuv422_odd_width_roundtrip() {
    roundtrip_one(synth_yuv422(17, 12), PixelFormat::Yuv422P, 17, 12);
}

#[test]
fn yuv444_32x32_roundtrip() {
    roundtrip_one(synth_yuv444(32, 32), PixelFormat::Yuv444P, 32, 32);
}

#[test]
fn yuv444_64x48_roundtrip() {
    roundtrip_one(synth_yuv444(64, 48), PixelFormat::Yuv444P, 64, 48);
}

#[test]
fn yuv420_all_zero_roundtrip() {
    // Highly-compressible flat content — stresses the highest-probability
    // states in the range coder and the context-0 (flat) pathway.
    let w = 128u32;
    let h = 80u32;
    let wu = w as usize;
    let hu = h as usize;
    let cw = wu.div_ceil(2);
    let ch = hu.div_ceil(2);
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: wu,
                data: vec![0u8; wu * hu],
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![128u8; cw * ch],
            },
        ],
    };
    roundtrip_one(frame, PixelFormat::Yuv420P, w, h);
}

/// Convert a `Vec<u16>` of samples into the little-endian byte buffer
/// layout that our `VideoPlane.data` carries for 10-bit formats.
fn u16_to_le(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Build a 10-bit YUV 4:2:0 frame whose luma is a ramp covering the full
/// [0, 1023] range. Chroma uses deterministic patterns too so we exercise
/// the full 10-bit sample space.
fn synth_yuv420p10(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            // Luma ramp: walks 0..1023 diagonally.
            let v = ((i * 16 + j * 11) as u32) & 0x3FF;
            y.push(v as u16);
        }
    }
    let mut u = Vec::with_capacity(cw * ch);
    let mut v = Vec::with_capacity(cw * ch);
    for j in 0..ch {
        for i in 0..cw {
            u.push((((i * 23 + j * 5 + 100) as u32) & 0x3FF) as u16);
            v.push((((i * 7 + j * 31 + 500) as u32) & 0x3FF) as u16);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&v),
            },
        ],
    }
}

fn synth_yuv444p10(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = Vec::with_capacity(w * h);
    let mut u = Vec::with_capacity(w * h);
    let mut v = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            y.push((((i * 9 + j * 13) as u32) & 0x3FF) as u16);
            u.push((((i * 17 + j * 3 + 200) as u32) & 0x3FF) as u16);
            v.push((((i * 5 + j * 29 + 700) as u32) & 0x3FF) as u16);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&v),
            },
        ],
    }
}

fn synth_yuv422p10(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h;
    let mut y = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            y.push((((i * 11 + j * 7) as u32) & 0x3FF) as u16);
        }
    }
    let mut u = Vec::with_capacity(cw * ch);
    let mut v = Vec::with_capacity(cw * ch);
    for j in 0..ch {
        for i in 0..cw {
            u.push((((i * 19 + j * 4 + 150) as u32) & 0x3FF) as u16);
            v.push((((i * 3 + j * 27 + 850) as u32) & 0x3FF) as u16);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&v),
            },
        ],
    }
}

#[test]
fn yuv420p10_64x64_roundtrip() {
    // Spec reference: FFV1 v3 §4.1 — `bits_per_raw_sample = 10`.
    // Full-range ramp covers 0..=1023 luma; FFV1 is lossless so the
    // decoded samples must reproduce the encoder's input exactly.
    roundtrip_one(synth_yuv420p10(64, 64), PixelFormat::Yuv420P10Le, 64, 64);
}

#[test]
fn yuv444p10_32x32_roundtrip() {
    roundtrip_one(synth_yuv444p10(32, 32), PixelFormat::Yuv444P10Le, 32, 32);
}

#[test]
fn yuv422p10_32x16_roundtrip() {
    roundtrip_one(synth_yuv422p10(32, 16), PixelFormat::Yuv422P10Le, 32, 16);
}

#[test]
fn yuv420p10_full_range_ramp() {
    // Walk every 10-bit luma value. The `width * height = 1024` buffer
    // encodes each value exactly once and verifies the mask/fold math.
    let width = 32u32;
    let height = 32u32;
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let y: Vec<u16> = (0..(w * h) as u32).map(|i| (i & 0x3FF) as u16).collect();
    let u: Vec<u16> = (0..(cw * ch) as u32).map(|i| (i & 0x3FF) as u16).collect();
    let v: Vec<u16> = (0..(cw * ch) as u32)
        .map(|i| ((1023 - (i & 0x3FF)) & 0x3FF) as u16)
        .collect();
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&v),
            },
        ],
    };
    roundtrip_one(frame, PixelFormat::Yuv420P10Le, width, height);
}

#[test]
fn yuv444_128x96_large_random_roundtrip() {
    // Bigger frame with pseudo-random samples — exercises many contexts.
    let w = 128u32;
    let h = 96u32;
    let wu = w as usize;
    let hu = h as usize;
    let mut rng: u32 = 0xc0ffee00;
    let mut rand = || {
        rng = rng.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (rng >> 16) as u8
    };
    let y: Vec<u8> = (0..wu * hu).map(|_| rand()).collect();
    let u: Vec<u8> = (0..wu * hu).map(|_| rand()).collect();
    let v: Vec<u8> = (0..wu * hu).map(|_| rand()).collect();
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: wu,
                data: y,
            },
            VideoPlane {
                stride: wu,
                data: u,
            },
            VideoPlane {
                stride: wu,
                data: v,
            },
        ],
    };
    roundtrip_one(frame, PixelFormat::Yuv444P, w, h);
}

// -------------------------------------------------------------------------
// Multi-slice encode round-trips
// -------------------------------------------------------------------------

fn roundtrip_with_slices(
    frame: VideoFrame,
    pix: PixelFormat,
    width: u32,
    height: u32,
    slices: u32,
) {
    let mut params = make_params(pix, width, height);
    params.options.insert("slices", slices.to_string());

    let mut enc = make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    assert!(pkt.flags.keyframe);

    let dec_params = enc.output_params().clone();
    let mut dec = make_decoder(&dec_params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    match out {
        Frame::Video(v) => assert_frames_equal(&v, &frame, pix, width, height),
        _ => panic!("decoder returned non-video frame"),
    }
}

#[test]
fn yuv420_multi_slice_2x2_roundtrip() {
    roundtrip_with_slices(synth_yuv420(64, 48), PixelFormat::Yuv420P, 64, 48, 4);
}

#[test]
fn yuv420_multi_slice_4_roundtrip() {
    roundtrip_with_slices(synth_yuv420(128, 96), PixelFormat::Yuv420P, 128, 96, 4);
}

#[test]
fn yuv444_multi_slice_9_roundtrip() {
    // 3x3 grid on a 96x96 frame — each slice is 32x32.
    roundtrip_with_slices(synth_yuv444(96, 96), PixelFormat::Yuv444P, 96, 96, 9);
}

#[test]
fn yuv420_multi_slice_2_horizontal_strips_roundtrip() {
    // 2x1 grid (single column of rows) on a 64x64 frame.
    roundtrip_with_slices(synth_yuv420(64, 64), PixelFormat::Yuv420P, 64, 64, 2);
}

#[test]
fn yuv422_multi_slice_4_roundtrip() {
    // 4:2:2 with 2x2 slice grid — interior boundary at x=32 is even, good.
    roundtrip_with_slices(synth_yuv422(64, 48), PixelFormat::Yuv422P, 64, 48, 4);
}

#[test]
fn yuv420p10_multi_slice_4_roundtrip() {
    // 10-bit YUV 4:2:0 with 2x2 slice grid — the headline combo.
    roundtrip_with_slices(synth_yuv420p10(64, 64), PixelFormat::Yuv420P10Le, 64, 64, 4);
}

#[test]
fn yuv420p10_multi_slice_16_roundtrip() {
    // 4x4 grid, 128x96 at 10-bit. Each slice is 32x24; chroma 16x12.
    roundtrip_with_slices(
        synth_yuv420p10(128, 96),
        PixelFormat::Yuv420P10Le,
        128,
        96,
        16,
    );
}

#[test]
fn yuv444p10_multi_slice_4_roundtrip() {
    roundtrip_with_slices(synth_yuv444p10(64, 48), PixelFormat::Yuv444P10Le, 64, 48, 4);
}

// ---------------------------------------------------------------------
// 12-bit YUV roundtrips (Yuv{420,422,444}P12Le)
// ---------------------------------------------------------------------

/// Build a 12-bit YUV 4:2:0 frame whose luma walks the full 0..=4095 range
/// diagonally. Chroma carries its own deterministic pattern so the U/V plane
/// ordering and the >8-bit fold/mask math is exercised end-to-end.
fn synth_yuv420p12(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            // Mask into the 12-bit sample range. Stride of 16 + 11 keeps it
            // non-aligned so consecutive pixels rarely repeat a context.
            let v = ((i * 16 + j * 11) as u32) & 0x0FFF;
            y.push(v as u16);
        }
    }
    let mut u = Vec::with_capacity(cw * ch);
    let mut v = Vec::with_capacity(cw * ch);
    for j in 0..ch {
        for i in 0..cw {
            u.push((((i * 23 + j * 5 + 1000) as u32) & 0x0FFF) as u16);
            v.push((((i * 7 + j * 31 + 3500) as u32) & 0x0FFF) as u16);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&v),
            },
        ],
    }
}

fn synth_yuv422p12(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h;
    let mut y = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            y.push((((i * 11 + j * 7) as u32) & 0x0FFF) as u16);
        }
    }
    let mut u = Vec::with_capacity(cw * ch);
    let mut v = Vec::with_capacity(cw * ch);
    for j in 0..ch {
        for i in 0..cw {
            u.push((((i * 19 + j * 4 + 600) as u32) & 0x0FFF) as u16);
            v.push((((i * 3 + j * 27 + 3400) as u32) & 0x0FFF) as u16);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&v),
            },
        ],
    }
}

fn synth_yuv444p12(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut y = Vec::with_capacity(w * h);
    let mut u = Vec::with_capacity(w * h);
    let mut v = Vec::with_capacity(w * h);
    for j in 0..h {
        for i in 0..w {
            y.push((((i * 9 + j * 13) as u32) & 0x0FFF) as u16);
            u.push((((i * 17 + j * 3 + 800) as u32) & 0x0FFF) as u16);
            v.push((((i * 5 + j * 29 + 2800) as u32) & 0x0FFF) as u16);
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&v),
            },
        ],
    }
}

#[test]
fn yuv420p12_64x64_roundtrip() {
    // Spec reference: FFV1 v3 §3.8 — `bits_per_raw_sample = 12`. The new
    // encoder path picks up `Yuv420P12Le`, stamps `bits_per_raw_sample = 12`
    // into the config record, and shares the same u16 plane encoder as the
    // 10-bit path. Decode reproduces the source bit-for-bit.
    roundtrip_one(synth_yuv420p12(64, 64), PixelFormat::Yuv420P12Le, 64, 64);
}

#[test]
fn yuv422p12_32x16_roundtrip() {
    roundtrip_one(synth_yuv422p12(32, 16), PixelFormat::Yuv422P12Le, 32, 16);
}

#[test]
fn yuv444p12_32x32_roundtrip() {
    roundtrip_one(synth_yuv444p12(32, 32), PixelFormat::Yuv444P12Le, 32, 32);
}

#[test]
fn yuv420p12_full_range_ramp() {
    // 64*64 = 4096 luma samples — visit every 12-bit value exactly once.
    let width = 64u32;
    let height = 64u32;
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let y: Vec<u16> = (0..(w * h) as u32).map(|i| (i & 0x0FFF) as u16).collect();
    let u: Vec<u16> = (0..(cw * ch) as u32).map(|i| (i & 0x0FFF) as u16).collect();
    let v: Vec<u16> = (0..(cw * ch) as u32)
        .map(|i| ((4095 - (i & 0x0FFF)) & 0x0FFF) as u16)
        .collect();
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: u16_to_le(&y),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&u),
            },
            VideoPlane {
                stride: cw * 2,
                data: u16_to_le(&v),
            },
        ],
    };
    roundtrip_one(frame, PixelFormat::Yuv420P12Le, width, height);
}

#[test]
fn yuv420p12_multi_slice_4_roundtrip() {
    // 12-bit YUV 4:2:0 split across a 2x2 grid — exercises both new format
    // wiring and the multi-slice u16 path.
    roundtrip_with_slices(synth_yuv420p12(64, 64), PixelFormat::Yuv420P12Le, 64, 64, 4);
}

#[test]
fn yuv444p12_multi_slice_4_roundtrip() {
    roundtrip_with_slices(synth_yuv444p12(64, 48), PixelFormat::Yuv444P12Le, 64, 48, 4);
}

#[test]
fn yuv420p12_golomb_64x64_roundtrip() {
    // Golomb-Rice (coder_type=0) at 12-bit — FFmpeg's `-coder 0 -pix_fmt
    // yuv420p12le` shape. Our u16 Golomb encoder already accepts 9..=16, so
    // the new pixel format slots straight in.
    let frame = synth_yuv420p12(64, 64);
    let mut params = make_params(PixelFormat::Yuv420P12Le, 64, 64);
    params.options.insert("coder_type", "0".to_string());

    let mut enc = make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    assert!(pkt.flags.keyframe);

    let dec_params = enc.output_params().clone();
    let mut dec = make_decoder(&dec_params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    match out {
        Frame::Video(v) => {
            assert_frames_equal(&v, &frame, PixelFormat::Yuv420P12Le, 64, 64);
        }
        _ => panic!("decoder returned non-video frame"),
    }
}

// ---------------------------------------------------------------------
// 8-bit RGB (JPEG 2000 RCT) roundtrip
// ---------------------------------------------------------------------

fn synth_rgb24(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let mut rgb = vec![0u8; w * h * 3];
    for j in 0..h {
        for i in 0..w {
            let base = (j * w + i) * 3;
            rgb[base] = ((i * 7 + j * 3 + 32) & 0xFF) as u8; // R
            rgb[base + 1] = ((i * 11 + j * 5 + 128) & 0xFF) as u8; // G
            rgb[base + 2] = ((i * 17 + j * 13 + 200) & 0xFF) as u8; // B
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: w * 3,
            data: rgb,
        }],
    }
}

#[test]
fn rgb24_16x16_roundtrip() {
    roundtrip_one(synth_rgb24(16, 16), PixelFormat::Rgb24, 16, 16);
}

#[test]
fn rgb24_64x48_roundtrip() {
    roundtrip_one(synth_rgb24(64, 48), PixelFormat::Rgb24, 64, 48);
}

// ---------------------------------------------------------------------
// Range-coded YUVA (extra_plane alpha on the range coder path)
// ---------------------------------------------------------------------

fn synth_yuva420(width: u32, height: u32) -> VideoFrame {
    // Yuva420P: Y at full res, U/V at half, A at full res. Mirrors the
    // shape used by the Golomb-Rice path tests; we re-use it here for the
    // range-coded path.
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    let mut a = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            y[j * w + i] = ((i * 7 + j * 11 + 16) & 0xFF) as u8;
            // Make alpha carry an actual edge so the predictor sees a real
            // signal, not a flat surface (which would compress trivially
            // and not exercise the new path).
            a[j * w + i] = if (i + j) & 1 == 0 {
                ((i * 3 + j * 5) & 0xFF) as u8
            } else {
                ((i * 5 + j * 3 + 64) & 0xFF) as u8
            };
        }
    }
    for j in 0..ch {
        for i in 0..cw {
            u[j * cw + i] = ((i * 19 + j * 3 + 64) & 0xFF) as u8;
            v[j * cw + i] = ((i * 5 + j * 23 + 128) & 0xFF) as u8;
        }
    }
    VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane { stride: w, data: y },
            VideoPlane {
                stride: cw,
                data: u,
            },
            VideoPlane {
                stride: cw,
                data: v,
            },
            VideoPlane { stride: w, data: a },
        ],
    }
}

fn assert_yuva_frames_equal(a: &VideoFrame, b: &VideoFrame, width: u32, height: u32) {
    assert_eq!(a.planes.len(), 4, "decoded yuva must have 4 planes");
    assert_eq!(b.planes.len(), 4, "input yuva must have 4 planes");
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let dims = [(w, h), (cw, ch), (cw, ch), (w, h)];
    for (i, (pa, pb)) in a.planes.iter().zip(b.planes.iter()).enumerate() {
        let (pw, ph) = dims[i];
        for y in 0..ph {
            let row_a = &pa.data[y * pa.stride..y * pa.stride + pw];
            let row_b = &pb.data[y * pb.stride..y * pb.stride + pw];
            assert_eq!(row_a, row_b, "yuva plane {} row {} mismatch", i, y);
        }
    }
}

fn yuva_roundtrip(frame: VideoFrame, coder_type: u32, slices: u32, width: u32, height: u32) {
    let mut params = make_params(PixelFormat::Yuva420P, width, height);
    params.options.insert("coder_type", coder_type.to_string());
    params.options.insert("slices", slices.to_string());

    let mut enc = make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    assert!(pkt.flags.keyframe);

    let dec_params = enc.output_params().clone();
    let mut dec = make_decoder(&dec_params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    match out {
        Frame::Video(v) => assert_yuva_frames_equal(&v, &frame, width, height),
        _ => panic!("decoder returned non-video frame"),
    }
}

#[test]
fn yuva420_range_coded_64x48_roundtrip() {
    // Single-slice range-coded YUVA — the primary new path.
    yuva_roundtrip(synth_yuva420(64, 48), 1, 1, 64, 48);
}

#[test]
fn yuva420_range_coded_multi_slice_roundtrip() {
    // 2x2 slice grid range-coded YUVA on a 64x48 frame: each slice 32x24
    // (chroma 16x12, alpha 32x24).
    yuva_roundtrip(synth_yuva420(64, 48), 1, 4, 64, 48);
}

#[test]
fn yuva420_range_coded_smaller_than_golomb() {
    // The motivating reason to add range-coded YUVA: the range coder gives a
    // materially smaller bitstream than Golomb-Rice on textured alpha. This
    // test pins the relationship — if range encode ever regresses to >=
    // Golomb size on this fixture something is wrong with state seeding.
    let frame = synth_yuva420(128, 96);
    let make_params_with = |coder_type: u32| {
        let mut p = make_params(PixelFormat::Yuva420P, 128, 96);
        p.options.insert("coder_type", coder_type.to_string());
        p
    };
    let encode = |coder_type: u32| -> usize {
        let params = make_params_with(coder_type);
        let mut enc = make_encoder(&params).expect("make_encoder");
        enc.send_frame(&Frame::Video(frame.clone()))
            .expect("send_frame");
        enc.receive_packet().expect("receive_packet").data.len()
    };
    let rc = encode(1);
    let golomb = encode(0);
    assert!(
        rc < golomb,
        "range-coded YUVA should be smaller than Golomb on textured fixture: rc={} golomb={}",
        rc,
        golomb
    );
}

// ---------------------------------------------------------------------
// 8-bit RGB (JPEG 2000 RCT) roundtrip continued
// ---------------------------------------------------------------------

#[test]
fn rgb24_solid_colors_roundtrip() {
    // Each colour is itself uniform — exercises the predictor paths more
    // aggressively than the procedural pattern.
    for &(r, g, b) in &[
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ] {
        let w = 16usize;
        let h = 12usize;
        let mut rgb = vec![0u8; w * h * 3];
        for j in 0..h {
            for i in 0..w {
                let base = (j * w + i) * 3;
                rgb[base] = r;
                rgb[base + 1] = g;
                rgb[base + 2] = b;
            }
        }
        let frame = VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: w * 3,
                data: rgb,
            }],
        };
        roundtrip_one(frame, PixelFormat::Rgb24, w as u32, h as u32);
    }
}
