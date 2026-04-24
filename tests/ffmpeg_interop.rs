//! Interoperability tests with FFmpeg. These write a MKV with FFV1 via our
//! encoder and verify that FFmpeg decodes it back to the same pixels (2),
//! then generate a real FFV1-in-MKV with FFmpeg and verify that our decoder
//! reproduces a sensible YUV plane (3).
//!
//! The tests are skipped (and print a message) when the `ffmpeg` binary is
//! not on `PATH`, so they don't break builds in sandboxed environments.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use oxideav_container::WriteSeek;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Frame, MediaType, Packet, PixelFormat, Rational, StreamInfo,
    TimeBase, VideoFrame,
};
use oxideav_ffv1::encoder::make_encoder;

/// Returns true if `ffmpeg` is available.
fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tmp_dir() -> PathBuf {
    let t = std::env::temp_dir().join(format!("oxideav-ffv1-{}", std::process::id()));
    let _ = fs::create_dir_all(&t);
    t
}

/// Run the encode→mux pipeline into a fresh file on disk. Returns the file
/// path.
fn encode_frame_to_mkv_file(frame: &VideoFrame, path: &Path) {
    encode_frame_to_mkv_file_with_slices(frame, path, 1);
}

fn encode_frame_to_mkv_file_with_slices(frame: &VideoFrame, path: &Path, slices: u32) {
    let mut params = CodecParameters::video(CodecId::new("ffv1"));
    params.width = Some(frame.width);
    params.height = Some(frame.height);
    params.pixel_format = Some(frame.format);
    params.frame_rate = Some(Rational::new(1, 1));
    if slices > 1 {
        params.options.insert("slices", slices.to_string());
    }

    let mut enc = make_encoder(&params).expect("make_encoder");
    enc.send_frame(&Frame::Video(frame.clone())).expect("send");
    let mut pkt = enc.receive_packet().expect("packet");
    pkt.stream_index = 0;
    pkt.pts = Some(0);
    pkt.dts = Some(0);
    pkt.duration = Some(1);

    let out_params = enc.output_params().clone();
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: None,
        start_time: None,
        params: out_params,
    };

    let file = std::fs::File::create(path).expect("create mkv");
    let sink: Box<dyn WriteSeek> = Box::new(file);
    let mut mux = oxideav_mkv::mux::open(sink, &[stream]).expect("muxer");
    mux.write_header().expect("header");
    mux.write_packet(&pkt).expect("packet");
    mux.write_trailer().expect("trailer");
}

fn synth_checkerboard(width: u32, height: u32) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut y = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            // 8-pixel checkerboard in luma.
            let on = ((i / 8) + (j / 8)) & 1 == 0;
            y[j * w + i] = if on { 30 } else { 220 };
        }
    }
    let u = vec![128u8; cw * ch];
    let v = vec![128u8; cw * ch];
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 1),
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

fn synth_flat(width: u32, height: u32, yy: u8, uu: u8, vv: u8) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    VideoFrame {
        format: PixelFormat::Yuv420P,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 1),
        planes: vec![
            VideoPlane {
                stride: w,
                data: vec![yy; w * h],
            },
            VideoPlane {
                stride: cw,
                data: vec![uu; cw * ch],
            },
            VideoPlane {
                stride: cw,
                data: vec![vv; cw * ch],
            },
        ],
    }
}

#[test]
fn ffmpeg_decodes_flat_gray_frame() {
    if !ffmpeg_available() {
        eprintln!("flat-gray: ffmpeg not on PATH, skipping");
        return;
    }
    let width = 8u32;
    let height = 8u32;
    let frame = synth_flat(width, height, 128, 128, 128);

    let dir = tmp_dir();
    let mkv = dir.join("flat.mkv");
    let yuv = dir.join("flat.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&yuv);
    encode_frame_to_mkv_file(&frame, &mkv);

    let output = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&yuv)
        .output()
        .expect("ffmpeg spawn");
    if !output.status.success() {
        panic!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let decoded = fs::read(&yuv).unwrap();
    assert!(
        decoded.iter().all(|&b| b == 128),
        "got {:?}",
        &decoded[..16]
    );
}

#[test]
fn ffmpeg_decodes_our_encoder_output() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg_decodes_our_encoder_output: ffmpeg binary not on PATH, skipping");
        return;
    }
    let width = 32u32;
    let height = 32u32;
    let frame = synth_checkerboard(width, height);

    let dir = tmp_dir();
    let mkv = dir.join("oxideav-out.mkv");
    let yuv = dir.join("oxideav-out.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&yuv);
    encode_frame_to_mkv_file(&frame, &mkv);

    // Run ffmpeg -i mkv -f rawvideo yuv
    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(&mkv)
        .arg("-f")
        .arg("rawvideo")
        .arg(&yuv)
        .output()
        .expect("ffmpeg spawn");
    if !output.status.success() {
        panic!(
            "ffmpeg refused to decode our FFV1: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let decoded = fs::read(&yuv).expect("read yuv");
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let expected_len = w * h + 2 * cw * ch;
    assert_eq!(
        decoded.len(),
        expected_len,
        "raw yuv size mismatch: expected {}, got {}",
        expected_len,
        decoded.len()
    );

    // Compare each plane to the source frame.
    let y_src = &frame.planes[0].data[..w * h];
    assert_eq!(
        &decoded[..w * h],
        y_src,
        "Y plane mismatch after ffmpeg decode"
    );
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    let u_src = &frame.planes[1].data[..cw * ch];
    let v_src = &frame.planes[2].data[..cw * ch];
    assert_eq!(&decoded[u_off..u_off + cw * ch], u_src, "U plane mismatch");
    assert_eq!(&decoded[v_off..v_off + cw * ch], v_src, "V plane mismatch");
}

/// Verify FFmpeg can decode an 8-bit YUV 4:2:0 stream that our encoder split
/// across a 2×2 slice grid. Exercises the multi-slice encode path
/// (`Ffv1EncoderOptions::slices = 4`) end-to-end against a third-party
/// reference decoder.
#[test]
fn ffmpeg_decodes_our_multi_slice_output() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg_decodes_our_multi_slice_output: ffmpeg not on PATH, skipping");
        return;
    }
    // 64x64 splits into a 2x2 grid of 32x32 slices. Chroma is 16x16 per
    // slice (multiple of 2 on each axis — see `factor_slice_count`).
    let width = 64u32;
    let height = 64u32;
    let frame = synth_checkerboard(width, height);

    let dir = tmp_dir();
    let mkv = dir.join("oxideav-multi-slice.mkv");
    let yuv = dir.join("oxideav-multi-slice.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&yuv);
    encode_frame_to_mkv_file_with_slices(&frame, &mkv, 4);

    let output = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&yuv)
        .output()
        .expect("ffmpeg spawn");
    if !output.status.success() {
        panic!(
            "ffmpeg refused our multi-slice FFV1: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let decoded = fs::read(&yuv).expect("read yuv");
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let expected_len = w * h + 2 * cw * ch;
    assert_eq!(decoded.len(), expected_len);

    let y_src = &frame.planes[0].data[..w * h];
    assert_eq!(&decoded[..w * h], y_src, "Y plane mismatch");
    let u_off = w * h;
    let v_off = u_off + cw * ch;
    let u_src = &frame.planes[1].data[..cw * ch];
    let v_src = &frame.planes[2].data[..cw * ch];
    assert_eq!(&decoded[u_off..u_off + cw * ch], u_src, "U plane mismatch");
    assert_eq!(&decoded[v_off..v_off + cw * ch], v_src, "V plane mismatch");
}

/// Verify FFmpeg can decode a 10-bit YUV 4:2:0 stream that our encoder split
/// across a 2×2 slice grid. This is the headline combo of round 5:
/// (10-bit YUV encode) ∩ (multi-slice encode).
#[test]
fn ffmpeg_decodes_our_10bit_multi_slice_output() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg_decodes_our_10bit_multi_slice_output: ffmpeg not on PATH, skipping");
        return;
    }
    let width = 64u32;
    let height = 64u32;
    let w = width as usize;
    let h = height as usize;
    let cw = w / 2;
    let ch = h / 2;

    // Deterministic 10-bit ramp. Luma walks the full 10-bit range;
    // chroma carries its own pattern to make sure U/V ordering is right.
    let y10: Vec<u16> = (0..w * h).map(|i| (i as u16) & 0x3FF).collect();
    let u10: Vec<u16> = (0..cw * ch).map(|i| ((i * 3) as u16) & 0x3FF).collect();
    let v10: Vec<u16> = (0..cw * ch).map(|i| (0x3FF - (i & 0x3FF)) as u16).collect();

    let y_bytes: Vec<u8> = y10.iter().flat_map(|&x| x.to_le_bytes()).collect();
    let u_bytes: Vec<u8> = u10.iter().flat_map(|&x| x.to_le_bytes()).collect();
    let v_bytes: Vec<u8> = v10.iter().flat_map(|&x| x.to_le_bytes()).collect();

    let frame = VideoFrame {
        format: PixelFormat::Yuv420P10Le,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 1),
        planes: vec![
            VideoPlane {
                stride: w * 2,
                data: y_bytes.clone(),
            },
            VideoPlane {
                stride: cw * 2,
                data: u_bytes.clone(),
            },
            VideoPlane {
                stride: cw * 2,
                data: v_bytes.clone(),
            },
        ],
    };

    let dir = tmp_dir();
    let mkv = dir.join("oxideav-10b-multi-slice.mkv");
    let yuv = dir.join("oxideav-10b-multi-slice.raw");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&yuv);
    encode_frame_to_mkv_file_with_slices(&frame, &mkv, 4);

    // Ask ffmpeg to decode back into raw yuv420p10le (planar, u16 LE).
    let output = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&mkv)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p10le"])
        .arg(&yuv)
        .output()
        .expect("ffmpeg spawn");
    if !output.status.success() {
        panic!(
            "ffmpeg refused our 10-bit multi-slice FFV1: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let decoded = fs::read(&yuv).expect("read raw");
    let y_len = w * h * 2;
    let c_len = cw * ch * 2;
    assert_eq!(decoded.len(), y_len + 2 * c_len);
    assert_eq!(&decoded[..y_len], &y_bytes[..], "Y plane (u16 LE) mismatch");
    assert_eq!(
        &decoded[y_len..y_len + c_len],
        &u_bytes[..],
        "U plane (u16 LE) mismatch"
    );
    assert_eq!(
        &decoded[y_len + c_len..y_len + 2 * c_len],
        &v_bytes[..],
        "V plane (u16 LE) mismatch"
    );
}

#[test]
fn our_decoder_accepts_ffmpeg_output() {
    if !ffmpeg_available() {
        eprintln!("our_decoder_accepts_ffmpeg_output: ffmpeg not on PATH, skipping");
        return;
    }

    let dir = tmp_dir();
    let mkv = dir.join("ref-ffv1-frame.mkv");
    let _ = fs::remove_file(&mkv);

    // FFmpeg-generated FFV1 reference: a testsrc colour-bar frame at 64×64.
    // FFmpeg's default coder is Golomb-Rice (`coder=0`); force the
    // range-coder path (`coder=1`, range_def) because that's what our
    // decoder supports.
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg("testsrc=d=1:s=64x64:r=1")
        .arg("-c:v")
        .arg("ffv1")
        .arg("-level")
        .arg("3")
        .arg("-coder")
        .arg("range_def")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-frames:v")
        .arg("1")
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to produce reference file");

    // Demux via oxideav-mkv. Pull the first video packet and the codec
    // parameters (including extradata).
    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let streams = demux.streams().to_vec();
    let vstream = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Video)
        .expect("no video stream");
    let vindex = vstream.index;
    let params = vstream.params.clone();

    let mut pkt: Option<Packet> = None;
    for _ in 0..64 {
        match demux.next_packet() {
            Ok(p) => {
                if p.stream_index == vindex {
                    pkt = Some(p);
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        }
    }
    let pkt = pkt.expect("no video packet demuxed");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("decoder returned non-video frame");
    };

    // testsrc produces a colour-bar pattern; the Y plane's mean must land
    // comfortably away from both extremes of the 0..=255 range.
    let y_plane = &vf.planes[0];
    let w = vf.width as usize;
    let h = vf.height as usize;
    let mut sum: u64 = 0;
    for row in 0..h {
        let off = row * y_plane.stride;
        for i in 0..w {
            sum += y_plane.data[off + i] as u64;
        }
    }
    let mean = sum as f64 / (w * h) as f64;
    assert!(
        (30.0..=230.0).contains(&mean),
        "Y plane mean = {mean} outside [30, 230] — decoder probably produced garbage"
    );
}

/// Generate a Golomb-Rice encoded FFV1 frame with ffmpeg (`-coder 0`) and
/// decode it bit-exactly with our decoder. We cross-check against the raw
/// YUV that ffmpeg itself produces from the same file — FFV1 is lossless,
/// so the two outputs must match byte-for-byte.
#[test]
fn our_decoder_accepts_ffmpeg_golomb_output() {
    if !ffmpeg_available() {
        eprintln!("our_decoder_accepts_ffmpeg_golomb_output: ffmpeg not on PATH, skipping");
        return;
    }

    let dir = tmp_dir();
    let mkv = dir.join("ref-ffv1-golomb.mkv");
    let ref_yuv = dir.join("ref-ffv1-golomb.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_yuv);

    // `-coder 0` selects Golomb-Rice. Use yuv420p at 64×64 for a compact
    // test — the testsrc filter yields a deterministic colour-bar pattern.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=64x64:r=1"])
        .args(["-c:v", "ffv1"])
        .args(["-level", "3"])
        .args(["-coder", "0"])
        .args(["-pix_fmt", "yuv420p"])
        .args(["-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(
        status.success(),
        "ffmpeg failed to produce Golomb reference file"
    );

    // Have ffmpeg decode the same file into raw YUV so we have a ground
    // truth to compare against.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_yuv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to decode Golomb reference");

    let ref_bytes = fs::read(&ref_yuv).expect("read ref yuv");
    let width = 64usize;
    let height = 64usize;
    let cw = width / 2;
    let ch = height / 2;
    let y_len = width * height;
    let c_len = cw * ch;
    assert_eq!(ref_bytes.len(), y_len + 2 * c_len);

    // Demux our way and feed into our decoder.
    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let streams = demux.streams().to_vec();
    let vstream = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Video)
        .expect("no video stream");
    let vindex = vstream.index;
    let params = vstream.params.clone();

    let mut pkt: Option<Packet> = None;
    for _ in 0..64 {
        match demux.next_packet() {
            Ok(p) => {
                if p.stream_index == vindex {
                    pkt = Some(p);
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        }
    }
    let pkt = pkt.expect("no video packet demuxed");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("decoder returned non-video frame");
    };

    // Compare each plane byte-for-byte with what ffmpeg decoded.
    let y_ref = &ref_bytes[..y_len];
    let u_ref = &ref_bytes[y_len..y_len + c_len];
    let v_ref = &ref_bytes[y_len + c_len..];

    let y_got = &vf.planes[0].data[..y_len];
    let u_got = &vf.planes[1].data[..c_len];
    let v_got = &vf.planes[2].data[..c_len];

    assert_eq!(
        y_got,
        y_ref,
        "Y plane mismatch (Golomb decode); first diff around byte {}",
        first_diff(y_got, y_ref)
    );
    assert_eq!(u_got, u_ref, "U plane mismatch (Golomb decode)");
    assert_eq!(v_got, v_ref, "V plane mismatch (Golomb decode)");
}

fn first_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .unwrap_or(a.len())
}

/// Multi-frame Golomb YUV 4:2:2 run. Exercises larger slice grids, chroma
/// subsampling on the horizontal axis only, and a bigger frame — a common
/// FFV1 production configuration.
#[test]
fn our_decoder_accepts_ffmpeg_golomb_yuv422_multiframe() {
    if !ffmpeg_available() {
        eprintln!("golomb_yuv422_multiframe: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("golomb-422.mkv");
    let ref_yuv = dir.join("golomb-422.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_yuv);

    // Force every frame to be a keyframe (`-g 1`): FFV1 is intra-only from
    // our decoder's point of view because we don't retain VLC state across
    // frames yet — `intra=0` streams with actual non-keyframes would need
    // that persistence.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=128x96:r=3"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "0"])
        .args(["-g", "1"])
        .args(["-pix_fmt", "yuv422p"])
        .args(["-frames:v", "3"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_yuv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_yuv).expect("read ref yuv");

    let width = 128usize;
    let height = 96usize;
    let cw = width / 2;
    let ch = height;
    let frame_bytes = width * height + 2 * cw * ch;

    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let streams = demux.streams().to_vec();
    let params = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Video)
        .expect("video stream")
        .params
        .clone();

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    let mut frame_idx = 0usize;
    loop {
        let pkt = match demux.next_packet() {
            Ok(p) => p,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux: {e:?}"),
        };
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
            panic!("non-video");
        };
        let off = frame_idx * frame_bytes;
        let y_ref = &ref_bytes[off..off + width * height];
        let u_ref = &ref_bytes[off + width * height..off + width * height + cw * ch];
        let v_ref = &ref_bytes[off + width * height + cw * ch..off + frame_bytes];

        // Compare per-plane respecting strides (may equal or differ from w).
        let y_got_stride = vf.planes[0].stride;
        for row in 0..height {
            let got = &vf.planes[0].data[row * y_got_stride..row * y_got_stride + width];
            let expected = &y_ref[row * width..row * width + width];
            assert_eq!(
                got,
                expected,
                "frame {frame_idx} Y row {row}: first diff {}",
                first_diff(got, expected)
            );
        }
        let u_stride = vf.planes[1].stride;
        for row in 0..ch {
            let got = &vf.planes[1].data[row * u_stride..row * u_stride + cw];
            let expected = &u_ref[row * cw..row * cw + cw];
            assert_eq!(got, expected, "frame {frame_idx} U row {row}");
        }
        let v_stride = vf.planes[2].stride;
        for row in 0..ch {
            let got = &vf.planes[2].data[row * v_stride..row * v_stride + cw];
            let expected = &v_ref[row * cw..row * cw + cw];
            assert_eq!(got, expected, "frame {frame_idx} V row {row}");
        }
        frame_idx += 1;
    }
    assert_eq!(frame_idx, 3, "expected 3 frames decoded");
}

/// Flat-gray Golomb file → our decoder. Every Y sample should equal the
/// reference colour (16 for TV-legal black). Useful for isolating the
/// bit-reader / run-mode path from the scalar VLC path.
#[test]
fn our_decoder_golomb_flat_gray() {
    if !ffmpeg_available() {
        eprintln!("golomb_flat_gray: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("golomb-flat.mkv");
    let ref_yuv = dir.join("golomb-flat.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_yuv);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "color=c=black:s=16x16:d=1:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "0"])
        .args(["-pix_fmt", "yuv420p", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_yuv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_yuv).expect("read ref yuv");

    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let streams = demux.streams().to_vec();
    let params = streams[0].params.clone();
    let pkt = demux.next_packet().expect("packet");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("non-video frame");
    };
    let width = 16usize;
    let height = 16usize;
    let cw = width / 2;
    let ch = height / 2;
    assert_eq!(
        &vf.planes[0].data[..width * height],
        &ref_bytes[..width * height],
        "Y plane mismatch — first diff at byte {}",
        first_diff(
            &vf.planes[0].data[..width * height],
            &ref_bytes[..width * height]
        )
    );
    let u_off = width * height;
    assert_eq!(
        &vf.planes[1].data[..cw * ch],
        &ref_bytes[u_off..u_off + cw * ch],
        "U plane mismatch"
    );
    let v_off = u_off + cw * ch;
    assert_eq!(
        &vf.planes[2].data[..cw * ch],
        &ref_bytes[v_off..v_off + cw * ch],
        "V plane mismatch"
    );
}

/// Generate an FFV1 file with `-pix_fmt gbrp` (JPEG 2000 RCT, 8-bit) and
/// assert our decoder reproduces the same packed RGB bytes ffmpeg itself
/// produces when asked for `rgb24` output. FFV1 is lossless, so the two
/// paths must match byte-for-byte.
#[test]
fn our_decoder_accepts_ffmpeg_rgb_rct() {
    if !ffmpeg_available() {
        eprintln!("rgb_rct: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("rgb-rct.mkv");
    let ref_rgb = dir.join("rgb-rct.rgb24");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_rgb);

    // Encode FFV1 in RCT mode. `-coder 1` forces the range coder (our
    // supported mode for RCT). `-pix_fmt gbrp` = FFmpeg's planar
    // Green/Blue/Red which is the native RCT plane order on the wire.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=48x32:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "1"])
        .args(["-pix_fmt", "gbrp", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to produce RCT reference");

    // Have ffmpeg decode the file into interleaved rgb24 as ground truth.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(&ref_rgb)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to decode RCT reference");
    let ref_bytes = fs::read(&ref_rgb).expect("read ref rgb");

    let width = 48usize;
    let height = 32usize;
    assert_eq!(ref_bytes.len(), width * height * 3);

    // Demux and feed into our decoder.
    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let streams = demux.streams().to_vec();
    let vstream = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Video)
        .expect("no video stream");
    let vindex = vstream.index;
    let params = vstream.params.clone();

    let mut pkt: Option<Packet> = None;
    for _ in 0..64 {
        match demux.next_packet() {
            Ok(p) => {
                if p.stream_index == vindex {
                    pkt = Some(p);
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        }
    }
    let pkt = pkt.expect("no video packet demuxed");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("decoder returned non-video frame");
    };
    assert_eq!(vf.format, PixelFormat::Rgb24);
    assert_eq!(vf.width as usize, width);
    assert_eq!(vf.height as usize, height);
    assert_eq!(
        vf.planes.len(),
        1,
        "RCT output should be single interleaved plane"
    );

    // Compare each row respecting the stride ffmpeg writes (= width * 3).
    let got_stride = vf.planes[0].stride;
    assert_eq!(got_stride, width * 3);
    assert_eq!(
        &vf.planes[0].data[..width * height * 3],
        &ref_bytes[..],
        "RGB mismatch (RCT decode); first diff at byte {}",
        first_diff(&vf.planes[0].data[..width * height * 3], &ref_bytes[..])
    );
}

/// Exercise the RCT path with a more "organic" colour distribution
/// (mandelbrot filter) to catch transforms we might get right on a flat
/// colour bar pattern but wrong when the RCT residuals cover the whole
/// signed range.
#[test]
fn our_decoder_accepts_ffmpeg_rgb_rct_mandelbrot() {
    if !ffmpeg_available() {
        eprintln!("rgb_rct_mandelbrot: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("rgb-rct-mandel.mkv");
    let ref_rgb = dir.join("rgb-rct-mandel.rgb24");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_rgb);
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "mandelbrot=s=80x60:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "1"])
        .args(["-pix_fmt", "gbrp", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(&ref_rgb)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_rgb).expect("read");

    let input: Box<dyn oxideav_container::ReadSeek> = Box::new(fs::File::open(&mkv).expect("open"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();
    let pkt = demux.next_packet().expect("pkt");
    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(vf) = dec.receive_frame().expect("recv") else {
        panic!("non-video")
    };
    assert_eq!(
        &vf.planes[0].data[..80 * 60 * 3],
        &ref_bytes[..],
        "mandelbrot RCT mismatch at byte {}",
        first_diff(&vf.planes[0].data[..80 * 60 * 3], &ref_bytes[..])
    );
}

/// RCT stream with a multi-slice grid (`-slices 4`) — exercises the slice
/// boundary arithmetic for RGB colourspace.
#[test]
fn our_decoder_accepts_ffmpeg_rgb_rct_multislice() {
    if !ffmpeg_available() {
        eprintln!("rgb_rct_multislice: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("rgb-rct-ms.mkv");
    let ref_rgb = dir.join("rgb-rct-ms.rgb24");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_rgb);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=64x64:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "1"])
        .args(["-pix_fmt", "gbrp", "-slices", "4", "-slicecrc", "1"])
        .args(["-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(&ref_rgb)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_rgb).expect("read ref rgb");

    let width = 64usize;
    let height = 64usize;

    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let streams = demux.streams().to_vec();
    let params = streams[0].params.clone();
    let pkt = demux.next_packet().expect("packet");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send_packet");
    let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
        panic!("non-video frame");
    };
    assert_eq!(
        &vf.planes[0].data[..width * height * 3],
        &ref_bytes[..],
        "multi-slice RCT mismatch; first diff at byte {}",
        first_diff(&vf.planes[0].data[..width * height * 3], &ref_bytes[..])
    );
}

/// Decode FFmpeg-produced `yuv420p10le` FFV1. Exercises the per-config quant
/// tables path (FFmpeg ships `quant9_10bit` for 10-bit streams, different
/// from the 8-bit quant11 default) together with the >8-bit sample path.
#[test]
fn our_decoder_accepts_ffmpeg_yuv420p10le() {
    if !ffmpeg_available() {
        eprintln!("yuv420p10le: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("yuv420p10le.mkv");
    let ref_raw = dir.join("yuv420p10le.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_raw);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=48x32:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "1"])
        .args(["-pix_fmt", "yuv420p10le", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_raw)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_raw).expect("read ref");

    let width = 48usize;
    let height = 32usize;
    let cw = width / 2;
    let ch = height / 2;
    let y_len = width * height * 2;
    let c_len = cw * ch * 2;
    assert_eq!(ref_bytes.len(), y_len + 2 * c_len);

    let input: Box<dyn oxideav_container::ReadSeek> = Box::new(fs::File::open(&mkv).expect("open"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();
    let pkt = demux.next_packet().expect("pkt");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(vf) = dec.receive_frame().expect("recv") else {
        panic!("non-video")
    };
    assert_eq!(vf.format, PixelFormat::Yuv420P10Le);

    // Compare plane-by-plane respecting strides.
    let y_ref = &ref_bytes[..y_len];
    let u_ref = &ref_bytes[y_len..y_len + c_len];
    let v_ref = &ref_bytes[y_len + c_len..];
    let y_got = &vf.planes[0].data[..y_len];
    let u_got = &vf.planes[1].data[..c_len];
    let v_got = &vf.planes[2].data[..c_len];
    assert_eq!(
        y_got,
        y_ref,
        "10-bit Y mismatch at byte {}",
        first_diff(y_got, y_ref)
    );
    assert_eq!(u_got, u_ref, "10-bit U mismatch");
    assert_eq!(v_got, v_ref, "10-bit V mismatch");
}

/// Decode FFmpeg-produced FFV1 built with `-context 1`, which ships
/// `initial_state_delta` overrides per RFC §4.2.15. Our decoder applies
/// the per-context state seed instead of starting from 128.
///
/// Ignored: our config-record parser now materialises `initial_state_delta`
/// per RFC §4.2.15 and the infrastructure threads it through the slice
/// decoders, but decode still diverges mid-plane against a real FFmpeg
/// `-context 1` file. FFmpeg's `-context 1` co-enables features beyond the
/// state seed (quant-table-set binding per plane, etc.) that need further
/// investigation — kept here as a regression to enable when fixed.
#[test]
#[ignore]
fn our_decoder_accepts_ffmpeg_context1() {
    if !ffmpeg_available() {
        eprintln!("context1: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("ctx1.mkv");
    let ref_raw = dir.join("ctx1.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_raw);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=64x48:r=1"])
        .args([
            "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
        ])
        .args(["-pix_fmt", "yuv420p", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_raw)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_raw).expect("read ref");

    let width = 64usize;
    let height = 48usize;
    let cw = width / 2;
    let ch = height / 2;
    let y_len = width * height;
    let c_len = cw * ch;
    assert_eq!(ref_bytes.len(), y_len + 2 * c_len);

    let input: Box<dyn oxideav_container::ReadSeek> = Box::new(fs::File::open(&mkv).expect("open"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();
    let pkt = demux.next_packet().expect("pkt");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(vf) = dec.receive_frame().expect("recv") else {
        panic!("non-video")
    };
    assert_eq!(vf.format, PixelFormat::Yuv420P);

    let y_ref = &ref_bytes[..y_len];
    let u_ref = &ref_bytes[y_len..y_len + c_len];
    let v_ref = &ref_bytes[y_len + c_len..];
    let y_got = &vf.planes[0].data[..y_len];
    let u_got = &vf.planes[1].data[..c_len];
    let v_got = &vf.planes[2].data[..c_len];
    assert_eq!(
        y_got,
        y_ref,
        "-context 1 Y mismatch at byte {}",
        first_diff(y_got, y_ref)
    );
    assert_eq!(u_got, u_ref, "-context 1 U mismatch");
    assert_eq!(v_got, v_ref, "-context 1 V mismatch");
}

/// Decode FFmpeg-produced `yuv420p10le` FFV1 with `-slices 4` — the
/// symmetric check to `ffmpeg_decodes_our_10bit_multi_slice_output`. This
/// is what `ffmpeg -c:v ffv1 -level 3 -slices 4 -g 1 -pix_fmt yuv420p10le`
/// ships and is the most common 10-bit archival flavour.
#[test]
fn our_decoder_accepts_ffmpeg_yuv420p10le_multislice() {
    if !ffmpeg_available() {
        eprintln!("yuv420p10le_multislice: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("yuv420p10le-slices4.mkv");
    let ref_raw = dir.join("yuv420p10le-slices4.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_raw);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=64x48:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "1"])
        .args(["-pix_fmt", "yuv420p10le", "-slices", "4", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    // Reference raw: decode via ffmpeg so we can compare byte-for-byte.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_raw)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_raw).expect("read ref");

    let width = 64usize;
    let height = 48usize;
    let cw = width / 2;
    let ch = height / 2;
    let y_len = width * height * 2;
    let c_len = cw * ch * 2;
    assert_eq!(ref_bytes.len(), y_len + 2 * c_len);

    let input: Box<dyn oxideav_container::ReadSeek> = Box::new(fs::File::open(&mkv).expect("open"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();
    let pkt = demux.next_packet().expect("pkt");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(vf) = dec.receive_frame().expect("recv") else {
        panic!("non-video")
    };
    assert_eq!(vf.format, PixelFormat::Yuv420P10Le);

    let y_ref = &ref_bytes[..y_len];
    let u_ref = &ref_bytes[y_len..y_len + c_len];
    let v_ref = &ref_bytes[y_len + c_len..];
    let y_got = &vf.planes[0].data[..y_len];
    let u_got = &vf.planes[1].data[..c_len];
    let v_got = &vf.planes[2].data[..c_len];
    assert_eq!(
        y_got,
        y_ref,
        "10-bit multi-slice Y mismatch at byte {}",
        first_diff(y_got, y_ref)
    );
    assert_eq!(u_got, u_ref, "10-bit multi-slice U mismatch");
    assert_eq!(v_got, v_ref, "10-bit multi-slice V mismatch");
}

/// Decode FFmpeg-produced `yuva420p` FFV1 — exercises the `extra_plane`
/// alpha channel decoding path on an 8-bit YUV 4:2:0 stream.
#[test]
fn our_decoder_accepts_ffmpeg_yuva420p() {
    if !ffmpeg_available() {
        eprintln!("yuva420p: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("yuva420p.mkv");
    let ref_raw = dir.join("yuva420p.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_raw);

    // The `format=yuva420p` filter on testsrc gives a full-resolution alpha
    // plane (fully opaque pixels) that's still non-trivial to encode — the
    // alpha plane's sample differences are all zero, i.e. a worst case for
    // the run-length in range mode.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=32x24:r=1"])
        .args(["-vf", "format=yuva420p"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "1"])
        .args(["-pix_fmt", "yuva420p", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_raw)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_raw).expect("read");

    let width = 32usize;
    let height = 24usize;
    let cw = width / 2;
    let ch = height / 2;
    let y_len = width * height;
    let c_len = cw * ch;
    let a_len = width * height;
    assert_eq!(ref_bytes.len(), y_len + 2 * c_len + a_len);

    let input: Box<dyn oxideav_container::ReadSeek> = Box::new(fs::File::open(&mkv).expect("open"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();
    let pkt = demux.next_packet().expect("pkt");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(vf) = dec.receive_frame().expect("recv") else {
        panic!("non-video")
    };
    assert_eq!(vf.format, PixelFormat::Yuva420P);
    assert_eq!(vf.planes.len(), 4);

    let y_ref = &ref_bytes[..y_len];
    let u_ref = &ref_bytes[y_len..y_len + c_len];
    let v_ref = &ref_bytes[y_len + c_len..y_len + 2 * c_len];
    let a_ref = &ref_bytes[y_len + 2 * c_len..];
    assert_eq!(
        &vf.planes[0].data[..y_len],
        y_ref,
        "YUVA Y mismatch; first diff {}",
        first_diff(&vf.planes[0].data[..y_len], y_ref)
    );
    assert_eq!(&vf.planes[1].data[..c_len], u_ref, "YUVA U mismatch");
    assert_eq!(&vf.planes[2].data[..c_len], v_ref, "YUVA V mismatch");
    assert_eq!(
        &vf.planes[3].data[..a_len],
        a_ref,
        "YUVA A mismatch; first diff {}",
        first_diff(&vf.planes[3].data[..a_len], a_ref)
    );
}

/// Decode FFmpeg-produced `gbrp10le` FFV1 — 10-bit planar RGB via the
/// JPEG 2000 RCT path *with the §3.7.2.1 BGR Exception*. The encoder puts
/// the blue component in the Y plane (not green), Cb = g − b, Cr = r − b,
/// per RFC 9043 §3.7.2.1. Our decoder must recognise the 9..=15-bit depth
/// with `extra_plane == 0` and apply the exception formula.
#[test]
fn our_decoder_accepts_ffmpeg_gbrp10le() {
    if !ffmpeg_available() {
        eprintln!("gbrp10le: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("gbrp10le.mkv");
    let ref_raw = dir.join("gbrp10le.planar");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_raw);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=48x32:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "1"])
        .args(["-pix_fmt", "gbrp10le", "-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    // Decode back to the same planar `gbrp10le` so we're comparing 10-bit
    // samples on both sides — FFmpeg's `rgb48le` output does a scale-up
    // replication (`(v << 4) | (v >> 6)`) that muddies the comparison.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo", "-pix_fmt", "gbrp10le"])
        .arg(&ref_raw)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_raw).expect("read ref");

    let width = 48usize;
    let height = 32usize;
    let plane_bytes = width * height * 2;
    assert_eq!(ref_bytes.len(), plane_bytes * 3);
    // FFmpeg writes G, B, R planes in that order for `gbrp*`.
    let g_ref = &ref_bytes[..plane_bytes];
    let b_ref = &ref_bytes[plane_bytes..2 * plane_bytes];
    let r_ref = &ref_bytes[2 * plane_bytes..];

    let input: Box<dyn oxideav_container::ReadSeek> = Box::new(fs::File::open(&mkv).expect("open"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();
    let pkt = demux.next_packet().expect("pkt");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(vf) = dec.receive_frame().expect("recv") else {
        panic!("non-video")
    };
    assert_eq!(vf.format, PixelFormat::Rgb48Le);

    // Un-interleave our packed Rgb48Le output into planar R, G, B 16-bit
    // samples for comparison.
    let mut r_got = vec![0u8; plane_bytes];
    let mut g_got = vec![0u8; plane_bytes];
    let mut b_got = vec![0u8; plane_bytes];
    let got = &vf.planes[0].data[..width * height * 3 * 2];
    for i in 0..(width * height) {
        r_got[i * 2] = got[i * 6];
        r_got[i * 2 + 1] = got[i * 6 + 1];
        g_got[i * 2] = got[i * 6 + 2];
        g_got[i * 2 + 1] = got[i * 6 + 3];
        b_got[i * 2] = got[i * 6 + 4];
        b_got[i * 2 + 1] = got[i * 6 + 5];
    }
    assert_eq!(
        g_got,
        g_ref,
        "G plane mismatch; first diff {}",
        first_diff(&g_got, g_ref)
    );
    assert_eq!(
        b_got,
        b_ref,
        "B plane mismatch; first diff {}",
        first_diff(&b_got, b_ref)
    );
    assert_eq!(
        r_got,
        r_ref,
        "R plane mismatch; first diff {}",
        first_diff(&r_got, r_ref)
    );
}

/// Long-GOP / `intra=0` regression test. `-g 10` tells ffmpeg to emit one
/// keyframe every 10 frames; with `-c:v ffv1` this produces an `intra=0`
/// stream where every non-keyframe carries state handed over from the
/// previous frame's range coder. Our decoder must retain state per slice
/// position and only reset on the leading keyframe bit of 1, otherwise
/// mid-GOP frames decode to garbage.
#[test]
fn our_decoder_accepts_ffmpeg_intra0_long_gop() {
    if !ffmpeg_available() {
        eprintln!("intra0_long_gop: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("intra0-longgop.mkv");
    let ref_yuv = dir.join("intra0-longgop.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_yuv);

    // `-g 10` + `-c:v ffv1` → intra=0 config record. `testsrc` gives a
    // time-varying pattern so each frame's samples really are different
    // (and so non-keyframes have work to do). 12 frames total exercises
    // the keyframe bit-carrying path twice: frames 0 and 10 are keyframes,
    // frames 1-9 and 11 are non-keyframes.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=2:s=64x48:r=6"])
        .args(["-c:v", "ffv1", "-level", "3"])
        .args(["-g", "10"])
        .args(["-pix_fmt", "yuv420p"])
        .args(["-frames:v", "12"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to produce intra=0 file");

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_yuv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to decode intra=0 file");
    let ref_bytes = fs::read(&ref_yuv).expect("read ref yuv");

    let width = 64usize;
    let height = 48usize;
    let cw = width / 2;
    let ch = height / 2;
    let frame_bytes = width * height + 2 * cw * ch;
    assert_eq!(
        ref_bytes.len(),
        frame_bytes * 12,
        "ffmpeg produced an unexpected number of frames"
    );

    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    let mut frame_idx = 0usize;
    loop {
        let pkt = match demux.next_packet() {
            Ok(p) => p,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        };
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
            panic!("non-video");
        };
        let off = frame_idx * frame_bytes;
        let y_ref = &ref_bytes[off..off + width * height];
        let u_ref = &ref_bytes[off + width * height..off + width * height + cw * ch];
        let v_ref = &ref_bytes[off + width * height + cw * ch..off + frame_bytes];

        // Compare per-plane respecting strides.
        let y_stride = vf.planes[0].stride;
        for row in 0..height {
            let got = &vf.planes[0].data[row * y_stride..row * y_stride + width];
            let want = &y_ref[row * width..row * width + width];
            assert_eq!(
                got,
                want,
                "intra=0 frame {frame_idx} Y row {row}: first diff at {}",
                first_diff(got, want)
            );
        }
        let u_stride = vf.planes[1].stride;
        for row in 0..ch {
            let got = &vf.planes[1].data[row * u_stride..row * u_stride + cw];
            let want = &u_ref[row * cw..row * cw + cw];
            assert_eq!(got, want, "intra=0 frame {frame_idx} U row {row} mismatch");
        }
        let v_stride = vf.planes[2].stride;
        for row in 0..ch {
            let got = &vf.planes[2].data[row * v_stride..row * v_stride + cw];
            let want = &v_ref[row * cw..row * cw + cw];
            assert_eq!(got, want, "intra=0 frame {frame_idx} V row {row} mismatch");
        }
        frame_idx += 1;
    }
    assert_eq!(frame_idx, 12);
}

/// Same long-GOP shape but for Golomb-Rice (`-coder 0`) — exercises VLC
/// state retention across frames in addition to the keyframe-bit handling.
#[test]
fn our_decoder_accepts_ffmpeg_intra0_golomb_long_gop() {
    if !ffmpeg_available() {
        eprintln!("intra0_golomb_long_gop: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("intra0-golomb-longgop.mkv");
    let ref_yuv = dir.join("intra0-golomb-longgop.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_yuv);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=2:s=64x48:r=6"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "0"])
        .args(["-g", "10"])
        .args(["-pix_fmt", "yuv420p"])
        .args(["-frames:v", "12"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo"])
        .arg(&ref_yuv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_yuv).expect("read ref yuv");

    let width = 64usize;
    let height = 48usize;
    let cw = width / 2;
    let ch = height / 2;
    let frame_bytes = width * height + 2 * cw * ch;
    assert_eq!(ref_bytes.len(), frame_bytes * 12);

    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    let mut frame_idx = 0usize;
    loop {
        let pkt = match demux.next_packet() {
            Ok(p) => p,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        };
        dec.send_packet(&pkt).expect("send_packet");
        let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
            panic!("non-video");
        };
        let off = frame_idx * frame_bytes;
        let y_ref = &ref_bytes[off..off + width * height];
        let y_stride = vf.planes[0].stride;
        for row in 0..height {
            let got = &vf.planes[0].data[row * y_stride..row * y_stride + width];
            let want = &y_ref[row * width..row * width + width];
            assert_eq!(
                got,
                want,
                "intra=0 golomb frame {frame_idx} Y row {row}: first diff at {}",
                first_diff(got, want)
            );
        }
        frame_idx += 1;
    }
    assert_eq!(frame_idx, 12);
}

/// Our RGB (JPEG 2000 RCT) encoder output must decode bit-exactly in
/// FFmpeg. Builds a synthetic 8-bit RGB frame, wraps it in MKV via our
/// muxer, feeds it to ffmpeg, and verifies the decoded packed `rgb24`
/// bytes match the original.
#[test]
fn ffmpeg_decodes_our_rgb_output() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg_decodes_our_rgb_output: ffmpeg not on PATH, skipping");
        return;
    }
    let width = 32u32;
    let height = 24u32;
    let w = width as usize;
    let h = height as usize;
    let mut rgb = vec![0u8; w * h * 3];
    for j in 0..h {
        for i in 0..w {
            let base = (j * w + i) * 3;
            rgb[base] = ((i * 7 + j * 3 + 32) & 0xFF) as u8;
            rgb[base + 1] = ((i * 11 + j * 5 + 128) & 0xFF) as u8;
            rgb[base + 2] = ((i * 17 + j * 13 + 200) & 0xFF) as u8;
        }
    }
    let expected = rgb.clone();
    let frame = VideoFrame {
        format: PixelFormat::Rgb24,
        width,
        height,
        pts: Some(0),
        time_base: TimeBase::new(1, 1),
        planes: vec![VideoPlane {
            stride: w * 3,
            data: rgb,
        }],
    };

    let dir = tmp_dir();
    let mkv = dir.join("ours-rgb.mkv");
    let out_raw = dir.join("ours-rgb-decoded.raw");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&out_raw);
    encode_frame_to_mkv_file(&frame, &mkv);

    // Ask ffmpeg to decode the MKV and write the single frame as raw rgb24.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(&out_raw)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success(), "ffmpeg failed to decode our rgb output");

    let got = fs::read(&out_raw).expect("read decoded raw");
    assert_eq!(
        got.len(),
        expected.len(),
        "decoded raw size mismatch — ffmpeg consumed our config record but output the wrong shape"
    );
    assert_eq!(
        got,
        expected,
        "ffmpeg decoded our RGB output but got different pixels; first diff at byte {}",
        first_diff(&got, &expected)
    );
}

/// `-coder 0 -pix_fmt yuv420p10le`: 10-bit samples coded as Golomb-Rice.
/// The RFC labels this combination SHOULD NOT, but ffmpeg historically emits
/// it for 10-bit YUV with `-coder 0` so we accept and decode it.
#[test]
fn our_decoder_accepts_ffmpeg_golomb_yuv420p10le() {
    if !ffmpeg_available() {
        eprintln!("golomb_yuv420p10le: ffmpeg not on PATH, skipping");
        return;
    }
    let dir = tmp_dir();
    let mkv = dir.join("golomb-10bit.mkv");
    let ref_yuv = dir.join("golomb-10bit.yuv");
    let _ = fs::remove_file(&mkv);
    let _ = fs::remove_file(&ref_yuv);

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(["-f", "lavfi", "-i", "testsrc=d=1:s=64x48:r=1"])
        .args(["-c:v", "ffv1", "-level", "3", "-coder", "0"])
        .args(["-g", "1"])
        .args(["-pix_fmt", "yuv420p10le"])
        .args(["-frames:v", "1"])
        .arg(&mkv)
        .status()
        .expect("ffmpeg spawn");
    assert!(
        status.success(),
        "ffmpeg failed to produce 10-bit golomb file"
    );

    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .arg("-i")
        .arg(&mkv)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p10le"])
        .arg(&ref_yuv)
        .status()
        .expect("ffmpeg spawn");
    assert!(status.success());
    let ref_bytes = fs::read(&ref_yuv).expect("read ref yuv");

    let width = 64usize;
    let height = 48usize;
    let cw = width / 2;
    let ch = height / 2;
    let y_bytes = width * height * 2;
    let c_bytes = cw * ch * 2;
    assert_eq!(ref_bytes.len(), y_bytes + 2 * c_bytes);

    let input: Box<dyn oxideav_container::ReadSeek> =
        Box::new(fs::File::open(&mkv).expect("open mkv"));
    let mut demux =
        oxideav_mkv::demux::open(input, &oxideav_core::NullCodecResolver).expect("demux");
    let params = demux.streams()[0].params.clone();
    let pkt = demux.next_packet().expect("pkt");

    let mut dec = oxideav_ffv1::decoder::make_decoder(&params).expect("make_decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(vf) = dec.receive_frame().expect("recv") else {
        panic!("non-video");
    };
    assert_eq!(vf.format, PixelFormat::Yuv420P10Le);

    let y_ref = &ref_bytes[..y_bytes];
    let u_ref = &ref_bytes[y_bytes..y_bytes + c_bytes];
    let v_ref = &ref_bytes[y_bytes + c_bytes..];
    let y_stride = vf.planes[0].stride;
    for row in 0..height {
        let got = &vf.planes[0].data[row * y_stride..row * y_stride + width * 2];
        let want = &y_ref[row * width * 2..row * width * 2 + width * 2];
        assert_eq!(
            got,
            want,
            "golomb 10-bit Y row {row}: first diff at {}",
            first_diff(got, want)
        );
    }
    let u_stride = vf.planes[1].stride;
    for row in 0..ch {
        let got = &vf.planes[1].data[row * u_stride..row * u_stride + cw * 2];
        let want = &u_ref[row * cw * 2..row * cw * 2 + cw * 2];
        assert_eq!(got, want, "golomb 10-bit U row {row}");
    }
    let v_stride = vf.planes[2].stride;
    for row in 0..ch {
        let got = &vf.planes[2].data[row * v_stride..row * v_stride + cw * 2];
        let want = &v_ref[row * cw * 2..row * cw * 2 + cw * 2];
        assert_eq!(got, want, "golomb 10-bit V row {row}");
    }
}
