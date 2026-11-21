//! Live black-box matrix (RFC 9043 coverage audit).
//!
//! When an `ffmpeg` executable is on `PATH` this test drives it purely
//! as a black box, in both directions, over every combination the tool
//! offers of stream version, coder, context model, pixel layout / bit
//! depth, Slice count, per-Slice CRC and keyframe interval:
//!
//! 1. **black-box encoder -> this decoder.** The tool encodes a
//!    synthetic source; this crate's registry decoder must reproduce,
//!    byte for byte, the samples the tool's own decoder produces from
//!    the same stream — at a serial and at a parallel thread budget.
//! 2. **this encoder -> black-box decoder.** The registry encoder (every
//!    option combination) encodes a synthetic source; the stream must
//!    round-trip here bit-exactly, and the tool must decode it to
//!    exactly the source samples.
//!
//! Streams cross the process boundary in AVI (RIFF) files, read and
//! written by the minimal chunk walker / writer below. Without the
//! executable the test is a runtime skip.

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, ExecutionContext, Frame, Packet, PixelFormat,
    RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{CODEC_ID_STR, register};

const TOOL: &str = "ffmpeg";

fn tool_available() -> bool {
    Command::new(TOOL)
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// One pixel layout: the tool's name for it, the framework format, the
/// chroma shifts, the Plane count and the bytes per Sample.
#[derive(Clone, Copy)]
struct Layout {
    name: &'static str,
    format: PixelFormat,
    bits: u32,
    shift: (u32, u32),
    planes: usize,
}

const fn layout(
    name: &'static str,
    format: PixelFormat,
    bits: u32,
    shift: (u32, u32),
    planes: usize,
) -> Layout {
    Layout {
        name,
        format,
        bits,
        shift,
        planes,
    }
}

const LAYOUTS: &[Layout] = &[
    layout("gray", PixelFormat::Gray8, 8, (0, 0), 1),
    layout("gray10le", PixelFormat::Gray10Le, 10, (0, 0), 1),
    layout("gray12le", PixelFormat::Gray12Le, 12, (0, 0), 1),
    layout("gray16le", PixelFormat::Gray16Le, 16, (0, 0), 1),
    layout("yuv420p", PixelFormat::Yuv420P, 8, (1, 1), 3),
    layout("yuv422p", PixelFormat::Yuv422P, 8, (1, 0), 3),
    layout("yuv444p", PixelFormat::Yuv444P, 8, (0, 0), 3),
    layout("yuv411p", PixelFormat::Yuv411P, 8, (2, 0), 3),
    layout("yuva420p", PixelFormat::Yuva420P, 8, (1, 1), 4),
    layout("yuva422p", PixelFormat::Yuva422P, 8, (1, 0), 4),
    layout("yuva444p", PixelFormat::Yuva444P, 8, (0, 0), 4),
    layout("yuv420p10le", PixelFormat::Yuv420P10Le, 10, (1, 1), 3),
    layout("yuv422p10le", PixelFormat::Yuv422P10Le, 10, (1, 0), 3),
    layout("yuv444p10le", PixelFormat::Yuv444P10Le, 10, (0, 0), 3),
    layout("yuv420p12le", PixelFormat::Yuv420P12Le, 12, (1, 1), 3),
    layout("yuv422p12le", PixelFormat::Yuv422P12Le, 12, (1, 0), 3),
    layout("yuv444p12le", PixelFormat::Yuv444P12Le, 12, (0, 0), 3),
    layout("yuv420p16le", PixelFormat::Yuv420P16Le, 16, (1, 1), 3),
    layout("yuv422p16le", PixelFormat::Yuv422P16Le, 16, (1, 0), 3),
    layout("yuv444p16le", PixelFormat::Yuv444P16Le, 16, (0, 0), 3),
    layout("yuva420p10le", PixelFormat::Yuva420P10Le, 10, (1, 1), 4),
    layout("yuva422p10le", PixelFormat::Yuva422P10Le, 10, (1, 0), 4),
    layout("yuva444p10le", PixelFormat::Yuva444P10Le, 10, (0, 0), 4),
    layout("yuva420p16le", PixelFormat::Yuva420P16Le, 16, (1, 1), 4),
    layout("yuva444p16le", PixelFormat::Yuva444P16Le, 16, (0, 0), 4),
    layout("gbrp", PixelFormat::Gbrp8, 8, (0, 0), 3),
    layout("gbrp10le", PixelFormat::Gbrp10Le, 10, (0, 0), 3),
    layout("gbrp12le", PixelFormat::Gbrp12Le, 12, (0, 0), 3),
    layout("gbrp14le", PixelFormat::Gbrp14Le, 14, (0, 0), 3),
    layout("gbrp16le", PixelFormat::Gbrp16Le, 16, (0, 0), 3),
    layout("gbrap10le", PixelFormat::Gbrap10Le, 10, (0, 0), 4),
    layout("gbrap12le", PixelFormat::Gbrap12Le, 12, (0, 0), 4),
    layout("gbrap16le", PixelFormat::Gbrap16Le, 16, (0, 0), 4),
];

impl Layout {
    fn is_rgb(&self) -> bool {
        self.name.starts_with("gbr")
    }

    /// Plane `p`'s size in a `w x h` Frame.
    fn plane_dims(&self, p: usize, w: u32, h: u32) -> (usize, usize) {
        let chroma = !self.is_rgb() && (p == 1 || p == 2);
        if chroma {
            (
                w.div_ceil(1 << self.shift.0) as usize,
                h.div_ceil(1 << self.shift.1) as usize,
            )
        } else {
            (w as usize, h as usize)
        }
    }

    fn bytes_per_sample(&self) -> usize {
        if self.bits > 8 { 2 } else { 1 }
    }

    /// Bytes of one raw Frame in the tool's planar layout.
    fn frame_bytes(&self, w: u32, h: u32) -> usize {
        (0..self.planes)
            .map(|p| {
                let (pw, ph) = self.plane_dims(p, w, h);
                pw * ph * self.bytes_per_sample()
            })
            .sum()
    }
}

// ───────────────────────── minimal RIFF / AVI ─────────────────────────

fn le32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 9);
    out.extend_from_slice(id);
    out.extend_from_slice(&le32(body.len() as u32));
    out.extend_from_slice(body);
    if body.len() % 2 == 1 {
        out.push(0);
    }
    out
}

fn list(kind: &[u8; 4], children: &[u8]) -> Vec<u8> {
    let mut body = kind.to_vec();
    body.extend_from_slice(children);
    chunk(b"LIST", &body)
}

/// Write one video stream (FourCC `FFV1`) as an AVI file.
fn write_avi(path: &Path, w: u32, h: u32, extradata: &[u8], packets: &[Vec<u8>]) {
    let n = packets.len() as u32;
    let mut avih = Vec::new();
    for v in [40_000, 0, 0, 0, n, 0, 1, 0, w, h, 0, 0, 0, 0] {
        avih.extend_from_slice(&le32(v));
    }
    let mut strh = Vec::new();
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(b"FFV1");
    for v in [0, 0, 0, 1, 25, 0, n, 0, 0, 0] {
        strh.extend_from_slice(&le32(v));
    }
    for v in [0u16, 0, w as u16, h as u16] {
        strh.extend_from_slice(&v.to_le_bytes());
    }
    let mut strf = Vec::new();
    strf.extend_from_slice(&le32(40 + extradata.len() as u32));
    strf.extend_from_slice(&le32(w));
    strf.extend_from_slice(&le32(h));
    strf.extend_from_slice(&1u16.to_le_bytes());
    strf.extend_from_slice(&24u16.to_le_bytes());
    strf.extend_from_slice(b"FFV1");
    for _ in 0..5 {
        strf.extend_from_slice(&le32(0));
    }
    strf.extend_from_slice(extradata);

    let mut strl = chunk(b"strh", &strh);
    strl.extend_from_slice(&chunk(b"strf", &strf));
    let mut hdrl = chunk(b"avih", &avih);
    hdrl.extend_from_slice(&list(b"strl", &strl));
    let mut movi = Vec::new();
    for p in packets {
        movi.extend_from_slice(&chunk(b"00dc", p));
    }
    let mut riff = b"AVI ".to_vec();
    riff.extend_from_slice(&list(b"hdrl", &hdrl));
    riff.extend_from_slice(&list(b"movi", &movi));
    std::fs::write(path, chunk(b"RIFF", &riff)).expect("write avi");
}

/// Walk a RIFF tree collecting the stream-format extradata and the video
/// chunks of the `movi` list.
fn walk_riff(data: &[u8], in_movi: bool, extradata: &mut Vec<u8>, packets: &mut Vec<Vec<u8>>) {
    let mut at = 0usize;
    while at + 8 <= data.len() {
        let id = &data[at..at + 4];
        let len = u32::from_le_bytes(data[at + 4..at + 8].try_into().expect("4 bytes")) as usize;
        let body_end = (at + 8 + len).min(data.len());
        let body = &data[at + 8..body_end];
        if id == b"RIFF" || id == b"LIST" {
            if body.len() >= 4 {
                walk_riff(&body[4..], &body[..4] == b"movi", extradata, packets);
            }
        } else if id == b"strf" && body.len() >= 40 && &body[16..20] == b"FFV1" {
            *extradata = body[40..].to_vec();
        } else if in_movi && (&id[2..] == b"dc" || &id[2..] == b"db") && !body.is_empty() {
            packets.push(body.to_vec());
        }
        at = body_end + (len & 1);
    }
}

fn read_avi(path: &Path) -> (Vec<u8>, Vec<Vec<u8>>) {
    let data = std::fs::read(path).expect("read avi");
    let (mut extradata, mut packets) = (Vec::new(), Vec::new());
    walk_riff(&data, false, &mut extradata, &mut packets);
    (extradata, packets)
}

// ─────────────────────────── helpers ───────────────────────────

fn run(args: &[String]) -> bool {
    Command::new(TOOL)
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y"])
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Whether to run `cell`. By default the sampled subset; with
/// `FFV1_MATRIX_FULL=1` the whole cross product; with
/// `FFV1_MATRIX_ONLY=<text>` exactly the cells whose label contains
/// every space-separated word of `<text>`.
fn selected(cell: &str, sampled: bool) -> bool {
    if let Ok(only) = std::env::var("FFV1_MATRIX_ONLY") {
        return only.split_whitespace().all(|word| cell.contains(word));
    }
    sampled || std::env::var_os("FFV1_MATRIX_FULL").is_some()
}

fn s(v: impl ToString) -> String {
    v.to_string()
}

fn path_arg(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn xorshift(x: &mut u32) -> u32 {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    *x
}

/// Deterministic source Frame in the tool's raw planar layout: smooth
/// structure, flat bands (run mode) and full-range noise bands (escape
/// codes, the 16-bit predictor wrap).
fn synth_raw(l: &Layout, w: u32, h: u32, frame: u32) -> Vec<u8> {
    let max = (1u32 << l.bits) - 1;
    let mut out = Vec::with_capacity(l.frame_bytes(w, h));
    for p in 0..l.planes {
        let (pw, ph) = l.plane_dims(p, w, h);
        let mut rng = 0x1234_5678u32 ^ (p as u32 * 7919) ^ (frame * 104_729) | 1;
        for y in 0..ph {
            for x in 0..pw {
                let v = match (y / 6) % 4 {
                    0 => ((x * 5 + y * 3 + frame as usize * 2) as u32) << (l.bits - 8),
                    1 => max / 3,
                    2 => xorshift(&mut rng),
                    _ => ((x as u32 * 9) << (l.bits - 8)) + (xorshift(&mut rng) & 3),
                } & max;
                if l.bits > 8 {
                    out.extend_from_slice(&(v as u16).to_le_bytes());
                } else {
                    out.push(v as u8);
                }
            }
        }
    }
    out
}

fn raw_to_frame(l: &Layout, w: u32, h: u32, raw: &[u8], pts: i64) -> VideoFrame {
    let mut at = 0usize;
    let planes = (0..l.planes)
        .map(|p| {
            let (pw, ph) = l.plane_dims(p, w, h);
            let len = pw * ph * l.bytes_per_sample();
            let plane = VideoPlane {
                stride: pw * l.bytes_per_sample(),
                data: raw[at..at + len].to_vec(),
            };
            at += len;
            plane
        })
        .collect();
    VideoFrame {
        pts: Some(pts),
        planes,
    }
}

fn frame_to_raw(v: &VideoFrame) -> Vec<u8> {
    v.planes
        .iter()
        .flat_map(|p| p.data.iter().copied())
        .collect()
}

fn decode_here(
    w: u32,
    h: u32,
    extradata: &[u8],
    packets: &[Vec<u8>],
    threads: usize,
) -> Result<Vec<u8>, String> {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.extradata = extradata.to_vec();
    let mut dec = ctx.codecs.first_decoder(&params).map_err(s)?;
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut out = Vec::new();
    for p in packets {
        let pkt = Packet::new(0, oxideav_core::TimeBase::new(1, 25), p.clone());
        dec.send_packet(&pkt).map_err(s)?;
        match dec.receive_frame().map_err(s)? {
            Frame::Video(v) => out.extend_from_slice(&frame_to_raw(&v)),
            _ => return Err(s("not a video frame")),
        }
    }
    Ok(out)
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("oxideav-ffv1-matrix-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        Scratch(dir)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ───────────────── black-box encoder -> this decoder ─────────────────

#[test]
fn blackbox_encoded_streams_decode_bit_exactly() {
    if !tool_available() {
        eprintln!("skip: no `{TOOL}` executable on PATH");
        return;
    }
    let scratch = Scratch::new("dec");
    let (w, h, frames) = (384u32, 290u32, 3u32);
    let mut failures: Vec<String> = Vec::new();
    let (mut ran, mut declined) = (0u32, 0u32);

    // (version, coder, context, slices, crc, gop)
    let mut cells: Vec<(u32, u32, u32, u32, u32, u32)> = Vec::new();
    for version in [0u32, 1, 3] {
        for coder in [0u32, 1, 2] {
            for context in [0u32, 1] {
                if version == 3 {
                    for (slices, crc, gop) in [(4, 1, 1), (9, 0, 3), (16, 1, 3), (24, 1, 1)] {
                        cells.push((version, coder, context, slices, crc, gop));
                    }
                } else {
                    cells.push((version, coder, context, 1, 0, 1));
                    cells.push((version, coder, context, 1, 0, 3));
                }
            }
        }
    }

    for (li, l) in LAYOUTS.iter().enumerate() {
        let src = scratch.file("src.raw");
        let raw: Vec<u8> = (0..frames).flat_map(|f| synth_raw(l, w, h, f)).collect();
        std::fs::write(&src, &raw).expect("write source");
        for (ci, &(version, coder, context, slices, crc, gop)) in cells.iter().enumerate() {
            let cell = format!(
                "{} v{version} coder{coder} ctx{context} slices{slices} crc{crc} gop{gop}",
                l.name
            );
            // By default a rotating ninth of the cross product per
            // layout (every cell family still meets every layout
            // family), to bound the run time of a plain `cargo test`.
            if !selected(&cell, (li + ci) % 9 == 0) {
                continue;
            }
            let avi = scratch.file("enc.avi");
            let mut args = vec![
                s("-f"),
                s("rawvideo"),
                s("-pix_fmt"),
                s(l.name),
                s("-s"),
                format!("{w}x{h}"),
                s("-i"),
                path_arg(&src),
                s("-c:v"),
                s("ffv1"),
                s("-level"),
                s(version),
                s("-coder"),
                s(coder),
                s("-context"),
                s(context),
                s("-g"),
                s(gop),
                s("-threads"),
                s(1),
            ];
            if version == 3 {
                args.extend([s("-slices"), s(slices), s("-slicecrc"), s(crc)]);
            }
            args.push(path_arg(&avi));
            if !run(&args) {
                // The tool does not offer this combination.
                declined += 1;
                continue;
            }
            // Ground truth: the tool's own decode of its own stream.
            let truth_path = scratch.file("truth.raw");
            if !run(&[
                s("-i"),
                path_arg(&avi),
                s("-f"),
                s("rawvideo"),
                s("-pix_fmt"),
                s(l.name),
                path_arg(&truth_path),
            ]) {
                declined += 1;
                continue;
            }
            let truth = std::fs::read(&truth_path).expect("read truth");
            let (extradata, packets) = read_avi(&avi);
            if truth != raw {
                // The tool does not reproduce its own input from this
                // stream, so the stream cannot serve as a reference.
                eprintln!("note: {cell}: the tool's own round trip is not lossless; skipped");
                declined += 1;
                continue;
            }
            ran += 1;
            for threads in [1usize, 5] {
                match decode_here(w, h, &extradata, &packets, threads) {
                    Ok(got) if got == truth => {}
                    Ok(got) => failures.push(format!(
                        "{cell} threads={threads}: {} of {} bytes differ",
                        got.iter().zip(&truth).filter(|(a, b)| a != b).count()
                            + got.len().abs_diff(truth.len()),
                        truth.len()
                    )),
                    Err(e) => failures.push(format!("{cell} threads={threads}: {e}")),
                }
            }
        }
    }
    eprintln!("black-box encoder -> this decoder: {ran} streams, {declined} declined by the tool");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ───────────────── this encoder -> black-box decoder ─────────────────

#[test]
fn this_encoder_streams_decode_in_the_blackbox_decoder() {
    if !tool_available() {
        eprintln!("skip: no `{TOOL}` executable on PATH");
        return;
    }
    let scratch = Scratch::new("enc");
    let (w, h, frames) = (384u32, 290u32, 4u32);
    let mut failures: Vec<String> = Vec::new();
    let mut ran = 0u32;

    // Option sets: the default (version 1) path with both coders, and
    // version 3 across coder / Slice count / CRC / keyframe interval /
    // thread budget (which also drives the layout election).
    let mut cells: Vec<(Vec<(&str, String)>, usize)> = vec![
        (vec![], 1),
        (vec![("coder", s("golomb"))], 1),
        (vec![("gop", s(1))], 1),
    ];
    for coder in ["range", "golomb"] {
        for (slices, ec, gop, threads) in [
            (0u32, true, 0u32, 1usize),
            (0, true, 3, 6),
            (0, false, 1, 16),
            (4, true, 2, 3),
            (9, false, 3, 2),
            (12, true, 1, 8),
            (30, true, 4, 4),
        ] {
            cells.push((
                vec![
                    ("version", s(3)),
                    ("coder", s(coder)),
                    ("slices", s(slices)),
                    ("ec", s(ec)),
                    ("gop", s(gop)),
                ],
                threads,
            ));
        }
    }

    for (li, l) in LAYOUTS.iter().enumerate() {
        let source: Vec<Vec<u8>> = (0..frames).map(|f| synth_raw(l, w, h, f)).collect();
        let raw: Vec<u8> = source.concat();
        for (ci, (options, threads)) in cells.iter().enumerate() {
            let cell = format!("{} {options:?} threads={threads}", l.name);
            if !selected(&cell, (li + ci) % 5 == 0) {
                continue;
            }
            let mut ctx = RuntimeContext::new();
            register(&mut ctx);
            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.width = Some(w);
            params.height = Some(h);
            params.pixel_format = Some(l.format);
            let mut bag = CodecOptions::new();
            for (k, v) in options {
                bag = bag.set(*k, v.clone());
            }
            params.options = bag;
            // RFC 9043 §4.2.3: Golomb-Rice SHOULD NOT be used above 8
            // bits — the registry options refuse to build such a record.
            let deep_golomb =
                l.bits > 8 && options.iter().any(|(k, v)| *k == "coder" && v == "golomb");
            let mut enc = match ctx.codecs.first_encoder(&params) {
                Ok(_) if deep_golomb => {
                    failures.push(format!("{cell}: deep Golomb-Rice was not refused"));
                    continue;
                }
                Ok(e) => e,
                Err(_) if deep_golomb => continue,
                Err(e) => {
                    failures.push(format!("{cell}: encoder refused: {e}"));
                    continue;
                }
            };
            enc.set_execution_context(&ExecutionContext::with_threads(*threads));
            let mut packets: Vec<Vec<u8>> = Vec::new();
            let mut ok = true;
            for (f, src) in source.iter().enumerate() {
                let frame = Frame::Video(raw_to_frame(l, w, h, src, f as i64));
                match enc.send_frame(&frame).and_then(|()| enc.receive_packet()) {
                    Ok(p) => packets.push(p.data.clone()),
                    Err(e) => {
                        failures.push(format!("{cell}: encode failed: {e}"));
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            ran += 1;
            let extradata = enc.output_params().extradata.clone();
            match decode_here(w, h, &extradata, &packets, *threads) {
                Ok(got) if got == raw => {}
                Ok(_) => failures.push(format!("{cell}: self round trip differs")),
                Err(e) => failures.push(format!("{cell}: self decode failed: {e}")),
            }
            let avi = scratch.file("ours.avi");
            write_avi(&avi, w, h, &extradata, &packets);
            let out = scratch.file("ours.raw");
            if !run(&[
                s("-i"),
                path_arg(&avi),
                s("-f"),
                s("rawvideo"),
                s("-pix_fmt"),
                s(l.name),
                path_arg(&out),
            ]) {
                failures.push(format!("{cell}: the black-box decoder rejected the stream"));
                continue;
            }
            let got = std::fs::read(&out).expect("read decode");
            if got != raw {
                failures.push(format!(
                    "{cell}: black-box decode differs ({} of {} bytes)",
                    got.iter().zip(&raw).filter(|(a, b)| a != b).count()
                        + got.len().abs_diff(raw.len()),
                    raw.len()
                ));
            }
        }
    }
    eprintln!("this encoder -> black-box decoder: {ran} streams");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ───── direct API: custom transitions, two Quantization Table Sets ─────

/// Version-3 streams built through the direct API with the features the
/// registry options do not reach — a §3.8.1.6 custom state-transition
/// table (`coder_type == 2`) and two Quantization Table Sets selected
/// per Plane (§4.6.6) — with and without the §4.9.3 Slice CRC, in both
/// traversals. Each must decode in the black-box decoder to exactly the
/// source samples, which also pins the Configuration Record layout
/// (per-table contexts, the `states_coded` / `ec` / `intra` tail).
#[test]
fn direct_api_records_decode_in_the_blackbox_decoder() {
    use oxideav_ffv1::{
        ColorspaceType, DecodedFrame, DecodedFramePlane, Ffv1ConfigurationRecord, Ffv1SliceHeader,
        Ffv1Version, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
        PictureStructure, QuantizationTableSet, encode_configuration_record_with_quant_tables,
        encode_frame,
    };
    if !tool_available() {
        eprintln!("skip: no `{TOOL}` executable on PATH");
        return;
    }
    let scratch = Scratch::new("direct");
    let (w, h) = (384u32, 290u32);

    // Two §4.1 sets: 11 levels on three differences (context_count
    // 666), and 11 / 11 / 5 / 5 / 5 levels on all five (7563).
    let level = |d: i32, steps: &[u32]| -> i32 {
        let l = steps.iter().filter(|&&t| d.unsigned_abs() >= t).count() as i32;
        if d < 0 { -l } else { l }
    };
    let build = |shape: [&[u32]; NUM_QUANT_SUBTABLES]| -> QuantizationTableSet {
        let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
        let mut scale = 1i32;
        for (table, steps) in tables.iter_mut().zip(shape) {
            for (idx, slot) in table.iter_mut().enumerate() {
                let d = if idx < 128 {
                    idx as i32
                } else {
                    idx as i32 - 256
                };
                *slot = scale * level(d, steps);
            }
            // §4.1 symmetry: entry 128 mirrors entry 127.
            table[128] = -table[127];
            scale *= 2 * steps.len() as i32 + 1;
        }
        QuantizationTableSet {
            tables,
            context_count: (scale as u32).div_ceil(2),
        }
    };
    let wide: &[u32] = &[1, 3, 7, 15, 31];
    let narrow: &[u32] = &[2, 12];
    let sets = [
        build([wide, wide, wide, &[], &[]]),
        build([wide, wide, narrow, narrow, narrow]),
    ];
    assert_eq!((sets[0].context_count, sets[1].context_count), (666, 7563));

    let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
    for (i, d) in deltas.iter_mut().enumerate().skip(8).take(240) {
        *d = [1, 0, -1, 0, 2, 0, -2][i % 7];
    }

    let mut failures: Vec<String> = Vec::new();
    for name in ["yuv420p", "yuv444p10le", "gbrp", "gbrp16le", "yuva420p"] {
        let l = LAYOUTS.iter().find(|l| l.name == name).expect("layout");
        let rgb = l.is_rgb();
        for (coder, ec) in [(2u32, true), (2, false), (1, false), (0, false)] {
            let cell = format!("{name} coder{coder} ec={ec}");
            let cr = Ffv1ConfigurationRecord {
                version: Ffv1Version::V3,
                micro_version: Some(4),
                coder_type: coder,
                state_transition_delta: if coder == 2 {
                    deltas
                } else {
                    [0; NUM_TRANSITION_DELTAS]
                },
                colorspace_type: if rgb {
                    ColorspaceType::Rgb
                } else {
                    ColorspaceType::YCbCr
                },
                bits_per_raw_sample: l.bits,
                chroma_planes: true,
                log2_h_chroma_subsample: l.shift.0,
                log2_v_chroma_subsample: l.shift.1,
                extra_plane: l.planes == 4,
                num_h_slices: Some(3),
                num_v_slices: Some(2),
                quant_table_set_count: Some(2),
                ec: Some(u32::from(ec)),
                intra: Some(true),
                initial_state_delta: None,
            };
            let extradata = match encode_configuration_record_with_quant_tables(&cr, &sets) {
                Ok(x) => x,
                Err(e) => {
                    failures.push(format!("{cell}: record: {e}"));
                    continue;
                }
            };
            // Luma on set 1, chroma on set 0, the extra Plane on set 1.
            let slots = 2 + usize::from(l.planes == 4);
            let headers: Vec<Ffv1SliceHeader> = (0..6u32)
                .map(|i| Ffv1SliceHeader {
                    slice_x: i % 3,
                    slice_y: i / 3,
                    slice_width: 1,
                    slice_height: 1,
                    quant_table_set_index_count: slots,
                    quant_table_set_index: {
                        let mut q = [0u32; MAX_QUANT_TABLE_SET_INDEXES];
                        q[0] = 1;
                        q[2] = 1;
                        q
                    },
                    picture_structure: PictureStructure::Progressive,
                    picture_structure_raw: 0,
                    sar_num: 0,
                    sar_den: 0,
                })
                .collect();
            let raw = synth_raw(l, w, h, 0);
            // The tool's planar order for RGB is G, B, R; this crate's
            // direct API takes R, G, B.
            let order: &[usize] = if rgb { &[2, 0, 1, 3] } else { &[0, 1, 2, 3] };
            let bps = l.bytes_per_sample();
            let mut offsets = Vec::new();
            let mut at = 0usize;
            for p in 0..l.planes {
                let (pw, ph) = l.plane_dims(p, w, h);
                offsets.push((at, pw, ph));
                at += pw * ph * bps;
            }
            let planes = (0..l.planes)
                .map(|p| {
                    let (start, pw, ph) = offsets[order[p]];
                    let bytes = &raw[start..start + pw * ph * bps];
                    let samples = if bps == 2 {
                        bytes
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|&c| i32::from(u16::from_le_bytes(c)))
                            .collect()
                    } else {
                        bytes.iter().map(|&b| i32::from(b)).collect()
                    };
                    DecodedFramePlane {
                        plane_index: p as u8,
                        width: pw as u32,
                        height: ph as u32,
                        samples,
                    }
                })
                .collect();
            let frame = DecodedFrame {
                planes,
                width: w,
                height: h,
                bits_per_raw_sample: l.bits,
                colorspace: cr.colorspace_type,
                keyframe: true,
                slice_headers: Vec::new(),
            };
            let packet = match encode_frame(&frame, &cr, &sets, &headers, ec) {
                Ok(p) => p,
                Err(e) => {
                    failures.push(format!("{cell}: encode: {e}"));
                    continue;
                }
            };
            match decode_here(w, h, &extradata, std::slice::from_ref(&packet), 2) {
                Ok(got) if got == raw => {}
                Ok(_) => failures.push(format!("{cell}: self round trip differs")),
                Err(e) => failures.push(format!("{cell}: self decode failed: {e}")),
            }
            let avi = scratch.file("direct.avi");
            write_avi(&avi, w, h, &extradata, &[packet]);
            let out = scratch.file("direct.raw");
            if !run(&[
                s("-i"),
                path_arg(&avi),
                s("-f"),
                s("rawvideo"),
                s("-pix_fmt"),
                s(l.name),
                path_arg(&out),
            ]) {
                failures.push(format!("{cell}: the black-box decoder rejected the stream"));
                continue;
            }
            if std::fs::read(&out).expect("read decode") != raw {
                failures.push(format!("{cell}: black-box decode differs"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
