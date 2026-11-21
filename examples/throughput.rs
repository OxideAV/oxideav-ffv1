//! End-to-end throughput probe for the registry encoder / decoder.
//!
//! ```text
//! cargo run --release --example throughput -- [--frames N] [--runs N]
//!     [--threads 1,2,4,8] [--res 1080,2160] [--fmt yuv420p8,yuv422p10,gbrp16]
//!     [--coder golomb,range] [--slices 4]
//! ```
//!
//! Every cell encodes `frames` intra frames of deterministic synthetic
//! picture content (smooth structure + texture + low-order-bit noise)
//! through the `oxideav_core::Encoder` trait object, then decodes the
//! packets back through the `oxideav_core::Decoder` trait object and
//! checks the round trip is bit-exact. Figures are **MiB of raw sample
//! bytes per second**, median of `runs` runs.

use std::time::Instant;

use oxideav_core::{
    CodecId, CodecParameters, ExecutionContext, Frame, RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{
    CODEC_ID_STR, ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS, QuantizationTableSet, encode_configuration_record_with_quant_tables,
    register,
};

#[derive(Clone, Copy)]
struct Format {
    name: &'static str,
    rgb: bool,
    bits: u32,
    log2_h: u32,
    log2_v: u32,
}

const FORMATS: [Format; 3] = [
    Format {
        name: "yuv420p8",
        rgb: false,
        bits: 8,
        log2_h: 1,
        log2_v: 1,
    },
    Format {
        name: "yuv422p10",
        rgb: false,
        bits: 10,
        log2_h: 1,
        log2_v: 0,
    },
    Format {
        name: "gbrp16",
        rgb: true,
        bits: 16,
        log2_h: 0,
        log2_v: 0,
    },
];

fn xorshift(x: &mut u32) -> u32 {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    *x
}

fn triangle(v: usize, period: usize) -> i64 {
    let p = v % period;
    let half = period / 2;
    (if p < half { p } else { period - p }) as i64
}

/// One plane of picture-like content at `bits` significant bits: two
/// slow triangle gradients, a mid-frequency texture and noise in the
/// low-order bits (about 2 bits at 8-bit depth, proportionally more at
/// higher depths, the way sensor noise scales).
fn synth_plane(w: usize, h: usize, bits: u32, seed: u32, frame: usize) -> Vec<u16> {
    let max = (1i64 << bits) - 1;
    let scale = 1i64 << (bits - 8);
    let mut rng = seed.wrapping_mul(2654435761).wrapping_add(frame as u32) | 1;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let slow = triangle(x + frame * 3, 640) * 96 / 320 + triangle(y, 400) * 96 / 200;
            let tex = triangle(x * 3 + y * 5, 48) / 4;
            let noise = (xorshift(&mut rng) % (4 * scale as u32)) as i64;
            let v = (slow + tex + 24) * scale + noise;
            out.push(v.clamp(0, max) as u16);
        }
    }
    out
}

fn pack(samples: &[u16], w: usize, wide: bool) -> VideoPlane {
    if wide {
        let mut data = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        VideoPlane {
            stride: w * 2,
            data,
        }
    } else {
        VideoPlane {
            stride: w,
            data: samples.iter().map(|&s| s as u8).collect(),
        }
    }
}

fn make_frame(f: Format, w: usize, h: usize, index: usize) -> VideoFrame {
    let wide = f.bits > 8;
    let (cw, ch) = (
        w.div_ceil(1 << f.log2_h as usize),
        h.div_ceil(1 << f.log2_v as usize),
    );
    let planes = vec![
        pack(&synth_plane(w, h, f.bits, 11, index), w, wide),
        pack(&synth_plane(cw, ch, f.bits, 23, index), cw, wide),
        pack(&synth_plane(cw, ch, f.bits, 37, index), cw, wide),
    ];
    VideoFrame {
        pts: Some(index as i64),
        planes,
    }
}

/// The 11-level three-difference Quantization Table Set (RFC 9043 §4.1;
/// `context_count == 666`), built from its §3.4 shape.
fn quant_tables() -> QuantizationTableSet {
    let level = |d: i32| -> i32 {
        let l = match d.unsigned_abs() {
            0 => 0,
            1..=2 => 1,
            3..=6 => 2,
            7..=14 => 3,
            15..=30 => 4,
            _ => 5,
        };
        if d < 0 { -l } else { l }
    };
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    for idx in 0..256usize {
        let d = if idx < 128 {
            idx as i32
        } else {
            idx as i32 - 256
        };
        let l = level(d);
        tables[0][idx] = l;
        tables[1][idx] = l * 11;
        tables[2][idx] = l * 121;
    }
    QuantizationTableSet {
        tables,
        context_count: 666,
    }
}

fn params(f: Format, w: usize, h: usize, coder: u32, grid: u32) -> CodecParameters {
    let cr = Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: coder,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: if f.rgb {
            ColorspaceType::Rgb
        } else {
            ColorspaceType::YCbCr
        },
        bits_per_raw_sample: f.bits,
        chroma_planes: true,
        log2_h_chroma_subsample: f.log2_h,
        log2_v_chroma_subsample: f.log2_v,
        extra_plane: false,
        num_h_slices: Some(grid),
        num_v_slices: Some(grid),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra: Some(true),
        initial_state_delta: None,
    };
    let extradata = encode_configuration_record_with_quant_tables(&cr, &[quant_tables()])
        .expect("configuration record encodes");
    let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    p.width = Some(w as u32);
    p.height = Some(h as u32);
    p.extradata = extradata;
    p
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

fn list(args: &[String], key: &str, default: &str) -> Vec<String> {
    let raw = args
        .iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map_or(default, |s| s.as_str());
    raw.split(',').map(str::to_owned).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num = |key: &str, default: usize| -> usize {
        list(&args, key, &default.to_string())[0]
            .parse()
            .expect("numeric argument")
    };
    let frames = num("--frames", 4);
    let runs = num("--runs", 5);
    let grid = num("--slices", 4) as u32;
    let threads: Vec<usize> = list(&args, "--threads", "1")
        .iter()
        .map(|s| s.parse().expect("thread count"))
        .collect();
    let resolutions = list(&args, "--res", "1080,2160");
    let formats = list(&args, "--fmt", "yuv420p8,yuv422p10,gbrp16");
    let coders = list(&args, "--coder", "golomb,range");

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    println!(
        "{:<28} {:>3} {:>12} {:>12} {:>7}",
        "case", "thr", "enc MiB/s", "dec MiB/s", "ratio"
    );
    for res in &resolutions {
        let (w, h) = match res.as_str() {
            "1080" => (1920usize, 1080usize),
            "2160" => (3840, 2160),
            "720" => (1280, 720),
            other => panic!("unknown resolution {other}"),
        };
        for f in FORMATS
            .iter()
            .filter(|f| formats.iter().any(|n| n == f.name))
        {
            let source: Vec<VideoFrame> = (0..frames).map(|i| make_frame(*f, w, h, i)).collect();
            let raw_bytes: usize = source
                .iter()
                .map(|v| v.planes.iter().map(|p| p.data.len()).sum::<usize>())
                .sum();
            let raw_mib = raw_bytes as f64 / (1024.0 * 1024.0);
            for coder in &coders {
                let coder_type = match coder.as_str() {
                    "golomb" => 0,
                    "range" => 1,
                    other => panic!("unknown coder {other}"),
                };
                let p = params(*f, w, h, coder_type, grid);
                for &t in &threads {
                    let exec = ExecutionContext::with_threads(t);
                    let mut enc_times = Vec::new();
                    let mut dec_times = Vec::new();
                    let mut coded_bytes = 0usize;
                    for _ in 0..runs {
                        let mut enc = ctx.codecs.first_encoder(&p).expect("encoder");
                        enc.set_execution_context(&exec);
                        let frames_in: Vec<Frame> =
                            source.iter().cloned().map(Frame::Video).collect();
                        let start = Instant::now();
                        let mut packets = Vec::new();
                        for fr in &frames_in {
                            enc.send_frame(fr).expect("send_frame");
                            packets.push(enc.receive_packet().expect("receive_packet"));
                        }
                        enc_times.push(start.elapsed().as_secs_f64());
                        coded_bytes = packets.iter().map(|k| k.data.len()).sum();

                        let mut dp = p.clone();
                        dp.extradata = enc.output_params().extradata.clone();
                        let mut dec = ctx.codecs.first_decoder(&dp).expect("decoder");
                        dec.set_execution_context(&exec);
                        let start = Instant::now();
                        let mut decoded = Vec::new();
                        for pkt in &packets {
                            dec.send_packet(pkt).expect("send_packet");
                            decoded.push(dec.receive_frame().expect("receive_frame"));
                        }
                        dec_times.push(start.elapsed().as_secs_f64());
                        for (d, s) in decoded.iter().zip(&source) {
                            let Frame::Video(v) = d else {
                                panic!("video frame expected")
                            };
                            for (a, b) in v.planes.iter().zip(&s.planes) {
                                assert!(a.data == b.data, "round trip must be bit-exact");
                            }
                        }
                    }
                    println!(
                        "{:<28} {:>3} {:>12.1} {:>12.1} {:>7.3}",
                        format!("{res}p/{}/{coder}", f.name),
                        t,
                        raw_mib / median(enc_times),
                        raw_mib / median(dec_times),
                        raw_bytes as f64 / coded_bytes as f64,
                    );
                }
            }
        }
    }
}
