//! Integration tests against the docs/video/ffv1/ fixture corpus.
//!
//! Each fixture under `../../docs/video/ffv1/fixtures/<name>/` ships an
//! `input.avi` (FFV1 versions 0/1, the version-2/3 path uses
//! `input.mkv`), an `expected.raw` byte-for-byte ground truth produced
//! by an instrumented FFmpeg `ffv1` decoder, a `notes.md` describing
//! the bitstream feature focus, and a `trace.txt` capturing the
//! per-step decode events. The reference is FFV1's intrinsic output
//! (FFV1 is lossless), so `expected.raw` is byte-identical to the
//! original raw input that was encoded.
//!
//! This driver demuxes every fixture (AVI for v0/v1, MKV for v2/v3),
//! decodes through the in-tree [`Ffv1Decoder`], and reports the
//! per-fixture pixel-match rate + per-plane PSNR against the expected
//! raw bytes.
//!
//! Acceptance:
//! * `Tier::BitExact` — must round-trip exactly. Failure = CI red.
//! * `Tier::ReportOnly` — divergence is logged but the test does NOT
//!   fail. All fixtures land here at first commit; promote individual
//!   cases to `BitExact` as the in-tree FFV1 decoder closes the
//!   underlying gap (RGB/RCT, bgra/bgr0 alpha drop, 16-bit YUV,
//!   grayscale, alternate quant tables, 4×4 slice grids, …).
//!
//! Workspace policy: NO external library code (libavcodec, x264,
//! …) was consulted while writing this driver. Spec authority:
//! RFC 9043 (FFV1 v0..v3). The `trace.txt` files are an aid for the
//! human implementer when localising divergences — the driver
//! references the trace path in the `eprintln!` header so a failing
//! run prints a clickable pointer.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{
    CodecParameters, Decoder, Error, Frame, MediaType, NullCodecResolver, Packet, ReadSeek,
};
use oxideav_ffv1::decoder::make_decoder;

/// Locate `docs/video/ffv1/fixtures/<name>/`. Tests run with CWD set
/// to the crate root, so we walk two levels up to reach the workspace
/// root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/ffv1/fixtures").join(name)
}

/// Per-frame decode result against `expected.raw`. Counters are in
/// reference-sample bytes (the reference is plain raw, layout-defined
/// per-fixture via `RefLayout` below).
#[derive(Default)]
struct PlaneDiff {
    total: usize,
    exact: usize,
    max: i32,
    sse: u64,
}

impl PlaneDiff {
    fn merge(&mut self, other: &PlaneDiff) {
        self.total += other.total;
        self.exact += other.exact;
        self.max = self.max.max(other.max);
        self.sse += other.sse;
    }

    /// PSNR in dB of an 8-bit-equivalent comparison. We always score
    /// in 8-bit space — 10/12/16-bit references are right-shifted to
    /// 8-bit before comparison so the dynamic range of the reported
    /// diff/max/PSNR numbers is consistent across pixel formats. A
    /// fully bit-exact plane reports `+inf`.
    fn psnr(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        if self.sse == 0 {
            return f64::INFINITY;
        }
        let mse = self.sse as f64 / self.total as f64;
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }
}

#[derive(Default)]
struct FrameDiff {
    y: PlaneDiff,
    uv: PlaneDiff,
    /// Optional alpha plane / packed extra channel diff (RGBA, BGR0).
    alpha: PlaneDiff,
}

impl FrameDiff {
    fn pct(&self) -> f64 {
        let total = self.y.total + self.uv.total + self.alpha.total;
        let exact = self.y.exact + self.uv.exact + self.alpha.exact;
        if total == 0 {
            0.0
        } else {
            exact as f64 / total as f64 * 100.0
        }
    }

    fn merge(&mut self, other: &FrameDiff) {
        self.y.merge(&other.y);
        self.uv.merge(&other.uv);
        self.alpha.merge(&other.alpha);
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // BitExact unused at first commit; promote fixtures over time.
enum Tier {
    BitExact,
    ReportOnly,
}

/// How `expected.raw` is laid out and which decoder output planes it
/// should be compared against.
#[derive(Clone, Copy, Debug)]
enum RefLayout {
    /// Planar YUV at the given chroma subsampling and bit depth.
    /// Decoder output is also planar (`vf.planes[0..3]`); for bit
    /// depths > 8 each plane is two bytes per sample (LE).
    Yuv {
        bits: u32,
        chroma_h_shift: u32,
        chroma_v_shift: u32,
    },
    /// Planar YUV+alpha (4:2:0 8-bit only — the only YUVA shape FFV1
    /// emits in the corpus). Reference layout matches yuv420p with an
    /// extra full-resolution alpha plane appended.
    Yuva420P8,
    /// Single-plane luma-only (`pix_fmt=gray`).
    Gray8,
    /// Packed BGR0 (3 components in a 32-bit container, 4th byte
    /// zeroed). Reference is one packed buffer of `w*h*4` bytes; our
    /// decoder produces the same packed shape via the RCT path.
    Bgr0,
    /// Packed BGRA (3 components + alpha, all 8-bit). Reference is
    /// `w*h*4` bytes; our decoder produces the same packed shape via
    /// the RCT path with `extra_plane=1`.
    Bgra,
}

impl RefLayout {
    /// Total reference-buffer size in bytes for one frame at
    /// `width × height`.
    fn frame_bytes(&self, width: usize, height: usize) -> usize {
        match self {
            RefLayout::Yuv {
                bits,
                chroma_h_shift,
                chroma_v_shift,
            } => {
                let bps = if *bits > 8 { 2 } else { 1 };
                let cw = width >> chroma_h_shift;
                let ch = height >> chroma_v_shift;
                (width * height + 2 * cw * ch) * bps
            }
            RefLayout::Yuva420P8 => {
                let cw = width.div_ceil(2);
                let ch = height.div_ceil(2);
                width * height + 2 * cw * ch + width * height
            }
            RefLayout::Gray8 => width * height,
            RefLayout::Bgr0 | RefLayout::Bgra => width * height * 4,
        }
    }

    fn bit_depth(&self) -> u32 {
        match self {
            RefLayout::Yuv { bits, .. } => *bits,
            _ => 8,
        }
    }
}

/// Container format of `input.*` for a fixture. v0/v1 sit in AVI;
/// v2/v3 sit in MKV.
#[derive(Clone, Copy, Debug)]
enum Container {
    Avi,
    Mkv,
}

impl Container {
    fn input_name(&self) -> &'static str {
        match self {
            Container::Avi => "input.avi",
            Container::Mkv => "input.mkv",
        }
    }
}

struct CorpusCase {
    name: &'static str,
    width: usize,
    height: usize,
    n_frames: usize,
    container: Container,
    layout: RefLayout,
    tier: Tier,
}

/// Compare two byte slices that the corpus stores in 8-bit-per-sample
/// form. Returns `(n, exact, max, sse)`.
fn diff_8bit(our: &[u8], refp: &[u8]) -> (usize, usize, i32, u64) {
    let mut ex = 0usize;
    let mut max = 0i32;
    let mut sse: u64 = 0;
    let n = our.len().min(refp.len());
    for i in 0..n {
        let d = (our[i] as i32 - refp[i] as i32).abs();
        if d == 0 {
            ex += 1;
        }
        if d > max {
            max = d;
        }
        sse += (d as u64) * (d as u64);
    }
    (n, ex, max, sse)
}

/// Compare a >8-bit plane stored as little-endian u16 samples in both
/// our buffer and the reference. We score in 8-bit space (right-shift
/// each sample by `bits - 8`) so the reported `max` / `PSNR` numbers
/// stay on the same scale as 8-bit fixtures. The total is in samples.
fn diff_u16le(our: &[u8], refp: &[u8], bits: u32) -> (usize, usize, i32, u64) {
    let shift = bits.saturating_sub(8);
    let n_samples = our.len().min(refp.len()) / 2;
    let mut ex = 0usize;
    let mut max = 0i32;
    let mut sse: u64 = 0;
    for i in 0..n_samples {
        let o = u16::from_le_bytes([our[i * 2], our[i * 2 + 1]]);
        let r = u16::from_le_bytes([refp[i * 2], refp[i * 2 + 1]]);
        let o8 = (o >> shift) as i32;
        let r8 = (r >> shift) as i32;
        let d = (o8 - r8).abs();
        if d == 0 {
            ex += 1;
        }
        if d > max {
            max = d;
        }
        sse += (d as u64) * (d as u64);
    }
    (n_samples, ex, max, sse)
}

/// Open the fixture container and pull out (params, first_video_packet).
fn demux_first_video_packet(case: &CorpusCase) -> Option<(CodecParameters, Packet)> {
    let dir = fixture_dir(case.name);
    let input_path = dir.join(case.container.input_name());
    let trace_path = dir.join("trace.txt");
    let raw_path = dir.join("expected.raw");
    if !input_path.exists() {
        eprintln!(
            "skip {}: missing {} (docs/ corpus is in the workspace umbrella repo)",
            case.name,
            input_path.display()
        );
        return None;
    }
    if !raw_path.exists() {
        eprintln!("skip {}: missing {}", case.name, raw_path.display());
        return None;
    }
    eprintln!(
        "fixture {}: {}={} bytes, expected.raw={} bytes, trace={}",
        case.name,
        case.container.input_name(),
        fs::metadata(&input_path).map(|m| m.len()).unwrap_or(0),
        fs::metadata(&raw_path).map(|m| m.len()).unwrap_or(0),
        trace_path.display(),
    );

    let file = match fs::File::open(&input_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip {}: open {} failed: {e}", case.name, input_path.display());
            return None;
        }
    };
    let input: Box<dyn ReadSeek> = Box::new(file);
    let mut demux = match case.container {
        Container::Avi => match oxideav_avi::demuxer::open(input, &NullCodecResolver) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {}: avi open failed: {e:?}", case.name);
                return None;
            }
        },
        Container::Mkv => match oxideav_mkv::demux::open(input, &NullCodecResolver) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {}: mkv open failed: {e:?}", case.name);
                return None;
            }
        },
    };

    let streams = demux.streams().to_vec();
    let vstream = streams
        .iter()
        .find(|s| s.params.media_type == MediaType::Video);
    let vstream = match vstream {
        Some(s) => s,
        None => {
            eprintln!("skip {}: no video stream in container", case.name);
            return None;
        }
    };
    let vindex = vstream.index;
    let mut params = vstream.params.clone();
    // The container demuxer may not always carry width/height through
    // — patch them in from the corpus metadata if missing.
    if params.width.is_none() {
        params.width = Some(case.width as u32);
    }
    if params.height.is_none() {
        params.height = Some(case.height as u32);
    }

    let mut pkt: Option<Packet> = None;
    for _ in 0..256 {
        match demux.next_packet() {
            Ok(p) => {
                if p.stream_index == vindex {
                    pkt = Some(p);
                    break;
                }
            }
            Err(Error::Eof) => break,
            Err(e) => {
                eprintln!("skip {}: demux error: {e:?}", case.name);
                return None;
            }
        }
    }
    match pkt {
        Some(p) => Some((params, p)),
        None => {
            eprintln!("skip {}: no video packet demuxed", case.name);
            None
        }
    }
}

struct DecodeReport {
    per_frame: Vec<Result<FrameDiff, String>>,
    visible_produced: usize,
    fatal: Option<String>,
}

fn decode_fixture(case: &CorpusCase) -> Option<DecodeReport> {
    let (params, first_pkt) = demux_first_video_packet(case)?;

    let dir = fixture_dir(case.name);
    let raw_path = dir.join("expected.raw");
    let raw = match fs::read(&raw_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: read {} failed: {e}", case.name, raw_path.display());
            return None;
        }
    };

    let frame_size = case.layout.frame_bytes(case.width, case.height);
    if raw.len() != case.n_frames * frame_size {
        eprintln!(
            "skip {}: expected.raw size mismatch (have {} bytes, expected {} = {} frames * {})",
            case.name,
            raw.len(),
            case.n_frames * frame_size,
            case.n_frames,
            frame_size,
        );
        return None;
    }

    let mut dec = match make_decoder(&params) {
        Ok(d) => d,
        Err(e) => {
            return Some(DecodeReport {
                per_frame: Vec::new(),
                visible_produced: 0,
                fatal: Some(format!("make_decoder: {e:?}")),
            });
        }
    };

    let mut visible_idx = 0usize;
    let mut per_frame: Vec<Result<FrameDiff, String>> = Vec::with_capacity(case.n_frames);
    let mut fatal: Option<String> = None;

    // Feed the single video packet. (FFV1 is intra-per-packet, the
    // corpus fixtures are 1-frame each — but the same scaffold scales
    // if the corpus is later extended to multi-frame fixtures.)
    if let Err(e) = dec.send_packet(&first_pkt) {
        let msg = format!("send_packet: {e:?}");
        per_frame.push(Err(msg.clone()));
        fatal = Some(msg);
    }
    let _ = dec.flush();
    // Drain every visible frame the decoder has buffered.
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(vf)) => {
                if visible_idx >= case.n_frames {
                    visible_idx += 1;
                    continue;
                }
                let ref_off = visible_idx * frame_size;
                let ref_slice = &raw[ref_off..ref_off + frame_size];
                match score_frame(case, &vf, ref_slice) {
                    Ok(d) => per_frame.push(Ok(d)),
                    Err(e) => per_frame.push(Err(e)),
                }
                visible_idx += 1;
            }
            Ok(_) => continue,
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => {
                let msg = format!("visible {visible_idx}: receive_frame: {e:?}");
                per_frame.push(Err(msg.clone()));
                if fatal.is_none() {
                    fatal = Some(msg);
                }
                break;
            }
        }
    }

    Some(DecodeReport {
        per_frame,
        visible_produced: visible_idx,
        fatal,
    })
}

/// Compare one decoded frame against its reference slice. The shape
/// of the comparison depends on `case.layout`.
fn score_frame(
    case: &CorpusCase,
    vf: &oxideav_core::VideoFrame,
    ref_slice: &[u8],
) -> Result<FrameDiff, String> {
    let mut diff = FrameDiff::default();
    match case.layout {
        RefLayout::Yuv {
            bits,
            chroma_h_shift,
            chroma_v_shift,
        } => {
            let bps = if bits > 8 { 2 } else { 1 };
            let y_bytes = case.width * case.height * bps;
            let cw = case.width >> chroma_h_shift;
            let ch = case.height >> chroma_v_shift;
            let c_bytes = cw * ch * bps;
            if vf.planes.len() < 3 {
                return Err(format!(
                    "decoder produced {} planes, expected >=3 (YUV)",
                    vf.planes.len()
                ));
            }
            let our_y = vf.planes[0].data.as_slice();
            let our_u = vf.planes[1].data.as_slice();
            let our_v = vf.planes[2].data.as_slice();
            if our_y.len() != y_bytes || our_u.len() != c_bytes || our_v.len() != c_bytes {
                return Err(format!(
                    "plane size mismatch (Y {} U {} V {} expected {} {} {})",
                    our_y.len(),
                    our_u.len(),
                    our_v.len(),
                    y_bytes,
                    c_bytes,
                    c_bytes
                ));
            }
            let ref_y = &ref_slice[..y_bytes];
            let ref_u = &ref_slice[y_bytes..y_bytes + c_bytes];
            let ref_v = &ref_slice[y_bytes + c_bytes..y_bytes + 2 * c_bytes];
            let (yt, ye, ym, ys) = if bits > 8 {
                diff_u16le(our_y, ref_y, bits)
            } else {
                diff_8bit(our_y, ref_y)
            };
            let (ut, ue, um, us) = if bits > 8 {
                diff_u16le(our_u, ref_u, bits)
            } else {
                diff_8bit(our_u, ref_u)
            };
            let (vt, ve, vm, vs) = if bits > 8 {
                diff_u16le(our_v, ref_v, bits)
            } else {
                diff_8bit(our_v, ref_v)
            };
            diff.y = PlaneDiff {
                total: yt,
                exact: ye,
                max: ym,
                sse: ys,
            };
            diff.uv = PlaneDiff {
                total: ut + vt,
                exact: ue + ve,
                max: um.max(vm),
                sse: us + vs,
            };
        }
        RefLayout::Yuva420P8 => {
            let y_bytes = case.width * case.height;
            let cw = case.width.div_ceil(2);
            let ch = case.height.div_ceil(2);
            let c_bytes = cw * ch;
            if vf.planes.len() < 4 {
                return Err(format!(
                    "decoder produced {} planes, expected 4 (YUVA)",
                    vf.planes.len()
                ));
            }
            let our_y = vf.planes[0].data.as_slice();
            let our_u = vf.planes[1].data.as_slice();
            let our_v = vf.planes[2].data.as_slice();
            let our_a = vf.planes[3].data.as_slice();
            let ref_y = &ref_slice[..y_bytes];
            let ref_u = &ref_slice[y_bytes..y_bytes + c_bytes];
            let ref_v = &ref_slice[y_bytes + c_bytes..y_bytes + 2 * c_bytes];
            let ref_a = &ref_slice[y_bytes + 2 * c_bytes..y_bytes + 2 * c_bytes + y_bytes];
            let (yt, ye, ym, ys) = diff_8bit(our_y, ref_y);
            let (ut, ue, um, us) = diff_8bit(our_u, ref_u);
            let (vt, ve, vm, vs) = diff_8bit(our_v, ref_v);
            let (at, ae, am, ass) = diff_8bit(our_a, ref_a);
            diff.y = PlaneDiff {
                total: yt,
                exact: ye,
                max: ym,
                sse: ys,
            };
            diff.uv = PlaneDiff {
                total: ut + vt,
                exact: ue + ve,
                max: um.max(vm),
                sse: us + vs,
            };
            diff.alpha = PlaneDiff {
                total: at,
                exact: ae,
                max: am,
                sse: ass,
            };
        }
        RefLayout::Gray8 => {
            // Grayscale: decoder may either expose a single plane or
            // emit YUV-with-empty-chroma (oxideav-ffv1's current path
            // produces 3 planes with U/V empty for `chroma_planes=0`).
            // Either way, compare plane 0 byte-for-byte.
            if vf.planes.is_empty() {
                return Err("decoder produced 0 planes".into());
            }
            let our_y = vf.planes[0].data.as_slice();
            let (yt, ye, ym, ys) = diff_8bit(our_y, ref_slice);
            diff.y = PlaneDiff {
                total: yt,
                exact: ye,
                max: ym,
                sse: ys,
            };
        }
        RefLayout::Bgr0 | RefLayout::Bgra => {
            // RCT path emits a single packed plane.
            if vf.planes.is_empty() {
                return Err("decoder produced 0 planes".into());
            }
            let our = vf.planes[0].data.as_slice();
            let (n, e, m, s) = diff_8bit(our, ref_slice);
            diff.y = PlaneDiff {
                total: n,
                exact: e,
                max: m,
                sse: s,
            };
        }
    }
    Ok(diff)
}

fn evaluate(case: &CorpusCase) {
    let report = match decode_fixture(case) {
        Some(r) => r,
        None => return,
    };

    let mut agg = FrameDiff::default();
    let mut errors: Vec<String> = Vec::new();
    for (i, r) in report.per_frame.iter().enumerate() {
        match r {
            Ok(d) => {
                eprintln!(
                    "  frame {i}: Y {}/{} exact (max {}, PSNR {:.2} dB), \
                     UV {}/{} exact (max {}, PSNR {:.2} dB), \
                     A {}/{} exact (max {}, PSNR {:.2} dB), pct={:.2}%",
                    d.y.exact,
                    d.y.total,
                    d.y.max,
                    d.y.psnr(),
                    d.uv.exact,
                    d.uv.total,
                    d.uv.max,
                    d.uv.psnr(),
                    d.alpha.exact,
                    d.alpha.total,
                    d.alpha.max,
                    d.alpha.psnr(),
                    d.pct(),
                );
                agg.merge(d);
            }
            Err(e) => {
                eprintln!("  frame {i}: ERROR {e}");
                errors.push(format!("frame {i}: {e}"));
            }
        }
    }

    let pct = agg.pct();
    eprintln!(
        "[{:?}] {}: aggregate {}/{} exact ({pct:.2}%), \
         Y max {} PSNR {:.2} dB, UV max {} PSNR {:.2} dB, A max {} PSNR {:.2} dB, \
         visible_produced={}/{}{} (bits={})",
        case.tier,
        case.name,
        agg.y.exact + agg.uv.exact + agg.alpha.exact,
        agg.y.total + agg.uv.total + agg.alpha.total,
        agg.y.max,
        agg.y.psnr(),
        agg.uv.max,
        agg.uv.psnr(),
        agg.alpha.max,
        agg.alpha.psnr(),
        report.visible_produced,
        case.n_frames,
        match &report.fatal {
            Some(f) => format!(", first_fatal=\"{f}\""),
            None => String::new(),
        },
        case.layout.bit_depth(),
    );

    match case.tier {
        Tier::BitExact => {
            assert!(
                errors.is_empty(),
                "{}: {} frame errors prevented bit-exact comparison: {:?}",
                case.name,
                errors.len(),
                errors
            );
            assert_eq!(
                agg.y.exact + agg.uv.exact + agg.alpha.exact,
                agg.y.total + agg.uv.total + agg.alpha.total,
                "{}: not bit-exact (Y max {}, UV max {}, A max {}; {:.4}% match)",
                case.name,
                agg.y.max,
                agg.uv.max,
                agg.alpha.max,
                pct
            );
        }
        Tier::ReportOnly => {
            // Don't fail. The eprintln output above is the report.
            // TODO(ffv1-corpus): tighten to BitExact once the
            // underlying decoder gap is closed.
            let _ = pct;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// All fixtures start as `Tier::ReportOnly`. As individual decoder
// gaps close (RGB/RCT roundtrip via JPEG2000-style RCT, BGRA alpha
// preservation, 16-bit YUV path, grayscale single-plane, alternate
// quant tables, 4×4 slice grids, …), promote the corresponding case
// to `BitExact`.
//
// Trace files (referenced in `evaluate()` via the `eprintln!` header)
// live alongside each fixture and capture the per-step decode events
// the reference decoder emitted on the bitstream.

/// FFV1 v0 with Golomb-Rice entropy coder (`coder_type=0`), 8-bit
/// YUV 4:2:0 single slice.
/// Trace: docs/video/ffv1/fixtures/v0-yuv420-golomb-rice/trace.txt
#[test]
fn corpus_v0_yuv420_golomb_rice() {
    evaluate(&CorpusCase {
        name: "v0-yuv420-golomb-rice",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Avi,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v0 with range coder (`coder_type=1`), 8-bit YUV 4:2:0 single
/// slice, default quant tables.
/// Trace: docs/video/ffv1/fixtures/v0-yuv420-rangecoder/trace.txt
#[test]
fn corpus_v0_yuv420_rangecoder() {
    evaluate(&CorpusCase {
        name: "v0-yuv420-rangecoder",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Avi,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v1: range coder, single slice, per-keyframe quant table
/// (v1 ships quant tables in the keyframe header).
/// Trace: docs/video/ffv1/fixtures/v1-single-slice/trace.txt
#[test]
fn corpus_v1_single_slice() {
    evaluate(&CorpusCase {
        name: "v1-single-slice",
        width: 128,
        height: 96,
        n_frames: 1,
        container: Container::Avi,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v2: 2×2 slice grid (4 slices), range coder, default quant
/// tables — slice geometry coded in the per-keyframe header.
/// Trace: docs/video/ffv1/fixtures/v2-multislice-2x2/trace.txt
#[test]
fn corpus_v2_multislice_2x2() {
    evaluate(&CorpusCase {
        name: "v2-multislice-2x2",
        width: 128,
        height: 96,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 default: 8-bit YUV 4:2:0, 2×2 slices, per-slice CRC, global
/// header in extradata.
/// Trace: docs/video/ffv1/fixtures/v3-default/trace.txt
#[test]
fn corpus_v3_default() {
    evaluate(&CorpusCase {
        name: "v3-default",
        width: 128,
        height: 96,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 with `-context 1` (large quant-table set with 5-input
/// contexts). Exercises the alternate quant-table set.
/// Trace: docs/video/ffv1/fixtures/v3-context-1/trace.txt
#[test]
fn corpus_v3_context_1() {
    evaluate(&CorpusCase {
        name: "v3-context-1",
        width: 128,
        height: 96,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 flat-color frame (extreme low-entropy input). Exercises
/// the run-mode / zero-residual paths in the entropy coder.
/// Trace: docs/video/ffv1/fixtures/v3-flat-color/trace.txt
#[test]
fn corpus_v3_flat_color() {
    evaluate(&CorpusCase {
        name: "v3-flat-color",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 with 4×4 slicing + per-slice CRC (frame-MT friendly).
/// Trace: docs/video/ffv1/fixtures/v3-frame-mt/trace.txt
#[test]
fn corpus_v3_frame_mt() {
    evaluate(&CorpusCase {
        name: "v3-frame-mt",
        width: 256,
        height: 192,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 4×4 slice grid (16 slices, each with own range coder + CRC).
/// Concurrency-friendly slicing path.
/// Trace: docs/video/ffv1/fixtures/v3-multislice-4x4/trace.txt
#[test]
fn corpus_v3_multislice_4x4() {
    evaluate(&CorpusCase {
        name: "v3-multislice-4x4",
        width: 128,
        height: 96,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 8,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 single-plane luma-only (`pix_fmt=gray`). Exercises the
/// no-chroma path through `decode_slice` (`chroma_planes=0`,
/// `transparency=0`, `plane_count=1`).
/// Trace: docs/video/ffv1/fixtures/v3-grayscale/trace.txt
#[test]
fn corpus_v3_grayscale() {
    evaluate(&CorpusCase {
        name: "v3-grayscale",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Gray8,
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 RGB packed BGR0 (`colorspace=1`, JPEG2000-style RCT).
/// 3 components in a 32-bit container, 4th byte zeroed.
/// Trace: docs/video/ffv1/fixtures/v3-rgb-bgr0/trace.txt
#[test]
fn corpus_v3_rgb_bgr0() {
    evaluate(&CorpusCase {
        name: "v3-rgb-bgr0",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Bgr0,
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 RGB+alpha packed BGRA (`transparency=1`, RCT path with
/// extra alpha plane).
/// Trace: docs/video/ffv1/fixtures/v3-rgba/trace.txt
#[test]
fn corpus_v3_rgba() {
    evaluate(&CorpusCase {
        name: "v3-rgba",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Bgra,
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 12-bit YUV 4:2:0 (`yuv420p12`). Exercises the
/// `bits_per_raw_sample > 8` path; samples stored at LSB end of a
/// 16-bit container.
/// Trace: docs/video/ffv1/fixtures/v3-yuv420p12/trace.txt
#[test]
fn corpus_v3_yuv420p12() {
    evaluate(&CorpusCase {
        name: "v3-yuv420p12",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 12,
            chroma_h_shift: 1,
            chroma_v_shift: 1,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 10-bit YUV 4:2:2 (`yuv422p10`). Exercises the >8-bit
/// path through the chroma half-horizontal grid.
/// Trace: docs/video/ffv1/fixtures/v3-yuv422p10/trace.txt
#[test]
fn corpus_v3_yuv422p10() {
    evaluate(&CorpusCase {
        name: "v3-yuv422p10",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 10,
            chroma_h_shift: 1,
            chroma_v_shift: 0,
        },
        tier: Tier::ReportOnly,
    });
}

/// FFV1 v3 16-bit YUV 4:4:4 (`yuv444p16`). Full-precision 16-bit
/// sample path, no chroma subsampling.
/// Trace: docs/video/ffv1/fixtures/v3-yuv444p16/trace.txt
#[test]
fn corpus_v3_yuv444p16() {
    evaluate(&CorpusCase {
        name: "v3-yuv444p16",
        width: 64,
        height: 48,
        n_frames: 1,
        container: Container::Mkv,
        layout: RefLayout::Yuv {
            bits: 16,
            chroma_h_shift: 0,
            chroma_v_shift: 0,
        },
        tier: Tier::ReportOnly,
    });
}
