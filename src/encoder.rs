//! FFV1 frame encoder.
//!
//! Emits an FFV1 v3 packet per input video frame. The output stream's
//! `extradata` (available via `output_params().extradata`) contains the
//! configuration record; muxers (e.g. Matroska) should read it from there.
//!
//! The encoder supports 8-bit and 10-bit YUV 4:2:0 / 4:2:2 / 4:4:4 input,
//! optionally split across a `num_h × num_v` slice grid (see
//! [`Ffv1EncoderOptions::slices`]). Golomb-Rice (`coder_type = 0`) is
//! supported for YUV (8-bit and 10-bit) and 8-bit YUVA (`Yuva420P`,
//! `extra_plane`). FFmpeg's decoder accepts both the single-slice and
//! multi-slice outputs bit-exactly.

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    parse_options, CodecId, CodecOptionsStruct, CodecParameters, Error, Frame, MediaType,
    OptionField, OptionKind, OptionValue, Packet, PixelFormat, Result, TimeBase, VideoFrame,
};

use crate::config::ConfigRecord;
use crate::slice::{
    encode_frame_golomb, encode_frame_golomb_u16, encode_multi_slice_frame,
    encode_multi_slice_frame_u16, encode_single_slice_frame, encode_single_slice_frame_rct,
    encode_single_slice_frame_u16, PlaneGeom, SlicePlanes, SlicePlanes16,
};

/// Encoder tuning knobs, attached via
/// [`CodecParameters::options`](oxideav_core::CodecParameters::options).
///
/// Recognised keys (see [`CodecOptionsStruct::SCHEMA`]):
/// - `slices` *(u32, default `1`)* — Total slice count. The encoder picks a
///   `num_h × num_v` factorisation that divides the frame: `num_v` is the
///   largest divisor of `slices` that is `≤ height`, and `num_h = slices /
///   num_v`. Passing `1` produces a single slice (the legacy shape).
/// - `coder_type` *(u32, default `1`)* — 1 = range coder with the default
///   state-transition table (most common); 0 = Golomb-Rice VLC (matches
///   FFmpeg's `-coder 0`). Golomb-Rice supports 8-bit and 10-bit YUV (with
///   or without alpha / `extra_plane`); RGB/RCT with Golomb is not yet
///   wired.
#[derive(Debug, Clone)]
pub struct Ffv1EncoderOptions {
    pub slices: u32,
    pub coder_type: u32,
}

impl Default for Ffv1EncoderOptions {
    fn default() -> Self {
        Self {
            slices: 1,
            coder_type: 1,
        }
    }
}

impl CodecOptionsStruct for Ffv1EncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "slices",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "Total number of slices to emit per frame (default 1). \
                   Factored into a num_h × num_v grid at encode time.",
        },
        OptionField {
            name: "coder_type",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "0 = Golomb-Rice VLC, 1 = range coder (default). Matches \
                   FFmpeg's `-coder` option.",
        },
    ];
    fn apply(&mut self, key: &str, v: &OptionValue) -> Result<()> {
        match key {
            "slices" => self.slices = v.as_u32()?,
            "coder_type" => self.coder_type = v.as_u32()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Factor `n` into `(num_h, num_v)` such that both dimensions divide the
/// frame sensibly. The heuristic mirrors what FFmpeg's `-slices N`
/// command-line gives for common counts: 2 → 1×2, 4 → 2×2, 6 → 2×3,
/// 9 → 3×3, 16 → 4×4, etc.
///
/// Constraints: `num_h ≤ width`, `num_v ≤ height`, `num_h * num_v == n`.
/// Falls back to `(n, 1)` if no better factorisation fits the frame.
fn factor_slice_count(n: u32, width: u32, height: u32) -> (u32, u32) {
    let mut best: Option<(u32, u32, i64)> = None;
    for v in 1..=n {
        if n % v != 0 {
            continue;
        }
        let h = n / v;
        if h > width || v > height {
            continue;
        }
        // Prefer "squarest" grids: minimise |h - v|. Among ties prefer
        // more rows than columns (matches FFmpeg's default tie-break).
        let aspect_score = (h as i64 - v as i64).abs();
        let tie_score = if h <= v { 0 } else { 1 };
        let score = aspect_score * 2 + tie_score;
        if best.map_or(true, |(_, _, s)| score < s) {
            best = Some((h, v, score));
        }
    }
    match best {
        Some((h, v, _)) => (h, v),
        None => (n.min(width).max(1), 1),
    }
}

/// Describe the stream shape implied by an input pixel format: bit depth
/// and chroma subsampling exponents. RGB inputs are represented as (8, 0, 0)
/// — chroma_planes but at full resolution — with colorspace_type selected
/// separately.
fn stream_shape(pix: PixelFormat) -> Option<(u32, u32, u32)> {
    match pix {
        PixelFormat::Yuv420P => Some((8, 1, 1)),
        PixelFormat::Yuv422P => Some((8, 1, 0)),
        PixelFormat::Yuv444P => Some((8, 0, 0)),
        PixelFormat::Yuv420P10Le => Some((10, 1, 1)),
        PixelFormat::Yuv422P10Le => Some((10, 1, 0)),
        PixelFormat::Yuv444P10Le => Some((10, 0, 0)),
        // 4-plane YUV (alpha). Only 8-bit 4:2:0 + alpha is exposed by
        // `oxideav-core`'s PixelFormat enum today.
        PixelFormat::Yuva420P => Some((8, 1, 1)),
        // RGB via JPEG 2000 RCT — no chroma subsampling possible.
        PixelFormat::Rgb24 => Some((8, 0, 0)),
        _ => None,
    }
}

/// True when the input pixel format carries a fourth alpha plane. This
/// drives the `extra_plane` flag in the configuration record, which tells
/// the decoder one more plane at luma resolution is coded after chroma.
fn has_alpha_plane(pix: PixelFormat) -> bool {
    matches!(pix, PixelFormat::Yuva420P)
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let opts = parse_options::<Ffv1EncoderOptions>(&params.options)?;
    let width = params
        .width
        .ok_or_else(|| Error::invalid("FFV1 encoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("FFV1 encoder: missing height"))?;
    let pix = params.pixel_format.unwrap_or(PixelFormat::Yuv420P);
    let (bits, log2_h, log2_v) = stream_shape(pix)
        .ok_or_else(|| Error::unsupported(format!("FFV1 encoder: pixel format {:?}", pix)))?;
    if opts.slices == 0 {
        return Err(Error::invalid("FFV1 encoder: slices must be >= 1"));
    }
    let (num_h_slices, num_v_slices) = factor_slice_count(opts.slices, width, height);
    // Subsampled chroma needs each *interior* slice boundary to land on a
    // chroma sample so the decoder's grid math (x0 = sx * W / num_h) lines
    // up with the rounded-up chroma plane. The frame edges (sx = 0 or
    // num_h) are always fine — they just adopt the chroma's rounding.
    let chroma_mult_x = 1u32 << log2_h;
    let chroma_mult_y = 1u32 << log2_v;
    for sx in 1..num_h_slices {
        let x = sx * width / num_h_slices;
        if x % chroma_mult_x != 0 {
            return Err(Error::invalid(format!(
                "FFV1 encoder: slice grid {num_h_slices}x{num_v_slices} doesn't \
                 align to {chroma_mult_x}:{chroma_mult_y} chroma on {width}x{height}"
            )));
        }
    }
    for sy in 1..num_v_slices {
        let y = sy * height / num_v_slices;
        if y % chroma_mult_y != 0 {
            return Err(Error::invalid(format!(
                "FFV1 encoder: slice grid {num_h_slices}x{num_v_slices} doesn't \
                 align to {chroma_mult_x}:{chroma_mult_y} chroma on {width}x{height}"
            )));
        }
    }

    // RGB inputs use the JPEG 2000 RCT config record (colorspace_type=1,
    // chroma_planes=1, no subsampling); YUV uses `new_yuv`. RGB encode
    // currently only handles a single slice; multi-slice RGB is a future
    // extension.
    let is_rgb = matches!(pix, PixelFormat::Rgb24);
    if is_rgb && (num_h_slices > 1 || num_v_slices > 1) {
        return Err(Error::unsupported(
            "FFV1 encoder: RGB multi-slice not yet implemented",
        ));
    }
    // Golomb-Rice (coder_type = 0) is currently 8-bit YUV only — no RGB/RCT,
    // no alpha, no 10-bit. Reject unsupported combos up front.
    if opts.coder_type != 0 && opts.coder_type != 1 {
        return Err(Error::unsupported(format!(
            "FFV1 encoder: coder_type={} not supported",
            opts.coder_type
        )));
    }
    if opts.coder_type == 0 && is_rgb {
        return Err(Error::unsupported(
            "FFV1 encoder: Golomb-Rice with RGB/RCT not yet implemented",
        ));
    }
    // `extra_plane` alpha is currently only wired in the Golomb-Rice encode
    // path — the single-slice / multi-slice range-coder encoders still emit
    // 3-plane frames only. Reject the combo up front to avoid a silent
    // mismatch between the config record (extra_plane=1) and the payload.
    if has_alpha_plane(pix) && opts.coder_type != 0 {
        return Err(Error::unsupported(
            "FFV1 encoder: range-coded YUVA (extra_plane) not yet implemented",
        ));
    }
    let mut config = if is_rgb {
        ConfigRecord::new_rgb_rct()
    } else {
        ConfigRecord::new_yuv(bits, log2_h, log2_v)
    };
    config.num_h_slices = num_h_slices;
    config.num_v_slices = num_v_slices;
    config.coder_type = opts.coder_type;
    config.extra_plane = has_alpha_plane(pix);
    let extradata = config.encode();

    let mut output_params = params.clone();
    output_params.media_type = MediaType::Video;
    output_params.codec_id = CodecId::new(super::CODEC_ID_STR);
    output_params.width = Some(width);
    output_params.height = Some(height);
    output_params.pixel_format = Some(pix);
    output_params.extradata = extradata;

    Ok(Box::new(Ffv1Encoder {
        output_params,
        width,
        height,
        pix,
        num_h_slices,
        num_v_slices,
        log2_h,
        log2_v,
        coder_type: opts.coder_type,
        time_base: params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num)),
        pending: VecDeque::new(),
        eof: false,
    }))
}

struct Ffv1Encoder {
    output_params: CodecParameters,
    width: u32,
    height: u32,
    pix: PixelFormat,
    num_h_slices: u32,
    num_v_slices: u32,
    log2_h: u32,
    log2_v: u32,
    coder_type: u32,
    time_base: TimeBase,
    pending: VecDeque<Packet>,
    eof: bool,
}

impl Encoder for Ffv1Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::invalid("FFV1 encoder: video frames only"));
        };
        let data = encode_frame(
            v,
            self.width,
            self.height,
            self.pix,
            self.num_h_slices,
            self.num_v_slices,
            self.log2_h,
            self.log2_v,
            self.coder_type,
        )?;
        let mut pkt = Packet::new(0, self.time_base, data);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = true;
        self.pending.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

fn encode_frame(
    v: &VideoFrame,
    width: u32,
    height: u32,
    pix: PixelFormat,
    num_h: u32,
    num_v: u32,
    cfg_log2_h: u32,
    cfg_log2_v: u32,
    coder_type: u32,
) -> Result<Vec<u8>> {
    // RGB input comes in a single packed plane; YUV has three. Validate up
    // front so downstream panics stay out of reach.
    if matches!(pix, PixelFormat::Rgb24) {
        if v.planes.len() != 1 {
            return Err(Error::invalid("FFV1 encoder: Rgb24 must have 1 plane"));
        }
        // Repack (honouring stride) into a contiguous w*h*3 byte buffer.
        let w = width as usize;
        let h = height as usize;
        let stride = v.planes[0].stride;
        if stride < w * 3 || v.planes[0].data.len() < (h - 1) * stride + w * 3 {
            return Err(Error::invalid(
                "FFV1 encoder: Rgb24 plane stride/buffer too small",
            ));
        }
        let mut rgb = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            let start = row * stride;
            rgb.extend_from_slice(&v.planes[0].data[start..start + w * 3]);
        }
        return encode_single_slice_frame_rct(&rgb, width, height, false);
    }
    if v.planes.len() != 3 && v.planes.len() != 4 {
        return Err(Error::invalid(
            "FFV1 encoder: expected 3 or 4 planes (Y/U/V[/A])",
        ));
    }
    let (bits, log2_h, log2_v) = stream_shape(pix)
        .ok_or_else(|| Error::unsupported(format!("FFV1 encoder: format {:?}", pix)))?;
    debug_assert_eq!(
        (log2_h, log2_v),
        (cfg_log2_h, cfg_log2_v),
        "encoder config and frame format must agree on chroma subsampling"
    );
    let cw = width.div_ceil(1 << log2_h);
    let ch = height.div_ceil(1 << log2_v);
    let multi = num_h > 1 || num_v > 1;

    if bits == 8 {
        // Flatten Y / U / V planes into contiguous w*h byte buffers.
        let y_flat = flatten_plane_u8(&v.planes[0].data, v.planes[0].stride, width, height);
        let u_flat = flatten_plane_u8(&v.planes[1].data, v.planes[1].stride, cw, ch);
        let v_flat = flatten_plane_u8(&v.planes[2].data, v.planes[2].stride, cw, ch);

        let a_flat_opt: Option<Vec<u8>> = if v.planes.len() >= 4 {
            Some(flatten_plane_u8(
                &v.planes[3].data,
                v.planes[3].stride,
                width,
                height,
            ))
        } else {
            None
        };
        let planes = SlicePlanes {
            y: &y_flat,
            u: Some(&u_flat),
            v: Some(&v_flat),
            a: a_flat_opt.as_deref(),
            y_geom: PlaneGeom { width, height },
            c_geom: PlaneGeom {
                width: cw,
                height: ch,
            },
        };
        if coder_type == 0 {
            // Golomb-Rice: both single and multi-slice share the same
            // per-slice Sentinel-mode termination and Golomb bit stream.
            encode_frame_golomb(&planes, num_h, num_v, log2_h, log2_v, false)
        } else if multi {
            Ok(encode_multi_slice_frame(
                &planes, num_h, num_v, log2_h, log2_v, false,
            ))
        } else {
            // RFC 9043 single-slice v3 packet: keyframe bit, slice header,
            // planes, 3-byte slice_size footer. We keep `ec = false` in the
            // config record — FFmpeg handles either form depending on the ec
            // flag it reads from extradata.
            Ok(encode_single_slice_frame(&planes, false))
        }
    } else {
        // 10-bit (or wider) path: samples are packed as little-endian u16
        // in the input `VideoPlane.data` byte buffer.
        let y_flat = flatten_plane_u16(&v.planes[0].data, v.planes[0].stride, width, height)?;
        let u_flat = flatten_plane_u16(&v.planes[1].data, v.planes[1].stride, cw, ch)?;
        let v_flat = flatten_plane_u16(&v.planes[2].data, v.planes[2].stride, cw, ch)?;

        let a_flat_opt: Option<Vec<u16>> = if v.planes.len() >= 4 {
            Some(flatten_plane_u16(
                &v.planes[3].data,
                v.planes[3].stride,
                width,
                height,
            )?)
        } else {
            None
        };
        let planes = SlicePlanes16 {
            y: &y_flat,
            u: Some(&u_flat),
            v: Some(&v_flat),
            a: a_flat_opt.as_deref(),
            y_geom: PlaneGeom { width, height },
            c_geom: PlaneGeom {
                width: cw,
                height: ch,
            },
            bit_depth: bits,
        };
        if coder_type == 0 {
            // We advertise Golomb u16 encode as unsupported at construction
            // time — this branch is defensive, kept so the match stays
            // exhaustive if someone lowers that restriction.
            encode_frame_golomb_u16(&planes, num_h, num_v, log2_h, log2_v, false)
        } else if multi {
            Ok(encode_multi_slice_frame_u16(
                &planes, num_h, num_v, log2_h, log2_v, false,
            ))
        } else {
            Ok(encode_single_slice_frame_u16(&planes, false))
        }
    }
}

fn flatten_plane_u8(data: &[u8], stride: usize, width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &data[y * stride..y * stride + w];
        out.extend_from_slice(row);
    }
    out
}

/// Read a `width * height` plane from a byte buffer holding little-endian
/// `u16` samples. `stride` is in bytes (usually `width * 2` plus padding).
fn flatten_plane_u16(data: &[u8], stride: usize, width: u32, height: u32) -> Result<Vec<u16>> {
    let w = width as usize;
    let h = height as usize;
    if data.len() < h.saturating_mul(stride) || stride < w * 2 {
        return Err(Error::invalid(
            "FFV1 encoder: u16 plane stride/buffer too small",
        ));
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &data[y * stride..y * stride + w * 2];
        for x in 0..w {
            out.push(u16::from_le_bytes([row[x * 2], row[x * 2 + 1]]));
        }
    }
    Ok(out)
}
