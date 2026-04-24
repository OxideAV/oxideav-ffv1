//! FFV1 packet decoder.
//!
//! Each compressed packet holds one frame: a leading range-coded keyframe
//! bit embedded in the first slice's range coder, followed by one or more
//! slices. Each slice ends with a trailer (`slice_size` + optional
//! `error_status` + `slice_crc_parity`) per RFC 9043 §4.5; the trailer
//! format is selected by the `ec` flag in the configuration record.
//!
//! Our decoder supports arbitrary `num_h_slices × num_v_slices` grids as
//! long as each slice lands on a single cell. It accepts 8-bit or 10-bit
//! YCbCr 4:2:0 / 4:2:2 / 4:4:4 streams with `coder_type = 1`, optional
//! `extra_plane` alpha channels (YUVA / RGBA), and 9..=15-bit RGB streams
//! following RFC §3.7.2.1's "RGB Exception" BGR plane order.

use oxideav_codec::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, VideoFrame,
};

use crate::config::ConfigRecord;
use crate::slice::{
    decode_frame_ex_with_states, decode_frame_golomb, decode_frame_rct_ex_with_states,
    decode_frame_u16_ex_with_states,
};

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let config = if params.extradata.is_empty() {
        // Allow extradata-less streams at our own default, inferring the
        // stream shape from the declared pixel format.
        match params.pixel_format {
            Some(PixelFormat::Yuv420P) | None => ConfigRecord::new_simple(false),
            Some(PixelFormat::Yuv422P) => ConfigRecord::new_yuv(8, 1, 0),
            Some(PixelFormat::Yuv444P) => ConfigRecord::new_simple(true),
            Some(PixelFormat::Yuv420P10Le) => ConfigRecord::new_yuv(10, 1, 1),
            Some(PixelFormat::Yuv422P10Le) => ConfigRecord::new_yuv(10, 1, 0),
            Some(PixelFormat::Yuv444P10Le) => ConfigRecord::new_yuv(10, 0, 0),
            Some(other) => {
                return Err(Error::unsupported(format!(
                    "FFV1 decoder: pixel format {:?}",
                    other
                )));
            }
        }
    } else {
        ConfigRecord::parse(&params.extradata)?
    };
    let width = params
        .width
        .ok_or_else(|| Error::invalid("FFV1 decoder: missing width"))?;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("FFV1 decoder: missing height"))?;
    Ok(Box::new(Ffv1Decoder {
        codec_id: params.codec_id.clone(),
        config,
        width,
        height,
        pending: None,
        eof: false,
    }))
}

struct Ffv1Decoder {
    codec_id: CodecId,
    config: ConfigRecord,
    width: u32,
    height: u32,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for Ffv1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "FFV1 decoder: receive_frame must be called before another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_packet(&self.config, &pkt, self.width, self.height)?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

fn decode_packet(
    config: &ConfigRecord,
    pkt: &Packet,
    width: u32,
    height: u32,
) -> Result<VideoFrame> {
    let data = &pkt.data;
    if data.is_empty() {
        return Err(Error::invalid("FFV1 decode: empty packet"));
    }

    let bits = config.bits_per_raw_sample;
    let ec = config.ec != 0;

    // JPEG 2000 RCT path: RGB stored as coded Y/Cb/Cr (plus optional alpha)
    // planes at the same resolution, then inverted to R, G, B [, A] per
    // pixel. Golomb-Rice with RCT isn't supported here.
    if config.is_rgb() {
        if config.coder_type == 0 {
            return Err(Error::unsupported("FFV1 RCT with Golomb-Rice coder_type=0"));
        }
        let transition = config.slice_state_transition();
        let quant_sets: Vec<_> = if config.quant_tables.is_empty() {
            vec![crate::state::default_quant_tables()]
        } else {
            config.quant_tables.clone()
        };
        let initial_states_slice: Option<&[Vec<[u8; 32]>]> = if config.initial_states.is_empty() {
            None
        } else {
            Some(&config.initial_states)
        };
        let decoded = decode_frame_rct_ex_with_states(
            data,
            width,
            height,
            config.num_h_slices,
            config.num_v_slices,
            bits,
            config.extra_plane,
            ec,
            &transition,
            &quant_sets,
            initial_states_slice,
        )?;
        // Pick an appropriate packed output format.
        let pix_fmt = match (decoded.bit_depth, decoded.channels) {
            (8, 3) => PixelFormat::Rgb24,
            (8, 4) => PixelFormat::Rgba,
            (d, 3) if (9..=16).contains(&d) => PixelFormat::Rgb48Le,
            (d, 4) if (9..=16).contains(&d) => PixelFormat::Rgba64Le,
            _ => {
                return Err(Error::unsupported(format!(
                    "FFV1 RCT: no packed output for bit_depth={}, channels={}",
                    decoded.bit_depth, decoded.channels
                )));
            }
        };
        let sample_bytes = if decoded.bit_depth > 8 { 2 } else { 1 };
        let stride = (width as usize) * (decoded.channels as usize) * sample_bytes;
        let plane = VideoPlane {
            stride,
            data: decoded.data,
        };
        return Ok(VideoFrame {
            format: pix_fmt,
            width,
            height,
            pts: pkt.pts,
            time_base: pkt.time_base,
            planes: vec![plane],
        });
    }

    // Non-RGB range-coded path honours `coder_type == 2` by picking up the
    // custom state transition from the config record. Golomb-Rice ignores
    // the range-coder state transition (it only applies to the slice header
    // range coder, which reuses the default for its short lifetime).
    let transition = config.slice_state_transition();

    // Map the pixel format from the config record.
    let pix_fmt = match (
        bits,
        config.is_yuv420(),
        config.is_yuv422(),
        config.is_yuv444(),
        config.extra_plane,
    ) {
        (8, true, _, _, false) => PixelFormat::Yuv420P,
        (8, _, true, _, false) => PixelFormat::Yuv422P,
        (8, _, _, true, false) => PixelFormat::Yuv444P,
        (10, true, _, _, false) => PixelFormat::Yuv420P10Le,
        (10, _, true, _, false) => PixelFormat::Yuv422P10Le,
        (10, _, _, true, false) => PixelFormat::Yuv444P10Le,
        // 8-bit YUV 4:2:0 + alpha → Yuva420P (our core type supports this
        // shape; 4:2:2 / 4:4:4 + alpha lands on Yuva420P for now since no
        // matching variant exists and alpha is the important part).
        (8, true, _, _, true) => PixelFormat::Yuva420P,
        _ => {
            return Err(Error::unsupported(format!(
                "FFV1: unsupported shape (bits={bits}, extra_plane={}, chroma_sub=({},{}))",
                config.extra_plane, config.log2_h_chroma_subsample, config.log2_v_chroma_subsample,
            )))
        }
    };

    let has_chroma = config.chroma_planes;
    let log2_h_sub = config.log2_h_chroma_subsample;
    let log2_v_sub = config.log2_v_chroma_subsample;
    let quant_sets: Vec<_> = if config.quant_tables.is_empty() {
        vec![crate::state::default_quant_tables()]
    } else {
        config.quant_tables.clone()
    };
    let initial_states_slice: Option<&[Vec<[u8; 32]>]> = if config.initial_states.is_empty() {
        None
    } else {
        Some(&config.initial_states)
    };

    if bits == 8 {
        let decoded = if config.coder_type == 0 {
            // Golomb-Rice path: only 8-bit, no alpha, default tables (the
            // RFC says Golomb SHOULD NOT be used with bits_per_raw_sample > 8
            // or with `extra_plane` — FFmpeg doesn't emit those shapes).
            if config.extra_plane {
                return Err(Error::unsupported(
                    "FFV1 Golomb-Rice with extra_plane alpha",
                ));
            }
            decode_frame_golomb(
                data,
                width,
                height,
                config.num_h_slices,
                config.num_v_slices,
                has_chroma,
                log2_h_sub,
                log2_v_sub,
                ec,
            )?
        } else {
            decode_frame_ex_with_states(
                data,
                width,
                height,
                config.num_h_slices,
                config.num_v_slices,
                has_chroma,
                config.extra_plane,
                log2_h_sub,
                log2_v_sub,
                ec,
                &quant_sets,
                &transition,
                initial_states_slice,
            )?
        };

        let y_plane = VideoPlane {
            stride: width as usize,
            data: decoded.y,
        };
        let cw = decoded.c_geom.width as usize;
        let u_plane = VideoPlane {
            stride: cw,
            data: decoded.u.unwrap_or_default(),
        };
        let v_plane = VideoPlane {
            stride: cw,
            data: decoded.v.unwrap_or_default(),
        };
        let mut planes = vec![y_plane, u_plane, v_plane];
        if let Some(a) = decoded.a {
            planes.push(VideoPlane {
                stride: width as usize,
                data: a,
            });
        }

        Ok(VideoFrame {
            format: pix_fmt,
            width,
            height,
            pts: pkt.pts,
            time_base: pkt.time_base,
            planes,
        })
    } else {
        if config.coder_type == 0 {
            return Err(Error::unsupported(
                "FFV1 Golomb-Rice decode with bits_per_raw_sample > 8",
            ));
        }
        // 10-bit (or wider) path: decode into u16 buffers and repack as
        // little-endian bytes. Stride is `width * 2` bytes.
        let decoded = decode_frame_u16_ex_with_states(
            data,
            width,
            height,
            config.num_h_slices,
            config.num_v_slices,
            has_chroma,
            config.extra_plane,
            log2_h_sub,
            log2_v_sub,
            ec,
            bits,
            &quant_sets,
            &transition,
            initial_states_slice,
        )?;

        let y_plane = VideoPlane {
            stride: (width as usize) * 2,
            data: u16_samples_to_le_bytes(&decoded.y),
        };
        let cw = decoded.c_geom.width as usize;
        let u_plane = VideoPlane {
            stride: cw * 2,
            data: decoded
                .u
                .as_deref()
                .map(u16_samples_to_le_bytes)
                .unwrap_or_default(),
        };
        let v_plane = VideoPlane {
            stride: cw * 2,
            data: decoded
                .v
                .as_deref()
                .map(u16_samples_to_le_bytes)
                .unwrap_or_default(),
        };
        let mut planes = vec![y_plane, u_plane, v_plane];
        if let Some(a) = &decoded.a {
            planes.push(VideoPlane {
                stride: (width as usize) * 2,
                data: u16_samples_to_le_bytes(a),
            });
        }

        Ok(VideoFrame {
            format: pix_fmt,
            width,
            height,
            pts: pkt.pts,
            time_base: pkt.time_base,
            planes,
        })
    }
}

fn u16_samples_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}
