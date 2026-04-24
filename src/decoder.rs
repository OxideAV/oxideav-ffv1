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
//! YCbCr 4:2:0 / 4:2:2 / 4:4:4 streams with `coder_type = 1`.

use oxideav_codec::Decoder;
use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, VideoFrame,
};

use crate::config::ConfigRecord;
use crate::slice::{decode_frame, decode_frame_golomb, decode_frame_u16};

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
    // Map the pixel format from the config record.
    let pix_fmt = match (
        bits,
        config.is_yuv420(),
        config.is_yuv422(),
        config.is_yuv444(),
    ) {
        (8, true, _, _) => PixelFormat::Yuv420P,
        (8, _, true, _) => PixelFormat::Yuv422P,
        (8, _, _, true) => PixelFormat::Yuv444P,
        (10, true, _, _) => PixelFormat::Yuv420P10Le,
        (10, _, true, _) => PixelFormat::Yuv422P10Le,
        (10, _, _, true) => PixelFormat::Yuv444P10Le,
        _ => return Err(Error::unsupported("FFV1: unsupported chroma subsampling")),
    };

    let has_chroma = config.chroma_planes;
    let log2_h_sub = config.log2_h_chroma_subsample;
    let log2_v_sub = config.log2_v_chroma_subsample;
    let ec = config.ec != 0;

    if bits == 8 {
        let decoded = if config.coder_type == 0 {
            // Golomb-Rice path: only 8-bit is supported at the moment
            // (RFC 9043 says Golomb SHOULD NOT be used with
            // bits_per_raw_sample > 8 anyway).
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
            decode_frame(
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

        Ok(VideoFrame {
            format: pix_fmt,
            width,
            height,
            pts: pkt.pts,
            time_base: pkt.time_base,
            planes: vec![y_plane, u_plane, v_plane],
        })
    } else {
        if config.coder_type == 0 {
            return Err(Error::unsupported(
                "FFV1 Golomb-Rice decode with bits_per_raw_sample > 8",
            ));
        }
        // 10-bit (or wider) path: decode into u16 buffers and repack as
        // little-endian bytes. Stride is `width * 2` bytes.
        let decoded = decode_frame_u16(
            data,
            width,
            height,
            config.num_h_slices,
            config.num_v_slices,
            has_chroma,
            log2_h_sub,
            log2_v_sub,
            ec,
            bits,
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

        Ok(VideoFrame {
            format: pix_fmt,
            width,
            height,
            pts: pkt.pts,
            time_base: pkt.time_base,
            planes: vec![y_plane, u_plane, v_plane],
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
