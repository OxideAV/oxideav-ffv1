//! `oxideav-core` framework integration: codec registration plus the
//! [`oxideav_core::Decoder`] implementation wrapping the crate's
//! [`decode_frame`] / [`decode_frame_rgb`] drivers.
//!
//! This is the registry half of the dual-API contract. The direct
//! `decode_frame*` / `encode_frame*` entry points stay the primary,
//! fully-typed surface (FFV1 carries no host pixel-buffer layout of its
//! own — see [`map_to_video_frame`] for the byte convention this module
//! picks for the framework's [`Frame`] surface); this module additionally
//! exposes the decoder behind the framework [`Decoder`] trait so a
//! container (`oxideav-avi`, `oxideav-mkv`, …) can route an FFV1 stream
//! to it through [`oxideav_core::CodecRegistry`] without hand-wiring the
//! `parse_quantization_table_sets` → `decode_frame*` plumbing.
//!
//! ## Configuration source
//!
//! Every FFV1 v3 decode needs the §4.2 Configuration Record (entropy
//! coder, colorspace, chroma subsampling, bit depth, slice grid,
//! Quantization Table Sets) and the §4 frame pixel dimensions. The
//! framework hands those in via [`CodecParameters`]:
//!
//! * `params.extradata` carries the Configuration Record. For the AVI
//!   mapping (RFC 9043 §4.3.3.1) it is the `strf`-chunk tail; for the
//!   Matroska mapping (RFC 9043 §4.3.3.4) it is the `CodecPrivate`
//!   element. Either way it is the bytes
//!   [`parse_quantization_table_sets`] consumes.
//! * `params.width` / `params.height` carry the §4 frame dimensions
//!   the container declares (FFV1 frames do not self-describe their
//!   pixel dimensions — RFC 9043 §4 "The whole Frame is provided by the
//!   underlying container").
//!
//! When any piece is missing the factory still constructs a decoder; the
//! diagnosable [`CoreError`] is surfaced at `receive_frame` time so a
//! caller driving the trait sees a precise reason rather than a panic.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Error as CoreError, Frame, Packet, Result as CoreResult, RuntimeContext, VideoFrame,
    VideoPlane,
};

use crate::{
    decode_frame, decode_frame_rgb, parse_quantization_table_sets, ColorspaceType, DecodedFrame,
    Ffv1ConfigurationRecord, FramePixelDimensions, ParametersWithQuantTables, QuantizationTableSet,
};

/// Canonical codec id string. `oxideav-meta`'s `register_all` reaches
/// this crate through `crate::__oxideav_entry` (the macro-generated
/// wrapper), which delegates to [`register`].
pub const CODEC_ID_STR: &str = "ffv1";

/// Register the FFV1 codec with `reg`.
///
/// Claims the two container tags RFC 9043 §4.3.3 names:
///
/// * the AVI FourCC `FFV1` (RFC 9043 §4.3.3.1 — the AVI mapping
///   extends the `strf` chunk with the Configuration Record; `FFV1` is
///   the codec's four-byte stream identifier), and
/// * the Matroska Codec ID `V_FFV1` (RFC 9043 §4.3.3.4 — "FFV1 SHOULD
///   use V_FFV1 as the Matroska Codec ID").
///
/// Only the decoder factory is wired this round; the encoder trait is a
/// follow-up (it needs §4.6 Slice-Header synthesis from the
/// Configuration Record's slice grid, which the direct `encode_frame*`
/// entry points currently take as an explicit argument).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("ffv1_sw")
        .with_decode()
        .with_lossless(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .tags([CodecTag::fourcc(b"FFV1"), CodecTag::matroska("V_FFV1")]),
    );
}

/// Unified entry point invoked by the macro-generated wrapper.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

// ──────────────────────── Decoder impl ────────────────────────

/// Build an FFV1 decoder for `params`.
///
/// The Configuration Record / dimensions are read at factory time (see
/// [`build_decode_config`]); if the container has not yet supplied them
/// the decoder is still constructed and surfaces a diagnosable error at
/// `receive_frame` time.
fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    let cfg = build_decode_config(params)?;
    Ok(Box::new(Ffv1Decoder {
        codec_id: params.codec_id.clone(),
        cfg,
        pending: None,
        eof: false,
    }))
}

/// Parsed, ready-to-decode configuration assembled from
/// [`CodecParameters`]. Holds the §4.2 Configuration Record, the §4.1
/// Quantization Table Sets, the §4 frame dimensions, and the §4.2.16
/// entropy-coder flag (`ec`).
struct DecodeConfig {
    record: Ffv1ConfigurationRecord,
    quant_table_sets: Vec<QuantizationTableSet>,
    dims: FramePixelDimensions,
    ec: bool,
}

/// Assemble a [`DecodeConfig`] from `params` when every required piece
/// is present.
///
/// Returns `Ok(None)` (a deferred decoder) when the container has not
/// yet supplied the Configuration Record extradata or the frame
/// dimensions — the decoder surfaces a precise error at first
/// `receive_frame`. Returns `Err` only when the supplied pieces are
/// themselves malformed (Configuration Record parse failure, zero
/// dimensions).
fn build_decode_config(params: &CodecParameters) -> CoreResult<Option<DecodeConfig>> {
    if params.extradata.is_empty() {
        return Ok(None);
    }
    let (Some(width), Some(height)) = (params.width, params.height) else {
        return Ok(None);
    };
    let ParametersWithQuantTables {
        record,
        quant_table_sets,
        ..
    } = parse_quantization_table_sets(&params.extradata)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: configuration record: {e}")))?;
    let dims = FramePixelDimensions::new(width, height)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: frame dimensions: {e}")))?;
    let ec = record.ec.is_some();
    Ok(Some(DecodeConfig {
        record,
        quant_table_sets,
        dims,
        ec,
    }))
}

struct Ffv1Decoder {
    codec_id: CodecId,
    /// Parsed decode configuration, built at factory time from the
    /// `CodecParameters` the container handed in. `None` when the
    /// container had not yet supplied extradata / dimensions; the
    /// decoder surfaces a diagnosable error at `receive_frame`.
    cfg: Option<DecodeConfig>,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for Ffv1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if self.pending.is_some() {
            return Err(CoreError::other(
                "oxideav-ffv1: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(CoreError::Eof)
            } else {
                Err(CoreError::NeedMore)
            };
        };
        let cfg = self.cfg.as_ref().ok_or_else(|| {
            CoreError::invalid(
                "oxideav-ffv1: stream not configured — CodecParameters needs the §4.2 \
                 Configuration Record in `extradata` plus `width` / `height`",
            )
        })?;
        // RFC 9043 §4.2.5: colorspace_type 0 is the YCbCr / plane-major
        // path, 1 is the RGB / JPEG 2000 RCT line-major path. Each has
        // its own driver; both produce plane-major-ordered output.
        let decoded = match cfg.record.colorspace_type {
            ColorspaceType::YCbCr => decode_frame(
                &pkt.data,
                &cfg.record,
                &cfg.quant_table_sets,
                cfg.dims,
                cfg.ec,
            ),
            ColorspaceType::Rgb => decode_frame_rgb(
                &pkt.data,
                &cfg.record,
                &cfg.quant_table_sets,
                cfg.dims,
                cfg.ec,
            ),
        }
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
        Ok(Frame::Video(map_to_video_frame(decoded, pkt.pts)))
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }
}

/// Map a reconstructed [`DecodedFrame`] onto the framework's
/// [`VideoFrame`].
///
/// FFV1's RFC 9043 §4.7 reconstruction yields one `i32`-per-Sample plane
/// per primary colour, each Sample in `0 .. 2^bits_per_raw_sample`; the
/// RFC does not prescribe a host pixel-buffer layout (that is the
/// container/renderer's concern). This module fixes the framework-surface
/// convention as:
///
/// * **8 bits or fewer** — one byte per Sample, `stride == width`.
/// * **more than 8 bits** — little-endian `u16` per Sample,
///   `stride == width * 2` (the low byte first, matching the
///   little-endian packing FFV1's `bits_per_raw_sample > 8` Samples use
///   throughout the codec).
///
/// Plane order is the §4.7.1 `primary_color_count` order the driver
/// already emits (luma / R / gray first; chroma 1 / 2 when present; the
/// extra / alpha plane last).
fn map_to_video_frame(frame: DecodedFrame, pts: Option<i64>) -> VideoFrame {
    let wide = frame.bits_per_raw_sample > 8;
    let planes = frame
        .planes
        .into_iter()
        .map(|p| {
            let width = p.width as usize;
            if wide {
                let mut data = Vec::with_capacity(p.samples.len() * 2);
                for s in &p.samples {
                    let v = *s as u16;
                    data.extend_from_slice(&v.to_le_bytes());
                }
                VideoPlane {
                    stride: width * 2,
                    data,
                }
            } else {
                let data = p.samples.iter().map(|&s| s as u8).collect();
                VideoPlane {
                    stride: width,
                    data,
                }
            }
        })
        .collect();
    VideoFrame { pts, planes }
}
