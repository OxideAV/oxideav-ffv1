//! `oxideav-core` framework integration: codec registration plus the
//! [`oxideav_core::Decoder`] implementation wrapping the crate's
//! frame-level decode drivers ([`crate::decode_frame`] /
//! [`crate::decode_frame_rgb`]).
//!
//! The decoder reads the FFV1 §4.2 Configuration Record from
//! [`CodecParameters::extradata`] (RFC 9043 §4.3.3: the Configuration
//! Record is carried by the surrounding container — Matroska
//! `CodecPrivate` §4.3.3.4 / AVI stream-format chunk §4.3.3.1), the
//! frame pixel dimensions from `params.width` / `params.height` (FFV1's
//! Configuration Record carries no width / height per §4.2), and each
//! compressed Frame payload from a [`Packet`]. It dispatches on the
//! Configuration Record's §4.2.5 `colorspace_type` between the
//! plane-major YCbCr driver and the line-major RGB driver, mirroring the
//! routing [`crate::Ffv1DecodeSession`] performs, and threads the
//! §3.8.1.3 / §3.8.2.5 per-context coder state across non-keyframes.
//!
//! Registration claims the two RFC 9043 §4.3.3 container tags: the AVI
//! / VfW FourCC `FFV1` (§4.3.3.1) and the Matroska Codec ID `V_FFV1`
//! (§4.3.3.4). The container crate's `CodecResolver` routes either
//! on-wire identifier to this codec.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Error as CoreError, Frame, Packet, Result as CoreResult, RuntimeContext, VideoFrame,
    VideoPlane,
};

use crate::config::{ColorspaceType, Ffv1ConfigurationRecord};
use crate::crc::validate_configuration_record_crc;
use crate::frame::{decode_frame_with_carry, DecodeOptions, DecodedFrame, Ffv1FrameCarry};
use crate::quant_table::{parse_quantization_table_sets, QuantizationTableSet};
use crate::rgb_reconstruct::decode_frame_rgb_with_carry;
use crate::slice_content::FramePixelDimensions;

/// Canonical codec id. `oxideav-meta::register_all` calls
/// `crate::__oxideav_entry`, which delegates to [`register`].
pub const CODEC_ID_STR: &str = "ffv1";

/// Register the FFV1 codec into `reg`.
///
/// Claims the RFC 9043 §4.3.3 container tags: the AVI / VfW FourCC
/// `FFV1` (§4.3.3.1) and the Matroska Codec ID `V_FFV1` (§4.3.3.4).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("ffv1_sw")
        .with_decode()
        .with_lossless(true);
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

/// Per-stream decode setup parsed once from [`CodecParameters`].
///
/// Built from `params.extradata` (the §4.2 Configuration Record per
/// RFC 9043 §4.3.3) plus `params.width` / `params.height`. Held by the
/// constructed decoder so each `receive_frame` reuses the parsed
/// cascade instead of re-walking the extradata.
struct StreamSetup {
    cr: Ffv1ConfigurationRecord,
    quant_table_sets: Vec<QuantizationTableSet>,
    frame_dims: FramePixelDimensions,
    /// §4.2.16 `ec` flag — `true` means a §4.9.3 CRC is present in each
    /// Slice (8-byte §4.9 Slice Footer). Derived as `cr.ec != 0` per
    /// RFC 9043 Table 13 (any non-zero `ec` puts a CRC in each Slice).
    ec: bool,
}

/// Assemble a [`StreamSetup`] from `params`, or return `None` if the
/// container has not yet supplied the extradata / dimensions (so the
/// decoder can be built ahead of a deferred configuration). Returns
/// `Err` when the supplied pieces are present but inconsistent
/// (malformed Configuration Record, failed §4.3.2 CRC, zero dims).
fn build_stream_setup(params: &CodecParameters) -> CoreResult<Option<StreamSetup>> {
    if params.extradata.is_empty() {
        return Ok(None);
    }
    let (Some(width), Some(height)) = (params.width, params.height) else {
        return Ok(None);
    };

    // RFC 9043 §4.3.2: the Configuration Record carries a 32-bit CRC
    // parity word over the whole record. Validate it before trusting the
    // parsed fields (a corrupt extradata blob is a hard error, not a
    // recoverable per-Slice corruption).
    validate_configuration_record_crc(&params.extradata)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;

    let parsed = parse_quantization_table_sets(&params.extradata)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
    let frame_dims = FramePixelDimensions::new(width, height)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
    // RFC 9043 §4.2.16 Table 13: `ec == 0` keeps the 32-bit CRC in the
    // Configuration Record only (3-byte §4.9 Slice Footer); any non-zero
    // `ec` puts a 32-bit CRC in each Slice (8-byte footer). The
    // footer-present condition is therefore `ec != 0`. An absent field
    // (`None`, versions 0/1) infers no per-Slice CRC.
    let ec = !matches!(parsed.record.ec, None | Some(0));
    Ok(Some(StreamSetup {
        cr: parsed.record,
        quant_table_sets: parsed.quant_table_sets,
        frame_dims,
        ec,
    }))
}

fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    let setup = build_stream_setup(params)?;
    Ok(Box::new(Ffv1FrameDecoder {
        codec_id: params.codec_id.clone(),
        setup,
        options: DecodeOptions::default(),
        carry: None,
        pending: None,
        eof: false,
    }))
}

struct Ffv1FrameDecoder {
    codec_id: CodecId,
    /// Per-stream decode setup. `None` until the container supplies
    /// extradata + dimensions; `receive_frame` then surfaces a
    /// diagnosable `Error::invalid` instead of decoding garbage.
    setup: Option<StreamSetup>,
    options: DecodeOptions,
    /// §3.8.1.3 / §3.8.2.5 per-context coder state carried across
    /// non-keyframes (the cross-Frame channel the single-Frame drivers
    /// cannot hold). `None` before the first Frame.
    carry: Option<Ffv1FrameCarry>,
    pending: Option<Packet>,
    eof: bool,
}

impl std::fmt::Debug for Ffv1FrameDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ffv1FrameDecoder")
            .field("codec_id", &self.codec_id)
            .field("configured", &self.setup.is_some())
            .field("eof", &self.eof)
            .finish()
    }
}

impl Decoder for Ffv1FrameDecoder {
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
        let setup = self.setup.as_ref().ok_or_else(|| {
            CoreError::invalid(
                "oxideav-ffv1: stream not configured (CodecParameters needs the §4.2 \
                 Configuration Record in extradata plus width / height)",
            )
        })?;

        // Route on §4.2.5 colorspace_type, mirroring Ffv1DecodeSession.
        // Decode into a working copy of the carry so a decode error
        // leaves the decoder's carry untouched.
        let mut working_carry = self.carry.clone();
        let decoded = match setup.cr.colorspace_type {
            ColorspaceType::YCbCr => decode_frame_with_carry(
                &pkt.data,
                &setup.cr,
                &setup.quant_table_sets,
                setup.frame_dims,
                setup.ec,
                self.options,
                &mut working_carry,
            ),
            ColorspaceType::Rgb => decode_frame_rgb_with_carry(
                &pkt.data,
                &setup.cr,
                &setup.quant_table_sets,
                setup.frame_dims,
                setup.ec,
                self.options,
                &mut working_carry,
            ),
        }
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
        self.carry = working_carry;

        Ok(Frame::Video(decoded_frame_to_video_frame(
            &decoded, pkt.pts,
        )))
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> CoreResult<()> {
        // RFC 9043 §3.8.1.3 / §3.8.2.5: per-context coder state carries
        // across non-keyframes. A seek invalidates that carry — the next
        // packet must be a keyframe and re-initialise state. Drop the
        // carry, any pending packet, and clear EOF.
        self.carry = None;
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

/// Convert a [`DecodedFrame`] (per-plane `i32` Samples) into an
/// `oxideav-core` [`VideoFrame`].
///
/// Each plane is packed into a tight row-major byte buffer: one byte per
/// Sample for `bits_per_raw_sample <= 8`, two little-endian bytes per
/// Sample otherwise (the LE packing every `*Le` `PixelFormat` in the
/// framework uses). Planes are emitted in `DecodedFrame::planes` order
/// (plane-major: luma / R, then chroma / G,B, then the optional extra /
/// alpha plane) so a consumer reading the chosen
/// [`pixel_format_for`] label finds them where it expects.
fn decoded_frame_to_video_frame(decoded: &DecodedFrame, pts: Option<i64>) -> VideoFrame {
    let wide = decoded.bits_per_raw_sample > 8;
    let planes = decoded
        .planes
        .iter()
        .map(|p| {
            let w = p.width as usize;
            if wide {
                let mut data = Vec::with_capacity(p.samples.len() * 2);
                for &s in &p.samples {
                    let v = s as u16;
                    data.extend_from_slice(&v.to_le_bytes());
                }
                VideoPlane {
                    stride: w * 2,
                    data,
                }
            } else {
                let data = p.samples.iter().map(|&s| s as u8).collect();
                VideoPlane { stride: w, data }
            }
        })
        .collect();
    VideoFrame { pts, planes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{ProbeContext, TimeBase};

    #[test]
    fn register_via_runtime_context_installs_decoder() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let codec_id = CodecId::new(CODEC_ID_STR);
        assert!(
            ctx.codecs.has_decoder(&codec_id),
            "codec registration should install a decoder factory"
        );
    }

    #[test]
    fn register_claims_avi_fourcc_and_matroska_tags() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);

        // RFC 9043 §4.3.3.1 — AVI / VfW FourCC.
        let fourcc = CodecTag::fourcc(b"FFV1");
        assert_eq!(
            reg.resolve_tag_ref(&ProbeContext::new(&fourcc))
                .map(|c| c.as_str()),
            Some(CODEC_ID_STR),
            "FourCC FFV1 must resolve to ffv1 (RFC 9043 §4.3.3.1)"
        );

        // RFC 9043 §4.3.3.4 — Matroska Codec ID.
        let mkv = CodecTag::matroska("V_FFV1");
        assert_eq!(
            reg.resolve_tag_ref(&ProbeContext::new(&mkv))
                .map(|c| c.as_str()),
            Some(CODEC_ID_STR),
            "Matroska V_FFV1 must resolve to ffv1 (RFC 9043 §4.3.3.4)"
        );
    }

    #[test]
    fn decoder_without_extradata_is_unconfigured() {
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut dec = make_decoder(&params).expect("factory builds an unconfigured decoder");
        let pkt = Packet::new(0, TimeBase::new(1, 1), vec![0u8; 8]);
        dec.send_packet(&pkt).unwrap();
        let err = dec.receive_frame().expect_err("unconfigured decode errors");
        assert!(matches!(err, CoreError::InvalidData(_)));
    }
}
