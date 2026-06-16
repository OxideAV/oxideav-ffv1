//! `oxideav-core` framework integration: codec registration plus the
//! [`oxideav_core::Decoder`] and [`oxideav_core::Encoder`]
//! implementations wrapping the crate's frame-level drivers
//! ([`crate::decode_frame`] / [`crate::decode_frame_rgb`] on the read
//! side, [`crate::encode_frame`] on the write side).
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
//! The encoder is the symmetric inverse: it reads the same
//! `CodecParameters`, derives the §4.6 Slice Header grid from the
//! Configuration Record's §4.2.11 / §4.2.12 `num_h_slices ×
//! num_v_slices` (one Slice per raster cell), converts an incoming
//! [`oxideav_core::VideoFrame`] back to the internal
//! [`crate::DecodedFrame`], and emits one coded keyframe per Frame
//! (FFV1 is intra-only).
//!
//! Registration claims the two RFC 9043 §4.3.3 container tags: the AVI
//! / VfW FourCC `FFV1` (§4.3.3.1) and the Matroska Codec ID `V_FFV1`
//! (§4.3.3.4). The container crate's `CodecResolver` routes either
//! on-wire identifier to this codec.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Encoder, Error as CoreError, Frame, Packet, PixelFormat, Result as CoreResult, RuntimeContext,
    TimeBase, VideoFrame, VideoPlane,
};

use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, PictureStructure};
use crate::crc::validate_configuration_record_crc;
use crate::frame::{
    decode_frame_with_carry, DecodeOptions, DecodedFrame, DecodedFramePlane, Ffv1FrameCarry,
};
use crate::frame_encode::encode_frame;
use crate::quant_table::{parse_quantization_table_sets, QuantizationTableSet};
use crate::rgb_reconstruct::decode_frame_rgb_with_carry;
use crate::slice_content::FramePixelDimensions;
use crate::slice_header::{Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES};

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
        .with_encode()
        .with_lossless(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
            .tags([CodecTag::fourcc(b"FFV1"), CodecTag::matroska("V_FFV1")]),
    );
}

/// Unified entry point invoked by the macro-generated wrapper.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

/// Derive the framework [`PixelFormat`] a Frame this Configuration Record
/// describes packs into, or `None` when the §4.2 Parameters select a
/// layout the framework's `PixelFormat` enum has no exact, plane-order-
/// and-packing-faithful variant for.
///
/// The mapping is read straight off the RFC 9043 §4.2 Parameters that
/// fix the Frame's plane geometry:
///
/// * §4.2.5 `colorspace_type` (Table 8) — `0` = YCbCr (the planes the
///   decoder emits are Y, then Cb / Cr when §4.2.6 `chroma_planes`, then
///   the optional §4.2.10 `extra_plane` alpha); `1` = RGB / JPEG 2000
///   RCT, which the [`crate::decode_frame_rgb`] driver emits in **R, G,
///   B** plane order.
/// * §4.2.6 `chroma_planes` — whether Cb / Cr Planes follow luma.
/// * §4.2.7 `bits_per_raw_sample` — the decoder packs one byte per
///   Sample at `<= 8` bits, two little-endian bytes otherwise (Table 10:
///   `0` is read as `8`, already normalised on the parsed record).
/// * §4.2.8 / §4.2.9 `log2_*_chroma_subsample` — the chroma subsampling
///   that distinguishes 4:2:0 / 4:2:2 / 4:4:4 / 4:1:1.
/// * §4.2.10 `extra_plane` — the optional alpha / transparency Plane
///   (Table 8 "extra Plane content" column = Transparency for both
///   colorspaces).
///
/// `None` is returned — rather than a near-miss variant — whenever an
/// exact framework `PixelFormat` does not exist (e.g. an 8-bit planar
/// RGB / GBR layout, a 16-bit YUV, a subsampled-plus-alpha YUV, or any
/// reserved subsample shift), so a caller leaves `CodecParameters::
/// pixel_format` unset instead of advertising a format whose plane order
/// or storage width would mislead a downstream muxer or filter. The §4.2.5
/// constraint that `colorspace_type == 1` always carries
/// `chroma_planes == 1 && log2_h == 0 && log2_v == 0` (full-resolution
/// 4:4:4 GBR) means the RGB path never subsamples; the framework enum
/// has no planar **R, G, B**-order variant (its `Gbrp*Le` family is
/// G, B, R order), so RGB consistently returns `None` here.
pub fn pixel_format_for(cr: &Ffv1ConfigurationRecord) -> Option<PixelFormat> {
    // RGB / JPEG 2000 RCT: the decoder emits R, G, B plane order, which
    // no framework PixelFormat matches (Gbrp*Le is G, B, R). Honest None.
    if cr.colorspace_type == ColorspaceType::Rgb {
        return None;
    }

    let bits = cr.bits_per_raw_sample;

    // Luma-only (no chroma Planes, no extra Plane) → grayscale.
    if !cr.chroma_planes {
        if cr.extra_plane {
            // Gray + alpha exists only as packed 8-bit Ya8 (interleaved
            // Y, A) — not the planar two-Plane layout the decoder emits.
            return None;
        }
        return match bits {
            8 => Some(PixelFormat::Gray8),
            10 => Some(PixelFormat::Gray10Le),
            12 => Some(PixelFormat::Gray12Le),
            16 => Some(PixelFormat::Gray16Le),
            _ => None,
        };
    }

    let h = cr.log2_h_chroma_subsample;
    let v = cr.log2_v_chroma_subsample;

    // Y + Cb + Cr (+ optional alpha). Match the subsample shift pair to a
    // named chroma layout, then the bit depth + alpha to a framework
    // variant. Only the combinations the enum represents exactly map.
    if cr.extra_plane {
        // The only planar YUV-with-alpha variant in the framework enum is
        // 8-bit Yuva420P (Y, Cb, Cr, A at 4:2:0).
        return match (bits, h, v) {
            (8, 1, 1) => Some(PixelFormat::Yuva420P),
            _ => None,
        };
    }

    match (h, v) {
        // 4:4:4 — no chroma subsampling.
        (0, 0) => match bits {
            8 => Some(PixelFormat::Yuv444P),
            10 => Some(PixelFormat::Yuv444P10Le),
            12 => Some(PixelFormat::Yuv444P12Le),
            _ => None,
        },
        // 4:2:2 — horizontal /2.
        (1, 0) => match bits {
            8 => Some(PixelFormat::Yuv422P),
            10 => Some(PixelFormat::Yuv422P10Le),
            12 => Some(PixelFormat::Yuv422P12Le),
            _ => None,
        },
        // 4:2:0 — horizontal /2, vertical /2.
        (1, 1) => match bits {
            8 => Some(PixelFormat::Yuv420P),
            10 => Some(PixelFormat::Yuv420P10Le),
            12 => Some(PixelFormat::Yuv420P12Le),
            _ => None,
        },
        // 4:1:1 — horizontal /4 (8-bit only in the framework enum).
        (2, 0) => match bits {
            8 => Some(PixelFormat::Yuv411P),
            _ => None,
        },
        _ => None,
    }
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
/// alpha plane) so a consumer reading the [`pixel_format_for`]-derived
/// label off `CodecParameters::pixel_format` finds them where it expects.
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

// ──────────────────────── Encoder impl ────────────────────────

/// Per-stream encode setup parsed once from [`CodecParameters`].
///
/// Built from the same `params.extradata` (the §4.2 Configuration
/// Record per RFC 9043 §4.3.3) plus `params.width` / `params.height` the
/// decoder side consumes, so a transcode that copies `CodecParameters`
/// from a decoder's `output_params` straight into an encoder reproduces
/// the identical stream layout. The derived [`Self::slice_headers`]
/// tile the §4.2.11 `num_h_slices × num_v_slices` raster grid one cell
/// per Slice.
struct EncodeSetup {
    cr: Ffv1ConfigurationRecord,
    quant_table_sets: Vec<QuantizationTableSet>,
    frame_dims: FramePixelDimensions,
    /// §4.2.16 `ec` — `true` puts a §4.9.3 per-Slice CRC in each Slice
    /// (8-byte §4.9 Slice Footer). Same derivation the decoder uses.
    ec: bool,
    /// One §4.6 Slice Header per Slice, in forward Slice-index order,
    /// tiling the `num_h_slices × num_v_slices` raster grid one cell per
    /// Slice (the grid the Configuration Record declares — §4.2.11 /
    /// §4.2.12). FFV1's `encode_frame` does NOT synthesise a raster from
    /// the Configuration Record (§4.6 admits any tiling), so the wiring
    /// layer supplies the canonical one-cell-per-Slice decomposition.
    slice_headers: Vec<Ffv1SliceHeader>,
}

/// Per-plane frame dimensions for plane `plane_index` (RFC 9043 §4.2.8 /
/// §4.2.9): luma + the optional extra plane run at full frame
/// resolution, chroma planes shrink by the §4.2.8 / §4.2.9
/// `log2_*_chroma_subsample` shifts (ceiling division so odd dimensions
/// round up). The arithmetic mirrors the decoder's frame-plane layout.
fn encode_plane_dims(
    frame: FramePixelDimensions,
    plane_index: u8,
    cr: &Ffv1ConfigurationRecord,
) -> (u32, u32) {
    if cr.chroma_planes && (plane_index == 1 || plane_index == 2) {
        let hdenom = 1u32 << cr.log2_h_chroma_subsample;
        let vdenom = 1u32 << cr.log2_v_chroma_subsample;
        (
            frame.width.saturating_add(hdenom - 1) / hdenom,
            frame.height.saturating_add(vdenom - 1) / vdenom,
        )
    } else {
        (frame.width, frame.height)
    }
}

/// Build the canonical one-cell-per-Slice raster decomposition for the
/// Configuration Record's `num_h_slices × num_v_slices` grid (RFC 9043
/// §4.2.11 / §4.2.12, defaulting to a single 1×1 Slice when the v3-only
/// fields are absent). Each Slice covers one raster cell:
/// `slice_x` / `slice_y` are the cell's grid coordinates and
/// `slice_width` / `slice_height` are both `1` (one cell). Slices are
/// emitted in raster order (row-major) — the §4.9.1 trailer-chain order
/// the decode drivers walk.
fn derive_slice_grid(cr: &Ffv1ConfigurationRecord) -> Vec<Ffv1SliceHeader> {
    let num_h = cr.num_h_slices.unwrap_or(1).max(1);
    let num_v = cr.num_v_slices.unwrap_or(1).max(1);

    // §4.6.5 quant_table_set_index_count: one entry for luma, one shared
    // chroma entry when chroma planes (or version <= 3) are present, one
    // for the optional extra plane. Every Slice carries the same count.
    let chroma_or_v3 = cr.chroma_planes
        || matches!(
            cr.version,
            crate::config::Ffv1Version::V0 | crate::config::Ffv1Version::V1
        )
        || cr.version == crate::config::Ffv1Version::V3;
    let qts_index_count = 1 + usize::from(chroma_or_v3) + usize::from(cr.extra_plane);

    let mut headers = Vec::with_capacity((num_h as usize) * (num_v as usize));
    for slice_y in 0..num_v {
        for slice_x in 0..num_h {
            headers.push(Ffv1SliceHeader {
                slice_x,
                slice_y,
                slice_width: 1,
                slice_height: 1,
                quant_table_set_index_count: qts_index_count,
                // Slot 0 → luma's set, the shared chroma slot → set 0,
                // the extra-plane slot → set 0. A single-Quantization-
                // Table-Set stream (the common case) selects set 0 for
                // every slot; multi-set streams are addressed by the
                // direct `encode_frame*` API where the caller supplies
                // bespoke headers.
                quant_table_set_index: [0u32; MAX_QUANT_TABLE_SET_INDEXES],
                picture_structure: PictureStructure::Progressive,
                picture_structure_raw: 0,
                sar_num: 0,
                sar_den: 0,
            });
        }
    }
    headers
}

/// Assemble an [`EncodeSetup`] from `params`, or `None` if the caller
/// has not yet supplied extradata / dimensions. Returns `Err` when the
/// supplied pieces are present but inconsistent.
fn build_encode_setup(params: &CodecParameters) -> CoreResult<Option<EncodeSetup>> {
    if params.extradata.is_empty() {
        return Ok(None);
    }
    let (Some(width), Some(height)) = (params.width, params.height) else {
        return Ok(None);
    };

    validate_configuration_record_crc(&params.extradata)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
    let parsed = parse_quantization_table_sets(&params.extradata)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
    let frame_dims = FramePixelDimensions::new(width, height)
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
    let ec = !matches!(parsed.record.ec, None | Some(0));
    let slice_headers = derive_slice_grid(&parsed.record);
    Ok(Some(EncodeSetup {
        cr: parsed.record,
        quant_table_sets: parsed.quant_table_sets,
        frame_dims,
        ec,
        slice_headers,
    }))
}

fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    let setup = build_encode_setup(params)?;
    // Surface the §4.2-derived pixel format on the parameters a downstream
    // muxer reads back via `output_params`. Derive it from the parsed
    // Configuration Record when an exact framework variant exists; keep
    // any caller-supplied value when the §4.2 layout has no faithful
    // `PixelFormat` (the helper returns `None`).
    let mut output_params = params.clone();
    if let Some(setup) = setup.as_ref() {
        if let Some(pf) = pixel_format_for(&setup.cr) {
            output_params.pixel_format = Some(pf);
        }
    }
    Ok(Box::new(Ffv1FrameEncoder {
        output_params,
        setup,
        queue: std::collections::VecDeque::new(),
    }))
}

struct Ffv1FrameEncoder {
    /// The stream parameters a downstream muxer reads. Carries the §4.2
    /// Configuration Record in `extradata` (RFC 9043 §4.3.3) plus the
    /// frame dimensions, unchanged from the caller-supplied parameters.
    output_params: CodecParameters,
    /// Per-stream encode setup. `None` until extradata + dimensions are
    /// supplied; `send_frame` then surfaces a diagnosable error.
    setup: Option<EncodeSetup>,
    /// Encoded packets awaiting `receive_packet`. FFV1 is intra-only —
    /// one coded keyframe per input Frame, no reordering — so this never
    /// holds more than one entry, but a queue keeps the
    /// send/receive contract uniform.
    queue: std::collections::VecDeque<Packet>,
}

impl std::fmt::Debug for Ffv1FrameEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ffv1FrameEncoder")
            .field("codec_id", &self.output_params.codec_id)
            .field("configured", &self.setup.is_some())
            .field("queued", &self.queue.len())
            .finish()
    }
}

/// Reconstruct a [`DecodedFrame`] from an `oxideav-core` [`VideoFrame`].
///
/// The exact inverse of [`decoded_frame_to_video_frame`]: each plane's
/// row-major byte buffer is unpacked into `i32` Samples (one byte per
/// Sample at `bits_per_raw_sample <= 8`, two little-endian bytes
/// otherwise) and laid out at the per-plane frame dimensions the
/// Configuration Record derives (§4.2.8 / §4.2.9 chroma subsample).
/// `keyframe` is always `true` — FFV1 is intra-only, so every coded
/// Frame this encoder builds is an independently decodable keyframe.
fn video_frame_to_decoded_frame(v: &VideoFrame, setup: &EncodeSetup) -> CoreResult<DecodedFrame> {
    let cr = &setup.cr;
    let bits = cr.bits_per_raw_sample;
    let wide = bits > 8;
    let bytes_per_sample = if wide { 2usize } else { 1usize };

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    if v.planes.len() != primary_color_count {
        return Err(CoreError::invalid(format!(
            "oxideav-ffv1: frame has {} planes but the Configuration Record \
             declares {primary_color_count} (chroma_planes={}, extra_plane={})",
            v.planes.len(),
            cr.chroma_planes,
            cr.extra_plane,
        )));
    }

    let mut planes = Vec::with_capacity(primary_color_count);
    for (p_idx, plane) in v.planes.iter().enumerate() {
        let (w, h) = encode_plane_dims(setup.frame_dims, p_idx as u8, cr);
        let want = (w as usize) * (h as usize) * bytes_per_sample;
        if plane.data.len() < want {
            return Err(CoreError::invalid(format!(
                "oxideav-ffv1: plane {p_idx} has {} bytes but {w}x{h} at \
                 {bits}-bit needs {want}",
                plane.data.len(),
            )));
        }
        // Unpack row by row honouring the plane's stride (the row pitch
        // may exceed the tight `w * bytes_per_sample` when the producer
        // padded rows). Samples are read tightly per row.
        let row_bytes = w as usize * bytes_per_sample;
        if plane.stride < row_bytes {
            return Err(CoreError::invalid(format!(
                "oxideav-ffv1: plane {p_idx} stride {} is shorter than the \
                 {row_bytes}-byte row width",
                plane.stride,
            )));
        }
        let mut samples = Vec::with_capacity(w as usize * h as usize);
        for row in 0..h as usize {
            let base = row * plane.stride;
            if base + row_bytes > plane.data.len() {
                return Err(CoreError::invalid(format!(
                    "oxideav-ffv1: plane {p_idx} row {row} runs past the \
                     {}-byte buffer",
                    plane.data.len(),
                )));
            }
            if wide {
                for col in 0..w as usize {
                    let off = base + col * 2;
                    let s = u16::from_le_bytes([plane.data[off], plane.data[off + 1]]);
                    samples.push(s as i32);
                }
            } else {
                for col in 0..w as usize {
                    samples.push(plane.data[base + col] as i32);
                }
            }
        }
        planes.push(DecodedFramePlane {
            plane_index: p_idx as u8,
            width: w,
            height: h,
            samples,
        });
    }

    Ok(DecodedFrame {
        planes,
        width: setup.frame_dims.width,
        height: setup.frame_dims.height,
        bits_per_raw_sample: bits,
        colorspace: cr.colorspace_type,
        // Intra-only codec — every coded Frame is a keyframe.
        keyframe: true,
        // `encode_frame` derives Slice geometry from `slice_headers`; this
        // field is ignored on the encode side.
        slice_headers: Vec::new(),
    })
}

impl Encoder for Ffv1FrameEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> CoreResult<()> {
        let setup = self.setup.as_ref().ok_or_else(|| {
            CoreError::invalid(
                "oxideav-ffv1: encoder not configured (CodecParameters needs the \
                 §4.2 Configuration Record in extradata plus width / height)",
            )
        })?;
        let Frame::Video(v) = frame else {
            return Err(CoreError::invalid(
                "oxideav-ffv1: FFV1 encodes video frames only",
            ));
        };

        let decoded = video_frame_to_decoded_frame(v, setup)?;
        let payload = encode_frame(
            &decoded,
            &setup.cr,
            &setup.quant_table_sets,
            &setup.slice_headers,
            setup.ec,
        )
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;

        let mut pkt = Packet::new(0, TimeBase::new(1, 1), payload).with_keyframe(true);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        self.queue.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        self.queue.pop_front().ok_or(CoreError::NeedMore)
    }

    fn flush(&mut self) -> CoreResult<()> {
        // FFV1 is intra-only with no internal frame buffering: every
        // `send_frame` already produced its packet, so there is nothing
        // to drain. Subsequent `receive_packet` calls return whatever
        // remains queued, then `NeedMore`.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Ffv1Version;
    use oxideav_core::{ProbeContext, TimeBase};

    /// A minimal v3 YCbCr Configuration Record for the §4.2 pixel-format
    /// mapping tests. Callers override the geometry-relevant fields
    /// (`colorspace_type`, `chroma_planes`, the subsample shifts, the bit
    /// depth, `extra_plane`).
    fn cr() -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            version: Ffv1Version::V3,
            micro_version: Some(4),
            coder_type: 1,
            state_transition_delta: [0; crate::config::NUM_TRANSITION_DELTAS],
            colorspace_type: ColorspaceType::YCbCr,
            bits_per_raw_sample: 8,
            chroma_planes: true,
            log2_h_chroma_subsample: 1,
            log2_v_chroma_subsample: 1,
            extra_plane: false,
            num_h_slices: Some(1),
            num_v_slices: Some(1),
            quant_table_set_count: Some(1),
            ec: Some(0),
            intra: Some(false),
            initial_state_delta: None,
        }
    }

    #[test]
    fn pixel_format_grayscale_by_bit_depth() {
        let mut c = cr();
        c.chroma_planes = false;
        c.extra_plane = false;
        for (bits, want) in [
            (8u32, PixelFormat::Gray8),
            (10, PixelFormat::Gray10Le),
            (12, PixelFormat::Gray12Le),
            (16, PixelFormat::Gray16Le),
        ] {
            c.bits_per_raw_sample = bits;
            assert_eq!(pixel_format_for(&c), Some(want), "{bits}-bit gray");
        }
        // 14-bit gray has no framework variant.
        c.bits_per_raw_sample = 14;
        assert_eq!(pixel_format_for(&c), None);
    }

    #[test]
    fn pixel_format_yuv_by_subsample_and_depth() {
        let mut c = cr();
        c.chroma_planes = true;
        c.extra_plane = false;
        // (h, v, bits) -> Some(variant)
        let cases = [
            (0, 0, 8, PixelFormat::Yuv444P),
            (0, 0, 10, PixelFormat::Yuv444P10Le),
            (0, 0, 12, PixelFormat::Yuv444P12Le),
            (1, 0, 8, PixelFormat::Yuv422P),
            (1, 0, 10, PixelFormat::Yuv422P10Le),
            (1, 0, 12, PixelFormat::Yuv422P12Le),
            (1, 1, 8, PixelFormat::Yuv420P),
            (1, 1, 10, PixelFormat::Yuv420P10Le),
            (1, 1, 12, PixelFormat::Yuv420P12Le),
            (2, 0, 8, PixelFormat::Yuv411P),
        ];
        for (h, v, bits, want) in cases {
            c.log2_h_chroma_subsample = h;
            c.log2_v_chroma_subsample = v;
            c.bits_per_raw_sample = bits;
            assert_eq!(pixel_format_for(&c), Some(want), "h={h} v={v} bits={bits}");
        }
    }

    #[test]
    fn pixel_format_yuv_unrepresented_combos_are_none() {
        let mut c = cr();
        c.chroma_planes = true;
        c.extra_plane = false;
        // 16-bit YUV 4:4:4 (the v3-yuv444p16 corpus shape) has no exact
        // framework variant.
        c.log2_h_chroma_subsample = 0;
        c.log2_v_chroma_subsample = 0;
        c.bits_per_raw_sample = 16;
        assert_eq!(pixel_format_for(&c), None, "16-bit yuv444");
        // 4:1:1 above 8-bit, and a reserved subsample shift.
        c.bits_per_raw_sample = 10;
        c.log2_h_chroma_subsample = 2;
        c.log2_v_chroma_subsample = 0;
        assert_eq!(pixel_format_for(&c), None, "10-bit 4:1:1");
        c.bits_per_raw_sample = 8;
        c.log2_h_chroma_subsample = 3;
        c.log2_v_chroma_subsample = 0;
        assert_eq!(pixel_format_for(&c), None, "reserved subsample");
    }

    #[test]
    fn pixel_format_extra_plane_only_8bit_yuva420() {
        let mut c = cr();
        c.chroma_planes = true;
        c.extra_plane = true;
        c.bits_per_raw_sample = 8;
        c.log2_h_chroma_subsample = 1;
        c.log2_v_chroma_subsample = 1;
        assert_eq!(pixel_format_for(&c), Some(PixelFormat::Yuva420P));
        // Any other alpha combination has no exact variant.
        c.log2_v_chroma_subsample = 0; // 4:2:2 + alpha
        assert_eq!(pixel_format_for(&c), None, "yuva422 unrepresented");
        c.log2_v_chroma_subsample = 1;
        c.bits_per_raw_sample = 10; // 10-bit 4:2:0 + alpha
        assert_eq!(pixel_format_for(&c), None, "10-bit yuva420 unrepresented");
        // Gray + alpha (planar two-plane) has no exact variant either.
        c.chroma_planes = false;
        c.bits_per_raw_sample = 8;
        assert_eq!(
            pixel_format_for(&c),
            None,
            "planar gray+alpha unrepresented"
        );
    }

    #[test]
    fn pixel_format_rgb_is_none() {
        // RGB / RCT decodes to R, G, B plane order — no framework variant
        // matches (Gbrp*Le is G, B, R). §4.2.5 fixes RGB at 4:4:4.
        let mut c = cr();
        c.colorspace_type = ColorspaceType::Rgb;
        c.chroma_planes = true;
        c.log2_h_chroma_subsample = 0;
        c.log2_v_chroma_subsample = 0;
        for bits in [8u32, 10, 12, 14, 16] {
            c.bits_per_raw_sample = bits;
            assert_eq!(pixel_format_for(&c), None, "{bits}-bit rgb");
            c.extra_plane = true;
            assert_eq!(pixel_format_for(&c), None, "{bits}-bit rgba");
            c.extra_plane = false;
        }
    }

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
