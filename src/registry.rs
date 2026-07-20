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
//! [`crate::DecodedFrame`], and emits the stream's first Frame as a
//! keyframe and every later Frame as a §4.4 non-keyframe whose §3.8.1.3 /
//! §3.8.2.5 per-context coder state continues from the previous Frame —
//! unless the §4.2.17 `intra` flag forces keyframe-only output. The
//! framework `Decoder` carries the matching state across `receive_frame`
//! calls, so a multi-Frame inter stream round-trips through the trait
//! surface.
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

use crate::config::{
    ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version, PictureStructure, NUM_TRANSITION_DELTAS,
};
use crate::crc::validate_configuration_record_crc;
use crate::frame::{
    decode_frame_with_carry, DecodeOptions, DecodedFrame, DecodedFramePlane, Ffv1FrameCarry,
};
use crate::frame_encode::{encode_frame_with_carry, Ffv1EncodeCarry};
use crate::frame_v0v1::{encode_frame_v0v1_inter_with_carry, encode_frame_v0v1_with_carry};
use crate::predictor::NUM_QUANT_SUBTABLES;
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
/// exact framework `PixelFormat` does not exist (e.g. an 8-bit or 16-bit
/// planar RGB layout, a deep 4:2:0-plus-alpha YUV, an off-grid depth
/// like 9 / 11 / 13 / 14-bit YCbCr, or any reserved subsample shift), so
/// a caller leaves `CodecParameters::pixel_format` unset instead of
/// advertising a format whose plane order or storage width would mislead
/// a downstream muxer or filter. Layouts whose *storage surface* exists
/// at a deeper named depth (those off-grid depths, and 8-bit planar RGB
/// on the 16-bit-word `Gbrp*` surface) are mapped by the side-channel-
/// aware [`pixel_format_mapping_for`] instead. The §4.2.5
/// constraint that `colorspace_type == 1` always carries
/// `chroma_planes == 1 && log2_h == 0 && log2_v == 0` (full-resolution
/// 4:4:4 RGB) means the RGB path never subsamples; the framework's planar
/// RGB formats are the `Gbrp*` / `Gbrap*` family (G, B, R (, A) order,
/// 2-byte little-endian Samples), so 10 / 12 / 14-bit RGB map to those
/// (the registry's plane converters reorder the decoder's R, G, B (, A)
/// Planes into that order via [`gbr_plane_order`]) while 8-bit and 16-bit
/// planar RGB — which have no `Gbrp` variant — stay `None`.
pub fn pixel_format_for(cr: &Ffv1ConfigurationRecord) -> Option<PixelFormat> {
    let bits = cr.bits_per_raw_sample;

    // RGB / JPEG 2000 RCT (§4.2.5 fixes RGB at 4:4:4): the decoder emits
    // three (R, G, B) or four (R, G, B, A) full-resolution planar colour
    // Planes. The framework's planar-RGB formats are the `Gbrp*` /
    // `Gbrap*` family, whose plane order is G, B, R (, A) and whose
    // Samples are 2-byte little-endian — so the registry's plane
    // converters reorder the decoder's R, G, B (, A) Planes into that
    // G, B, R (, A) order (see [`gbr_plane_order`]). Only the depths the
    // enum names exactly map (10 / 12 / 14): 8-bit and 16-bit planar RGB
    // have no `Gbrp` variant (the framework's 8/16-bit RGB formats are
    // packed, not planar), so those stay honestly unmapped.
    if cr.colorspace_type == ColorspaceType::Rgb {
        return match (bits, cr.extra_plane) {
            (10, false) => Some(PixelFormat::Gbrp10Le),
            (12, false) => Some(PixelFormat::Gbrp12Le),
            (14, false) => Some(PixelFormat::Gbrp14Le),
            (10, true) => Some(PixelFormat::Gbrap10Le),
            (12, true) => Some(PixelFormat::Gbrap12Le),
            (14, true) => Some(PixelFormat::Gbrap14Le),
            _ => None,
        };
    }

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
        // Planar YUV-with-alpha: the 8-bit trio (`Yuva420P` / `Yuva422P` /
        // `Yuva444P`) plus the deep 4:2:2 / 4:4:4 family at 10 / 12 / 16
        // bits — 4 planes ordered Y, U, V, A with the alpha Plane at full
        // resolution, exactly the plane layout the §4.2.10 extra Plane
        // decodes to. Deep 4:2:0 + alpha has no framework variant.
        return match (bits, h, v) {
            (8, 1, 1) => Some(PixelFormat::Yuva420P),
            (8, 1, 0) => Some(PixelFormat::Yuva422P),
            (8, 0, 0) => Some(PixelFormat::Yuva444P),
            (10, 1, 0) => Some(PixelFormat::Yuva422P10Le),
            (12, 1, 0) => Some(PixelFormat::Yuva422P12Le),
            (16, 1, 0) => Some(PixelFormat::Yuva422P16Le),
            (10, 0, 0) => Some(PixelFormat::Yuva444P10Le),
            (12, 0, 0) => Some(PixelFormat::Yuva444P12Le),
            (16, 0, 0) => Some(PixelFormat::Yuva444P16Le),
            _ => None,
        };
    }

    match (h, v) {
        // 4:4:4 — no chroma subsampling.
        (0, 0) => match bits {
            8 => Some(PixelFormat::Yuv444P),
            10 => Some(PixelFormat::Yuv444P10Le),
            12 => Some(PixelFormat::Yuv444P12Le),
            16 => Some(PixelFormat::Yuv444P16Le),
            _ => None,
        },
        // 4:2:2 — horizontal /2.
        (1, 0) => match bits {
            8 => Some(PixelFormat::Yuv422P),
            10 => Some(PixelFormat::Yuv422P10Le),
            12 => Some(PixelFormat::Yuv422P12Le),
            16 => Some(PixelFormat::Yuv422P16Le),
            _ => None,
        },
        // 4:2:0 — horizontal /2, vertical /2.
        (1, 1) => match bits {
            8 => Some(PixelFormat::Yuv420P),
            10 => Some(PixelFormat::Yuv420P10Le),
            12 => Some(PixelFormat::Yuv420P12Le),
            16 => Some(PixelFormat::Yuv420P16Le),
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

/// A framework pixel-format mapping for one §4.2 Parameters layout: the
/// storage surface plus the per-plane significant-bits record (RFC 9043
/// §4.2.7 `bits_per_raw_sample`, uniform across Planes) that
/// [`oxideav_core::VideoFrame::set_significant_bits`] carries when the
/// surface's named depth is deeper than the coded depth.
///
/// Produced by [`pixel_format_mapping_for`]; consumed by the registry
/// `Decoder` (which attaches `significant_bits` to every emitted frame)
/// and `Encoder` (which derives the input plane word size from
/// `format`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ffv1PixelFormatMapping {
    /// The storage surface: plane order, chroma geometry and word size
    /// are exactly this `PixelFormat`'s.
    pub format: PixelFormat,
    /// `Some(record)` when the §4.2.7 coded depth is below `format`'s
    /// named depth — one byte per image Plane (all equal to
    /// `bits_per_raw_sample`), in `format` plane order, ready for
    /// [`oxideav_core::VideoFrame::set_significant_bits`]. `None` when
    /// `format` names the coded depth exactly (attach nothing — the
    /// format's own documented depth already tells the truth).
    pub significant_bits: Option<Vec<u8>>,
}

/// Derive the framework storage mapping for this Configuration Record,
/// side-channel-aware: like [`pixel_format_for`], but additionally
/// mapping the §4.2 layouts whose exact depth has no named framework
/// variant onto the smallest named surface that holds it, paired with
/// the per-plane significant-bits record that
/// [`oxideav_core::VideoFrame`] carries for exactly this purpose
/// (LSB-anchored values in the low bits of the surface word — the same
/// convention the decoder's little-endian plane packing already
/// produces).
///
/// Beyond the exact matches (`significant_bits == None`, identical to
/// [`pixel_format_for`]), this maps:
///
/// * **off-grid YCbCr depths** — 9 / 11 / 13 / 14 / 15-bit gray, YUV
///   4:2:0 / 4:2:2 / 4:4:4, and (4:2:2 / 4:4:4 only) YUV + alpha, onto
///   the 10 / 12 / 16-bit surface family (9 → `*10Le`, 11 → `*12Le`,
///   13..15 → `*16Le`), e.g. 14-bit 4:4:4 → `Yuv444P16Le` +
///   `[14, 14, 14]`;
/// * **sub-8-bit YCbCr depths** (1..=7) onto the 8-bit byte surfaces
///   (`Gray8`, `Yuv*P`, `Yuva*P`) with the depth recorded, matching the
///   decoder's one-byte-per-Sample packing at `bits <= 8`;
/// * **8 / 9 / 11 / 13-bit planar RGB / RCT** (± the §4.2.10 alpha
///   Plane) onto the `Gbrp*Le` / `Gbrap*Le` 16-bit-word surfaces
///   (8, 9 → `*10Le`, 11 → `*12Le`, 13 → `*14Le`) — closing the 8-bit
///   planar-RGB gap: the Samples ride the low bits of each 2-byte word
///   and the record says how many are significant.
///
/// Still unmapped (`None`): 15 / 16-bit planar RGB (no 16-bit `Gbrp`
/// surface exists in the framework enum), deep 4:2:0-plus-alpha YUV,
/// planar gray + alpha, 4:1:1 above 8 bits, and reserved subsample
/// shifts — no surface exists whose plane geometry and word size are
/// faithful, so the honest answer stays "no format".
pub fn pixel_format_mapping_for(cr: &Ffv1ConfigurationRecord) -> Option<Ffv1PixelFormatMapping> {
    // Exact variant first: no side-channel needed.
    if let Some(format) = pixel_format_for(cr) {
        return Some(Ffv1PixelFormatMapping {
            format,
            significant_bits: None,
        });
    }

    let bits = cr.bits_per_raw_sample;
    // Build the record for `planes` image Planes at the coded depth.
    let record = |format: PixelFormat, planes: usize| {
        Some(Ffv1PixelFormatMapping {
            format,
            significant_bits: Some(vec![bits as u8; planes]),
        })
    };

    if cr.colorspace_type == ColorspaceType::Rgb {
        // §4.2.5 fixes RGB at full-resolution 4:4:4; the `Gbrp*` /
        // `Gbrap*` surfaces store 2-byte LE words, so any depth up to
        // the surface's named depth rides the low bits. 15/16-bit have
        // no 16-bit `Gbrp` surface — honestly unmapped.
        let (no_alpha, alpha) = match bits {
            8 | 9 => (PixelFormat::Gbrp10Le, PixelFormat::Gbrap10Le),
            11 => (PixelFormat::Gbrp12Le, PixelFormat::Gbrap12Le),
            13 => (PixelFormat::Gbrp14Le, PixelFormat::Gbrap14Le),
            _ => return None,
        };
        return if cr.extra_plane {
            record(alpha, 4)
        } else {
            record(no_alpha, 3)
        };
    }

    // YCbCr: pick the smallest named surface holding `bits` for the
    // record's chroma / alpha geometry, mirroring `pixel_format_for`'s
    // (h, v, extra_plane) dispatch. `surface_bits` is the depth family
    // to look the surface up at.
    let surface_bits = match bits {
        1..=7 => 8,
        9 => 10,
        11 => 12,
        13..=15 => 16,
        // 8 / 10 / 12 / 16 were exact matches (or unmapped geometry).
        _ => return None,
    };
    let mut probe = cr.clone();
    probe.bits_per_raw_sample = surface_bits;
    let format = pixel_format_for(&probe)?;
    let planes = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    record(format, planes)
}

/// The exact inverse of [`pixel_format_for`] over its mapped range: build
/// the FFV1 **version 1** §4.2 Parameters (`Ffv1ConfigurationRecord`) that
/// describe `pf`, or `None` when `pf` is not a format [`pixel_format_for`]
/// produces (packed formats, depths with no FFV1 framework mapping, …).
///
/// Used by the framework `Encoder` to synthesise the inline §4.4
/// Parameters of a versions-0/1 stream from `CodecParameters` that carry
/// no §4.2 Configuration Record (RFC 9043 §4.3.3: v0/v1 streams have
/// none — their Parameters ride inline in each keyframe Frame). Version 1
/// is chosen over version 0 because it carries the explicit
/// `bits_per_raw_sample` field (§4.4), letting every mapped depth
/// round-trip; the entropy coder is the §3.8.1 range coder with the
/// default state-transition table (`coder_type == 1`), which carries any
/// mapped depth unrestricted (§4.2.3 confines only Golomb-Rice to
/// `bits <= 8`). The v3-only fields (`quant_table_set_count`, `ec`,
/// `intra`, `initial_state_delta`, `micro_version`) are absent, exactly
/// as [`crate::parse_v0v1_frame_parameters`] leaves them.
///
/// For every `Some(cr)` this returns, `pixel_format_for(&cr) == Some(pf)`
/// holds (covered by a round-trip unit test), so the plane order the
/// registry's converters apply on encode and decode is consistent by
/// construction — including the R, G, B (, A) ⇄ G, B, R (, A) reorder on
/// the planar `Gbr*` formats.
fn record_for_pixel_format(pf: PixelFormat) -> Option<Ffv1ConfigurationRecord> {
    // (colorspace, bits, chroma_planes, log2_h, log2_v, extra_plane)
    let (cs, bits, chroma, h, v, extra) = match pf {
        PixelFormat::Gray8 => (ColorspaceType::YCbCr, 8, false, 0, 0, false),
        PixelFormat::Gray10Le => (ColorspaceType::YCbCr, 10, false, 0, 0, false),
        PixelFormat::Gray12Le => (ColorspaceType::YCbCr, 12, false, 0, 0, false),
        PixelFormat::Gray16Le => (ColorspaceType::YCbCr, 16, false, 0, 0, false),
        PixelFormat::Yuv444P => (ColorspaceType::YCbCr, 8, true, 0, 0, false),
        PixelFormat::Yuv444P10Le => (ColorspaceType::YCbCr, 10, true, 0, 0, false),
        PixelFormat::Yuv444P12Le => (ColorspaceType::YCbCr, 12, true, 0, 0, false),
        PixelFormat::Yuv422P => (ColorspaceType::YCbCr, 8, true, 1, 0, false),
        PixelFormat::Yuv422P10Le => (ColorspaceType::YCbCr, 10, true, 1, 0, false),
        PixelFormat::Yuv422P12Le => (ColorspaceType::YCbCr, 12, true, 1, 0, false),
        PixelFormat::Yuv420P => (ColorspaceType::YCbCr, 8, true, 1, 1, false),
        PixelFormat::Yuv420P10Le => (ColorspaceType::YCbCr, 10, true, 1, 1, false),
        PixelFormat::Yuv420P12Le => (ColorspaceType::YCbCr, 12, true, 1, 1, false),
        PixelFormat::Yuv444P16Le => (ColorspaceType::YCbCr, 16, true, 0, 0, false),
        PixelFormat::Yuv422P16Le => (ColorspaceType::YCbCr, 16, true, 1, 0, false),
        PixelFormat::Yuv420P16Le => (ColorspaceType::YCbCr, 16, true, 1, 1, false),
        PixelFormat::Yuv411P => (ColorspaceType::YCbCr, 8, true, 2, 0, false),
        PixelFormat::Yuva420P => (ColorspaceType::YCbCr, 8, true, 1, 1, true),
        PixelFormat::Yuva422P => (ColorspaceType::YCbCr, 8, true, 1, 0, true),
        PixelFormat::Yuva444P => (ColorspaceType::YCbCr, 8, true, 0, 0, true),
        PixelFormat::Yuva422P10Le => (ColorspaceType::YCbCr, 10, true, 1, 0, true),
        PixelFormat::Yuva422P12Le => (ColorspaceType::YCbCr, 12, true, 1, 0, true),
        PixelFormat::Yuva422P16Le => (ColorspaceType::YCbCr, 16, true, 1, 0, true),
        PixelFormat::Yuva444P10Le => (ColorspaceType::YCbCr, 10, true, 0, 0, true),
        PixelFormat::Yuva444P12Le => (ColorspaceType::YCbCr, 12, true, 0, 0, true),
        PixelFormat::Yuva444P16Le => (ColorspaceType::YCbCr, 16, true, 0, 0, true),
        PixelFormat::Gbrp10Le => (ColorspaceType::Rgb, 10, true, 0, 0, false),
        PixelFormat::Gbrp12Le => (ColorspaceType::Rgb, 12, true, 0, 0, false),
        PixelFormat::Gbrp14Le => (ColorspaceType::Rgb, 14, true, 0, 0, false),
        PixelFormat::Gbrap10Le => (ColorspaceType::Rgb, 10, true, 0, 0, true),
        PixelFormat::Gbrap12Le => (ColorspaceType::Rgb, 12, true, 0, 0, true),
        PixelFormat::Gbrap14Le => (ColorspaceType::Rgb, 14, true, 0, 0, true),
        _ => return None,
    };
    Some(Ffv1ConfigurationRecord {
        version: Ffv1Version::V1,
        micro_version: None,
        coder_type: 1,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: cs,
        bits_per_raw_sample: bits,
        chroma_planes: chroma,
        log2_h_chroma_subsample: h,
        log2_v_chroma_subsample: v,
        extra_plane: extra,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: None,
        ec: None,
        intra: None,
        initial_state_delta: None,
    })
}

/// Build the default §4.1 Quantization Table Set the framework `Encoder`
/// installs for a synthesised v0/v1 stream (which must carry its single
/// Set inline in each keyframe, RFC 9043 §4.4).
///
/// The construction follows the RFC 9043 §4.1 decoder fill exactly, so
/// the Set is wire-serializable by definition: each sub-table's first
/// half (`k = 0..127`) is a sequence of runs, the `v`-th run holding
/// `len[v]` consecutive copies of `scale * v`; the second half is the
/// §4.1 sign-flipped reflection (`table[256 - k] = -table[k]`,
/// `table[128] = -table[127]`); and the scale chain multiplies
/// `2 * len_count - 1` per sub-table with
/// `context_count = ceil(scale / 2)` (§4.1.2).
///
/// The *choice* of run lengths is encoder freedom (the RFC specifies the
/// schema, not the values). This Set quantizes the three §3.5 Figure 5
/// primary neighbour differences (`Q0[l-tl]`, `Q1[tl-t]`, `Q2[t-tr]`)
/// into 11 symmetric levels each — power-of-two magnitude buckets
/// `{0}, 1..=2, 3..=6, 7..=14, 15..=30, 31..=127` and their negative
/// reflections — and leaves the two second-order differences (`Q3[L-l]`,
/// `Q4[T-t]`) flat (one level). Scale chain: `11 × 11 × 11 = 1331` →
/// `context_count == 666`.
fn default_quantization_table_set() -> QuantizationTableSet {
    // First-half run lengths. Active sub-tables (Q0..Q2): 6 runs of
    // scale·{0,1,2,3,4,5} spanning power-of-two buckets; flat sub-tables
    // (Q3, Q4): one 128-long run of 0.
    const ACTIVE_LENS: [u32; 6] = [1, 2, 4, 8, 16, 97];
    const FLAT_LENS: [u32; 1] = [128];

    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    let mut scale: i64 = 1;
    for (i, table) in tables.iter_mut().enumerate() {
        let lens: &[u32] = if i < 3 { &ACTIVE_LENS } else { &FLAT_LENS };
        // §4.1 first-half fill: len[v] consecutive copies of scale * v.
        let mut k = 0usize;
        for (v, &len) in lens.iter().enumerate() {
            for _ in 0..len {
                if k >= 128 {
                    break;
                }
                table[k] = (scale * v as i64) as i32;
                k += 1;
            }
        }
        debug_assert_eq!(k, 128, "run lengths must cover the first half");
        // §4.1 second-half sign-flipped reflection.
        for k in 1..128 {
            table[256 - k] = -table[k];
        }
        table[128] = -table[127];
        // §4.1 scale chain: scale *= 2 * len_count - 1.
        scale *= 2 * lens.len() as i64 - 1;
    }
    QuantizationTableSet {
        tables,
        // §4.1.2: context_count = ceil(scale / 2).
        context_count: ((scale as u64).div_ceil(2)) as u32,
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
    // RFC 9043 §4.3.3 / §4.4: versions 0 and 1 carry NO Configuration
    // Record — their §4.2 Parameters are inline in each keyframe Frame.
    // A v0/v1 container therefore supplies no extradata, only the frame
    // dimensions. Detect that shape (empty extradata + dims present) and
    // build a v0/v1-mode decoder that parses the §4.4 prologue off the
    // first keyframe packet.
    let v0v1_dims = if setup.is_none() && params.extradata.is_empty() {
        match (params.width, params.height) {
            (Some(w), Some(h)) => FramePixelDimensions::new(w, h).ok(),
            _ => None,
        }
    } else {
        None
    };
    Ok(Box::new(Ffv1FrameDecoder {
        codec_id: params.codec_id.clone(),
        setup,
        v0v1_dims,
        v0v1_config: None,
        options: DecodeOptions::default(),
        carry: None,
        ec_resolved: false,
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
    /// Frame dimensions for a v0/v1 stream (no Configuration Record).
    /// `Some` when the container supplied dims but no extradata (the
    /// versions-0/1 carriage shape); the v0/v1 §4.4 prologue is then
    /// parsed off the first keyframe packet.
    v0v1_dims: Option<FramePixelDimensions>,
    /// The §4.2 Parameters + single §4.1 Quantization Table Set parsed
    /// from the first v0/v1 keyframe (RFC 9043 §4.4). Cached so later
    /// non-keyframes (which carry no inline Parameters) can reuse it.
    v0v1_config: Option<(Ffv1ConfigurationRecord, QuantizationTableSet)>,
    options: DecodeOptions,
    /// §3.8.1.3 / §3.8.2.5 per-context coder state carried across
    /// non-keyframes (the cross-Frame channel the single-Frame drivers
    /// cannot hold). `None` before the first Frame.
    carry: Option<Ffv1FrameCarry>,
    /// `true` once a Frame has decoded successfully under the current
    /// `setup.ec` hypothesis, locking it in for the rest of the stream.
    ///
    /// r416 black-box finding (module doc of
    /// `tests/reference_inter_decode.rs`): the current reference
    /// encoder's Configuration Record TAIL does not read back under the
    /// RFC 9043 Figure 28 layout its own parser accepts for records this
    /// crate writes, so the record-declared §4.2.16 `ec` is unreliable
    /// on such streams — a `-slicecrc 0` stream can misdeclare
    /// `ec != 0`, sending a §4.9-faithful decoder hunting for Slice
    /// Footers that are not there. Until the first Frame succeeds, a
    /// decode failure under the record-derived hypothesis triggers ONE
    /// retry with the opposite §4.9 footer shape; whichever hypothesis
    /// first yields a fully-validated Frame (§4.9.1 trailer chain, §4.9
    /// footer size cross-check, §4.9.3 CRC residue, and §5 raster
    /// coverage all gate it) is locked. Streams whose record parses
    /// truthfully decode on the first attempt and never retry.
    ec_resolved: bool,
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

        // RFC 9043 §4.4: versions 0 / 1 carry their §4.2 Parameters inline
        // in each keyframe Frame (no Configuration Record). When the
        // container configured us with dims but no extradata, route to the
        // v0/v1 driver.
        if let Some(dims) = self.v0v1_dims {
            return self.receive_v0v1_frame(&pkt, dims);
        }

        let setup = self.setup.as_ref().ok_or_else(|| {
            CoreError::invalid(
                "oxideav-ffv1: stream not configured (CodecParameters needs the §4.2 \
                 Configuration Record in extradata plus width / height)",
            )
        })?;

        // Route on §4.2.5 colorspace_type, mirroring Ffv1DecodeSession.
        // Decode into a working copy of the carry so a decode error
        // leaves the decoder's carry untouched.
        let decode_with_ec =
            |ec: bool, working_carry: &mut Option<Ffv1FrameCarry>| match setup.cr.colorspace_type {
                ColorspaceType::YCbCr => decode_frame_with_carry(
                    &pkt.data,
                    &setup.cr,
                    &setup.quant_table_sets,
                    setup.frame_dims,
                    ec,
                    self.options,
                    working_carry,
                ),
                ColorspaceType::Rgb => decode_frame_rgb_with_carry(
                    &pkt.data,
                    &setup.cr,
                    &setup.quant_table_sets,
                    setup.frame_dims,
                    ec,
                    self.options,
                    working_carry,
                ),
            };

        let mut working_carry = self.carry.clone();
        let mut resolved_ec = setup.ec;
        let decoded = match decode_with_ec(setup.ec, &mut working_carry) {
            Ok(frame) => Ok(frame),
            // Until the first Frame has decoded, the record-derived
            // §4.2.16 `ec` is a HYPOTHESIS (the current reference
            // encoder's record tail can misdeclare it — see the
            // `ec_resolved` field doc). Retry the packet once with the
            // opposite §4.9 footer shape; every §4.9 / §4.9.3 / §5 gate
            // still applies, so the retry only sticks when the stream
            // genuinely validates under it.
            Err(first_err) if !self.ec_resolved => {
                working_carry = self.carry.clone();
                match decode_with_ec(!setup.ec, &mut working_carry) {
                    Ok(frame) => {
                        resolved_ec = !setup.ec;
                        Ok(frame)
                    }
                    // Surface the ORIGINAL error: the record-derived
                    // hypothesis is the declared one, so its failure is
                    // the meaningful diagnostic.
                    Err(_) => Err(first_err),
                }
            }
            Err(e) => Err(e),
        }
        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
        self.carry = working_carry;
        self.ec_resolved = true;
        let mapping = pixel_format_mapping_for(&setup.cr);
        if let Some(s) = self.setup.as_mut() {
            s.ec = resolved_ec;
        }
        Ok(Frame::Video(decoded_frame_to_video_frame(
            &decoded,
            pkt.pts,
            mapping.as_ref(),
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
        // carry, any pending packet, and clear EOF. The v0/v1 inline
        // config cache is also invalidated: after a seek the next packet
        // must be a keyframe (which re-supplies the inline Parameters).
        self.carry = None;
        self.v0v1_config = None;
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

impl Ffv1FrameDecoder {
    /// Decode one v0/v1 Frame (RFC 9043 §4.4). A keyframe carries the
    /// inline §4.2 Parameters + §4.1 cascade (parsed via
    /// [`crate::decode_frame_v0v1`], which also caches the record + single
    /// Quantization Table Set for later non-keyframes); a non-keyframe
    /// reuses the cached config via [`crate::decode_frame_v0v1_inter`].
    fn receive_v0v1_frame(
        &mut self,
        pkt: &Packet,
        dims: FramePixelDimensions,
    ) -> CoreResult<Frame> {
        // RFC 9043 §4.4: the Frame opens with the `keyframe` boolean. We
        // peek the parsed prologue to decide whether this Frame carries
        // its own inline Parameters (keyframe) or reuses the cached config
        // (non-keyframe). `parse_v0v1_frame_prologue` reads the keyframe
        // bit + (when set) the inline Parameters; a non-keyframe surfaces
        // `NonKeyframeHasNoInFrameParameters`.
        match crate::parse_v0v1_frame_prologue(&pkt.data) {
            Ok(prologue) => {
                // Keyframe: cache the record + Quantization Table Set, then
                // decode through the keyframe entry.
                self.v0v1_config =
                    Some((prologue.record.clone(), prologue.quant_table_set.clone()));
                // Decode into a working copy of the §3.8.1.3 / §3.8.2.5
                // carry so a decode error leaves the decoder's carry
                // untouched — mirroring the v3 route.
                let mut working_carry = self.carry.clone();
                let decoded =
                    crate::decode_frame_v0v1_with_carry(&pkt.data, dims, &mut working_carry)
                        .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
                self.carry = working_carry;
                let mapping = pixel_format_mapping_for(&prologue.record);
                Ok(Frame::Video(decoded_frame_to_video_frame(
                    &decoded,
                    pkt.pts,
                    mapping.as_ref(),
                )))
            }
            Err(crate::Error::NonKeyframeHasNoInFrameParameters) => {
                // Non-keyframe: it inherits the most recent keyframe's
                // inline Parameters + Quantization Table Set.
                let (cr, qts) = self.v0v1_config.as_ref().ok_or_else(|| {
                    CoreError::invalid(
                        "oxideav-ffv1: v0/v1 non-keyframe Frame arrived before any keyframe \
                         (no inline Parameters to inherit)",
                    )
                })?;
                let mapping = pixel_format_mapping_for(cr);
                // RFC 9043 §3.8.1.3 / §3.8.2.5: a v0/v1 non-keyframe
                // resumes the previous Frame's per-context coder state,
                // exactly as the v3 route carries it across packets.
                let mut working_carry = self.carry.clone();
                let decoded = crate::decode_frame_v0v1_inter_with_carry(
                    &pkt.data,
                    cr,
                    qts,
                    dims,
                    &mut working_carry,
                )
                .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
                self.carry = working_carry;
                Ok(Frame::Video(decoded_frame_to_video_frame(
                    &decoded,
                    pkt.pts,
                    mapping.as_ref(),
                )))
            }
            Err(e) => Err(CoreError::invalid(format!("oxideav-ffv1: {e}"))),
        }
    }
}

/// When `pf` is one of the framework's planar-RGB (`Gbrp*` / `Gbrap*`)
/// formats, return the plane permutation that maps the FFV1 RGB driver's
/// native `R, G, B (, A)` `DecodedFrame::planes` order onto the framework's
/// `G, B, R (, A)` order, expressed as *the source `DecodedFrame` plane
/// index for each output VideoFrame plane*:
///
/// | output plane | channel | source (`DecodedFrame`) plane |
/// |---|---|---|
/// | 0 | G | 1 |
/// | 1 | B | 2 |
/// | 2 | R | 0 |
/// | 3 | A | 3 |
///
/// (RFC 9043 §3.7 recovers R, G, B; oxideav-core's `Gbr` planar formats
/// store G, B, R.) Returns `None` for every non-`Gbr` format — YCbCr
/// (identity plane order) and the RGB depths `pixel_format_for` leaves
/// unmapped — so those paths emit / consume Planes in native
/// `DecodedFrame` order unchanged.
///
/// The permutation is an involution's near-inverse: the encode side reads
/// input planes through [`gbr_input_order`], the exact inverse of this.
fn gbr_plane_order(pf: Option<PixelFormat>) -> Option<[usize; 4]> {
    matches!(
        pf,
        Some(
            PixelFormat::Gbrp10Le
                | PixelFormat::Gbrp12Le
                | PixelFormat::Gbrp14Le
                | PixelFormat::Gbrap10Le
                | PixelFormat::Gbrap12Le
                | PixelFormat::Gbrap14Le
        )
    )
    .then_some([1, 2, 0, 3])
}

/// `true` when the mapped storage surface packs each Sample as a 2-byte
/// little-endian word (`*Le` formats), `false` for the one-byte 8-bit
/// planar surfaces.
///
/// Only meaningful for the formats [`pixel_format_mapping_for`]
/// produces — for those, every non-byte surface is a 16-bit-word `*Le`
/// planar format, so the byte-sized surfaces are the closed list below.
/// This is what decides the trait-boundary plane packing: an 8-bit
/// RGB / RCT stream mapped onto the `Gbrp10Le` surface packs (and
/// consumes) 2-byte words even though `bits_per_raw_sample == 8`.
fn surface_uses_two_byte_words(pf: PixelFormat) -> bool {
    !matches!(
        pf,
        PixelFormat::Gray8
            | PixelFormat::Yuv420P
            | PixelFormat::Yuv422P
            | PixelFormat::Yuv444P
            | PixelFormat::Yuv411P
            | PixelFormat::Yuva420P
            | PixelFormat::Yuva422P
            | PixelFormat::Yuva444P
    )
}

/// The exact inverse of [`gbr_plane_order`]: the source *input*
/// (`VideoFrame`) plane index for each output `DecodedFrame` plane, i.e.
/// how the encoder reads a framework `G, B, R (, A)` frame back into the
/// FFV1 driver's `R, G, B (, A)` plane order. `None` for non-`Gbr`
/// formats (native order).
///
/// | output (`DecodedFrame`) plane | channel | source (`VideoFrame`) plane |
/// |---|---|---|
/// | 0 | R | 2 |
/// | 1 | G | 0 |
/// | 2 | B | 1 |
/// | 3 | A | 3 |
fn gbr_input_order(pf: Option<PixelFormat>) -> Option<[usize; 4]> {
    gbr_plane_order(pf).map(|_| [2, 0, 1, 3])
}

/// Pack one `DecodedFrame` plane into a tight row-major
/// [`VideoPlane`]: one byte per Sample for `bits_per_raw_sample <= 8`,
/// two little-endian bytes per Sample otherwise (the LE packing every
/// `*Le` `PixelFormat` in the framework uses).
fn pack_plane(p: &DecodedFramePlane, wide: bool) -> VideoPlane {
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
}

/// Convert a [`DecodedFrame`] (per-plane `i32` Samples) into an
/// `oxideav-core` [`VideoFrame`].
///
/// Each plane is packed by [`pack_plane`]. Planes are emitted in the
/// order the frame's [`pixel_format_mapping_for`]-derived `mapping`
/// implies: native `DecodedFrame::planes` order (plane-major luma / R,
/// chroma / G,B, then the optional extra / alpha plane) for YCbCr and
/// unmapped RGB depths, or the `G, B, R (, A)` reorder for the
/// planar-RGB `Gbr` formats (see [`gbr_plane_order`]) — so a consumer
/// reading the mapped format label finds each channel's Plane where
/// that format says it is.
///
/// The per-Sample byte width follows the mapped storage surface
/// ([`surface_uses_two_byte_words`]): an 8-bit RGB stream mapped onto
/// the 16-bit-word `Gbrp10Le` surface packs 2-byte LE words. With no
/// mapping at all (`None`), the historical `bits > 8` rule applies.
/// When the mapping carries a significant-bits record (the coded depth
/// is below the surface's named depth), it is attached to the emitted
/// frame via [`VideoFrame::set_significant_bits`], so every consumer
/// sees the true §4.2.7 depth on the frame itself.
fn decoded_frame_to_video_frame(
    decoded: &DecodedFrame,
    pts: Option<i64>,
    mapping: Option<&Ffv1PixelFormatMapping>,
) -> VideoFrame {
    let wide = match mapping {
        Some(m) => surface_uses_two_byte_words(m.format),
        None => decoded.bits_per_raw_sample > 8,
    };
    let planes = if let Some(order) = gbr_plane_order(mapping.map(|m| m.format)) {
        (0..decoded.planes.len())
            .map(|out_idx| pack_plane(&decoded.planes[order[out_idx]], wide))
            .collect()
    } else {
        decoded.planes.iter().map(|p| pack_plane(p, wide)).collect()
    };
    let mut frame = VideoFrame { pts, planes };
    if let Some(bits) = mapping.and_then(|m| m.significant_bits.clone()) {
        frame.set_significant_bits(bits);
    }
    frame
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
/// has not yet supplied enough configuration. Returns `Err` when the
/// supplied pieces are present but inconsistent.
///
/// Two configuration shapes are accepted, mirroring the decoder side:
///
/// * **v3** — `params.extradata` carries the §4.2 Configuration Record
///   (RFC 9043 §4.3.3) plus width / height.
/// * **v0/v1** — empty `extradata` plus `params.pixel_format` and
///   width / height. Versions 0/1 have no Configuration Record (their
///   §4.2 Parameters ride inline in each keyframe Frame, §4.4), so the
///   encoder synthesises a version-1 record from the pixel format
///   ([`record_for_pixel_format`]) and installs the
///   [`default_quantization_table_set`] as the stream's single inline
///   §4.1 Set. A pixel format with no FFV1 mapping is a diagnosable
///   error rather than a silently unconfigured encoder.
fn build_encode_setup(params: &CodecParameters) -> CoreResult<Option<EncodeSetup>> {
    let (Some(width), Some(height)) = (params.width, params.height) else {
        return Ok(None);
    };

    if params.extradata.is_empty() {
        // RFC 9043 §4.4: versions 0/1 carry their Parameters inline. Build
        // the v0/v1 encode setup from the caller's pixel format.
        let Some(pf) = params.pixel_format else {
            return Ok(None);
        };
        let Some(cr) = record_for_pixel_format(pf) else {
            return Err(CoreError::invalid(format!(
                "oxideav-ffv1: pixel format {pf:?} has no FFV1 §4.2 Parameters \
                 mapping (supply a §4.2 Configuration Record in extradata to \
                 encode a v3 stream instead)",
            )));
        };
        let frame_dims = FramePixelDimensions::new(width, height)
            .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?;
        let slice_headers = derive_slice_grid(&cr);
        return Ok(Some(EncodeSetup {
            cr,
            quant_table_sets: vec![default_quantization_table_set()],
            frame_dims,
            // v0/v1 has no §4.9 Slice Footer, hence no per-Slice CRC.
            ec: false,
            slice_headers,
        }));
    }

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
    // Configuration Record when a faithful framework storage surface
    // exists (`pixel_format_mapping_for` — for off-grid depths this is
    // the storage surface; the true coded depth rides each frame's
    // significant-bits record); keep any caller-supplied value when the
    // §4.2 layout has no faithful surface (the helper returns `None`).
    let mut output_params = params.clone();
    if let Some(setup) = setup.as_ref() {
        if let Some(m) = pixel_format_mapping_for(&setup.cr) {
            output_params.pixel_format = Some(m.format);
        }
    }
    // §4.2.17 `intra` (Table 14): `intra == 1` forces keyframe-only
    // output. Absent / `Some(false)` admits inter Frames.
    let intra_only = setup.as_ref().is_some_and(|s| s.cr.intra == Some(true));
    Ok(Box::new(Ffv1FrameEncoder {
        output_params,
        setup,
        queue: std::collections::VecDeque::new(),
        carry: None,
        intra_only,
        first_frame: true,
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
    /// Encoded packets awaiting `receive_packet`. FFV1 emits one coded
    /// Frame per input Frame, no reordering — so this never holds more
    /// than one entry, but a queue keeps the send/receive contract
    /// uniform.
    queue: std::collections::VecDeque<Packet>,
    /// RFC 9043 §3.8.1.3 / §3.8.2.5 per-context coder state carried
    /// across non-keyframes — the write-side mirror of the decoder's
    /// `Ffv1FrameCarry`. `None` before the first Frame; the first
    /// `send_frame` emits a keyframe (re-initialising state) and seeds
    /// this, and every later Frame emits a non-keyframe whose per-context
    /// state continues from here, so the registry produces (and
    /// round-trips) a genuine multi-Frame inter stream — unless the §4.2.17
    /// `intra` flag forces keyframe-only output (see `intra_only`).
    carry: Option<Ffv1EncodeCarry>,
    /// `true` when the Configuration Record's §4.2.17 `intra` flag is set
    /// (Table 14: "keyframe MUST be 1 (keyframes only)"). Forces every
    /// coded Frame to a keyframe so the encoder never produces a stream
    /// the decoder's §4.2.17 intra gate would reject.
    intra_only: bool,
    /// Whether the next `send_frame` is the first Frame of the stream. The
    /// first Frame is always a keyframe (no previous Frame to carry from);
    /// `flush` of an end-of-stream does not reset this — a `reset` does.
    first_frame: bool,
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
/// Sample on the 8-bit byte surfaces, two little-endian bytes on the
/// `*Le` 16-bit-word surfaces — per the mapped storage surface, falling
/// back to the `bits_per_raw_sample <= 8` rule when the §4.2 layout has
/// no mapping) and laid out at the per-plane frame dimensions the
/// Configuration Record derives (§4.2.8 / §4.2.9 chroma subsample).
/// `keyframe` is always `true` — FFV1 is intra-only, so every coded
/// Frame this encoder builds is an independently decodable keyframe.
///
/// Side-channel entries at the tail of `planes` (palette,
/// significant bits — see [`VideoFrame`]) are not image Planes and are
/// skipped via [`VideoFrame::image_planes`]. An attached
/// significant-bits record must agree with the stream's §4.2.7
/// `bits_per_raw_sample` on every covered image Plane: FFV1 codes one
/// uniform depth, so a conflicting record would silently lose the
/// producer's depth metadata — that is a diagnosable error, not a
/// coercion.
fn video_frame_to_decoded_frame(v: &VideoFrame, setup: &EncodeSetup) -> CoreResult<DecodedFrame> {
    let cr = &setup.cr;
    let bits = cr.bits_per_raw_sample;
    let mapping = pixel_format_mapping_for(cr);
    let wide = match mapping.as_ref() {
        Some(m) => surface_uses_two_byte_words(m.format),
        None => bits > 8,
    };
    let bytes_per_sample = if wide { 2usize } else { 1usize };

    let primary_color_count = 1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane);
    let image_planes = v.image_planes();
    if image_planes.len() != primary_color_count {
        return Err(CoreError::invalid(format!(
            "oxideav-ffv1: frame has {} image planes but the Configuration Record \
             declares {primary_color_count} (chroma_planes={}, extra_plane={})",
            image_planes.len(),
            cr.chroma_planes,
            cr.extra_plane,
        )));
    }

    // §4.2.7: FFV1 codes one uniform depth. A frame-attached per-plane
    // significant-bits record that disagrees on any covered image Plane
    // cannot be represented in this stream.
    if let Some(rec) = v.significant_bits() {
        for (k, &b) in rec.iter().take(primary_color_count).enumerate() {
            if u32::from(b) != bits {
                return Err(CoreError::invalid(format!(
                    "oxideav-ffv1: frame declares {b} significant bits on plane {k} \
                     but the stream's §4.2.7 bits_per_raw_sample is {bits}",
                )));
            }
        }
    }

    // For the planar-RGB `Gbr` formats the framework hands us Planes in
    // G, B, R (, A) order; the FFV1 driver wants R, G, B (, A). Read each
    // output DecodedFrame plane from its source input plane through the
    // inverse permutation (identity for YCbCr and unmapped RGB depths, so
    // those consume Planes in native order). `src` is bounded by
    // `primary_color_count` (the plane-count check above), so the index is
    // always in range.
    let input_order = gbr_input_order(mapping.as_ref().map(|m| m.format));
    let mut planes = Vec::with_capacity(primary_color_count);
    for p_idx in 0..primary_color_count {
        let src = input_order.map_or(p_idx, |o| o[p_idx]);
        let plane = &image_planes[src];
        let (w, h) = encode_plane_dims(setup.frame_dims, p_idx as u8, cr);
        let want = (w as usize) * (h as usize) * bytes_per_sample;
        if plane.data.len() < want {
            return Err(CoreError::invalid(format!(
                "oxideav-ffv1: input plane {src} has {} bytes but {w}x{h} at \
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
                "oxideav-ffv1: input plane {src} stride {} is shorter than the \
                 {row_bytes}-byte row width",
                plane.stride,
            )));
        }
        let mut samples = Vec::with_capacity(w as usize * h as usize);
        for row in 0..h as usize {
            let base = row * plane.stride;
            if base + row_bytes > plane.data.len() {
                return Err(CoreError::invalid(format!(
                    "oxideav-ffv1: input plane {src} row {row} runs past the \
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

        // RFC 9043 §3.8.1.3 / §3.8.2.5: the first Frame is always a
        // keyframe (no previous Frame to carry from); subsequent Frames
        // are non-keyframes — unless the §4.2.17 `intra` flag forces
        // keyframe-only output.
        let keyframe = self.first_frame || self.intra_only;
        let payload = match setup.cr.version {
            // Versions 0/1 (empty-extradata setup): a keyframe carries the
            // inline §4.4 Parameters + single §4.1 Set; a non-keyframe
            // reuses them AND resumes the previous Frame's §3.8.1.3 /
            // §3.8.2.5 per-context coder state (RFC 9043 re-initialises
            // only on keyframes, on every version) — the same carry
            // discipline as the v3 route, over the implied single Slice.
            Ffv1Version::V0 | Ffv1Version::V1 => if keyframe {
                encode_frame_v0v1_with_carry(
                    &decoded,
                    &setup.cr,
                    &setup.quant_table_sets[0],
                    &mut self.carry,
                )
            } else {
                encode_frame_v0v1_inter_with_carry(
                    &decoded,
                    &setup.cr,
                    &setup.quant_table_sets[0],
                    &mut self.carry,
                )
            }
            .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?,
            // v3: `encode_frame_with_carry` dispatches on §4.2.5
            // `colorspace_type` + §4.2.3 `coder_type` to the matching
            // carry-aware driver and updates `self.carry` with this
            // Frame's end-of-Frame snapshot for the next non-keyframe.
            Ffv1Version::V3 => encode_frame_with_carry(
                &decoded,
                &setup.cr,
                &setup.quant_table_sets,
                &setup.slice_headers,
                setup.ec,
                keyframe,
                &mut self.carry,
            )
            .map_err(|e| CoreError::invalid(format!("oxideav-ffv1: {e}")))?,
        };
        self.first_frame = false;

        let mut pkt = Packet::new(0, TimeBase::new(1, 1), payload).with_keyframe(keyframe);
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
            (0, 0, 16, PixelFormat::Yuv444P16Le),
            (1, 0, 16, PixelFormat::Yuv422P16Le),
            (1, 1, 16, PixelFormat::Yuv420P16Le),
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
        // 14-bit YUV 4:4:4 has no exact framework variant (it maps onto
        // the 16-bit surface only through `pixel_format_mapping_for`'s
        // significant-bits side-channel).
        c.log2_h_chroma_subsample = 0;
        c.log2_v_chroma_subsample = 0;
        c.bits_per_raw_sample = 14;
        assert_eq!(pixel_format_for(&c), None, "14-bit yuv444");
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
    fn pixel_format_extra_plane_yuva_family() {
        let mut c = cr();
        c.chroma_planes = true;
        c.extra_plane = true;
        // (bits, h, v) -> the Yuva variant the §4.2.10 extra Plane maps to.
        let cases = [
            (8u32, 1u32, 1u32, PixelFormat::Yuva420P),
            (8, 1, 0, PixelFormat::Yuva422P),
            (8, 0, 0, PixelFormat::Yuva444P),
            (10, 1, 0, PixelFormat::Yuva422P10Le),
            (12, 1, 0, PixelFormat::Yuva422P12Le),
            (16, 1, 0, PixelFormat::Yuva422P16Le),
            (10, 0, 0, PixelFormat::Yuva444P10Le),
            (12, 0, 0, PixelFormat::Yuva444P12Le),
            (16, 0, 0, PixelFormat::Yuva444P16Le),
        ];
        for (bits, h, v, want) in cases {
            c.bits_per_raw_sample = bits;
            c.log2_h_chroma_subsample = h;
            c.log2_v_chroma_subsample = v;
            assert_eq!(
                pixel_format_for(&c),
                Some(want),
                "{bits}-bit h={h} v={v} + alpha"
            );
        }
        // Deep 4:2:0 + alpha has no framework variant.
        c.log2_h_chroma_subsample = 1;
        c.log2_v_chroma_subsample = 1;
        for bits in [10u32, 12, 16] {
            c.bits_per_raw_sample = bits;
            assert_eq!(
                pixel_format_for(&c),
                None,
                "{bits}-bit yuva420 unrepresented"
            );
        }
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
    fn pixel_format_rgb_maps_to_planar_gbr_at_named_depths() {
        // RGB / RCT (§4.2.5 fixes RGB at 4:4:4) decodes to R, G, B (, A)
        // planar Planes; the registry reorders them into the framework's
        // planar `Gbrp*` / `Gbrap*` G, B, R (, A) order. Only 10 / 12 /
        // 14-bit have a planar framework variant.
        let mut c = cr();
        c.colorspace_type = ColorspaceType::Rgb;
        c.chroma_planes = true;
        c.log2_h_chroma_subsample = 0;
        c.log2_v_chroma_subsample = 0;
        // (bits, no-alpha variant, alpha variant)
        let cases = [
            (10u32, PixelFormat::Gbrp10Le, PixelFormat::Gbrap10Le),
            (12, PixelFormat::Gbrp12Le, PixelFormat::Gbrap12Le),
            (14, PixelFormat::Gbrp14Le, PixelFormat::Gbrap14Le),
        ];
        for (bits, rgb, rgba) in cases {
            c.bits_per_raw_sample = bits;
            c.extra_plane = false;
            assert_eq!(pixel_format_for(&c), Some(rgb), "{bits}-bit rgb");
            c.extra_plane = true;
            assert_eq!(pixel_format_for(&c), Some(rgba), "{bits}-bit rgba");
        }
    }

    #[test]
    fn pixel_format_rgb_unnamed_depths_stay_none() {
        // 8-bit and 16-bit planar RGB have no `Gbrp` variant (the
        // framework's 8/16-bit RGB formats are packed, not planar);
        // odd depths likewise. Honest None.
        let mut c = cr();
        c.colorspace_type = ColorspaceType::Rgb;
        c.chroma_planes = true;
        c.log2_h_chroma_subsample = 0;
        c.log2_v_chroma_subsample = 0;
        for bits in [8u32, 9, 11, 13, 15, 16] {
            c.bits_per_raw_sample = bits;
            c.extra_plane = false;
            assert_eq!(pixel_format_for(&c), None, "{bits}-bit rgb");
            c.extra_plane = true;
            assert_eq!(pixel_format_for(&c), None, "{bits}-bit rgba");
        }
    }

    #[test]
    fn mapping_exact_formats_carry_no_side_channel() {
        // Wherever `pixel_format_for` names the depth exactly, the
        // side-channel-aware mapping must agree and attach nothing.
        let mut c = cr();
        for (bits, h, v, extra) in [
            (8u32, 1u32, 1u32, false),
            (10, 1, 0, false),
            (12, 0, 0, false),
            (16, 0, 0, false),
            (8, 1, 1, true),
            (16, 0, 0, true),
        ] {
            c.bits_per_raw_sample = bits;
            c.log2_h_chroma_subsample = h;
            c.log2_v_chroma_subsample = v;
            c.extra_plane = extra;
            let want = pixel_format_for(&c).expect("exact variant exists");
            let mapping = pixel_format_mapping_for(&c).expect("mapping exists");
            assert_eq!(mapping.format, want, "bits={bits} h={h} v={v}");
            assert_eq!(
                mapping.significant_bits, None,
                "exact match must not attach a record"
            );
        }
    }

    #[test]
    fn mapping_off_grid_ycbcr_depths_ride_deeper_surfaces() {
        let mut c = cr();
        // (bits, h, v, extra, chroma, surface) — the smallest named
        // surface holding the depth; the record repeats the coded depth
        // once per Plane.
        let cases = [
            (9u32, 1u32, 1u32, false, true, PixelFormat::Yuv420P10Le),
            (11, 1, 0, false, true, PixelFormat::Yuv422P12Le),
            (13, 0, 0, false, true, PixelFormat::Yuv444P16Le),
            (14, 0, 0, false, true, PixelFormat::Yuv444P16Le),
            (15, 1, 1, false, true, PixelFormat::Yuv420P16Le),
            (14, 0, 0, false, false, PixelFormat::Gray16Le),
            (9, 0, 0, false, false, PixelFormat::Gray10Le),
            (13, 1, 0, true, true, PixelFormat::Yuva422P16Le),
            (14, 0, 0, true, true, PixelFormat::Yuva444P16Le),
            (9, 0, 0, true, true, PixelFormat::Yuva444P10Le),
            // Sub-8-bit rides the 8-bit byte surfaces.
            (6, 1, 1, false, true, PixelFormat::Yuv420P),
            (7, 0, 0, false, false, PixelFormat::Gray8),
        ];
        for (bits, h, v, extra, chroma, surface) in cases {
            c.bits_per_raw_sample = bits;
            c.log2_h_chroma_subsample = h;
            c.log2_v_chroma_subsample = v;
            c.extra_plane = extra;
            c.chroma_planes = chroma;
            assert_eq!(
                pixel_format_for(&c),
                None,
                "{bits}-bit must not claim an exact variant"
            );
            let mapping = pixel_format_mapping_for(&c)
                .unwrap_or_else(|| panic!("{bits}-bit h={h} v={v} extra={extra} maps"));
            assert_eq!(mapping.format, surface, "bits={bits} h={h} v={v}");
            let planes = 1 + usize::from(chroma) * 2 + usize::from(extra);
            assert_eq!(
                mapping.significant_bits,
                Some(vec![bits as u8; planes]),
                "bits={bits}: record repeats the coded depth per Plane"
            );
        }
    }

    #[test]
    fn mapping_rgb_low_depths_ride_gbr_surfaces() {
        let mut c = cr();
        c.colorspace_type = ColorspaceType::Rgb;
        c.chroma_planes = true;
        c.log2_h_chroma_subsample = 0;
        c.log2_v_chroma_subsample = 0;
        let cases = [
            (8u32, PixelFormat::Gbrp10Le, PixelFormat::Gbrap10Le),
            (9, PixelFormat::Gbrp10Le, PixelFormat::Gbrap10Le),
            (11, PixelFormat::Gbrp12Le, PixelFormat::Gbrap12Le),
            (13, PixelFormat::Gbrp14Le, PixelFormat::Gbrap14Le),
        ];
        for (bits, no_alpha, alpha) in cases {
            c.bits_per_raw_sample = bits;
            c.extra_plane = false;
            let m = pixel_format_mapping_for(&c).expect("rgb maps");
            assert_eq!(m.format, no_alpha, "{bits}-bit rgb surface");
            assert_eq!(m.significant_bits, Some(vec![bits as u8; 3]));
            c.extra_plane = true;
            let m = pixel_format_mapping_for(&c).expect("rgba maps");
            assert_eq!(m.format, alpha, "{bits}-bit rgba surface");
            assert_eq!(m.significant_bits, Some(vec![bits as u8; 4]));
        }
        // 15 / 16-bit planar RGB: no 16-bit `Gbrp` surface exists.
        for bits in [15u32, 16] {
            c.bits_per_raw_sample = bits;
            c.extra_plane = false;
            assert_eq!(pixel_format_mapping_for(&c), None, "{bits}-bit rgb");
        }
    }

    #[test]
    fn mapping_unmapped_geometries_stay_none() {
        let mut c = cr();
        // Deep 4:2:0 + alpha.
        c.extra_plane = true;
        c.bits_per_raw_sample = 10;
        c.log2_h_chroma_subsample = 1;
        c.log2_v_chroma_subsample = 1;
        assert_eq!(pixel_format_mapping_for(&c), None, "deep yuva420");
        // Planar gray + alpha.
        c.chroma_planes = false;
        c.bits_per_raw_sample = 8;
        assert_eq!(pixel_format_mapping_for(&c), None, "gray+alpha");
        // 4:1:1 above 8 bits.
        c.chroma_planes = true;
        c.extra_plane = false;
        c.bits_per_raw_sample = 10;
        c.log2_h_chroma_subsample = 2;
        c.log2_v_chroma_subsample = 0;
        assert_eq!(pixel_format_mapping_for(&c), None, "deep 4:1:1");
        // Reserved subsample shift.
        c.bits_per_raw_sample = 8;
        c.log2_h_chroma_subsample = 3;
        assert_eq!(pixel_format_mapping_for(&c), None, "reserved shift");
    }

    #[test]
    fn gbr_plane_orders_are_mutual_inverses() {
        // The decode (`gbr_plane_order`) and encode (`gbr_input_order`)
        // permutations must compose to the identity so a decode → encode
        // round trip through the framework recovers the original Planes.
        for pf in [
            Some(PixelFormat::Gbrp10Le),
            Some(PixelFormat::Gbrp12Le),
            Some(PixelFormat::Gbrp14Le),
            Some(PixelFormat::Gbrap10Le),
            Some(PixelFormat::Gbrap12Le),
            Some(PixelFormat::Gbrap14Le),
        ] {
            let out = gbr_plane_order(pf).expect("Gbr format reorders");
            let inp = gbr_input_order(pf).expect("Gbr format reorders");
            // out maps output(G,B,R,A) -> source(DecodedFrame R,G,B,A);
            // inp maps output(DecodedFrame R,G,B,A) -> source(G,B,R,A).
            // Composing inp∘out and out∘inp must both be the identity.
            for i in 0..4 {
                assert_eq!(inp[out[i]], i, "inp∘out identity at {i}");
                assert_eq!(out[inp[i]], i, "out∘inp identity at {i}");
            }
            // The concrete channel mapping: output plane 0 (G) reads
            // DecodedFrame plane 1 (the green Plane FFV1 recovers second).
            assert_eq!(out, [1, 2, 0, 3]);
            assert_eq!(inp, [2, 0, 1, 3]);
        }
    }

    #[test]
    fn record_for_pixel_format_inverts_pixel_format_for() {
        // For every format `record_for_pixel_format` maps, the §4.2
        // Parameters it builds must map straight back — the identity that
        // keeps the encode- and decode-side plane order consistent.
        let mapped = [
            PixelFormat::Gray8,
            PixelFormat::Gray10Le,
            PixelFormat::Gray12Le,
            PixelFormat::Gray16Le,
            PixelFormat::Yuv444P,
            PixelFormat::Yuv444P10Le,
            PixelFormat::Yuv444P12Le,
            PixelFormat::Yuv422P,
            PixelFormat::Yuv422P10Le,
            PixelFormat::Yuv422P12Le,
            PixelFormat::Yuv420P,
            PixelFormat::Yuv420P10Le,
            PixelFormat::Yuv420P12Le,
            PixelFormat::Yuv444P16Le,
            PixelFormat::Yuv422P16Le,
            PixelFormat::Yuv420P16Le,
            PixelFormat::Yuv411P,
            PixelFormat::Yuva420P,
            PixelFormat::Yuva422P,
            PixelFormat::Yuva444P,
            PixelFormat::Yuva422P10Le,
            PixelFormat::Yuva422P12Le,
            PixelFormat::Yuva422P16Le,
            PixelFormat::Yuva444P10Le,
            PixelFormat::Yuva444P12Le,
            PixelFormat::Yuva444P16Le,
            PixelFormat::Gbrp10Le,
            PixelFormat::Gbrp12Le,
            PixelFormat::Gbrp14Le,
            PixelFormat::Gbrap10Le,
            PixelFormat::Gbrap12Le,
            PixelFormat::Gbrap14Le,
        ];
        for pf in mapped {
            let rec = record_for_pixel_format(pf)
                .unwrap_or_else(|| panic!("{pf:?} must map to §4.2 Parameters"));
            assert_eq!(rec.version, Ffv1Version::V1, "{pf:?} builds a v1 record");
            assert_eq!(
                pixel_format_for(&rec),
                Some(pf),
                "{pf:?} round-trips through the §4.2 mapping"
            );
        }
        // Formats outside the mapped set stay None (packed / unmapped).
        for pf in [
            PixelFormat::Rgb24,
            PixelFormat::Rgba,
            PixelFormat::Bgr24,
            PixelFormat::YuvJ420P,
        ] {
            assert!(record_for_pixel_format(pf).is_none(), "{pf:?} unmapped");
        }
    }

    #[test]
    fn default_quantization_table_set_is_section_4_1_well_formed() {
        let qts = default_quantization_table_set();
        // Scale chain 11 × 11 × 11 = 1331 → §4.1.2 ceil(1331 / 2) = 666.
        assert_eq!(qts.context_count, 666);
        for (i, table) in qts.tables.iter().enumerate() {
            // §4.1: the v = 0 run starts at k = 0, so table[0] == 0.
            assert_eq!(table[0], 0, "sub-table {i} first entry");
            // First half is non-decreasing (runs of scale * v, v growing).
            for k in 1..128 {
                assert!(
                    table[k] >= table[k - 1],
                    "sub-table {i} first half must be non-decreasing at {k}"
                );
            }
            // §4.1 second-half sign-flipped reflection.
            for k in 1..128 {
                assert_eq!(table[256 - k], -table[k], "sub-table {i} reflection at {k}");
            }
            assert_eq!(table[128], -table[127], "sub-table {i} midpoint");
        }
        // The two second-order sub-tables (Q3, Q4) are flat.
        for i in 3..NUM_QUANT_SUBTABLES {
            assert!(
                qts.tables[i].iter().all(|&v| v == 0),
                "sub-table {i} must be flat"
            );
        }
        // The three active sub-tables span 11 distinct levels: scale·{0..5}
        // in the first half plus the 5 negative reflections.
        let mut levels: Vec<i32> = qts.tables[0].to_vec();
        levels.sort_unstable();
        levels.dedup();
        assert_eq!(levels.len(), 11, "Q0 must quantize into 11 levels");
    }

    #[test]
    fn gbr_plane_orders_none_for_non_gbr_formats() {
        for pf in [
            None,
            Some(PixelFormat::Yuv420P),
            Some(PixelFormat::Gray16Le),
            Some(PixelFormat::Yuva420P),
            Some(PixelFormat::Rgb24),
        ] {
            assert!(gbr_plane_order(pf).is_none());
            assert!(gbr_input_order(pf).is_none());
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
