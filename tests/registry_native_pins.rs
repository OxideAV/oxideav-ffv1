//! Round 430 — whole-loop encoder byte-exactness pins for the six
//! newly-native deep / RGB formats (core 0.1.33: `Gbrp8`, `Gbrp16Le`,
//! `Gbrap16Le`, `Yuva420P10Le` / `Yuva420P12Le` / `Yuva420P16Le`),
//! driven end-to-end through the **registry trait surface** on the
//! v0/v1 empty-extradata route.
//!
//! These are streams the framework `Encoder` could not synthesise
//! before the remap: `record_for_pixel_format` now builds the inline
//! version-1 §4.2 Parameters for each of the six formats, so the whole
//! loop — pixel format → synthesised v1 record + §4.1-constructed
//! default Quantization Table Set → §4.4 keyframe + carried
//! non-keyframes → trait decode — is new observable surface. Each
//! packet's FNV-1a-64 hash is pinned (same drift-detection convention
//! as `tests/optimization_pins.rs`: FFV1 is deterministic, so the
//! encoder's bytes for a fixed input are part of the crate contract),
//! and every frame must decode back bit-exact (RFC 9043 §1,
//! `decode(encode(x)) == x`) with the native format advertised and no
//! significant-bits record attached.
//!
//! If an intentional wire-conformance change alters these bytes,
//! re-pin AFTER re-running the external black-box validation procedure
//! of `tests/external_conformance_notes.md` on the affected shapes.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{parse_v0v1_frame_prologue, register, ColorspaceType, CODEC_ID_STR};

// ---------------------------------------------------------------------
// FNV-1a-64 (same pin convention as `tests/optimization_pins.rs`)
// ---------------------------------------------------------------------

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

// ---------------------------------------------------------------------
// Deterministic source synthesis
// ---------------------------------------------------------------------

/// Deterministic per-plane test pattern spread over the FULL coded
/// depth (multiplicative hash, top `bits` bits) so each depth's
/// content — and therefore each pinned stream — is distinct, packed at
/// the surface word width (1 byte per Sample on the byte surfaces,
/// 2-byte LE words on the `*Le` surfaces).
fn plane(n: usize, w: u32, mul: u32, add: u32, bits: u32, wide: bool) -> VideoPlane {
    let samples: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * mul + add).wrapping_mul(0x9E37_79B1) >> (32 - bits)) as u16)
        .collect();
    if wide {
        let mut data = Vec::with_capacity(n * 2);
        for &s in &samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        VideoPlane {
            stride: w as usize * 2,
            data,
        }
    } else {
        VideoPlane {
            stride: w as usize,
            data: samples.iter().map(|&s| s as u8).collect(),
        }
    }
}

/// Build frame `f` for the format's geometry: distinct per-plane and
/// per-frame patterns so a swapped plane or a broken §3.8.1.3 carry is
/// caught by the round-trip, at frame-varying content so the inter
/// packets differ from the keyframe.
fn source_frame(
    f: u32,
    w: u32,
    h: u32,
    bits: u32,
    planes: usize,
    shifts: (u32, u32),
    wide: bool,
) -> VideoFrame {
    let (cw, ch) = (w.div_ceil(1 << shifts.0), h.div_ceil(1 << shifts.1));
    let full = (w * h) as usize;
    let sub = (cw * ch) as usize;
    const MULS: [u32; 4] = [31, 41, 59, 23];
    const ADDS: [u32; 4] = [2, 9, 4, 6];
    let mut fplanes = Vec::with_capacity(planes);
    for p in 0..planes {
        // Planes 1 / 2 of a subsampled YCbCr layout shrink; the planar
        // RGB layouts pass shifts (0, 0) so every plane is full-size.
        let (n, pw) = if (p == 1 || p == 2) && (shifts.0 > 0 || shifts.1 > 0) {
            (sub, cw)
        } else {
            (full, w)
        };
        fplanes.push(plane(n, pw, MULS[p], ADDS[p] + f * 13, bits, wide));
    }
    VideoFrame {
        pts: Some(f as i64),
        planes: fplanes,
    }
}

// ---------------------------------------------------------------------
// The pin matrix
// ---------------------------------------------------------------------

/// One pinned whole-loop stream. Dims are fixed at 9×7 (odd on both
/// axes, so the 4:2:0 layouts exercise the §4.2.8 / §4.2.9 ceiling
/// division and the RGB layouts an off-word-boundary raster). Pins
/// recorded at the r430 remap commit, immediately after the six
/// formats became reachable.
struct Pin {
    pf: PixelFormat,
    bits: u32,
    planes: usize,
    /// (log2_h, log2_v) chroma shifts; (0, 0) for the RGB layouts.
    shifts: (u32, u32),
    /// 2-byte LE words (`*Le` surfaces) vs one byte per Sample.
    wide: bool,
    rgb: bool,
    /// FNV-1a-64 of packets 0 / 1 / 2.
    hashes: [u64; 3],
}

const PINS: &[Pin] = &[
    Pin {
        pf: PixelFormat::Gbrp8,
        bits: 8,
        planes: 3,
        shifts: (0, 0),
        wide: false,
        rgb: true,
        hashes: [0x32d0f75c18e65276, 0x61e328bf0f9b14ed, 0xdbe1a915982eff1a],
    },
    Pin {
        pf: PixelFormat::Gbrp16Le,
        bits: 16,
        planes: 3,
        shifts: (0, 0),
        wide: true,
        rgb: true,
        hashes: [0x0c39fa0e79cf6051, 0xa204eda7a696900a, 0x61adb409d0266ab0],
    },
    Pin {
        pf: PixelFormat::Gbrap16Le,
        bits: 16,
        planes: 4,
        shifts: (0, 0),
        wide: true,
        rgb: true,
        hashes: [0x519ef739ff62c8a4, 0xbd5ce7aeb93909c4, 0x7c8f2c2eafb6f598],
    },
    Pin {
        pf: PixelFormat::Yuva420P10Le,
        bits: 10,
        planes: 4,
        shifts: (1, 1),
        wide: true,
        rgb: false,
        hashes: [0xca6e161e65c629ef, 0xd9ae0cf23dfad53a, 0x3bdd4f43e2e28162],
    },
    Pin {
        pf: PixelFormat::Yuva420P12Le,
        bits: 12,
        planes: 4,
        shifts: (1, 1),
        wide: true,
        rgb: false,
        hashes: [0x5190b10611567949, 0x1b0f2c744c819325, 0x9ab0c37c8c10c89d],
    },
    Pin {
        pf: PixelFormat::Yuva420P16Le,
        bits: 16,
        planes: 4,
        shifts: (1, 1),
        wide: true,
        rgb: false,
        hashes: [0x73bf2579ce1bfb61, 0xa9a5b7886575bcec, 0x7297f317ee8b6f70],
    },
];

const W: u32 = 9;
const H: u32 = 7;

#[test]
fn native_formats_v0v1_whole_loop_pins() {
    for &Pin {
        pf,
        bits,
        planes,
        shifts,
        wide,
        rgb,
        hashes: want_hashes,
    } in PINS
    {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(W);
        params.height = Some(H);
        params.pixel_format = Some(pf);

        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut enc = ctx
            .codecs
            .first_encoder(&params)
            .unwrap_or_else(|e| panic!("{pf:?}: v0/v1 encoder: {e:?}"));
        assert_eq!(
            enc.output_params().pixel_format,
            Some(pf),
            "{pf:?}: the native format is advertised (no surface detour)"
        );

        let mut dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        dparams.width = Some(W);
        dparams.height = Some(H);
        let mut dec = ctx
            .codecs
            .first_decoder(&dparams)
            .unwrap_or_else(|e| panic!("{pf:?}: v0/v1 decoder: {e:?}"));

        for f in 0..3u32 {
            let src = source_frame(f, W, H, bits, planes, shifts, wide);
            enc.send_frame(&Frame::Video(src.clone()))
                .unwrap_or_else(|e| panic!("{pf:?} frame {f}: encode: {e:?}"));
            let pkt = enc
                .receive_packet()
                .unwrap_or_else(|e| panic!("{pf:?} frame {f}: packet: {e:?}"));
            assert_eq!(pkt.flags.keyframe, f == 0, "{pf:?} frame {f}: §4.4 flag");

            // Byte-exactness pin.
            assert_eq!(
                fnv1a64(&pkt.data),
                want_hashes[f as usize],
                "{pf:?} frame {f}: emitted bytes drifted from the r430 pin \
                 (got 0x{:016x})",
                fnv1a64(&pkt.data),
            );

            // The keyframe's inline §4.4 Parameters restate the format.
            if f == 0 {
                let prologue = parse_v0v1_frame_prologue(&pkt.data)
                    .unwrap_or_else(|e| panic!("{pf:?}: keyframe prologue: {e:?}"));
                assert_eq!(prologue.record.bits_per_raw_sample, bits, "{pf:?}");
                assert_eq!(
                    prologue.record.colorspace_type,
                    if rgb {
                        ColorspaceType::Rgb
                    } else {
                        ColorspaceType::YCbCr
                    },
                    "{pf:?}: §4.2.5 colorspace"
                );
                assert_eq!(prologue.record.extra_plane, planes == 4, "{pf:?}: §4.2.10");
            }

            // Whole-loop lossless identity through the trait decoder.
            dec.send_packet(&pkt)
                .unwrap_or_else(|e| panic!("{pf:?} frame {f}: send: {e:?}"));
            let Frame::Video(out) = dec
                .receive_frame()
                .unwrap_or_else(|e| panic!("{pf:?} frame {f}: decode: {e:?}"))
            else {
                panic!("{pf:?} frame {f}: expected video");
            };
            assert_eq!(
                out.significant_bits(),
                None,
                "{pf:?} frame {f}: native format carries no record"
            );
            for (p, (got, want)) in out
                .image_planes()
                .iter()
                .zip(src.image_planes().iter())
                .enumerate()
            {
                assert_eq!(got.data, want.data, "{pf:?} frame {f} plane {p}");
            }
        }
    }
}
