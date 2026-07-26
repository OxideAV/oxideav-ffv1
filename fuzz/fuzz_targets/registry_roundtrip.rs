#![no_main]
//! `registry_roundtrip` fuzz target — encode → decode bit-exactness
//! **through the framework trait surface**.
//!
//! The `roundtrip` target sweeps the direct `encode_frame*` /
//! `decode_frame*` API. This one drives the registry-installed
//! [`oxideav_core::Encoder`] / [`oxideav_core::Decoder`] pair instead, so
//! everything the trait wiring adds on top of the codec core is under
//! the lossless-identity contract too:
//!
//!   * the §4.2 pixel-format mapping and its inverse (the encoder
//!     synthesises a version-1 record from `CodecParameters::pixel_format`
//!     when `extradata` is empty — RFC 9043 §4.4 inline Parameters);
//!   * the §4.1-constructed default Quantization Table Set's wire
//!     round-trip (encoded inline in the keyframe, re-parsed by the
//!     decoder);
//!   * the plane packing (one byte per Sample at ≤ 8-bit, two
//!     little-endian bytes otherwise) and its inverse;
//!   * the R, G, B (, A) ⇄ G, B, R (, A) plane reorder on the planar
//!     `Gbr*` formats;
//!   * the keyframe → non-keyframe Frame sequencing (two Frames are
//!     encoded, so the §4.4 non-keyframe path and — for v0/v1 — the
//!     cached-record reuse are reached).
//!
//! Layout of the attacker buffer:
//!   byte 0  — width  (1..=MAX_DIM)
//!   byte 1  — height (1..=MAX_DIM)
//!   byte 2  — pixel-format selector (modulo the mapped-format count)
//!   bytes 3.. — folded into the Plane-sample PRNG seed
//!
//! The contract under test: for every mapped pixel format, a well-formed
//! two-Frame stream encoded through the trait decodes back bit-exact
//! (`decode(encode(x)) == x`, RFC 9043 §1), with no panic anywhere in
//! the wiring. Construction errors bail; only a panic or a divergence is
//! a finding.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{register, CODEC_ID_STR};

/// Cap on each frame dimension (same rationale as the `roundtrip`
/// target: peak allocation stays finite, off-multiple shapes remain
/// reachable).
const MAX_DIM: u32 = 48;

/// Every `PixelFormat` the registry maps to §4.2 Parameters, with its
/// plane geometry: `(pf, bits, plane_count, log2_h, log2_v)`. Chroma
/// shifts apply to planes 1 and 2 of the subsampled YCbCr layouts only;
/// `Gbr*` and 4:4:4 planes are all full-resolution.
const MAPPED: &[(PixelFormat, u32, usize, u32, u32)] = &[
    (PixelFormat::Gray8, 8, 1, 0, 0),
    (PixelFormat::Gray10Le, 10, 1, 0, 0),
    (PixelFormat::Gray12Le, 12, 1, 0, 0),
    (PixelFormat::Gray16Le, 16, 1, 0, 0),
    (PixelFormat::Yuv444P, 8, 3, 0, 0),
    (PixelFormat::Yuv444P10Le, 10, 3, 0, 0),
    (PixelFormat::Yuv444P12Le, 12, 3, 0, 0),
    (PixelFormat::Yuv422P, 8, 3, 1, 0),
    (PixelFormat::Yuv422P10Le, 10, 3, 1, 0),
    (PixelFormat::Yuv422P12Le, 12, 3, 1, 0),
    (PixelFormat::Yuv420P, 8, 3, 1, 1),
    (PixelFormat::Yuv420P10Le, 10, 3, 1, 1),
    (PixelFormat::Yuv420P12Le, 12, 3, 1, 1),
    (PixelFormat::Yuv444P16Le, 16, 3, 0, 0),
    (PixelFormat::Yuv422P16Le, 16, 3, 1, 0),
    (PixelFormat::Yuv420P16Le, 16, 3, 1, 1),
    (PixelFormat::Yuv411P, 8, 3, 2, 0),
    (PixelFormat::Yuva420P, 8, 4, 1, 1),
    (PixelFormat::Yuva422P, 8, 4, 1, 0),
    (PixelFormat::Yuva444P, 8, 4, 0, 0),
    (PixelFormat::Yuva422P10Le, 10, 4, 1, 0),
    (PixelFormat::Yuva422P12Le, 12, 4, 1, 0),
    (PixelFormat::Yuva422P16Le, 16, 4, 1, 0),
    (PixelFormat::Yuva444P10Le, 10, 4, 0, 0),
    (PixelFormat::Yuva444P12Le, 12, 4, 0, 0),
    (PixelFormat::Yuva444P16Le, 16, 4, 0, 0),
    (PixelFormat::Gbrp10Le, 10, 3, 0, 0),
    (PixelFormat::Gbrp12Le, 12, 3, 0, 0),
    (PixelFormat::Gbrp14Le, 14, 3, 0, 0),
    (PixelFormat::Gbrap10Le, 10, 4, 0, 0),
    (PixelFormat::Gbrap12Le, 12, 4, 0, 0),
    (PixelFormat::Gbrap14Le, 14, 4, 0, 0),
    // Native deep 4:2:0 + alpha and the Gbrp depth-ladder ends
    // (core 0.1.33).
    (PixelFormat::Yuva420P10Le, 10, 4, 1, 1),
    (PixelFormat::Yuva420P12Le, 12, 4, 1, 1),
    (PixelFormat::Yuva420P16Le, 16, 4, 1, 1),
    (PixelFormat::Gbrp8, 8, 3, 0, 0),
    (PixelFormat::Gbrp16Le, 16, 3, 0, 0),
    (PixelFormat::Gbrap16Le, 16, 4, 0, 0),
];

/// Deterministic SplitMix64-style PRNG byte stream for one plane,
/// confined to `[0, 1 << bits)` per Sample and packed tight (1 byte per
/// Sample at ≤ 8-bit, 2 little-endian bytes otherwise) — the exact
/// contract the registry's plane converters expect.
fn synth_plane(seed: u64, w: u32, h: u32, bits: u32) -> VideoPlane {
    let mask: u64 = (1u64 << bits) - 1;
    let n = (w as usize) * (h as usize);
    let wide = bits > 8;
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut data = Vec::with_capacity(n * if wide { 2 } else { 1 });
    for _ in 0..n {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let v = (z & mask) as u16;
        if wide {
            data.extend_from_slice(&v.to_le_bytes());
        } else {
            data.push(v as u8);
        }
    }
    VideoPlane {
        stride: (w as usize) * if wide { 2 } else { 1 },
        data,
    }
}

/// Per-plane dimensions: planes 1 / 2 of a subsampled YCbCr layout
/// shrink by the ceiling-divided chroma shifts; every other plane runs
/// at frame resolution.
fn plane_dims(p: usize, w: u32, h: u32, shifts: (u32, u32), planes: usize) -> (u32, u32) {
    // Chroma shifts only ever apply to the 3-or-4-plane YCbCr layouts.
    if (p == 1 || p == 2) && planes >= 3 && (shifts.0 > 0 || shifts.1 > 0) {
        (w.div_ceil(1 << shifts.0), h.div_ceil(1 << shifts.1))
    } else {
        (w, h)
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let width = (u32::from(data[0]) % MAX_DIM) + 1;
    let height = (u32::from(data[1]) % MAX_DIM) + 1;
    let (pf, bits, planes, hs, vs) = MAPPED[usize::from(data[2]) % MAPPED.len()];

    let mut seed: u64 = 0x0BAD_5EED_CAFE_F00D;
    for &b in &data[3..] {
        seed = seed.rotate_left(7) ^ u64::from(b).wrapping_mul(0x100_0000_01B3);
    }
    seed ^= u64::from(width) << 40 ^ u64::from(height) << 24 ^ u64::from(data[2]) << 8;

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pf);

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let Ok(mut enc) = ctx.codecs.first_encoder(&params) else {
        // Every MAPPED entry must construct; a refusal is a finding.
        panic!("encoder construction failed for mapped format {pf:?}");
    };
    let Ok(mut dec) = ctx.codecs.first_decoder(&params) else {
        panic!("decoder construction failed for {pf:?}");
    };

    // Two Frames: the §4.4 keyframe (inline Parameters + Set) then a
    // non-keyframe (cached-record reuse on the decode side).
    for f in 0..2u64 {
        let frame = VideoFrame {
            pts: Some(f as i64),
            planes: (0..planes)
                .map(|p| {
                    let (pw, ph) = plane_dims(p, width, height, (hs, vs), planes);
                    synth_plane(seed ^ (f << 32) ^ (p as u64) << 16, pw, ph, bits)
                })
                .collect(),
        };

        enc.send_frame(&Frame::Video(frame.clone()))
            .expect("well-formed frame must encode through the trait");
        let pkt = enc
            .receive_packet()
            .expect("one packet per frame must be queued");
        assert_eq!(pkt.flags.keyframe, f == 0, "keyframe flag sequencing");

        dec.send_packet(&pkt).expect("decoder accepts own packet");
        let Ok(Frame::Video(out)) = dec.receive_frame() else {
            panic!("own stream must decode ({pf:?} frame {f})");
        };
        assert_eq!(out.planes.len(), frame.planes.len(), "plane count");
        for (p, (got, want)) in out.planes.iter().zip(frame.planes.iter()).enumerate() {
            assert_eq!(
                got.data, want.data,
                "{pf:?} frame {f} plane {p}: lossless round-trip violated"
            );
        }
    }
});
