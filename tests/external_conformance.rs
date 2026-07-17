//! Self-encoded external-conformance corpus (RFC 9043 encoder axis).
//!
//! Every stream in [`corpus`] is encoded by THIS crate's encoder from a
//! deterministic synthetic source, then:
//!
//! 1. **self round-trip** — decoded back with this crate's decoder and
//!    required to be bit-exact against the source Planes (RFC 9043 §1:
//!    FFV1 is lossless, `decode(encode(x)) == x`);
//! 2. **pinned** — each emitted packet's SHA-256 must equal the value
//!    recorded in [`STREAM_PINS`]. The pinned bytes are the exact
//!    streams that were validated against an external black-box
//!    reference decoder (see `tests/external_conformance_notes.md` for
//!    the wrap + validation procedure and results); any encoder change
//!    that alters emitted bytes must re-run that external validation
//!    before re-pinning.
//!
//! Setting `FFV1_CONFORMANCE_EXPORT_DIR` exports every stream
//! (packets, §4.3.3 extradata when version 3, ffmpeg-layout raw source
//! frames, and a `manifest.txt`) for the out-of-tree wrap-and-validate
//! step; the test behaves identically otherwise.

use std::io::Write as _;

use oxideav_ffv1::{
    decode_frame_rgb_with_carry, decode_frame_v0v1_inter_with_carry, decode_frame_v0v1_with_carry,
    decode_frame_with_carry, encode_configuration_record_with_quant_tables,
    encode_frame_v0v1_inter_with_carry, encode_frame_v0v1_with_carry, encode_frame_with_carry,
    ColorspaceType, DecodeOptions, DecodedFrame, DecodedFramePlane, Ffv1ConfigurationRecord,
    Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure, QuantizationTableSet,
    DEFAULT_ONE_STATE, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

// ────────────────────────── SHA-256 (FIPS 180-4) ──────────────────────────
//
// Inlined so the corpus pins carry no external dependency. Standard
// SHA-256 as published in FIPS 180-4.

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

// ────────────────────────── corpus definition ──────────────────────────

/// One matrix point of the encoder-conformance corpus.
struct StreamSpec {
    name: &'static str,
    /// The exact raw pixel layout `srcN.raw` is written in — a planar
    /// layout the external validator can be asked to emit losslessly.
    pix_fmt: &'static str,
    version: u32,
    coder: u32,
    colorspace: ColorspaceType,
    bits: u32,
    chroma_planes: bool,
    hss: u32,
    vss: u32,
    alpha: bool,
    slices: (u32, u32),
    ec: bool,
    /// §4.2.17 `intra` (v3 only): when set, every Frame is coded as a
    /// §4.4 keyframe.
    intra: bool,
    frames: u32,
    width: u32,
    height: u32,
}

/// The corpus matrix: version (0/1/3) × coder (Golomb-Rice / range
/// default / range custom, §4.2.3) × colorspace (§4.2.5 YCbCr / RGB) ×
/// depth (8/10/12/14/16) × subsampling × alpha × slice grid × ec.
fn corpus() -> Vec<StreamSpec> {
    use ColorspaceType::{Rgb, YCbCr};
    vec![
        // ── version 3, range coder (default table) ──
        StreamSpec {
            name: "v3-yuv420p8-range-2x2",
            pix_fmt: "yuv420p",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (2, 2),
            ec: true,
            intra: false,
            frames: 2,
            width: 128,
            height: 96,
        },
        StreamSpec {
            name: "v3-gray8-range",
            pix_fmt: "gray",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: false,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-gray16-range",
            pix_fmt: "gray16le",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 16,
            chroma_planes: false,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-yuv422p10-range",
            pix_fmt: "yuv422p10le",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 10,
            chroma_planes: true,
            hss: 1,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-yuv420p12-range",
            pix_fmt: "yuv420p12le",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 12,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-yuv444p16-range",
            pix_fmt: "yuv444p16le",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 16,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-yuva420p8-range",
            pix_fmt: "yuva420p",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: true,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-rgb8-range",
            pix_fmt: "gbrp",
            version: 3,
            coder: 1,
            colorspace: Rgb,
            bits: 8,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-rgba8-range",
            pix_fmt: "gbrap",
            version: 3,
            coder: 1,
            colorspace: Rgb,
            bits: 8,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: true,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-rgb10-range",
            pix_fmt: "gbrp10le",
            version: 3,
            coder: 1,
            colorspace: Rgb,
            bits: 10,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-rgb14-range",
            pix_fmt: "gbrp14le",
            version: 3,
            coder: 1,
            colorspace: Rgb,
            bits: 14,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        // ── version 3, Golomb-Rice (§3.8.2) ──
        StreamSpec {
            name: "v3-yuv420p8-golomb",
            pix_fmt: "yuv420p",
            version: 3,
            coder: 0,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-rgb8-golomb",
            pix_fmt: "gbrp",
            version: 3,
            coder: 0,
            colorspace: Rgb,
            bits: 8,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        // ── version 3, range coder with §3.8.1.6 custom table ──
        StreamSpec {
            name: "v3-yuv420p8-custom",
            pix_fmt: "yuv420p",
            version: 3,
            coder: 2,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        // ── version 3, structural edges ──
        StreamSpec {
            name: "v3-yuv444p8-multislice-3x2-odd",
            pix_fmt: "yuv444p",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (3, 2),
            ec: true,
            intra: false,
            frames: 2,
            width: 97,
            height: 61,
        },
        StreamSpec {
            name: "v3-gray8-range-ec0",
            pix_fmt: "gray",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: false,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        // ── versions 0 / 1 (inline §4.4 Parameters, single Slice) ──
        StreamSpec {
            name: "v0-yuv420p8-range",
            pix_fmt: "yuv420p",
            version: 0,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v0-yuv420p8-golomb",
            pix_fmt: "yuv420p",
            version: 0,
            coder: 0,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v0-yuv420p8-custom",
            pix_fmt: "yuv420p",
            version: 0,
            coder: 2,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v1-yuv420p8-range",
            pix_fmt: "yuv420p",
            version: 1,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v1-gray8-golomb",
            pix_fmt: "gray",
            version: 1,
            coder: 0,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: false,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v1-rgb8-range",
            pix_fmt: "gbrp",
            version: 1,
            coder: 1,
            colorspace: Rgb,
            bits: 8,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v0-rgb8-golomb",
            pix_fmt: "gbrp",
            version: 0,
            coder: 0,
            colorspace: Rgb,
            bits: 8,
            chroma_planes: true,
            hss: 0,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v1-yuv422p10-range",
            pix_fmt: "yuv422p10le",
            version: 1,
            coder: 1,
            colorspace: YCbCr,
            bits: 10,
            chroma_planes: true,
            hss: 1,
            vss: 0,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 2,
            width: 64,
            height: 48,
        },
        // ── deeper temporal chains + §4.2.17 intra ──
        StreamSpec {
            name: "v3-yuv420p8-range-2x2-4f",
            pix_fmt: "yuv420p",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (2, 2),
            ec: true,
            intra: false,
            frames: 4,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v3-yuv420p8-range-intra",
            pix_fmt: "yuv420p",
            version: 3,
            coder: 1,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: true,
            intra: true,
            frames: 2,
            width: 64,
            height: 48,
        },
        StreamSpec {
            name: "v0-yuv420p8-golomb-3f",
            pix_fmt: "yuv420p",
            version: 0,
            coder: 0,
            colorspace: YCbCr,
            bits: 8,
            chroma_planes: true,
            hss: 1,
            vss: 1,
            alpha: false,
            slices: (1, 1),
            ec: false,
            intra: false,
            frames: 3,
            width: 64,
            height: 48,
        },
    ]
}

// ────────────────────────── deterministic source ──────────────────────────

struct XorShift32(u32);

impl XorShift32 {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// Deterministic per-plane source content: a horizontal ramp band
/// (smooth gradients), a noise band (high entropy), a flat band (§3.8.2
/// run mode / low-entropy range paths), and a texture band.
fn synth_plane(w: u32, h: u32, depth: u32, plane: u32, frame_idx: u32) -> Vec<i32> {
    let mask: u32 = if depth >= 32 {
        u32::MAX
    } else {
        (1 << depth) - 1
    };
    let mut rng = XorShift32(
        0x9e37_79b9
            ^ plane.wrapping_mul(0x85eb_ca6b)
            ^ frame_idx.wrapping_mul(0xc2b2_ae35)
            ^ w.wrapping_mul(0x27d4_eb2f)
            ^ (h << 16)
            | 1,
    );
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let band = if h >= 4 { y * 4 / h } else { y % 4 };
        for x in 0..w {
            let v = match band {
                0 => (x * mask / w.max(1)).wrapping_add(frame_idx * 3),
                1 => rng.next(),
                2 => (mask / 2).wrapping_add(frame_idx),
                _ => (x ^ y)
                    .wrapping_mul(3)
                    .wrapping_add(plane * 5)
                    .wrapping_add(frame_idx * 7),
            } & mask;
            out.push(v as i32);
        }
    }
    out
}

fn ceil_shift(v: u32, s: u32) -> u32 {
    v.div_ceil(1 << s)
}

/// Build the `frame_idx`-th source frame for `spec` (plane geometry per
/// RFC 9043 §4.7.1: luma / R / gray full-size, chroma subsampled,
/// trailing alpha full-size).
fn synth_frame(spec: &StreamSpec, frame_idx: u32) -> DecodedFrame {
    let mut planes = Vec::new();
    let plane_count = 1 + if spec.chroma_planes { 2 } else { 0 } + if spec.alpha { 1 } else { 0 };
    for p in 0..plane_count {
        let is_chroma = spec.chroma_planes && (p == 1 || p == 2);
        let (w, h) = if is_chroma {
            (
                ceil_shift(spec.width, spec.hss),
                ceil_shift(spec.height, spec.vss),
            )
        } else {
            (spec.width, spec.height)
        };
        planes.push(DecodedFramePlane {
            plane_index: p as u8,
            width: w,
            height: h,
            samples: synth_plane(w, h, spec.bits, p as u32, frame_idx),
        });
    }
    DecodedFrame {
        planes,
        width: spec.width,
        height: spec.height,
        bits_per_raw_sample: spec.bits,
        colorspace: spec.colorspace,
        keyframe: frame_idx == 0,
        slice_headers: Vec::new(),
    }
}

// ────────────────────────── record / tables / headers ──────────────────────────

/// §4.2.4 custom state-transition deltas for the `coder_type == 2`
/// entries: every reachable default transition (`DEFAULT_ONE_STATE[i]
/// != 0`) is nudged by ±1, keeping the resulting `one_state` in
/// `1..=255` (values ≤ 128 move up, values > 128 move down), so the
/// custom table differs from the §3.8.1.5 default on every live state.
///
/// The zero entries of the default table (`one_state[1..=8]` and
/// `one_state[249..=255]`, unreachable from the §3.8.1.3 initial state
/// 128) are additionally lifted to the self-loop `i` so that EVERY
/// transmitted transition is nonzero. Both choices are encoder freedom
/// under RFC 9043 (those states are never visited by a valid stream),
/// but a fully-live table is the interoperable one: black-box probing
/// (r416) showed the external reference decoder accepts a v0/v1
/// `coder_type == 2` stream only when no transmitted `one_state` entry
/// is zero — it rejects any zero transition on the v0/v1 inline
/// Parameters path ("invalid state transition 0"), including the
/// all-zero-delta table that equals the §3.8.1.5 default, while its
/// version-3 Configuration Record path accepts the same bytes.
fn custom_transition_deltas() -> [i32; NUM_TRANSITION_DELTAS] {
    let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
    for (i, delta) in deltas.iter_mut().enumerate().skip(1) {
        let def = DEFAULT_ONE_STATE[i] as i32;
        *delta = if def == 0 {
            // Lift zero-default (unreachable) entries to the self-loop
            // `i`, making the transmitted table fully live.
            i as i32
        } else if def <= 128 {
            1
        } else {
            -1
        };
    }
    deltas
}

fn build_record(spec: &StreamSpec) -> Ffv1ConfigurationRecord {
    let v3 = spec.version == 3;
    Ffv1ConfigurationRecord {
        version: match spec.version {
            0 => Ffv1Version::V0,
            1 => Ffv1Version::V1,
            _ => Ffv1Version::V3,
        },
        micro_version: if v3 { Some(4) } else { None },
        coder_type: spec.coder,
        state_transition_delta: if spec.coder == 2 {
            custom_transition_deltas()
        } else {
            [0; NUM_TRANSITION_DELTAS]
        },
        colorspace_type: spec.colorspace,
        bits_per_raw_sample: spec.bits,
        chroma_planes: spec.chroma_planes || spec.colorspace == ColorspaceType::Rgb,
        log2_h_chroma_subsample: spec.hss,
        log2_v_chroma_subsample: spec.vss,
        extra_plane: spec.alpha,
        // v0/v1 carry no slice grid on the wire, but the implied single
        // Slice is modelled as a 1×1 grid by `compute_slice_content`.
        num_h_slices: Some(spec.slices.0),
        num_v_slices: Some(spec.slices.1),
        quant_table_set_count: v3.then_some(1),
        ec: v3.then_some(u32::from(spec.ec)),
        intra: v3.then_some(spec.intra),
        initial_state_delta: None,
    }
}

/// A §4.1-well-formed Quantization Table Set: 11 symmetric levels on
/// the three §3.5 Figure 5 primary differences (power-of-two magnitude
/// buckets), flat on the two second-order differences. Scale chain
/// `11 × 11 × 11 = 1331` → `context_count == 666` (§4.1.2). The run
/// lengths are encoder freedom; the schema is the RFC's.
fn default_quant_table_set() -> QuantizationTableSet {
    const ACTIVE_LENS: [u32; 6] = [1, 2, 4, 8, 16, 97];
    const FLAT_LENS: [u32; 1] = [128];

    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    let mut scale: i64 = 1;
    for (i, table) in tables.iter_mut().enumerate() {
        let lens: &[u32] = if i < 3 { &ACTIVE_LENS } else { &FLAT_LENS };
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
        assert_eq!(k, 128, "run lengths must cover the first half");
        for k in 1..128 {
            table[256 - k] = -table[k];
        }
        table[128] = -table[127];
        scale *= 2 * lens.len() as i64 - 1;
    }
    QuantizationTableSet {
        tables,
        context_count: ((scale as u64).div_ceil(2)) as u32,
    }
}

/// One §4.6 Slice Header per raster cell of the `num_h × num_v` grid
/// (slice raster coordinates, §4.6.1-§4.6.4).
fn build_slice_headers(spec: &StreamSpec) -> Vec<Ffv1SliceHeader> {
    let slot_count = 2 + usize::from(spec.alpha);
    let mut headers = Vec::new();
    for j in 0..spec.slices.1 {
        for i in 0..spec.slices.0 {
            headers.push(Ffv1SliceHeader {
                slice_x: i,
                slice_y: j,
                slice_width: 1,
                slice_height: 1,
                quant_table_set_index_count: slot_count,
                quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
                picture_structure: PictureStructure::Progressive,
                picture_structure_raw: 0,
                sar_num: 0,
                sar_den: 0,
            });
        }
    }
    headers
}

// ────────────────────────── build + self round-trip ──────────────────────────

struct BuiltStream {
    spec: StreamSpec,
    extradata: Option<Vec<u8>>,
    packets: Vec<Vec<u8>>,
    /// ffmpeg-layout raw bytes per frame (planar, LE16 above 8 bits;
    /// RGB exported in G, B, R (, A) plane order to match `gbrp*`).
    source_raw: Vec<Vec<u8>>,
}

fn plane_export_order(spec: &StreamSpec, plane_count: usize) -> Vec<usize> {
    if spec.colorspace == ColorspaceType::Rgb {
        // RFC 9043 §3.7 recovers R, G, B (, A); `gbrp*` stores
        // G, B, R (, A).
        if plane_count == 4 {
            vec![1, 2, 0, 3]
        } else {
            vec![1, 2, 0]
        }
    } else {
        (0..plane_count).collect()
    }
}

fn frame_to_raw(spec: &StreamSpec, frame: &DecodedFrame) -> Vec<u8> {
    let mut out = Vec::new();
    for &p in &plane_export_order(spec, frame.planes.len()) {
        let plane = &frame.planes[p];
        if spec.bits <= 8 {
            out.extend(plane.samples.iter().map(|&s| s as u8));
        } else {
            for &s in &plane.samples {
                out.extend_from_slice(&(s as u16).to_le_bytes());
            }
        }
    }
    out
}

fn assert_planes_equal(name: &str, frame_idx: u32, got: &DecodedFrame, want: &DecodedFrame) {
    assert_eq!(
        got.planes.len(),
        want.planes.len(),
        "{name} frame {frame_idx}: plane count"
    );
    for (g, w) in got.planes.iter().zip(want.planes.iter()) {
        assert_eq!(
            (g.width, g.height),
            (w.width, w.height),
            "{name} frame {frame_idx}: plane {} geometry",
            w.plane_index
        );
        assert_eq!(
            g.samples, w.samples,
            "{name} frame {frame_idx}: plane {} samples diverged",
            w.plane_index
        );
    }
}

fn build_stream(spec: StreamSpec) -> BuiltStream {
    let record = build_record(&spec);
    let qts = vec![default_quant_table_set()];
    let dims = FramePixelDimensions::new(spec.width, spec.height).unwrap();
    let sources: Vec<DecodedFrame> = (0..spec.frames).map(|i| synth_frame(&spec, i)).collect();

    let mut packets = Vec::new();
    if spec.version == 3 {
        let headers = build_slice_headers(&spec);
        let mut carry = None;
        for (i, src) in sources.iter().enumerate() {
            // §4.2.17: an intra stream codes every Frame as a keyframe.
            let keyframe = i == 0 || spec.intra;
            packets.push(
                encode_frame_with_carry(
                    src, &record, &qts, &headers, spec.ec, keyframe, &mut carry,
                )
                .unwrap_or_else(|e| panic!("{}: encode frame {i}: {e:?}", spec.name)),
            );
        }
        // Self round-trip through the matching carry-aware decoder.
        let mut dcarry = None;
        for (i, src) in sources.iter().enumerate() {
            let decoded = if spec.colorspace == ColorspaceType::Rgb {
                decode_frame_rgb_with_carry(
                    &packets[i],
                    &record,
                    &qts,
                    dims,
                    spec.ec,
                    DecodeOptions::strict(),
                    &mut dcarry,
                )
            } else {
                decode_frame_with_carry(
                    &packets[i],
                    &record,
                    &qts,
                    dims,
                    spec.ec,
                    DecodeOptions::strict(),
                    &mut dcarry,
                )
            }
            .unwrap_or_else(|e| panic!("{}: self-decode frame {i}: {e:?}", spec.name));
            assert_planes_equal(spec.name, i as u32, &decoded, src);
        }
    } else {
        // Conforming v0/v1 inter Frames resume the previous Frame's
        // §3.8.1.3 / §3.8.2.5 per-context coder state (r411).
        let mut ecarry = None;
        for (i, src) in sources.iter().enumerate() {
            let bytes = if i == 0 {
                encode_frame_v0v1_with_carry(src, &record, &qts[0], &mut ecarry)
            } else {
                encode_frame_v0v1_inter_with_carry(src, &record, &qts[0], &mut ecarry)
            }
            .unwrap_or_else(|e| panic!("{}: encode frame {i}: {e:?}", spec.name));
            packets.push(bytes);
        }
        let mut dcarry = None;
        for (i, src) in sources.iter().enumerate() {
            let decoded = if i == 0 {
                decode_frame_v0v1_with_carry(&packets[i], dims, &mut dcarry)
            } else {
                decode_frame_v0v1_inter_with_carry(&packets[i], &record, &qts[0], dims, &mut dcarry)
            }
            .unwrap_or_else(|e| panic!("{}: self-decode frame {i}: {e:?}", spec.name));
            assert_planes_equal(spec.name, i as u32, &decoded, src);
        }
    }

    let extradata = (spec.version == 3).then(|| {
        encode_configuration_record_with_quant_tables(&record, &qts)
            .unwrap_or_else(|e| panic!("{}: encode configuration record: {e:?}", spec.name))
    });
    let source_raw = sources.iter().map(|f| frame_to_raw(&spec, f)).collect();

    BuiltStream {
        spec,
        extradata,
        packets,
        source_raw,
    }
}

// ────────────────────────── pins ──────────────────────────

/// SHA-256 pins of every emitted packet (and the §4.3.3 extradata blob
/// for version-3 streams, pinned as the first entry prefixed `x:`).
/// These exact bytes passed the black-box external-decoder validation
/// recorded in `tests/external_conformance_notes.md` — all 27 streams
/// decode bit-exactly in the reference decoder (r416: the former
/// `v0-yuv420p8-custom` exception was root-caused to the zero
/// transitions the r411 table inherited from the §3.8.1.5 default and
/// closed by transmitting a fully-live custom table; see
/// [`custom_transition_deltas`]).
const STREAM_PINS: &[(&str, &[&str])] = &[
    (
        "v3-yuv420p8-range-2x2",
        &[
            "x:3b43cce48edf3000673d76e9a9f978c8d9eae144194bfa3bbafcd9679e53f7be",
            "d8f082c2d35f84f8212d4982e07630bf8f386318a6d2a6c4493001d91e2c8fa5",
            "89b4dfc046e98f36b11d31fab880b35e9620de148b5c4e70682b58f8f04970c4",
        ],
    ),
    (
        "v3-gray8-range",
        &[
            "x:32c2d095315a568543e1bd2086184c656b1dc62dc306a24d525d8a24c78a6ebc",
            "59c285a6d8bd7f254b877948b6d978a497b906f95660e2578e5c625070d30252",
            "1a13b95afc4766cfc7378d751b5d34d907c4a77ab679459aa37cdf158a3ab09d",
        ],
    ),
    (
        "v3-gray16-range",
        &[
            "x:260dc2514e1493d7242e06aa6e84cf8fb735e7c4928f41cd6a8a6e4b97cc4838",
            "ab8e5614679793970fac93f34321a97c0d3f6f4e2af895e8dd9e5f9dace67ad6",
            "6cf9b48a211042672a0dfc47c5e1dac23f191ce7f4adad6d1f213cd8665e982c",
        ],
    ),
    (
        "v3-yuv422p10-range",
        &[
            "x:7d32ac76f2b068bd36a788ad633e2ed1ed3a27f4b3f5ba85f2d36e7f1eaf292f",
            "3dfec52697cf73f236a79570311bb2e4d2842eedf8e1b676c48fdd4aa57adc57",
            "6a0f29d0766fbc750482c2908157ef3414704223a8d530aaa2930bdba9d45186",
        ],
    ),
    (
        "v3-yuv420p12-range",
        &[
            "x:d28358d40e4042ce01d950063c3407af93b4546728a85286a4d8ffd733cd9047",
            "33f364da6f205ea6d0cb22a1cc0e44b44ccfa7d58df32863e42e8ef062114942",
            "9dd454f520688d0515f3ee5e24602df12b7e9f23410a7872d86ee9113ea28621",
        ],
    ),
    (
        "v3-yuv444p16-range",
        &[
            "x:6b4c4a4832f3a705b580773fd14b6c843d768b77283a94f9d1d3dbaf7b1724f0",
            "bbee475ebbdad678dbc482c826d6e34feb19093ee97eafe03ecae03547afb7d0",
            "84df8ed93c086fda392edc5049cdbdb23837ddaabd468b8e4398027897dbc7db",
        ],
    ),
    (
        "v3-yuva420p8-range",
        &[
            "x:26579bc4c5f6155959fdd6a9a09c4b4da5a8b6faa3e909ed32dfb069cc16b746",
            "e12affdcaf0870b03bcb974ac1bfd2b78a149b7c6cb13d2ce8460d4a595cc92f",
            "d6fdd003be904e594464271cd8e4670cabc96f0bee0f7eed4f2b7257b26ce87c",
        ],
    ),
    (
        "v3-rgb8-range",
        &[
            "x:d50f3dc596753b2089011cdeffe9e32248824279f2741104376f4e6ffc1fe82e",
            "e06bea48f649289775798a7c27899fd6cf08aee93e0b6bfd891b26a92445b808",
            "e014c4a3c2943b0762bd968c5b6e9748c0731983f07fbc44a0df5f26e6e526ad",
        ],
    ),
    (
        "v3-rgba8-range",
        &[
            "x:9526bb31ed2c75c8a43278b9bf16496834256afc917cc4ecd3f167bba3d2b473",
            "c1a7eca7013876f01c4aeb41c2558c6067ed7bcfdd0a4c5c721cfe94e484dbea",
            "4400519454eebea729373ebcf5b1725a9598506c8edd8f80cae715ee50b10660",
        ],
    ),
    (
        "v3-rgb10-range",
        &[
            "x:788ac4f4d56d15faffdd8f1e21b2885efe6b6cbdd4e068dd79e825fbf4cbb8f4",
            "d18a839b8c7a16da8d0245af6027f6f07f786d281e42ab4ca656695b9ff86d46",
            "1da393ee90c6b179d218329c626c4e0082d6a79f0afadfaf78ac841eb77a4f1f",
        ],
    ),
    (
        "v3-rgb14-range",
        &[
            "x:394b4bb249a306d60dafda5eed80e9040f81342600592657815207a201836ff9",
            "e13e7c64049bc3f107f01e5d8f5f9bb2b92438f6440b1cc17d9ea7c281f97b31",
            "3fce1d4194b10ce1559d98e97e6b72cec22750ecc7b781b86ff4ce207808f42d",
        ],
    ),
    (
        "v3-yuv420p8-golomb",
        &[
            "x:a1c6451bca6c44807d8e71f82e0cc4b5b77bb7d61c81892b261663b1b1a60fe7",
            "0c57c24d4cc5b7e25dfce293878e73930be14c60a642b7cfd35897a0e54d2b4c",
            "81c88f910b20cd959dcaedb3f687379b017623bf454e71322e91e98dbedc190b",
        ],
    ),
    (
        "v3-rgb8-golomb",
        &[
            "x:f30e3843c46f2588eab559f9e5a37234363789dab6888fcf9e8a0a838d5fa2ce",
            "7e96fec080bc737d03e8b63a9d1accb2a513c5e11fe8d4c8919695477ea7f643",
            "316cc230322fcd0db43f899a5628c9a4ef983926ca0baf25aaacad937d92b8f6",
        ],
    ),
    (
        "v3-yuv420p8-custom",
        &[
            "x:b73251eecae4c22ac47cf252a033c66916844b92d5c98aa15bbda0db14ba81ea",
            "a86c4c704cb1f50cd9b10c8aab7bdaf21af49a745fa8b71d3b6fc153107c060a",
            "399a1ec98327e0aa9a3b56d74bc594096e9f3a72b0d4e0834ea50dd639573508",
        ],
    ),
    (
        "v3-yuv444p8-multislice-3x2-odd",
        &[
            "x:63174dd70716d6c932cc386af82c0ebd938c7c75c171fd22b9e8f6d900f040b2",
            "a6e78f6860a165dc8a5c29479d287205bae1e4574f2c6f31b32b3ec381e0df2f",
            "17fea66046cc3da187c2b7399249dad66097d7acb6ad21810ea23b790999e1e3",
        ],
    ),
    (
        "v3-gray8-range-ec0",
        &[
            "x:5bbad0be9316587d67e0ef2c13738794e595073b137835baca9c302bae63b34f",
            "38fdd50ab3a081440608a935394b4b0bde91d76a0fbe6d506d60181f00e2c654",
            "ffd9c55194ffdd95579a60af5767db7eaf4aa39059e07f7a6d57e99cc03bcca0",
        ],
    ),
    (
        "v0-yuv420p8-range",
        &[
            "bbc8763e6097cbe49fb2fc35baec734d8cd10c8169dc57f3f5dc120498ac523e",
            "2f00724a8533311ece6de98442f30dabe78ac0e067bb898a6d0b2fa2e97072cf",
        ],
    ),
    (
        "v0-yuv420p8-golomb",
        &[
            "f43fa466c04f07458244eed471e9c1ef09536c0fe55df6cfaf7042ce3cf9282d",
            "455d02ea1b574557617dc0a23cfe02fda851b05d1c2bcdeb5dc72799bd08d939",
        ],
    ),
    (
        "v0-yuv420p8-custom",
        &[
            "95514d2ba70d48e7845fd0f4e632fd2f1c26ea0b8a6e665b009d57a2f4329803",
            "dc5a5c4af38428128caa78382d2483a4b56abacd7456d2ee33acd63b4f4cc162",
        ],
    ),
    (
        "v1-yuv420p8-range",
        &[
            "4d745d290ebc5bb4329b797cf1e9bba2d02021646433f7b0124728c441fcdbc4",
            "2f00724a8533311ece6de98442f30dabe78ac0e067bb898a6d0b2fa2e97072cf",
        ],
    ),
    (
        "v1-gray8-golomb",
        &[
            "e414c17f2d8f4f8d757a0478b5c03b4549d5b2a1d32aba06693921680c861a84",
            "d7777c7cdc5154be145c159ae9d7f8752533a44e16ee6dbb9e1e92f0b4bfd7aa",
        ],
    ),
    (
        "v1-rgb8-range",
        &[
            "cd7db663738e6966f7e78c860c6e03e483362e48c407777c648c11291eecf0fa",
            "231691fe6500215dbc3ad1452aa46c8db836808a1921b299f600914c8c329290",
        ],
    ),
    (
        "v0-rgb8-golomb",
        &[
            "4165b6faf478c4398332fd27ed0483d588d7691bcfb123f3252c3f2303ad6171",
            "9556c4fb7276a15af0a28d5ca4261bd0aa33de605f53be2139e55007c49c5826",
        ],
    ),
    (
        "v1-yuv422p10-range",
        &[
            "86a91ffb351c46402db50e649e1a2eddc4b1122ac028949100587f627854b11a",
            "81ff423552a9d3c489410edf4b8945f6fc8c1611a53696b8072363abe56eae3d",
        ],
    ),
    (
        "v3-yuv420p8-range-2x2-4f",
        &[
            "x:3b43cce48edf3000673d76e9a9f978c8d9eae144194bfa3bbafcd9679e53f7be",
            "7ed197e6155655a6e92e8d6d6265f0c5d4cce20fc835216450163440d0a1b7f1",
            "6c1d9fd3f6e18cefc69df430b2b4cfbeded45dda1c9c52fb1e9611d387950031",
            "1e2d76bf8dfe81455cdd496495751bd830c4468c551fb1708db65debffcaee02",
            "525ba1478a61909569ad49c8a09d7ca2e072f17cbb24360b6dfd705eb3efb8de",
        ],
    ),
    (
        "v3-yuv420p8-range-intra",
        &[
            "x:86bb9a90653ca62bbbcfdddfe4d723870fba94f0dc25c066880b499b436c718a",
            "872fc4df1947b0c95d9e24ea39f7beb5341a3407d69cc11335b7519f05f4cbd8",
            "9a4327b3df4986b73946eddbbbf8c7fe39068f0230cc0339b21758e35c4dc8aa",
        ],
    ),
    (
        "v0-yuv420p8-golomb-3f",
        &[
            "f43fa466c04f07458244eed471e9c1ef09536c0fe55df6cfaf7042ce3cf9282d",
            "455d02ea1b574557617dc0a23cfe02fda851b05d1c2bcdeb5dc72799bd08d939",
            "284675e0b3e5e379615c0c4674ca965d85f618f21cf8b1997f62d20e54ba339e",
        ],
    ),
];

// ────────────────────────── the test ──────────────────────────

#[test]
fn external_conformance_corpus() {
    let export_dir = std::env::var_os("FFV1_CONFORMANCE_EXPORT_DIR").map(std::path::PathBuf::from);
    let mut missing_pins = Vec::new();

    for spec in corpus() {
        let name = spec.name;
        let built = build_stream(spec);

        // Assemble the hash list: extradata first (v3), then packets.
        let mut hashes: Vec<String> = Vec::new();
        if let Some(x) = &built.extradata {
            hashes.push(format!("x:{}", sha256_hex(x)));
        }
        for pkt in &built.packets {
            hashes.push(sha256_hex(pkt));
        }

        match STREAM_PINS.iter().find(|(n, _)| *n == name) {
            Some((_, pinned)) => {
                let got: Vec<&str> = hashes.iter().map(String::as_str).collect();
                assert_eq!(
                    &got[..],
                    *pinned,
                    "{name}: emitted bytes diverged from the externally-validated pin — \
                     re-run the black-box validation (tests/external_conformance_notes.md) \
                     before re-pinning"
                );
            }
            None => {
                for h in &hashes {
                    println!("PIN {name} {h}");
                }
                missing_pins.push(name);
            }
        }

        if let Some(dir) = &export_dir {
            let sdir = dir.join(name);
            std::fs::create_dir_all(&sdir).unwrap();
            if let Some(x) = &built.extradata {
                std::fs::write(sdir.join("extradata.bin"), x).unwrap();
            }
            for (i, pkt) in built.packets.iter().enumerate() {
                std::fs::write(sdir.join(format!("pkt{i}.bin")), pkt).unwrap();
            }
            for (i, raw) in built.source_raw.iter().enumerate() {
                std::fs::write(sdir.join(format!("src{i}.raw")), raw).unwrap();
            }
            let mut mf = std::fs::File::create(sdir.join("manifest.txt")).unwrap();
            writeln!(mf, "name={name}").unwrap();
            writeln!(mf, "pix_fmt={}", built.spec.pix_fmt).unwrap();
            writeln!(mf, "width={}", built.spec.width).unwrap();
            writeln!(mf, "height={}", built.spec.height).unwrap();
            writeln!(mf, "version={}", built.spec.version).unwrap();
            writeln!(mf, "coder={}", built.spec.coder).unwrap();
            writeln!(mf, "frames={}", built.spec.frames).unwrap();
            writeln!(mf, "intra={}", u32::from(built.spec.intra)).unwrap();
            writeln!(
                mf,
                "extradata={}",
                if built.extradata.is_some() { 1 } else { 0 }
            )
            .unwrap();
        }
    }

    assert!(
        missing_pins.is_empty(),
        "streams without externally-validated pins: {missing_pins:?} \
         (hashes printed above; validate black-box, then pin)"
    );
}

#[test]
fn sha256_known_answer() {
    // FIPS 180-4 known-answer vectors.
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
