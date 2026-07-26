//! Round 420 — reference-encoded streams through the framework
//! `oxideav_core::Decoder`, pinning the deep-format trait wiring against
//! *reference* bytes (not just this crate's own encoder):
//!
//! * the §4.2 mapping surfaces the fixtures decode onto —
//!   `Gray16Le`, `Yuv444P16Le`, `Yuv422P10Le`, `Yuv420P12Le` and (as
//!   of core 0.1.33) the native one-byte `Gbrp8` for the 8-bit RGB
//!   fixture, all exact; the 8-bit RGBA fixture stays on the
//!   `Gbrap10Le` storage surface with an `[8, 8, 8, 8]`
//!   significant-bits record on every emitted frame (no `Gbrap8`
//!   variant exists);
//! * the trait-boundary plane packing (surface word width + `Gbr`
//!   reorder) reproducing the reference decoder's raw layout SHA-256
//!   pins bit-exactly across keyframe + carried non-keyframes;
//! * the §4.2.16 `ec` first-frame resolution running against the
//!   reference writer's record tails (see
//!   `tests/reference_inter_decode.rs` for the finding).
//!
//! Fixtures: `tests/data/reference_inter_fixtures.rs` (inlined packets;
//! per-frame SHA-256 of the reference decoder's `expected.raw`, staged
//! under `docs/video/ffv1/fixtures/inter-*`).

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, RuntimeContext, TimeBase,
};
use oxideav_ffv1::{register, CODEC_ID_STR};

#[path = "data/reference_inter_fixtures.rs"]
mod fx;

// ────────────────────────── SHA-256 (FIPS 180-4) ──────────────────────────
//
// Inlined so the fixture pins carry no external dependency (same
// convention as `tests/reference_inter_decode.rs`).

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

/// `(fixture name, mapped surface, significant-bits record, low-byte
/// repack)`: `low_byte` is set for the 8-bit RGBA fixture, whose
/// reference raw layout is one byte per Sample while the mapped
/// `Gbrap10Le` surface emits 2-byte LE words — the low byte of each
/// word is the Sample (the record says only 8 bits are significant).
/// The 8-bit RGB (no alpha) fixture decodes onto the native one-byte
/// `Gbrp8` (core 0.1.33), so its emitted planes ARE the reference raw
/// layout — same pinned hash, no repack, no record.
const CASES: &[(&str, PixelFormat, Option<&[u8]>, bool)] = &[
    ("inter-v3-gray16-range", PixelFormat::Gray16Le, None, false),
    (
        "inter-v3-yuv444p16-range",
        PixelFormat::Yuv444P16Le,
        None,
        false,
    ),
    (
        "inter-v3-yuv422p10-range",
        PixelFormat::Yuv422P10Le,
        None,
        false,
    ),
    (
        "inter-v3-yuv420p12-range",
        PixelFormat::Yuv420P12Le,
        None,
        false,
    ),
    ("inter-v3-rgb-bgr0-range", PixelFormat::Gbrp8, None, false),
    (
        "inter-v3-rgba-range",
        PixelFormat::Gbrap10Le,
        Some(&[8, 8, 8, 8]),
        true,
    ),
];

#[test]
fn reference_streams_decode_bit_exact_through_the_trait() {
    for &(name, want_pf, want_sig, low_byte) in CASES {
        let fixture = fx::INTER_FIXTURES
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: fixture present"));
        let extradata = fixture.extradata.expect("v3 fixture has a record");
        // Corpus ground truth: all six streams open with a §4.4 keyframe
        // and were generated with a per-Slice CRC (`-slicecrc 1`), so the
        // registry's first-frame `ec` resolution must land on the
        // footer-present hypothesis despite the reference record-tail
        // misparse.
        assert!(fixture.keyframes[0], "{name}: first frame is a keyframe");
        assert!(fixture.slice_crc, "{name}: generated with -slicecrc 1");

        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(fixture.width);
        params.height = Some(fixture.height);
        params.extradata = extradata.to_vec();

        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut dec = ctx
            .codecs
            .first_decoder(&params)
            .unwrap_or_else(|e| panic!("{name}: decoder: {e:?}"));

        for (i, pkt_bytes) in fixture.packets.iter().enumerate() {
            let pkt = Packet::new(0, TimeBase::new(1, 25), pkt_bytes.to_vec())
                .with_keyframe(fixture.keyframes[i]);
            dec.send_packet(&pkt)
                .unwrap_or_else(|e| panic!("{name} frame {i}: send: {e:?}"));
            let frame = dec
                .receive_frame()
                .unwrap_or_else(|e| panic!("{name} frame {i}: decode: {e:?}"));
            let Frame::Video(out) = frame else {
                panic!("{name} frame {i}: expected video");
            };

            assert_eq!(
                out.significant_bits(),
                want_sig,
                "{name} frame {i}: significant-bits record for {want_pf:?}"
            );

            // Reassemble the reference raw layout from the emitted
            // planes (already in surface order — `Gbr` reorder applied
            // by the registry).
            let mut raw = Vec::with_capacity(fixture.frame_raw_len);
            for plane in out.image_planes() {
                if low_byte {
                    // 2-byte LE words, 8 significant bits → low byte.
                    for pair in plane.data.chunks_exact(2) {
                        assert_eq!(pair[1], 0, "{name}: high byte of an 8-bit Sample");
                        raw.push(pair[0]);
                    }
                } else {
                    raw.extend_from_slice(&plane.data);
                }
            }
            assert_eq!(
                raw.len(),
                fixture.frame_raw_len,
                "{name} frame {i}: raw length in {} layout",
                fixture.pix_fmt
            );
            assert_eq!(
                sha256_hex(&raw),
                fixture.frame_sha256[i],
                "{name} frame {i}: trait-surface bytes diverged from the reference decoder"
            );
        }
    }
}
