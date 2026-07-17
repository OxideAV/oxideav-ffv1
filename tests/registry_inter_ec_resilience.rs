//! Registry-level §4.2.16 `ec` resolution against a real
//! reference-encoded stream whose Configuration Record MISDECLARES it.
//!
//! r416 black-box finding (full write-up in the module doc of
//! `tests/reference_inter_decode.rs`): the current reference encoder's
//! record TAIL does not read back under the RFC 9043 Figure 28 layout
//! its own parser accepts, so the record-declared `ec` is unreliable —
//! the `inter-v3-yuv420p8-range-crc0` fixture was generated with
//! `-slicecrc 0` (no §4.9 Slice Footer CRC) yet its record parses to
//! `ec != 0` / `intra == true`. A decoder trusting the record then looks
//! for §4.9.3 CRCs that are not there and fails on the very first
//! Slice.
//!
//! The registry `oxideav_core::Decoder` treats the record-derived `ec`
//! as a hypothesis until the first Frame decodes: on failure it retries
//! the packet once with the opposite §4.9 footer shape and locks in
//! whichever hypothesis yields a fully-validated Frame. These tests pin
//! that behaviour with real reference bytes:
//!
//! * the misdeclared `-slicecrc 0` stream decodes bit-exactly through
//!   the trait surface (keyframe + two carried non-keyframes), and
//! * a truthfully-declared `-slicecrc 1` stream from the same corpus
//!   still decodes bit-exactly (the retry never fires on it).

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, RuntimeContext, TimeBase};
use oxideav_ffv1::{parse_quantization_table_sets, register, CODEC_ID_STR};

// `dead_code`: this binary reads only the subset of `InterFixture`
// fields it needs; `tests/reference_inter_decode.rs` consumes the rest.
#[allow(dead_code)]
#[path = "data/reference_inter_fixtures.rs"]
mod fx;

// ────────────────────────── SHA-256 (FIPS 180-4) ──────────────────────────

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

/// Decode `fixture` end-to-end through the registry `Decoder` trait and
/// pin every frame's packed plane bytes against the reference decoder's
/// per-frame SHA-256.
fn decode_through_registry(fixture: &fx::InterFixture) {
    let name = fixture.name;
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(fixture.width);
    params.height = Some(fixture.height);
    params.extradata = fixture.extradata.expect("v3 fixture").to_vec();
    let mut dec = ctx
        .codecs
        .first_decoder(&params)
        .unwrap_or_else(|e| panic!("{name}: build decoder: {e:?}"));

    for (i, pkt) in fixture.packets.iter().enumerate() {
        let packet = Packet::new(0, TimeBase::new(1, 25), pkt.to_vec());
        dec.send_packet(&packet)
            .unwrap_or_else(|e| panic!("{name} frame {i}: send: {e:?}"));
        let frame = dec
            .receive_frame()
            .unwrap_or_else(|e| panic!("{name} frame {i}: receive: {e:?}"));
        let Frame::Video(video) = frame else {
            panic!("{name} frame {i}: expected a video frame");
        };
        let mut raw = Vec::new();
        for plane in &video.planes {
            raw.extend_from_slice(&plane.data);
        }
        assert_eq!(
            raw.len(),
            fixture.frame_raw_len,
            "{name} frame {i}: packed byte length"
        );
        assert_eq!(
            sha256_hex(&raw),
            fixture.frame_sha256[i],
            "{name} frame {i}: trait-surface bytes diverged from the reference decoder"
        );
    }
}

/// The `-slicecrc 0` stream whose record misdeclares `ec != 0` decodes
/// bit-exactly through the registry: the first-packet retry resolves the
/// true §4.9 footer shape and the carried non-keyframes reuse it.
#[test]
fn misdeclared_crc0_stream_decodes_through_registry() {
    let fixture = fx::INTER_FIXTURES
        .iter()
        .find(|f| f.name == "inter-v3-yuv420p8-range-crc0")
        .expect("crc0 fixture present");
    // Precondition for the test to be meaningful: the record really does
    // misdeclare ec (otherwise no retry would be exercised).
    let parsed = parse_quantization_table_sets(fixture.extradata.unwrap()).expect("extradata");
    assert!(
        parsed.record.ec.is_some_and(|ec| ec != 0),
        "fixture record must misdeclare ec != 0 (got {:?})",
        parsed.record.ec
    );
    decode_through_registry(fixture);
}

/// A truthfully-declared `-slicecrc 1` stream from the same reference
/// encoder decodes identically (the record-derived hypothesis succeeds
/// on the first packet; the retry path never runs).
#[test]
fn truthful_crc1_stream_decodes_through_registry() {
    let fixture = fx::INTER_FIXTURES
        .iter()
        .find(|f| f.name == "inter-v3-yuv420p8-range-2x2")
        .expect("2x2 fixture present");
    decode_through_registry(fixture);
}

/// The mid-stream-keyframe stream (`-g 2`, keyframes at Frames 0 and 2)
/// also holds through the trait surface: the locked `ec` survives the
/// later keyframe's state re-initialisation.
#[test]
fn g2_stream_decodes_through_registry() {
    let fixture = fx::INTER_FIXTURES
        .iter()
        .find(|f| f.name == "inter-v3-yuv420p8-range-g2")
        .expect("g2 fixture present");
    decode_through_registry(fixture);
}
