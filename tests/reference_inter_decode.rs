//! Inter-frame (keyframe + non-keyframe) reference-stream decode tests.
//!
//! Every fixture in `tests/data/reference_inter_fixtures.rs` is a
//! reference-encoded FFV1 stream of one §4.4 keyframe plus carried
//! non-keyframes (staged with generation commands + SHA-256 under
//! `docs/video/ffv1/fixtures/inter-*`). Decoding drives the §3.8.1.3 /
//! §3.8.2.5 inter-Frame coder-state carry against *reference-produced*
//! bytes — the decode-side mirror of the self-encoded
//! `tests/external_conformance.rs` corpus — across:
//!
//! * versions 0 / 1 (inline §4.4 Parameters, range + Golomb-Rice) and
//!   version 3 (§4.3 Configuration Record);
//! * all three §4.2.3 coders, including a reference-encoded
//!   `coder_type == 2` custom state-transition-table stream
//!   (`inter-v3-yuv420p8-rangetab`) — previously the custom-table
//!   decode path was validated only against this crate's own encoder;
//! * 8 / 10 / 12 / 16-bit depths, 4:2:0 / 4:2:2 / 4:4:4 chroma, gray,
//!   RGB and RGBA (JPEG 2000 RCT), a 2×2 slice grid, and the large
//!   `-context 1` Quantization Table Set;
//! * a mid-stream keyframe (`inter-v3-yuv420p8-range-g2`: keyframes at
//!   Frames 0 and 2), proving the carry re-initialises on a later §4.4
//!   keyframe, not only on the first Frame.
//!
//! Each decoded Frame is packed into the reference raw layout (planar,
//! little-endian 16-bit above 8 bits, RGB in G, B, R (, A) plane order)
//! and pinned by SHA-256 against the reference decoder's
//! `expected.raw`; the §4.4 keyframe flag of every Frame must match the
//! reference toolchain's report. Version-3 Slices decode under
//! [`DecodeOptions::pedantic`], so the §3.8.1.1.1 Sentinel-termination
//! gate is also exercised against reference-encoded bytes.
//!
//! # The reference-writer record-tail finding (r416)
//!
//! The §4.2.16 `ec` / §4.2.17 `intra` values these tests DRIVE the
//! decode with come from each fixture's ground truth (the recorded
//! generation command), NOT from the parsed Configuration Record —
//! because the current reference encoder's record TAIL does not
//! Figure-28-parse. Black-box findings, all r416:
//!
//! * The reference **parser** accepts records this crate writes per
//!   Figure 28 (per-set §4.2.14 `states_coded` flags on the shared
//!   Parameters window, then `ec`, then `intra`): re-authoring a
//!   fixture's record with this crate's writer (same field values,
//!   `ec == 1`) and feeding the reference packets back decodes
//!   bit-exact with zero warnings, while the same re-authoring with
//!   `ec == 0` fails hard — so the parsed `ec` is honoured. The r411/r416
//!   28-stream self-encoded corpus (single-set records) passes the same
//!   way.
//! * The reference **writer**'s own records (Lavc62, two Quantization
//!   Table Sets), read under that very layout, yield non-physical tail
//!   values — `ec` up to 7 (RFC 9043 Table 13 defines 0/1), `intra == 1`
//!   on streams that contain non-keyframes — varying with unrelated
//!   record content, while every field up to and including both
//!   quant-table cascades parses exactly (the Frames decode bit-exact
//!   with those tables). Earlier reference-writer records (the pinned
//!   single-frame corpus under `docs/video/ffv1/fixtures/`, and the
//!   reference-validated `states-coded-1` hand-authored record) parse
//!   cleanly under the same layout.
//!
//! In practice a misdeclared `ec` is the dangerous half: a `-slicecrc 0`
//! stream whose record misreads as `ec != 0` makes a §4.9-faithful
//! decoder look for Slice Footers that are not there (see
//! `inter-v3-yuv420p8-range-crc0` + the registry-level first-packet
//! resolution in `tests/registry_inter_ec_resilience.rs`); `intra` is
//! only consumed by the opt-in `Ffv1DecodeSession` conformance gate.
//! The exact symbol/window layout the current reference writer uses for
//! the tail remains unresolved black-box (probed extensively: single- /
//! dual- / zero-flag shapes, fresh / shared / per-symbol-reset windows,
//! boundary fill variants — no candidate reproduces every stream).

use oxideav_ffv1::{
    decode_frame_rgb_with_carry, decode_frame_v0v1_inter_with_carry, decode_frame_v0v1_with_carry,
    decode_frame_with_carry, parse_quantization_table_sets, parse_v0v1_frame_prologue,
    ColorspaceType, DecodeOptions, DecodedFrame, FramePixelDimensions,
};

#[path = "data/reference_inter_fixtures.rs"]
mod fx;

// ────────────────────────── SHA-256 (FIPS 180-4) ──────────────────────────
//
// Inlined so the fixture pins carry no external dependency (same
// convention as `tests/external_conformance.rs`).

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

/// Pack a decoded Frame into the reference raw layout: planar, one byte
/// per Sample at ≤ 8-bit depth, two little-endian bytes otherwise; RGB
/// Planes (recovered in R, G, B (, A) order per §3.7) reordered to the
/// G, B, R (, A) order the planar `gbrp*` layouts store.
fn pack_reference_raw(frame: &DecodedFrame, rgb: bool, bits: u32) -> Vec<u8> {
    let order: Vec<usize> = if rgb {
        if frame.planes.len() == 4 {
            vec![1, 2, 0, 3]
        } else {
            vec![1, 2, 0]
        }
    } else {
        (0..frame.planes.len()).collect()
    };
    let mut out = Vec::new();
    for &p in &order {
        let plane = &frame.planes[p];
        if bits <= 8 {
            out.extend(plane.samples.iter().map(|&s| s as u8));
        } else {
            for &s in &plane.samples {
                out.extend_from_slice(&(s as u16).to_le_bytes());
            }
        }
    }
    out
}

fn check_frame(
    fixture: &fx::InterFixture,
    idx: usize,
    decoded: &DecodedFrame,
    rgb: bool,
    bits: u32,
) {
    let name = fixture.name;
    assert_eq!(
        decoded.keyframe, fixture.keyframes[idx],
        "{name} frame {idx}: §4.4 keyframe flag"
    );
    let raw = pack_reference_raw(decoded, rgb, bits);
    assert_eq!(
        raw.len(),
        fixture.frame_raw_len,
        "{name} frame {idx}: raw byte length in {} layout",
        fixture.pix_fmt
    );
    assert_eq!(
        sha256_hex(&raw),
        fixture.frame_sha256[idx],
        "{name} frame {idx}: decoded bytes diverged from the reference decoder's expected.raw"
    );
}

#[test]
fn v3_inter_streams_decode_bit_exact() {
    let mut seen = 0;
    for fixture in fx::INTER_FIXTURES {
        let Some(extra) = fixture.extradata else {
            continue;
        };
        seen += 1;
        let name = fixture.name;
        let parsed = parse_quantization_table_sets(extra)
            .unwrap_or_else(|e| panic!("{name}: extradata: {e:?}"));
        let record = parsed.record;
        let rgb = record.colorspace_type == ColorspaceType::Rgb;
        // Ground truth from the generation command, NOT `record.ec` —
        // see the record-tail finding in the module doc.
        let ec = fixture.slice_crc;
        let dims = FramePixelDimensions::new(fixture.width, fixture.height).expect("dims");
        let mut carry = None;
        for (i, pkt) in fixture.packets.iter().enumerate() {
            let decoded = if rgb {
                decode_frame_rgb_with_carry(
                    pkt,
                    &record,
                    &parsed.quant_table_sets,
                    dims,
                    ec,
                    DecodeOptions::pedantic(),
                    &mut carry,
                )
            } else {
                decode_frame_with_carry(
                    pkt,
                    &record,
                    &parsed.quant_table_sets,
                    dims,
                    ec,
                    DecodeOptions::pedantic(),
                    &mut carry,
                )
            }
            .unwrap_or_else(|e| panic!("{name} frame {i}: decode: {e:?}"));
            check_frame(fixture, i, &decoded, rgb, record.bits_per_raw_sample);
        }
    }
    assert_eq!(seen, 12, "version-3 inter fixture count");
}

/// The `-slicecrc 0` fixture pins the record-tail finding from the
/// module doc: the Figure-28 parse of the reference writer's record
/// yields `ec != 0` even though the stream carries NO §4.9 Slice
/// Footers, so a decoder trusting `record.ec` fails on the very first
/// Slice — while the ground-truth `ec = false` decode is bit-exact
/// (covered by [`v3_inter_streams_decode_bit_exact`]).
#[test]
fn crc0_record_misdeclares_ec_nonzero() {
    let fixture = fx::INTER_FIXTURES
        .iter()
        .find(|f| f.name == "inter-v3-yuv420p8-range-crc0")
        .expect("crc0 fixture present");
    assert!(!fixture.slice_crc, "generated with -slicecrc 0");
    let parsed = parse_quantization_table_sets(fixture.extradata.unwrap()).expect("extradata");
    assert!(
        parsed.record.ec.is_some_and(|ec| ec != 0),
        "the reference-writer record tail misreads as ec != 0 under the \
         Figure 28 layout (got {:?}); if this ever starts parsing as \
         Some(0), the reference writer changed and the module-doc finding \
         must be re-probed",
        parsed.record.ec
    );
    // Trusting the misdeclared record must fail: the footer parse looks
    // for a §4.9.3 CRC that is not there.
    let dims = FramePixelDimensions::new(fixture.width, fixture.height).expect("dims");
    let mut carry = None;
    let err = decode_frame_with_carry(
        fixture.packets[0],
        &parsed.record,
        &parsed.quant_table_sets,
        dims,
        true, // record-derived (wrong) hypothesis
        DecodeOptions::strict(),
        &mut carry,
    );
    assert!(
        err.is_err(),
        "decoding a footer-less stream under ec=true must surface a typed error"
    );
}

#[test]
fn v0v1_inter_streams_decode_bit_exact() {
    let mut seen = 0;
    for fixture in fx::INTER_FIXTURES {
        if fixture.extradata.is_some() {
            continue;
        }
        seen += 1;
        let name = fixture.name;
        assert!(
            fixture.keyframes[0],
            "{name}: first Frame must be a keyframe"
        );
        // The §4.4 keyframe carries the inline Parameters + single §4.1
        // Quantization Table Set the later non-keyframes inherit.
        let prologue = parse_v0v1_frame_prologue(fixture.packets[0])
            .unwrap_or_else(|e| panic!("{name}: keyframe prologue: {e:?}"));
        let record = prologue.record;
        let qts = prologue.quant_table_set;
        let dims = FramePixelDimensions::new(fixture.width, fixture.height).expect("dims");
        let mut carry = None;
        for (i, pkt) in fixture.packets.iter().enumerate() {
            let decoded = if fixture.keyframes[i] {
                decode_frame_v0v1_with_carry(pkt, dims, &mut carry)
            } else {
                decode_frame_v0v1_inter_with_carry(pkt, &record, &qts, dims, &mut carry)
            }
            .unwrap_or_else(|e| panic!("{name} frame {i}: decode: {e:?}"));
            check_frame(fixture, i, &decoded, false, record.bits_per_raw_sample);
        }
    }
    assert_eq!(seen, 4, "version-0/1 inter fixture count");
}

#[test]
fn rangetab_fixture_uses_custom_state_transition_table() {
    // The reference encoder's `range_tab` mode must actually have
    // produced a §4.2.3 `coder_type == 2` record with a non-trivial
    // §4.2.4 delta block — otherwise the fixture would silently stop
    // covering the §3.8.1.6 custom-table decode path.
    let fixture = fx::INTER_FIXTURES
        .iter()
        .find(|f| f.name == "inter-v3-yuv420p8-rangetab")
        .expect("rangetab fixture present");
    let parsed = parse_quantization_table_sets(fixture.extradata.unwrap()).expect("extradata");
    assert_eq!(parsed.record.coder_type, 2, "coder_type");
    assert!(
        parsed.record.state_transition_delta.iter().any(|&d| d != 0),
        "custom table must differ from the §3.8.1.5 default"
    );
}
