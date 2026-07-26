//! Round 420 — reference-decode tests for the r420 corpus
//! (`tests/data/nonuniform_deep_fixtures.rs`): **non-uniform RFC 9043
//! §4.8 floor-division slice grids** and **deep / alpha-carrying
//! formats**, every stream a §4.4 keyframe plus carried non-keyframes.
//!
//! * `nonuni-*` — slice grids the frame dimensions do not divide
//!   evenly, so the §4.8.2 / §4.8.3 floor divisions
//!   (`slice_pixel_x = floor(slice_x * width / num_h_slices)`, width =
//!   the difference of two consecutive floors) yield Slices of
//!   genuinely different pixel sizes. The tests recompute the §4.8
//!   geometry from the decoded §4.6 Slice Headers and assert the grid
//!   is non-uniform, covers the raster exactly, and decodes bit-exact
//!   against the reference decoder's per-frame SHA-256 pins — for the
//!   range coder AND the §3.8.2 Golomb-Rice coder, on YCbCr, gray,
//!   YUVA and RGB / RCT (line-major), at 8 / 10 / 16 bits, on odd
//!   frame dimensions where the format admits them.
//! * `deep-*` — the deep Yuva family (4:2:2 / 4:4:4 at 10 / 12 / 16
//!   bits) and the off-grid 9 / 14-bit depths, reference-encoded.
//!
//! Both the direct `decode_frame*_with_carry` API (under
//! [`DecodeOptions::pedantic`]) and the framework
//! [`oxideav_core::Decoder`] trait (asserting the §4.2 mapping surface
//! plus the attached significant-bits record) are driven over every
//! stream.

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, RuntimeContext, TimeBase,
};
use oxideav_ffv1::{
    decode_frame_rgb_with_carry, decode_frame_with_carry, parse_quantization_table_sets, register,
    ColorspaceType, DecodeOptions, DecodedFrame, Ffv1ConfigurationRecord, Ffv1SliceHeader,
    FramePixelDimensions, CODEC_ID_STR,
};

#[path = "data/nonuniform_deep_fixtures.rs"]
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

/// Pack a decoded Frame into the reference raw layout: planar, one byte
/// per Sample at ≤ 8-bit depth, two little-endian bytes otherwise; RGB
/// Planes (recovered in R, G, B order per §3.7) reordered to the
/// G, B, R order the planar `gbrp` layout stores.
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

/// The §4.8.2 / §4.8.3 slice-pixel geometry of one §4.6 Slice Header:
/// `(x, y, width, height)` in pixels, from the floor divisions over the
/// §4.2.11 / §4.2.12 raster grid.
fn slice_pixel_geometry(
    header: &Ffv1SliceHeader,
    frame_w: u32,
    frame_h: u32,
    num_h: u32,
    num_v: u32,
) -> (u32, u32, u32, u32) {
    let (sx, sy) = (header.slice_x, header.slice_y);
    let (sw, sh) = (header.slice_width, header.slice_height);
    // §4.8.3: slice_pixel_x = floor(slice_x * frame_pixel_width / num_h_slices)
    let px = (u64::from(sx) * u64::from(frame_w) / u64::from(num_h)) as u32;
    let py = (u64::from(sy) * u64::from(frame_h) / u64::from(num_v)) as u32;
    // §4.8.2: slice_pixel_width =
    //   floor((slice_x + slice_width) * frame_pixel_width / num_h_slices) - slice_pixel_x
    let pw = (u64::from(sx + sw) * u64::from(frame_w) / u64::from(num_h)) as u32 - px;
    let ph = (u64::from(sy + sh) * u64::from(frame_h) / u64::from(num_v)) as u32 - py;
    (px, py, pw, ph)
}

/// Assert the decoded Frame's §4.6 Slice Headers form a genuinely
/// NON-uniform §4.8 grid: the floor divisions yield at least two
/// distinct slice pixel widths or heights, and the per-slice areas tile
/// the frame exactly.
fn assert_nonuniform_grid(name: &str, cr: &Ffv1ConfigurationRecord, frame: &DecodedFrame) {
    let num_h = cr.num_h_slices.expect("v3 record has num_h_slices");
    let num_v = cr.num_v_slices.expect("v3 record has num_v_slices");
    assert_eq!(
        frame.slice_headers.len(),
        (num_h * num_v) as usize,
        "{name}: one Slice per raster cell"
    );
    let mut widths = Vec::new();
    let mut heights = Vec::new();
    let mut area = 0u64;
    for sh in &frame.slice_headers {
        let (_, _, pw, ph) = slice_pixel_geometry(sh, frame.width, frame.height, num_h, num_v);
        widths.push(pw);
        heights.push(ph);
        area += u64::from(pw) * u64::from(ph);
    }
    assert_eq!(
        area,
        u64::from(frame.width) * u64::from(frame.height),
        "{name}: §4.8 slice areas tile the frame"
    );
    widths.sort_unstable();
    widths.dedup();
    heights.sort_unstable();
    heights.dedup();
    assert!(
        widths.len() > 1 || heights.len() > 1,
        "{name}: grid {num_h}x{num_v} over {}x{} must be NON-uniform \
         (widths {widths:?}, heights {heights:?}) — fixture no longer \
         covers the §4.8 floor-division inequality",
        frame.width,
        frame.height,
    );
}

/// Direct-API decode of every fixture: bit-exact against the reference
/// SHA-256 pins under `DecodeOptions::pedantic()`, keyframe flags
/// matching the reference toolchain, and — for the `nonuni-*` streams —
/// the §4.8 non-uniform grid assertions on every Frame.
#[test]
fn corpus_decodes_bit_exact_with_nonuniform_grids() {
    let mut nonuni_seen = 0;
    for fixture in fx::FIXTURES {
        let name = fixture.name;
        let parsed = parse_quantization_table_sets(fixture.extradata)
            .unwrap_or_else(|e| panic!("{name}: extradata: {e:?}"));
        let record = parsed.record;
        let rgb = record.colorspace_type == ColorspaceType::Rgb;
        let dims = FramePixelDimensions::new(fixture.width, fixture.height).expect("dims");
        // Ground truth: every stream was generated with `-slicecrc 1`
        // (the record-tail `ec` parse is unreliable on current
        // reference-writer records — see tests/reference_inter_decode.rs).
        let ec = true;
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

            assert_eq!(
                decoded.keyframe, fixture.keyframes[i],
                "{name} frame {i}: §4.4 keyframe flag"
            );
            let raw = pack_reference_raw(&decoded, rgb, record.bits_per_raw_sample);
            assert_eq!(
                raw.len(),
                fixture.frame_raw_len,
                "{name} frame {i}: raw byte length in {} layout",
                fixture.pix_fmt
            );
            assert_eq!(
                sha256_hex(&raw),
                fixture.frame_sha256[i],
                "{name} frame {i}: decoded bytes diverged from the reference decoder"
            );

            if name.starts_with("nonuni-") {
                assert_nonuniform_grid(name, &record, &decoded);
            }
        }
        if name.starts_with("nonuni-") {
            // The coded grid must be the multi-slice one the generation
            // command requested — a silently 1×1 stream would gut the
            // §4.8 coverage.
            let requested = fixture.requested_slices.expect("nonuni requests slices");
            let (num_h, num_v) = (record.num_h_slices.unwrap(), record.num_v_slices.unwrap());
            assert_eq!(num_h * num_v, requested, "{name}: coded slice count");
            // The name's `-NxM` suffix documents the coded grid.
            let suffix = name.rsplit('-').next().unwrap();
            assert_eq!(
                suffix,
                format!("{num_h}x{num_v}"),
                "{name}: name suffix matches the coded §4.2.11/§4.2.12 grid"
            );
            nonuni_seen += 1;
        }
    }
    assert_eq!(nonuni_seen, 8, "non-uniform fixture count");
}

/// `(fixture name, mapped surface, significant-bits record, low-byte
/// repack)` for the framework-trait decode. The 8-bit RGB streams
/// decode onto the native one-byte `Gbrp8` (core 0.1.33) — emitted
/// planes are already the reference raw layout, so no `low_byte`
/// repack and no significant-bits record (retired r420 `Gbrp10Le`
/// surface detour; the pinned hashes are unchanged).
const TRAIT_CASES: &[(&str, PixelFormat, Option<&[u8]>, bool)] = &[
    (
        "nonuni-v3-yuv420p8-range-3x2",
        PixelFormat::Yuv420P,
        None,
        false,
    ),
    (
        "nonuni-v3-yuv420p8-golomb-3x2",
        PixelFormat::Yuv420P,
        None,
        false,
    ),
    (
        "nonuni-v3-yuv422p10-range-3x3",
        PixelFormat::Yuv422P10Le,
        None,
        false,
    ),
    (
        "nonuni-v3-yuv444p16-range-2x2",
        PixelFormat::Yuv444P16Le,
        None,
        false,
    ),
    (
        "nonuni-v3-gray16-range-2x2",
        PixelFormat::Gray16Le,
        None,
        false,
    ),
    ("nonuni-v3-rgb8-range-2x2", PixelFormat::Gbrp8, None, false),
    ("nonuni-v3-rgb8-golomb-2x2", PixelFormat::Gbrp8, None, false),
    (
        "nonuni-v3-yuva420p8-range-3x2",
        PixelFormat::Yuva420P,
        None,
        false,
    ),
    (
        "deep-v3-yuva422p10-range",
        PixelFormat::Yuva422P10Le,
        None,
        false,
    ),
    (
        "deep-v3-yuva444p12-range",
        PixelFormat::Yuva444P12Le,
        None,
        false,
    ),
    (
        "deep-v3-yuva444p16-range",
        PixelFormat::Yuva444P16Le,
        None,
        false,
    ),
    (
        "deep-v3-yuv444p14-range",
        PixelFormat::Yuv444P16Le,
        Some(&[14, 14, 14]),
        false,
    ),
    (
        "deep-v3-yuv420p9-range",
        PixelFormat::Yuv420P10Le,
        Some(&[9, 9, 9]),
        false,
    ),
];

/// Every fixture also decodes bit-exactly through the framework
/// `oxideav_core::Decoder`, with the §4.2 mapping surface and the
/// attached significant-bits record asserted per stream — the deep /
/// non-uniform corpus running end-to-end on the trait wiring.
#[test]
fn corpus_decodes_bit_exact_through_the_trait() {
    assert_eq!(
        TRAIT_CASES.len(),
        fx::FIXTURES.len(),
        "every fixture has a trait-decode case"
    );
    for &(name, want_pf, want_sig, low_byte) in TRAIT_CASES {
        let fixture = fx::FIXTURES
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("{name}: fixture present"));

        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(fixture.width);
        params.height = Some(fixture.height);
        params.extradata = fixture.extradata.to_vec();

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
            let Frame::Video(out) = dec
                .receive_frame()
                .unwrap_or_else(|e| panic!("{name} frame {i}: decode: {e:?}"))
            else {
                panic!("{name} frame {i}: expected video");
            };

            assert_eq!(
                out.significant_bits(),
                want_sig,
                "{name} frame {i}: significant-bits record for {want_pf:?}"
            );
            let mut raw = Vec::with_capacity(fixture.frame_raw_len);
            for plane in out.image_planes() {
                if low_byte {
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
