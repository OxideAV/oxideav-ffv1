//! Round 434 — the staged corpus C under `docs/video/ffv1/fixtures/`
//! (13 reference-encoded streams: 8 non-uniform §4.8 floor-division
//! slice grids on odd frame dimensions + 5 deep-colour pins), decoded
//! straight from the staged bytes.
//!
//! Unlike the inlined corpora (`tests/data/*_fixtures.rs`) these tests
//! read the staged `input.mkv` / `expected.raw` files at run time and
//! pin them by full SHA-256 + byte count, so no stream bytes are copied
//! into this repository. The suite is **gated on docs presence**: when
//! the workspace `docs/` checkout is absent (e.g. the standalone-crate
//! CI), every test states so and passes vacuously; when the corpus is
//! present, any drift from the pinned hashes or any decode divergence
//! fails.
//!
//! The coded Frames are extracted from `input.mkv` with a minimal
//! container-layer walk (EBML element tree → track private data +
//! block payloads + keyframe flags). Container parsing is independent
//! of the FFV1 bitstream — the same separation as the RIFF/AVI chunk
//! walk used when the earlier corpora were inlined.
//!
//! Every stream is a §4.4 keyframe plus one carried non-keyframe, so
//! the §3.8.1.3 / §3.8.2.5 inter-Frame coder-state carry runs on all
//! 13. Both the direct `decode_frame*_with_carry` API (under
//! [`DecodeOptions::pedantic`]) and the framework
//! [`oxideav_core::Decoder`] trait (asserting the §4.2 mapping surface
//! and the attached significant-bits record) are driven over every
//! stream, and — for the `nonuniform-*` streams — the recomputed
//! §4.8.2 / §4.8.3 slice-pixel geometry must tile the raster exactly
//! and be non-uniform on exactly the axes the divisibility arithmetic
//! dictates (six of the eight are genuinely non-uniform; the
//! `3x3`-over-`99x75` pair divides evenly — see
//! `assert_grid_geometry`).

use std::path::{Path, PathBuf};

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, RuntimeContext, TimeBase,
};
use oxideav_ffv1::{
    decode_frame_rgb_with_carry, decode_frame_with_carry, parse_quantization_table_sets,
    pixel_format_mapping_for, register, ColorspaceType, DecodeOptions, DecodedFrame,
    Ffv1ConfigurationRecord, Ffv1DecodeSession, Ffv1SliceHeader, FramePixelDimensions,
    CODEC_ID_STR,
};

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

// ─────────────────── container-layer EBML / Matroska walk ───────────────────
//
// Minimal, read-only, and specific to what a conformance test needs:
// the video track's private data (the §4.3.3 Configuration Record),
// the coded Frame payload of every block, and each block's keyframe
// flag. No timestamps, lacing, cues or seeking — the staged streams
// are single-track, one Frame per block, no lacing (asserted).

struct MkvStream {
    codec_private: Vec<u8>,
    /// `(coded Frame bytes, container keyframe flag)` in stored order.
    frames: Vec<(Vec<u8>, bool)>,
}

/// Read an EBML element ID (1..=4 bytes, marker bits kept).
fn read_id(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    let b0 = *data.get(*pos).ok_or("EOF at element ID")?;
    let len = b0.leading_zeros() as usize + 1;
    if len > 4 {
        return Err(format!("bad ID lead byte {b0:#04x} at {pos}"));
    }
    let mut id = 0u32;
    for i in 0..len {
        id = (id << 8) | u32::from(*data.get(*pos + i).ok_or("EOF inside element ID")?);
    }
    *pos += len;
    Ok(id)
}

/// Read an EBML data-size field (1..=8 bytes, marker removed).
/// `None` = the reserved all-ones "unknown size".
fn read_size(data: &[u8], pos: &mut usize) -> Result<Option<u64>, String> {
    let b0 = *data.get(*pos).ok_or("EOF at element size")?;
    let len = b0.leading_zeros() as usize + 1;
    if len > 8 {
        return Err(format!("bad size lead byte {b0:#04x} at {pos}"));
    }
    let mut v = u64::from(b0) & (0xffu64 >> len);
    for i in 1..len {
        v = (v << 8) | u64::from(*data.get(*pos + i).ok_or("EOF inside element size")?);
    }
    *pos += len;
    let unknown = v == (1u64 << (7 * len)) - 1;
    Ok(if unknown { None } else { Some(v) })
}

/// Read the element header and return the payload slice, requiring a
/// known size (everything below Segment level in the staged files has
/// one).
fn read_child<'a>(data: &'a [u8], pos: &mut usize) -> Result<(u32, &'a [u8]), String> {
    let id = read_id(data, pos)?;
    let size =
        read_size(data, pos)?.ok_or_else(|| format!("unexpected unknown-size element {id:#x}"))?;
    let end = pos
        .checked_add(size as usize)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| format!("element {id:#x} overruns the file"))?;
    let payload = &data[*pos..end];
    *pos = end;
    Ok((id, payload))
}

/// Split a block payload (SimpleBlock or Block — same layout) into the
/// coded Frame, asserting the no-lacing shape, and return the
/// SimpleBlock keyframe flag bit.
fn parse_block(payload: &[u8]) -> Result<(Vec<u8>, bool), String> {
    let mut pos = 0usize;
    // Track number is an EBML variable-width integer; the staged files
    // have one track, so one byte, but decode generally.
    let b0 = *payload.first().ok_or("empty block")?;
    let tn_len = b0.leading_zeros() as usize + 1;
    if tn_len > 8 || payload.len() < tn_len + 3 {
        return Err("short block header".into());
    }
    pos += tn_len; // track number value unused beyond skipping
    pos += 2; // relative timestamp (s16)
    let flags = payload[pos];
    pos += 1;
    if (flags >> 1) & 0x3 != 0 {
        return Err(format!("unexpected lacing (flags {flags:#04x})"));
    }
    Ok((payload[pos..].to_vec(), flags & 0x80 != 0))
}

fn parse_mkv(data: &[u8]) -> Result<MkvStream, String> {
    let mut pos = 0usize;
    let mut codec_private = None;
    let mut frames = Vec::new();
    while pos < data.len() {
        let id = read_id(data, &mut pos)?;
        let size = read_size(data, &mut pos)?;
        let end = match size {
            Some(s) => pos
                .checked_add(s as usize)
                .filter(|&e| e <= data.len())
                .ok_or_else(|| format!("top-level element {id:#x} overruns the file"))?,
            // An unknown-size Segment extends to end of file.
            None if id == 0x1853_8067 => data.len(),
            None => return Err(format!("unknown-size non-Segment element {id:#x}")),
        };
        if id == 0x1853_8067 {
            // Segment: walk its children in place.
            while pos < end {
                let (cid, payload) = read_child(data, &mut pos)?;
                match cid {
                    // Tracks
                    0x1654_AE6B => {
                        let mut tp = 0usize;
                        while tp < payload.len() {
                            let (tid, entry) = read_child(payload, &mut tp)?;
                            if tid != 0xAE {
                                continue; // Void / CRC-32
                            }
                            let (mut ep, mut codec_id, mut private) = (0usize, None, None);
                            while ep < entry.len() {
                                let (eid, v) = read_child(entry, &mut ep)?;
                                match eid {
                                    0x86 => codec_id = Some(v.to_vec()),
                                    0x63A2 => private = Some(v.to_vec()),
                                    _ => {}
                                }
                            }
                            match codec_id.as_deref() {
                                // Native tag: private data IS the §4.3.3
                                // Configuration Record.
                                Some(b"V_FFV1") => codec_private = private,
                                // VfW-compat tag (§4.3.3.1 FourCC route):
                                // a 40-byte BITMAPINFOHEADER whose
                                // biCompression (bytes 16..20) is `FFV1`,
                                // followed by the Configuration Record.
                                Some(b"V_MS/VFW/FOURCC") => {
                                    let p = private.ok_or("VfW track without private data")?;
                                    if p.len() < 40 || &p[16..20] != b"FFV1" {
                                        return Err("VfW private data is not FFV1".into());
                                    }
                                    codec_private = Some(p[40..].to_vec());
                                }
                                _ => {}
                            }
                        }
                    }
                    // Cluster
                    0x1F43_B675 => {
                        let mut cp = 0usize;
                        while cp < payload.len() {
                            let (bid, block) = read_child(payload, &mut cp)?;
                            match bid {
                                // SimpleBlock: keyframe flag in the block header.
                                0xA3 => frames.push(parse_block(block)?),
                                // BlockGroup: keyframe ⇔ no ReferenceBlock child.
                                0xA0 => {
                                    let (mut gp, mut fr, mut has_ref) = (0usize, None, false);
                                    while gp < block.len() {
                                        let (gid, v) = read_child(block, &mut gp)?;
                                        match gid {
                                            0xA1 => fr = Some(parse_block(v)?.0),
                                            0xFB => has_ref = true,
                                            _ => {}
                                        }
                                    }
                                    let fr = fr.ok_or("BlockGroup without Block")?;
                                    frames.push((fr, !has_ref));
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {} // SeekHead / Info / Cues / Tags / Void / …
                }
            }
        }
        pos = end;
    }
    Ok(MkvStream {
        codec_private: codec_private.ok_or("no V_FFV1 track with private data")?,
        frames,
    })
}

// ────────────────────────────── fixture pins ──────────────────────────────

struct StagedFixture {
    dir: &'static str,
    /// Raw layout `expected.raw` was emitted in (planar; two
    /// little-endian bytes per Sample above 8 bits; RGB planar as
    /// G, B, R (, A)).
    pix_fmt: &'static str,
    width: u32,
    height: u32,
    /// §4.4 keyframe flags in stored order (all corpus-C streams:
    /// keyframe + one carried non-keyframe).
    keyframes: &'static [bool],
    /// The coded §4.2.11 × §4.2.12 grid (`None` for the deep pins'
    /// uniform 2×2 grids — still asserted, just not required to be
    /// non-uniform).
    nonuniform_grid: Option<(u32, u32)>,
    input_len: usize,
    input_sha256: &'static str,
    expected_len: usize,
    expected_sha256: &'static str,
    /// Trait-surface expectations: mapped `PixelFormat` + attached
    /// significant-bits record.
    surface: PixelFormat,
    significant_bits: Option<&'static [u8]>,
    /// `(ec, intra)` as the §4.2.14/§4.2.16/§4.2.17 record TAIL of this
    /// stream's Configuration Record parses under the RFC 9043
    /// Figure 28 layout. Every corpus-C record is a TWO-SET record
    /// (`quant_table_set_count == 2`), the shape whose reference-writer
    /// tail does not reliably read back under Figure 28 (the r416
    /// interop finding, `tests/reference_inter_decode.rs`): the ground
    /// truth for every stream is `ec == 1` (all were generated with
    /// `-slicecrc 1`) and `intra == 0` (all carry a non-keyframe), yet
    /// three of the thirteen tails parse to misdeclared values —
    /// `deep-yuva420p9-range` even to the non-physical `ec == 2`.
    /// Pinned so a parser or corpus change that shifts the observed
    /// tails surfaces here.
    parsed_tail: (u32, bool),
}

const KEY_NONKEY: &[bool] = &[true, false];

const FIXTURES: &[StagedFixture] = &[
    StagedFixture {
        dir: "deep-gbrap14-range",
        pix_fmt: "gbrap14le",
        width: 64,
        height: 48,
        keyframes: KEY_NONKEY,
        nonuniform_grid: None,
        input_len: 5975,
        input_sha256: "ed47bc54c07fa3fe78c4c34e6cc98c885f5706a3c880dac664fd129ca03db54f",
        expected_len: 49152,
        expected_sha256: "fd733cdab058b77150fb754484094b95bfdfc96486d0011da74aa312504d4b6e",
        surface: PixelFormat::Gbrap14Le,
        significant_bits: None,
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "deep-gray14-range",
        pix_fmt: "gray14le",
        width: 64,
        height: 48,
        keyframes: KEY_NONKEY,
        nonuniform_grid: None,
        input_len: 1863,
        input_sha256: "1a8635cfff951675a468980cc5a2b41f0dc9f3ca1bc95e5768d8f2605e68563e",
        expected_len: 12288,
        expected_sha256: "0147767e62dbc49de7f9a85ffe86ea010e1f73d9a75abe7ff82eb3f7a6b3179d",
        surface: PixelFormat::Gray16Le,
        significant_bits: Some(&[14]),
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "deep-yuv444p14-range",
        pix_fmt: "yuv444p14le",
        width: 64,
        height: 48,
        keyframes: KEY_NONKEY,
        nonuniform_grid: None,
        input_len: 11586,
        input_sha256: "960cd38743982fc19aa3f76db53bef81c267d2078dfff69a4124aba4fa765fe3",
        expected_len: 36864,
        expected_sha256: "60ba2c2232a61f60716619fc76adb98476a2ca0ff2a57b556d08bb043f4561e7",
        surface: PixelFormat::Yuv444P16Le,
        significant_bits: Some(&[14, 14, 14]),
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "deep-yuva420p9-range",
        pix_fmt: "yuva420p9le",
        width: 64,
        height: 48,
        keyframes: KEY_NONKEY,
        nonuniform_grid: None,
        input_len: 3564,
        input_sha256: "f7294f274d5f8579ea5f2ce98591439ea359319a19e6684995d75ca6d2b8a457",
        expected_len: 30720,
        expected_sha256: "faeb53b4ea24b81a6c23ff1cdcfdbb9664c61866f4bf4341ab2cc22af21a7ff6",
        surface: PixelFormat::Yuva420P10Le,
        significant_bits: Some(&[9, 9, 9, 9]),
        parsed_tail: (2, true),
    },
    StagedFixture {
        dir: "deep-yuva444p9-range",
        pix_fmt: "yuva444p9le",
        width: 64,
        height: 48,
        keyframes: KEY_NONKEY,
        nonuniform_grid: None,
        input_len: 3828,
        input_sha256: "3951338ac57232c1013922e64b5d08bf32bdf35ecf12c3bc8efc10a3a57f9a2a",
        expected_len: 49152,
        expected_sha256: "6f15a9cb5e8af3a606b7cd189d53e145f68571faf4b52e2a6c0fd594c5216ca8",
        surface: PixelFormat::Yuva444P10Le,
        significant_bits: Some(&[9, 9, 9, 9]),
        parsed_tail: (1, true),
    },
    StagedFixture {
        dir: "nonuniform-2x2-61x47-gray-golomb",
        pix_fmt: "gray",
        width: 61,
        height: 47,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((2, 2)),
        input_len: 1894,
        input_sha256: "349b25cca57fc02d5c9a8dd078e1be1cc556a6010da4d70037700ee66307d86b",
        expected_len: 5734,
        expected_sha256: "8a65f14b158c3824faa3372d28b678327f623260923c73aa611b9110a6b2bb50",
        surface: PixelFormat::Gray8,
        significant_bits: None,
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "nonuniform-2x2-61x47-yuv444p-range",
        pix_fmt: "yuv444p",
        width: 61,
        height: 47,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((2, 2)),
        input_len: 5368,
        input_sha256: "17dfacb7242ca04155c0ec69f3eabcfeb87ce2883142deb9db27fcdacc543f6e",
        expected_len: 17202,
        expected_sha256: "8fe4cbf0904b145ab9d77869bea15d9e740513123d661f803b976739fcb3d02d",
        surface: PixelFormat::Yuv444P,
        significant_bits: None,
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "nonuniform-3x2-97x65-yuv444p-golomb",
        pix_fmt: "yuv444p",
        width: 97,
        height: 65,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((3, 2)),
        input_len: 11020,
        input_sha256: "1d497cfcbc0b73a0ca624fb11615f2b1fa531013b9439c55170da6eb66189955",
        expected_len: 37830,
        expected_sha256: "0c1f152eb375b30859e7f91cedce2bb7914d008e828ec10e94fcbed78d667504",
        surface: PixelFormat::Yuv444P,
        significant_bits: None,
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "nonuniform-3x2-97x65-yuva444p-range",
        pix_fmt: "yuva444p",
        width: 97,
        height: 65,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((3, 2)),
        input_len: 6280,
        input_sha256: "1b5279cdd4b30b2518daaa7a68f636295c16842538a0ef2e35a8b1fe110a1e74",
        expected_len: 50440,
        expected_sha256: "46d433f3603ebedb8949d8aede9f946c10a8ee39d55e0e10536e7ee1b46618dc",
        surface: PixelFormat::Yuva444P,
        significant_bits: None,
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "nonuniform-3x2-99x75-gray16-range",
        pix_fmt: "gray16le",
        width: 99,
        height: 75,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((3, 2)),
        input_len: 5238,
        input_sha256: "1d91d1b06fed30ef8cc8c744dfcd9ba25a2eb2b2812bead1f04b6a050df54ec3",
        expected_len: 29700,
        expected_sha256: "f30590cc5b829291d371ed24ea99b8466178138617950a7e57f0a3aa8c28a700",
        surface: PixelFormat::Gray16Le,
        significant_bits: None,
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "nonuniform-3x3-97x65-yuv444p16-range",
        pix_fmt: "yuv444p16le",
        width: 97,
        height: 65,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((3, 3)),
        input_len: 31986,
        input_sha256: "c5e6ee9b0c9d211f86041319f4dbd08657c857e3e53a4e24944a2f6ca88a72bd",
        expected_len: 75660,
        expected_sha256: "157f62dbff444e74f4c0bb5414dfe9f38ffe3f660c34d2d56e3b2097b01b8204",
        surface: PixelFormat::Yuv444P16Le,
        significant_bits: None,
        parsed_tail: (1, true),
    },
    StagedFixture {
        dir: "nonuniform-3x3-99x75-rgb-bgr0-range",
        pix_fmt: "gbrp",
        width: 99,
        height: 75,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((3, 3)),
        input_len: 6979,
        input_sha256: "ca28a37853cd6979ca521be56299d1de2c3c69e839137b272c178c677fdca219",
        expected_len: 44550,
        expected_sha256: "8542c6e32b6598be3989404317e2d3c594fd901aa745c43a749872475c6ffb0e",
        surface: PixelFormat::Gbrp8,
        significant_bits: None,
        parsed_tail: (1, false),
    },
    StagedFixture {
        dir: "nonuniform-3x3-99x75-yuv444p10-range",
        pix_fmt: "yuv444p10le",
        width: 99,
        height: 75,
        keyframes: KEY_NONKEY,
        nonuniform_grid: Some((3, 3)),
        input_len: 17841,
        input_sha256: "e00e03405c19e96267fd4e6122e97f8b2c18e001fba6fb7c97d6c7bbdf23a89a",
        expected_len: 89100,
        expected_sha256: "fe86e7a5a56bfb61b61dd22b474fce4f23e3d94f56fc73c165921dbdcabe674c",
        surface: PixelFormat::Yuv444P10Le,
        significant_bits: None,
        parsed_tail: (1, false),
    },
];

/// Locate the staged corpus. `None` (with a note on stderr) when this
/// checkout has no workspace `docs/` sibling — the standalone-crate CI.
fn corpus_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/video/ffv1/fixtures");
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!(
            "staged_corpus_c: docs/video/ffv1/fixtures/ not present in this \
             checkout — staged-corpus conformance gated off"
        );
        None
    }
}

fn load(fixture: &StagedFixture, dir: &Path) -> (MkvStream, Vec<u8>) {
    let name = fixture.dir;
    let input = std::fs::read(dir.join(name).join("input.mkv"))
        .unwrap_or_else(|e| panic!("{name}: read input.mkv: {e}"));
    assert_eq!(input.len(), fixture.input_len, "{name}: input.mkv size pin");
    assert_eq!(
        sha256_hex(&input),
        fixture.input_sha256,
        "{name}: input.mkv SHA-256 pin"
    );
    let expected = std::fs::read(dir.join(name).join("expected.raw"))
        .unwrap_or_else(|e| panic!("{name}: read expected.raw: {e}"));
    assert_eq!(
        expected.len(),
        fixture.expected_len,
        "{name}: expected.raw size pin"
    );
    assert_eq!(
        sha256_hex(&expected),
        fixture.expected_sha256,
        "{name}: expected.raw SHA-256 pin"
    );
    let stream = parse_mkv(&input).unwrap_or_else(|e| panic!("{name}: container walk: {e}"));
    assert_eq!(
        stream.frames.len(),
        fixture.keyframes.len(),
        "{name}: stored Frame count"
    );
    for (i, (_, kf)) in stream.frames.iter().enumerate() {
        assert_eq!(
            *kf, fixture.keyframes[i],
            "{name} frame {i}: container keyframe flag"
        );
    }
    (stream, expected)
}

/// Pack a decoded Frame into the reference raw layout: planar, one byte
/// per Sample at ≤ 8-bit depth, two little-endian bytes otherwise; RGB
/// Planes (recovered in R, G, B order per §3.7) reordered to the
/// G, B, R (, A) order the planar `gbrp` / `gbrap` layouts store.
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

/// The §4.8.2 / §4.8.3 slice-pixel geometry of one §4.6 Slice Header.
fn slice_pixel_geometry(
    header: &Ffv1SliceHeader,
    frame_w: u32,
    frame_h: u32,
    num_h: u32,
    num_v: u32,
) -> (u32, u32) {
    let (sx, sy) = (header.slice_x, header.slice_y);
    let (sw, sh) = (header.slice_width, header.slice_height);
    let px = (u64::from(sx) * u64::from(frame_w) / u64::from(num_h)) as u32;
    let py = (u64::from(sy) * u64::from(frame_h) / u64::from(num_v)) as u32;
    let pw = (u64::from(sx + sw) * u64::from(frame_w) / u64::from(num_h)) as u32 - px;
    let ph = (u64::from(sy + sh) * u64::from(frame_h) / u64::from(num_v)) as u32 - py;
    (pw, ph)
}

/// Assert the §4.8 grid tiles the raster exactly and is non-uniform on
/// exactly the axes the floor-division arithmetic makes non-uniform
/// (an axis yields unequal slice extents iff the slice count does not
/// divide the frame extent). Returns `true` when at least one axis is
/// genuinely non-uniform.
///
/// Corpus-C staging note: the two `nonuniform-3x3-99x75-*` fixtures are
/// actually UNIFORM — 99 = 3 × 33 and 75 = 3 × 25, so the 3×3 grid
/// divides evenly on both axes despite the odd frame dimensions the
/// fixture name advertises. The §4.8 floor divisions still run; they
/// just produce equal extents. The remaining six `nonuniform-*`
/// fixtures are non-uniform on at least one axis.
fn assert_grid_geometry(name: &str, cr: &Ffv1ConfigurationRecord, frame: &DecodedFrame) -> bool {
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
        let (pw, ph) = slice_pixel_geometry(sh, frame.width, frame.height, num_h, num_v);
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
    let want_w_nonuniform = frame.width % num_h != 0;
    let want_h_nonuniform = frame.height % num_v != 0;
    assert_eq!(
        widths.len() > 1,
        want_w_nonuniform,
        "{name}: grid {num_h}x{num_v} over {}x{} — horizontal §4.8 extents \
         {widths:?} vs the divisibility arithmetic",
        frame.width,
        frame.height,
    );
    assert_eq!(
        heights.len() > 1,
        want_h_nonuniform,
        "{name}: grid {num_h}x{num_v} over {}x{} — vertical §4.8 extents \
         {heights:?} vs the divisibility arithmetic",
        frame.width,
        frame.height,
    );
    want_w_nonuniform || want_h_nonuniform
}

/// Direct-API decode of every staged stream: bit-exact against the
/// staged `expected.raw` under `DecodeOptions::pedantic()`, §4.4
/// keyframe flags matching the container, and the §4.8 non-uniform
/// geometry asserted on every `nonuniform-*` Frame.
#[test]
fn staged_corpus_decodes_bit_exact() {
    let Some(dir) = corpus_dir() else { return };
    let mut nonuni_seen = 0;
    for fixture in FIXTURES {
        let name = fixture.dir;
        let (stream, expected) = load(fixture, &dir);
        let parsed = parse_quantization_table_sets(&stream.codec_private)
            .unwrap_or_else(|e| panic!("{name}: extradata: {e:?}"));
        let record = parsed.record;
        let rgb = record.colorspace_type == ColorspaceType::Rgb;
        let dims = FramePixelDimensions::new(fixture.width, fixture.height).expect("dims");
        if let Some((gh, gv)) = fixture.nonuniform_grid {
            assert_eq!(
                (record.num_h_slices, record.num_v_slices),
                (Some(gh), Some(gv)),
                "{name}: coded §4.2.11 × §4.2.12 grid"
            );
        }
        // Every corpus-C stream was generated with `-slicecrc 1`; the
        // record-tail `ec` parse on current reference-writer records is
        // unreliable (see tests/reference_inter_decode.rs).
        let ec = true;
        let mut carry = None;
        let mut nonuni_this = false;
        let frame_len = fixture.expected_len / fixture.keyframes.len();
        for (i, (pkt, _)) in stream.frames.iter().enumerate() {
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
                raw,
                expected[i * frame_len..(i + 1) * frame_len],
                "{name} frame {i}: decoded bytes diverged from the reference \
                 decoder ({} layout)",
                fixture.pix_fmt
            );
            if fixture.nonuniform_grid.is_some() && assert_grid_geometry(name, &record, &decoded) {
                nonuni_this = true;
            }
        }
        if nonuni_this {
            nonuni_seen += 1;
        }
    }
    // Six of the eight `nonuniform-*` fixtures are genuinely
    // non-uniform; the 3×3-over-99×75 pair divides evenly (see
    // `assert_grid_geometry`).
    assert_eq!(nonuni_seen, 6, "genuinely non-uniform fixture count");
    assert_eq!(FIXTURES.len(), 13, "corpus-C stream count");
}

/// Every staged stream also decodes bit-exactly through the framework
/// `oxideav_core::Decoder`, with the §4.2 mapping surface and the
/// attached significant-bits record asserted per stream.
#[test]
fn staged_corpus_decodes_bit_exact_through_the_trait() {
    let Some(dir) = corpus_dir() else { return };
    for fixture in FIXTURES {
        let name = fixture.dir;
        let (stream, expected) = load(fixture, &dir);
        let frame_len = fixture.expected_len / fixture.keyframes.len();

        // The §4.2 mapping surface this stream's record lands on.
        let record = parse_quantization_table_sets(&stream.codec_private)
            .unwrap_or_else(|e| panic!("{name}: extradata: {e:?}"))
            .record;
        let mapping = pixel_format_mapping_for(&record)
            .unwrap_or_else(|| panic!("{name}: layout maps to a surface"));
        assert_eq!(
            mapping.format, fixture.surface,
            "{name}: mapped §4.2 storage surface"
        );

        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(fixture.width);
        params.height = Some(fixture.height);
        params.extradata = stream.codec_private.clone();

        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut dec = ctx
            .codecs
            .first_decoder(&params)
            .unwrap_or_else(|e| panic!("{name}: decoder: {e:?}"));

        for (i, (pkt_bytes, kf)) in stream.frames.iter().enumerate() {
            let pkt = Packet::new(0, TimeBase::new(1, 25), pkt_bytes.clone()).with_keyframe(*kf);
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
                fixture.significant_bits,
                "{name} frame {i}: significant-bits record for {:?}",
                fixture.surface
            );
            let mut raw = Vec::with_capacity(frame_len);
            for plane in out.image_planes() {
                raw.extend_from_slice(&plane.data);
            }
            assert_eq!(
                raw,
                expected[i * frame_len..(i + 1) * frame_len],
                "{name} frame {i}: trait-surface bytes diverged from the \
                 reference decoder ({} layout)",
                fixture.pix_fmt
            );
        }
    }
}

/// The third decode surface: every staged stream through the stateful
/// [`Ffv1DecodeSession`] under `DecodeOptions::pedantic()`. Beyond the
/// per-Frame decode this applies the session's two stream-scope
/// conformance gates to reference bytes — the §5 third-paragraph
/// non-keyframe slice-geometry-stability walk (each corpus-C stream is
/// keyframe + non-keyframe over a multi-slice grid, so the stability
/// tracker compares real §4.6 header sets across Frames) and the
/// §4.2.17 `intra` gate.
///
/// **Record-tail handling.** Every corpus-C record is a TWO-SET record
/// (`quant_table_set_count == 2`) — the shape whose reference-writer
/// tail does not read back reliably under the RFC 9043 Figure 28
/// layout (the r416 interop finding pinned in
/// `tests/reference_inter_decode.rs`). The test first asserts each
/// record's parsed `(ec, intra)` tail equals the
/// [`StagedFixture::parsed_tail`] pin — three of the thirteen are
/// misdeclared, one of them with the non-physical `ec == 2` — and
/// then, like any container-level caller holding reliable stream
/// metadata, constructs the session from the ground-truth-corrected
/// record (`ec = 1`: every stream was generated with `-slicecrc 1`;
/// `intra = 0`: every stream carries a non-keyframe). With a truthful
/// tail the §4.2.17 gate and the §4.9 footer walk both hold on all 13
/// streams; feeding the three misdeclared tails uncorrected is
/// exactly what the registry `Decoder`'s first-frame `ec` retry (and
/// its untrusting treatment of `intra`) exists to absorb.
#[test]
fn staged_corpus_decodes_bit_exact_through_the_session() {
    let Some(dir) = corpus_dir() else { return };
    let mut misdeclared = 0;
    for fixture in FIXTURES {
        let name = fixture.dir;
        let (stream, expected) = load(fixture, &dir);
        let parsed = parse_quantization_table_sets(&stream.codec_private)
            .unwrap_or_else(|e| panic!("{name}: extradata: {e:?}"));
        let mut record = parsed.record;
        assert_eq!(
            record.quant_table_set_count,
            Some(2),
            "{name}: corpus-C records are two-set records"
        );
        assert_eq!(
            (record.ec.unwrap_or(0), record.intra.unwrap_or(false)),
            fixture.parsed_tail,
            "{name}: parsed §4.2.14/§4.2.16/§4.2.17 record tail"
        );
        if fixture.parsed_tail != (1, false) {
            misdeclared += 1;
        }
        // Ground truth (generation commands + stream contents): CRCs
        // present, non-keyframes present.
        record.ec = Some(1);
        record.intra = Some(false);
        let rgb = record.colorspace_type == ColorspaceType::Rgb;
        let bits = record.bits_per_raw_sample;
        let dims = FramePixelDimensions::new(fixture.width, fixture.height).expect("dims");
        let mut session = Ffv1DecodeSession::with_options(
            record,
            parsed.quant_table_sets,
            dims,
            true, // every corpus-C stream was generated with `-slicecrc 1`
            DecodeOptions::pedantic(),
        );
        let frame_len = fixture.expected_len / fixture.keyframes.len();
        for (i, (pkt, _)) in stream.frames.iter().enumerate() {
            let decoded = session
                .decode_next_frame(pkt)
                .unwrap_or_else(|e| panic!("{name} frame {i}: session decode: {e:?}"));
            assert_eq!(
                decoded.keyframe, fixture.keyframes[i],
                "{name} frame {i}: §4.4 keyframe flag"
            );
            let raw = pack_reference_raw(&decoded, rgb, bits);
            assert_eq!(
                raw,
                expected[i * frame_len..(i + 1) * frame_len],
                "{name} frame {i}: session-decoded bytes diverged from the \
                 reference decoder ({} layout)",
                fixture.pix_fmt
            );
        }
        assert_eq!(
            session.frames_observed(),
            fixture.keyframes.len() as u64,
            "{name}: session frame counter"
        );
    }
    // The reference-writer two-set tail misdeclaration count observed
    // on this corpus (r434): deep-yuva420p9 (ec=2 + intra), deep-yuva444p9
    // (intra), nonuniform-3x3-97x65-yuv444p16 (intra).
    assert_eq!(misdeclared, 3, "misdeclared two-set record tails");
}
