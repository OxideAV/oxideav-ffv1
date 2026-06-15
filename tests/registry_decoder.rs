//! Round 317 — end-to-end bit-exact decode through the
//! [`oxideav_core::Decoder`] trait surface, driven entirely through the
//! registry (`register` → `CodecRegistry::first_decoder` →
//! `Decoder::send_packet` / `receive_frame`).
//!
//! This locks in the framework-integration wiring:
//!
//! * `register` installs an `ffv1` decoder factory plus the two
//!   RFC 9043 §4.3.3 container tags (AVI / VfW FourCC `FFV1` §4.3.3.1,
//!   Matroska Codec ID `V_FFV1` §4.3.3.4).
//! * The decoder reads the §4.2 Configuration Record from
//!   `CodecParameters::extradata` (RFC 9043 §4.3.3) and the frame
//!   dimensions from `params.width` / `params.height`.
//! * A compressed Frame fed as one `Packet` decodes — through the trait
//!   — to a `VideoFrame` whose plane bytes match the reference decoder's
//!   `expected.raw` byte-for-byte.
//!
//! The fixture is `v3-default` (FFV1 v3, range coder, 8-bit YUV 4:2:0,
//! 2×2 slices, 128×96). Its frame bytes and reference output are the
//! same proven-bit-exact constants `tests/frame_driver.rs` validates
//! through the direct `decode_frame` API; here the path runs through the
//! `Decoder` trait instead.

use oxideav_core::{
    CodecId, CodecParameters, CodecTag, Frame, Packet, ProbeContext, RuntimeContext, TimeBase,
};
use oxideav_ffv1::{register, CODEC_ID_STR};

// v3-default Matroska CodecPrivate (the §4.2 Configuration Record).
// Same blob as `tests/fixture_v3_default.rs` / `tests/frame_driver.rs`.
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

// Whole-frame byte payload: concatenation of the four whole-Slice byte
// constants, the same bytes `tests/frame_driver.rs` walks.
#[allow(dead_code)]
mod shared_fixtures {
    include!("data/slice_footer_fixtures.rs");
    pub(super) const SLICE0: &[u8] = V3_DEFAULT_FULL_SLICE0;
    pub(super) const SLICE1: &[u8] = V3_DEFAULT_FULL_SLICE1;
    pub(super) const SLICE2: &[u8] = V3_DEFAULT_FULL_SLICE2;
    pub(super) const SLICE3: &[u8] = V3_DEFAULT_FULL_SLICE3;
}

// Inlined reference decoder output for v3-default (Y / Cb / Cr).
include!("data/v3_default_expected.rs");

fn v3_default_frame_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    for s in [
        shared_fixtures::SLICE0,
        shared_fixtures::SLICE1,
        shared_fixtures::SLICE2,
        shared_fixtures::SLICE3,
    ] {
        out.extend_from_slice(s);
    }
    out
}

fn v3_default_params() -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(128);
    params.height = Some(96);
    params.extradata = V3_DEFAULT_EXTRADATA.to_vec();
    params
}

/// The registry installs an `ffv1` decoder reachable by codec id, and
/// resolves both RFC 9043 §4.3.3 container tags to it.
#[test]
fn register_installs_decoder_and_container_tags() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    assert!(
        ctx.codecs.has_decoder(&CodecId::new(CODEC_ID_STR)),
        "register must install an ffv1 decoder factory"
    );

    let fourcc = CodecTag::fourcc(b"FFV1");
    assert_eq!(
        ctx.codecs
            .resolve_tag_ref(&ProbeContext::new(&fourcc))
            .map(|c| c.as_str()),
        Some(CODEC_ID_STR),
        "AVI FourCC FFV1 must resolve to ffv1 (RFC 9043 §4.3.3.1)"
    );

    let mkv = CodecTag::matroska("V_FFV1");
    assert_eq!(
        ctx.codecs
            .resolve_tag_ref(&ProbeContext::new(&mkv))
            .map(|c| c.as_str()),
        Some(CODEC_ID_STR),
        "Matroska V_FFV1 must resolve to ffv1 (RFC 9043 §4.3.3.4)"
    );
}

/// Decode the v3-default Frame through the `Decoder` trait (registry
/// factory → `send_packet` → `receive_frame`) and verify the resulting
/// `VideoFrame`'s three planes are bit-exact against the reference
/// `expected.raw`.
#[test]
fn registry_decode_is_bit_exact_against_expected_raw() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let params = v3_default_params();
    let mut dec = ctx
        .codecs
        .first_decoder(&params)
        .expect("registry builds an ffv1 decoder from the configuration record");

    assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);

    let pkt = Packet::new(0, TimeBase::new(1, 1), v3_default_frame_bytes());
    dec.send_packet(&pkt)
        .expect("send_packet accepts the frame");
    let frame = dec
        .receive_frame()
        .expect("receive_frame decodes the frame");

    let video = match frame {
        Frame::Video(v) => v,
        other => panic!("expected a video frame, got {other:?}"),
    };

    // YUV 4:2:0: Y (128×96), Cb (64×48), Cr (64×48), 8-bit → one byte
    // per Sample, tight stride.
    assert_eq!(video.planes.len(), 3, "Y + Cb + Cr");
    assert_eq!(video.planes[0].stride, 128);
    assert_eq!(video.planes[1].stride, 64);
    assert_eq!(video.planes[2].stride, 64);

    assert_eq!(
        video.planes[0].data, V3_DEFAULT_EXPECTED_Y,
        "Y plane must be bit-exact against expected.raw"
    );
    assert_eq!(
        video.planes[1].data, V3_DEFAULT_EXPECTED_CB,
        "Cb plane must be bit-exact against expected.raw"
    );
    assert_eq!(
        video.planes[2].data, V3_DEFAULT_EXPECTED_CR,
        "Cr plane must be bit-exact against expected.raw"
    );
}

/// `receive_frame` before any `send_packet` reports `NeedMore`, and
/// after `flush` with nothing pending reports `Eof` — the standard
/// `Decoder` drain contract.
#[test]
fn decoder_drain_contract() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut dec = ctx.codecs.first_decoder(&v3_default_params()).unwrap();

    assert!(matches!(
        dec.receive_frame(),
        Err(oxideav_core::Error::NeedMore)
    ));

    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame(), Err(oxideav_core::Error::Eof)));
}

/// A decoder built from parameters with no extradata is unconfigured;
/// `receive_frame` surfaces a diagnosable error rather than decoding
/// garbage.
#[test]
fn unconfigured_decoder_errors_cleanly() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let bare = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = ctx.codecs.first_decoder(&bare).unwrap();
    let pkt = Packet::new(0, TimeBase::new(1, 1), vec![0u8; 8]);
    dec.send_packet(&pkt).unwrap();
    assert!(dec.receive_frame().is_err());
}
