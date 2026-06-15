//! Framework-integration tests for the `oxideav-core` registry path.
//!
//! These exercise the round-317 wiring: the FFV1 decoder behind the
//! [`oxideav_core::Decoder`] trait, reachable through a populated
//! [`oxideav_core::RuntimeContext`] / [`oxideav_core::CodecRegistry`] by
//! the codec id `"ffv1"` and by the RFC 9043 §4.3.3 container tags
//! (the AVI FourCC `FFV1` and the Matroska Codec ID `V_FFV1`).
//!
//! The bit-exact assertion reuses the v3-default reference fixture
//! (`docs/video/ffv1/fixtures/v3-default/expected.raw`) already proven
//! byte-for-byte against the direct `decode_frame` entry point in
//! `frame_driver.rs`; here it travels the trait surface
//! (`send_packet` → `receive_frame` → `Frame::Video`) instead, so the
//! `CodecParameters` → Configuration Record plumbing and the
//! plane-to-`VideoFrame` byte mapping are covered end to end.

use oxideav_core::{
    CodecId, CodecParameters, CodecTag, Frame, Packet, ProbeContext, RuntimeContext, TimeBase,
};
use oxideav_ffv1::{register, CODEC_ID_STR};

// The v3-default Configuration Record + per-Slice payloads, inlined the
// same way `frame_driver.rs` does (the fixtures live in the workspace
// `docs/` submodule, not in this crate's standalone CI checkout).
#[allow(dead_code)]
mod shared_fixtures {
    include!("data/slice_footer_fixtures.rs");
    pub(super) const S0: &[u8] = V3_DEFAULT_FULL_SLICE0;
    pub(super) const S1: &[u8] = V3_DEFAULT_FULL_SLICE1;
    pub(super) const S2: &[u8] = V3_DEFAULT_FULL_SLICE2;
    pub(super) const S3: &[u8] = V3_DEFAULT_FULL_SLICE3;
}
use shared_fixtures::{S0, S1, S2, S3};

include!("data/v3_default_expected.rs");

const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

fn v3_default_frame_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    for s in [S0, S1, S2, S3] {
        out.extend_from_slice(s);
    }
    out
}

/// Build `CodecParameters` for the v3-default fixture exactly as a
/// container demuxer would: the Configuration Record in `extradata`,
/// the §4 frame dimensions in `width` / `height`, and the AVI FourCC
/// tag.
fn v3_default_params() -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(128);
    params.height = Some(96);
    params.extradata = V3_DEFAULT_EXTRADATA.to_vec();
    params.tag = Some(CodecTag::fourcc(b"FFV1"));
    params
}

#[test]
fn register_installs_ffv1_decoder() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let id = CodecId::new(CODEC_ID_STR);
    assert!(
        ctx.codecs.has_decoder(&id),
        "register() must install an FFV1 decoder factory"
    );
    // Encoder is intentionally not wired this round.
    assert!(
        !ctx.codecs.has_encoder(&id),
        "the encoder trait is a documented follow-up; no encoder should register yet"
    );
}

#[test]
fn registry_resolves_avi_and_matroska_tags() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let avi = CodecTag::fourcc(b"FFV1");
    assert_eq!(
        ctx.codecs
            .resolve_tag_ref(&ProbeContext::new(&avi))
            .map(|c| c.as_str()),
        Some("ffv1"),
        "AVI FourCC FFV1 (RFC 9043 §4.3.3.1) must resolve to the ffv1 codec id"
    );

    let mkv = CodecTag::matroska("V_FFV1");
    assert_eq!(
        ctx.codecs
            .resolve_tag_ref(&ProbeContext::new(&mkv))
            .map(|c| c.as_str()),
        Some("ffv1"),
        "Matroska Codec ID V_FFV1 (RFC 9043 §4.3.3.4) must resolve to the ffv1 codec id"
    );
}

#[test]
fn trait_decode_v3_default_is_bit_exact() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let params = v3_default_params();
    let mut decoder = ctx
        .codecs
        .first_decoder(&params)
        .expect("ffv1 decoder factory builds from v3-default params");

    let pkt = Packet::new(0, TimeBase::new(1, 25), v3_default_frame_bytes());
    decoder
        .send_packet(&pkt)
        .expect("send_packet accepts frame");
    let frame = decoder
        .receive_frame()
        .expect("receive_frame yields a frame");

    let video = match frame {
        Frame::Video(v) => v,
        other => panic!("expected a video frame, got {other:?}"),
    };

    // §4.7.1 primary_color_count == 3 for yuv420p (Y + Cb + Cr).
    assert_eq!(video.planes.len(), 3, "Y + Cb + Cr");

    // 8-bit fixture → one byte per Sample, stride == plane width.
    assert_eq!(video.planes[0].stride, 128, "Y stride == width");
    assert_eq!(video.planes[1].stride, 64, "Cb stride == width");
    assert_eq!(video.planes[2].stride, 64, "Cr stride == width");

    assert_eq!(
        video.planes[0].data, V3_DEFAULT_EXPECTED_Y,
        "Y plane bytes must be bit-exact against expected.raw via the trait path"
    );
    assert_eq!(
        video.planes[1].data, V3_DEFAULT_EXPECTED_CB,
        "Cb plane bytes must be bit-exact against expected.raw via the trait path"
    );
    assert_eq!(
        video.planes[2].data, V3_DEFAULT_EXPECTED_CR,
        "Cr plane bytes must be bit-exact against expected.raw via the trait path"
    );

    // pts threads through unchanged.
    assert_eq!(video.pts, pkt.pts);
}

#[test]
fn trait_decode_without_config_surfaces_error_not_panic() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    // No extradata / dimensions: the factory builds a deferred decoder
    // that must surface a diagnosable error at receive_frame time.
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut decoder = ctx
        .codecs
        .first_decoder(&params)
        .expect("factory still builds a deferred decoder");

    let pkt = Packet::new(0, TimeBase::new(1, 25), v3_default_frame_bytes());
    decoder
        .send_packet(&pkt)
        .expect("send_packet accepts frame");
    let err = decoder
        .receive_frame()
        .expect_err("an unconfigured decoder must error, not panic");
    let msg = format!("{err}");
    assert!(
        msg.contains("not configured"),
        "error should explain the missing Configuration Record / dimensions, got: {msg}"
    );
}

#[test]
fn need_more_before_first_packet() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let params = v3_default_params();
    let mut decoder = ctx.codecs.first_decoder(&params).unwrap();
    let err = decoder
        .receive_frame()
        .expect_err("no packet sent yet → NeedMore");
    assert!(matches!(err, oxideav_core::Error::NeedMore));
}
