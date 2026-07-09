//! Decode the real `v1-golomb` reference fixture through the framework
//! [`oxideav_core::Decoder`] trait (the empty-extradata v0/v1 route).
//!
//! `tests/reference_fixture_decode.rs` decodes this fixture through the
//! direct `decode_frame_v0v1` API. This complements it by driving the
//! identical real Golomb-coded FFV1 version-1 Frame through the full
//! framework surface: `CodecParameters` with dims but **empty**
//! `extradata` (RFC 9043 §4.3.3 / §4.4 — v0/v1 carry no Configuration
//! Record), a coded `Packet` through `send_packet` / `receive_frame`,
//! and the little-endian plane packing into a `VideoFrame`. The
//! recovered plane bytes must match the reference decoder's
//! `expected.yuv` (planar yuv420p) byte-for-byte.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, RuntimeContext, TimeBase};
use oxideav_ffv1::{register, CODEC_ID_STR};

// Only the `V1GR_*` fixtures are read here; the rest of the shared module
// is consumed by `reference_fixture_decode.rs`.
#[allow(dead_code)]
#[path = "data/reference_fixtures.rs"]
mod fx;

#[test]
fn registry_decodes_real_v1_golomb_fixture() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    // v0/v1 carriage shape: dims present, extradata empty.
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(64);
    params.height = Some(48);

    let mut dec = ctx
        .codecs
        .first_decoder(&params)
        .expect("registry builds a v0/v1 ffv1 decoder from dims alone");

    let pkt = Packet::new(0, TimeBase::new(1, 1), fx::V1GR_FRAME.to_vec());
    dec.send_packet(&pkt).expect("send_packet");
    let video = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };

    assert_eq!(video.planes.len(), 3, "YUV 4:2:0 — three planes");

    // Luma: 64×48, one byte per Sample, tight stride.
    let want_y: Vec<u8> = fx::V1GR_Y.iter().map(|&s| s as u8).collect();
    assert_eq!(video.planes[0].stride, 64, "Y stride");
    assert_eq!(video.planes[0].data, want_y, "Y plane bytes");

    // Chroma: 32×24 each.
    let want_u: Vec<u8> = fx::V1GR_U.iter().map(|&s| s as u8).collect();
    let want_v: Vec<u8> = fx::V1GR_V.iter().map(|&s| s as u8).collect();
    assert_eq!(video.planes[1].stride, 32, "Cb stride");
    assert_eq!(video.planes[1].data, want_u, "Cb plane bytes");
    assert_eq!(video.planes[2].stride, 32, "Cr stride");
    assert_eq!(video.planes[2].data, want_v, "Cr plane bytes");
}
