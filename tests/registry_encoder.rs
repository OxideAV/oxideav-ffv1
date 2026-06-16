//! Round 324 — end-to-end encode through the [`oxideav_core::Encoder`]
//! trait surface, driven entirely through the registry
//! (`register` → `CodecRegistry::first_encoder` → `Encoder::send_frame`
//! / `receive_packet`), then a registry round-trip back through the
//! [`oxideav_core::Decoder`].
//!
//! This locks in the encode-side framework wiring the README named as
//! the headline follow-up: "deriving §4.6 Slice Headers from the
//! Configuration Record's slice grid". The encoder:
//!
//! * reads the §4.2 Configuration Record from
//!   `CodecParameters::extradata` (RFC 9043 §4.3.3) plus the frame
//!   dimensions, exactly like the decoder;
//! * derives the §4.6 Slice Header grid one cell per Slice from the
//!   Configuration Record's §4.2.11 / §4.2.12 `num_h_slices ×
//!   num_v_slices` (here a 2×2 grid);
//! * converts an incoming `Frame::Video` to the internal `DecodedFrame`
//!   (the inverse of the decoder's plane packing) and emits one coded
//!   keyframe per Frame (FFV1 is intra-only).
//!
//! The fixture is `v3-default` (FFV1 v3, range coder, 8-bit YUV 4:2:0,
//! 2×2 slices, 128×96): the same Configuration Record and reference
//! pixel buffers `tests/registry_decoder.rs` uses. A full encode →
//! decode round-trip must reproduce those pixels bit-for-bit.

use oxideav_core::{
    CodecId, CodecParameters, Frame, RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{register, CODEC_ID_STR};

// v3-default Matroska CodecPrivate (the §4.2 Configuration Record):
// FFV1 v3, range coder (coder_type 1), 8-bit YUV 4:2:0, 2×2 slices.
// Same blob as `tests/registry_decoder.rs` / `tests/frame_driver.rs`.
const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

// Reference decoder output for v3-default (Y / Cb / Cr) — the pixels we
// feed the encoder.
include!("data/v3_default_expected.rs");

fn v3_default_params() -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(128);
    params.height = Some(96);
    params.extradata = V3_DEFAULT_EXTRADATA.to_vec();
    params
}

/// Build the source `VideoFrame` from the reference pixel buffers: Y
/// (128×96), Cb (64×48), Cr (64×48), 8-bit → one byte per Sample, tight
/// stride.
fn v3_default_video_frame() -> VideoFrame {
    VideoFrame {
        pts: Some(7),
        planes: vec![
            VideoPlane {
                stride: 128,
                data: V3_DEFAULT_EXPECTED_Y.to_vec(),
            },
            VideoPlane {
                stride: 64,
                data: V3_DEFAULT_EXPECTED_CB.to_vec(),
            },
            VideoPlane {
                stride: 64,
                data: V3_DEFAULT_EXPECTED_CR.to_vec(),
            },
        ],
    }
}

/// The registry installs an `ffv1` encoder reachable by codec id, and
/// `Encoder::output_params` reports the configured stream.
#[test]
fn register_installs_encoder() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    assert!(
        ctx.codecs.has_encoder(&CodecId::new(CODEC_ID_STR)),
        "register must install an ffv1 encoder factory"
    );

    let enc = ctx
        .codecs
        .first_encoder(&v3_default_params())
        .expect("registry builds an ffv1 encoder from the configuration record");
    assert_eq!(enc.codec_id().as_str(), CODEC_ID_STR);
    assert_eq!(enc.output_params().width, Some(128));
    assert_eq!(enc.output_params().height, Some(96));
    assert_eq!(
        enc.output_params().extradata,
        V3_DEFAULT_EXTRADATA,
        "output_params carries the §4.2 Configuration Record for the muxer"
    );
}

/// Encode the v3-default pixels through the `Encoder` trait, then decode
/// the emitted packet back through the `Decoder` trait: the round-trip
/// must reproduce the original pixels bit-for-bit, proving the encoder's
/// 2×2 slice-grid derivation and `VideoFrame` → `DecodedFrame`
/// conversion are correct.
#[test]
fn encode_then_decode_round_trips_pixels_bit_exact() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let params = v3_default_params();

    // ---- encode ----
    let mut enc = ctx
        .codecs
        .first_encoder(&params)
        .expect("registry builds an ffv1 encoder");
    let frame = Frame::Video(v3_default_video_frame());
    enc.send_frame(&frame)
        .expect("send_frame encodes the frame");
    let pkt = enc
        .receive_packet()
        .expect("receive_packet yields the coded keyframe");
    assert!(pkt.is_keyframe(), "every FFV1 coded Frame is a keyframe");
    assert_eq!(pkt.pts, Some(7), "the input pts propagates to the packet");
    assert!(
        !pkt.data.is_empty(),
        "the coded payload must carry the Slice byte stream"
    );

    // No more packets queued (FFV1 is intra-only, one packet per frame).
    enc.flush().unwrap();
    assert!(matches!(
        enc.receive_packet(),
        Err(oxideav_core::Error::NeedMore)
    ));

    // ---- decode the encoder's output back ----
    let mut dec = ctx
        .codecs
        .first_decoder(&params)
        .expect("registry builds an ffv1 decoder");
    let in_pkt = oxideav_core::Packet::new(0, TimeBase::new(1, 1), pkt.data.clone());
    dec.send_packet(&in_pkt).unwrap();
    let decoded = match dec.receive_frame().expect("decode succeeds") {
        Frame::Video(v) => v,
        other => panic!("expected a video frame, got {other:?}"),
    };

    assert_eq!(decoded.planes.len(), 3, "Y + Cb + Cr");
    assert_eq!(
        decoded.planes[0].data, V3_DEFAULT_EXPECTED_Y,
        "Y plane survives the encode → decode round-trip bit-exactly"
    );
    assert_eq!(
        decoded.planes[1].data, V3_DEFAULT_EXPECTED_CB,
        "Cb plane survives the round-trip bit-exactly"
    );
    assert_eq!(
        decoded.planes[2].data, V3_DEFAULT_EXPECTED_CR,
        "Cr plane survives the round-trip bit-exactly"
    );
}

/// `receive_packet` before any `send_frame` reports `NeedMore` — the
/// standard `Encoder` drain contract.
#[test]
fn encoder_drain_contract() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx.codecs.first_encoder(&v3_default_params()).unwrap();
    assert!(matches!(
        enc.receive_packet(),
        Err(oxideav_core::Error::NeedMore)
    ));
    enc.flush().unwrap();
    assert!(matches!(
        enc.receive_packet(),
        Err(oxideav_core::Error::NeedMore)
    ));
}

/// An encoder built from parameters with no extradata is unconfigured;
/// `send_frame` surfaces a diagnosable error rather than producing
/// garbage.
#[test]
fn unconfigured_encoder_errors_cleanly() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let bare = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut enc = ctx.codecs.first_encoder(&bare).unwrap();
    let frame = Frame::Video(v3_default_video_frame());
    assert!(enc.send_frame(&frame).is_err());
}

/// A `Frame::Video` whose plane count disagrees with the Configuration
/// Record's `primary_color_count` is rejected with a diagnosable error
/// (the conversion guards the plane shape before encoding).
#[test]
fn wrong_plane_count_errors_cleanly() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx.codecs.first_encoder(&v3_default_params()).unwrap();
    // Only the Y plane — the 4:2:0 record expects Y + Cb + Cr.
    let frame = Frame::Video(VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: 128,
            data: V3_DEFAULT_EXPECTED_Y.to_vec(),
        }],
    });
    let err = enc
        .send_frame(&frame)
        .expect_err("a single-plane frame cannot satisfy a 3-plane record");
    assert!(matches!(err, oxideav_core::Error::InvalidData(_)));
}
