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
//!   (the inverse of the decoder's plane packing) and emits the stream's
//!   first Frame as a keyframe and every later Frame as a §4.4
//!   non-keyframe whose §3.8.1.3 / §3.8.2.5 per-context coder state
//!   continues from the previous Frame — unless the §4.2.17 `intra` flag
//!   forces keyframe-only output (round 338).
//!
//! The fixture is `v3-default` (FFV1 v3, range coder, 8-bit YUV 4:2:0,
//! 2×2 slices, 128×96): the same Configuration Record and reference
//! pixel buffers `tests/registry_decoder.rs` uses. A full encode →
//! decode round-trip must reproduce those pixels bit-for-bit.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{
    encode_configuration_record_with_quant_tables, parse_quantization_table_sets, pixel_format_for,
    register, CODEC_ID_STR,
};

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
    // The §4.2-derived pixel format is surfaced for a downstream muxer:
    // v3-default is 8-bit YUV 4:2:0 → Yuv420P.
    assert_eq!(
        enc.output_params().pixel_format,
        Some(PixelFormat::Yuv420P),
        "output_params advertises the §4.2-derived 8-bit YUV 4:2:0 format"
    );
}

/// `pixel_format_for` reads the §4.2 Parameters off the v3-default
/// Configuration Record (8-bit YUV 4:2:0) and the encoder propagates the
/// derived format onto `output_params`, leaving a caller-supplied format
/// in place only when the §4.2 layout has no exact framework variant.
#[test]
fn encoder_surfaces_section_4_2_pixel_format() {
    use oxideav_ffv1::parse_quantization_table_sets;

    // Direct derivation off the parsed Configuration Record.
    let parsed =
        parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).expect("v3-default extradata parses");
    assert_eq!(
        pixel_format_for(&parsed.record),
        Some(PixelFormat::Yuv420P),
        "v3-default Configuration Record maps to 8-bit YUV 4:2:0"
    );

    // A caller that pre-set a (wrong) pixel format is overridden by the
    // §4.2-derived one when an exact variant exists.
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut params = v3_default_params();
    params.pixel_format = Some(PixelFormat::Rgb24);
    let enc = ctx
        .codecs
        .first_encoder(&params)
        .expect("registry builds an ffv1 encoder");
    assert_eq!(
        enc.output_params().pixel_format,
        Some(PixelFormat::Yuv420P),
        "the §4.2-derived format takes precedence over a caller's guess"
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
    assert!(
        pkt.is_keyframe(),
        "the first coded Frame of a stream is always a keyframe"
    );
    assert_eq!(pkt.pts, Some(7), "the input pts propagates to the packet");
    assert!(
        !pkt.data.is_empty(),
        "the coded payload must carry the Slice byte stream"
    );

    // No more packets queued (one packet per frame, no reordering).
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

// -- multi-Frame inter (non-keyframe) round-trip through the trait -------
//
// The framework `Encoder` now emits the first Frame of a stream as a
// keyframe and every subsequent Frame as a §4.4 non-keyframe whose
// §3.8.1.3 / §3.8.2.5 per-context coder state continues from the prior
// Frame (unless the §4.2.17 `intra` flag forces keyframe-only output).
// The framework `Decoder` already carries that state across `receive_frame`
// calls, so a multi-Frame inter stream produced through the trait surface
// round-trips back through it bit-exactly — the encode-side end-to-end
// inter-Frame milestone.

/// Byte-rotate each plane by `shift` so successive Frames differ — a
/// distinct (but still lossless-codable) image per Frame, so the
/// inter-Frame coder state genuinely evolves Frame-to-Frame.
fn rotated_video_frame(shift: usize, pts: i64) -> VideoFrame {
    let rot = |src: &[u8]| -> Vec<u8> {
        let n = src.len();
        (0..n).map(|i| src[(i + shift) % n]).collect()
    };
    VideoFrame {
        pts: Some(pts),
        planes: vec![
            VideoPlane {
                stride: 128,
                data: rot(V3_DEFAULT_EXPECTED_Y),
            },
            VideoPlane {
                stride: 64,
                data: rot(V3_DEFAULT_EXPECTED_CB),
            },
            VideoPlane {
                stride: 64,
                data: rot(V3_DEFAULT_EXPECTED_CR),
            },
        ],
    }
}

/// Encode a multi-Frame stream through the `Encoder` trait — first Frame
/// keyframe, the rest §4.4 non-keyframes — then decode it back through the
/// `Decoder` trait. Each Frame reconstructs bit-exactly, proving the
/// registry produces a genuine inter stream and the read side carries the
/// §3.8.1.3 / §3.8.2.5 per-context coder state across packets. The
/// v3-default Configuration Record declares `intra == 0`, so the encoder
/// is allowed to emit non-keyframes (it would force keyframe-only output
/// under `intra == 1`).
#[test]
fn multi_frame_inter_stream_round_trips_through_trait_surface() {
    let params = v3_default_params();
    // Frame 0 = the reference pixels; Frames 1 / 2 = distinct rotations.
    let frames: Vec<VideoFrame> = vec![
        v3_default_video_frame(),
        rotated_video_frame(101, 1),
        rotated_video_frame(257, 2),
    ];

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    // ---- encode each Frame through the trait, collecting the packets ----
    let mut enc = ctx
        .codecs
        .first_encoder(&params)
        .expect("registry builds an ffv1 encoder");
    let mut packets = Vec::new();
    for (i, vframe) in frames.iter().enumerate() {
        enc.send_frame(&Frame::Video(vframe.clone()))
            .expect("send_frame encodes");
        let pkt = enc.receive_packet().expect("one packet per Frame");
        assert_eq!(
            pkt.is_keyframe(),
            i == 0,
            "Frame 0 is a keyframe; later Frames are non-keyframes"
        );
        packets.push(pkt);
    }

    // A non-keyframe must differ in bytes from the keyframe encode of the
    // same Frame (proving the carry actually changed the coded stream, not
    // just the §4.4 flag): re-encode Frame 1 standalone as a keyframe and
    // confirm the payloads differ.
    let mut fresh_enc = ctx
        .codecs
        .first_encoder(&params)
        .expect("registry builds a second ffv1 encoder");
    fresh_enc
        .send_frame(&Frame::Video(frames[1].clone()))
        .expect("encode Frame 1 standalone as a keyframe");
    let standalone_kf = fresh_enc.receive_packet().expect("keyframe packet");
    assert!(standalone_kf.is_keyframe());
    assert_ne!(
        standalone_kf.data, packets[1].data,
        "the inter-coded Frame 1 (non-keyframe, carried state) must differ \
         from its standalone keyframe encode"
    );

    // ---- decode the stream back through the trait ----
    let mut dec = ctx
        .codecs
        .first_decoder(&params)
        .expect("registry builds an ffv1 decoder");
    for (i, (pkt, expected)) in packets.iter().zip(frames.iter()).enumerate() {
        let in_pkt = oxideav_core::Packet::new(0, TimeBase::new(1, 1), pkt.data.clone());
        dec.send_packet(&in_pkt).unwrap();
        let decoded = match dec.receive_frame().expect("decode succeeds") {
            Frame::Video(v) => v,
            other => panic!("expected a video frame, got {other:?}"),
        };
        assert_eq!(decoded.planes.len(), 3, "Y + Cb + Cr");
        assert_eq!(
            decoded.planes[0].data, expected.planes[0].data,
            "Frame {i} Y plane survives the inter-Frame encode → decode \
             round-trip bit-exactly (the §3.8.1.3 carry stayed in lockstep \
             across the trait surface)"
        );
        assert_eq!(
            decoded.planes[1].data, expected.planes[1].data,
            "Frame {i} Cb plane survives the inter round-trip bit-exactly"
        );
        assert_eq!(
            decoded.planes[2].data, expected.planes[2].data,
            "Frame {i} Cr plane survives the inter round-trip bit-exactly"
        );
    }
}

/// RFC 9043 §4.2.17 (Table 14): a Configuration Record with `intra == 1`
/// means "keyframe MUST be 1 (keyframes only)". The framework encoder must
/// then emit **every** Frame as a keyframe — never a non-keyframe the
/// decoder's §4.2.17 intra gate would reject. Build an `intra == 1`
/// Configuration Record (re-encoded off the v3-default parse so its §4.1
/// quant-table cascade + §4.3.2 CRC stay valid) and confirm all coded
/// Frames are keyframes and still round-trip bit-exactly.
#[test]
fn intra_one_configuration_forces_keyframe_only_output() {
    let parsed =
        parse_quantization_table_sets(V3_DEFAULT_EXTRADATA).expect("v3-default extradata parses");
    let mut record = parsed.record.clone();
    record.intra = Some(true);
    let extradata =
        encode_configuration_record_with_quant_tables(&record, &parsed.quant_table_sets)
            .expect("intra==1 Configuration Record encodes with a solved CRC");

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(128);
    params.height = Some(96);
    params.extradata = extradata;

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx
        .codecs
        .first_encoder(&params)
        .expect("registry builds an ffv1 encoder");

    let frames = [
        v3_default_video_frame(),
        rotated_video_frame(101, 1),
        rotated_video_frame(257, 2),
    ];
    let mut packets = Vec::new();
    for vframe in &frames {
        enc.send_frame(&Frame::Video(vframe.clone()))
            .expect("send_frame encodes");
        let pkt = enc.receive_packet().expect("one packet per Frame");
        assert!(
            pkt.is_keyframe(),
            "intra == 1 forces every coded Frame to a keyframe (§4.2.17)"
        );
        packets.push(pkt);
    }

    // Every Frame decodes standalone (no carry dependence) — the proof the
    // encoder really emitted keyframes.
    let mut dec = ctx.codecs.first_decoder(&params).unwrap();
    for (pkt, expected) in packets.iter().zip(frames.iter()) {
        let in_pkt = oxideav_core::Packet::new(0, TimeBase::new(1, 1), pkt.data.clone());
        dec.send_packet(&in_pkt).unwrap();
        let decoded = match dec.receive_frame().expect("decode succeeds") {
            Frame::Video(v) => v,
            other => panic!("expected video, got {other:?}"),
        };
        assert_eq!(decoded.planes[0].data, expected.planes[0].data);
        assert_eq!(decoded.planes[1].data, expected.planes[1].data);
        assert_eq!(decoded.planes[2].data, expected.planes[2].data);
    }
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
