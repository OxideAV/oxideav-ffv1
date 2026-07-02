//! Round 382 — FFV1 **versions 0/1** encoding through the
//! `oxideav-core` `Encoder` trait.
//!
//! RFC 9043 §4.3.3 / §4.4: v0/v1 streams carry no §4.2 Configuration
//! Record — their Parameters ride inline in each keyframe Frame. The
//! framework encoder therefore accepts `CodecParameters` with **empty
//! extradata** plus a `pixel_format` and dimensions, synthesises a
//! version-1 record from the pixel format, installs a §4.1-constructed
//! default Quantization Table Set, and emits the stream's first Frame as
//! a §4.4 keyframe (inline Parameters + Set) and later Frames as
//! non-keyframes. The registry *decoder* already accepts exactly this
//! configuration shape (dims + empty extradata), so encoder → decoder
//! round-trips end-to-end through the trait surface with no
//! out-of-band configuration at all.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{parse_v0v1_frame_prologue, register, CODEC_ID_STR};

/// v0/v1 `CodecParameters`: dimensions + pixel format, empty extradata.
fn v0v1_params(pf: PixelFormat, w: u32, h: u32) -> CodecParameters {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(pf);
    params
}

/// Deterministic 8-bit plane bytes.
fn plane8(seed: u32, w: u32, h: u32) -> VideoPlane {
    let data: Vec<u8> = (0..w * h)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(seed)) & 0xFF) as u8)
        .collect();
    VideoPlane {
        stride: w as usize,
        data,
    }
}

/// Deterministic `bits`-deep little-endian plane bytes.
fn plane_le(seed: u32, w: u32, h: u32, bits: u32) -> VideoPlane {
    let mask = (1u32 << bits) - 1;
    let mut data = Vec::with_capacity((w * h) as usize * 2);
    for i in 0..w * h {
        let v = (i.wrapping_mul(197).wrapping_add(seed) & mask) as u16;
        data.extend_from_slice(&v.to_le_bytes());
    }
    VideoPlane {
        stride: w as usize * 2,
        data,
    }
}

/// Encode `frames` through the registry encoder built from `params`,
/// decode each packet back through the registry decoder built from the
/// *same* parameters, and assert every plane round-trips bit-exactly.
/// Returns the encoded packets for further stream-shape assertions.
fn assert_v0v1_trait_round_trip(
    params: &CodecParameters,
    frames: &[VideoFrame],
) -> Vec<oxideav_core::Packet> {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let mut enc = ctx
        .codecs
        .first_encoder(params)
        .expect("registry builds a v0/v1 encoder from pixel_format + dims");
    // No Configuration Record exists for v0/v1: extradata must stay empty
    // so a muxer writes no CodecPrivate (RFC 9043 §4.3.3).
    assert!(
        enc.output_params().extradata.is_empty(),
        "v0/v1 output_params must carry no extradata"
    );

    let mut dec = ctx
        .codecs
        .first_decoder(params)
        .expect("registry builds the v0/v1 decoder from the same parameters");

    let mut packets = Vec::with_capacity(frames.len());
    for (idx, src) in frames.iter().enumerate() {
        enc.send_frame(&Frame::Video(src.clone()))
            .expect("v0/v1 frame encodes through the trait");
        let pkt = enc.receive_packet().expect("one packet per frame");
        assert_eq!(
            pkt.flags.keyframe,
            idx == 0,
            "frame {idx}: first Frame is the §4.4 keyframe, later Frames are not"
        );

        dec.send_packet(&pkt).expect("decoder accepts the packet");
        let Frame::Video(out) = dec.receive_frame().expect("decoder emits a frame") else {
            panic!("expected a video frame");
        };
        assert_eq!(out.planes.len(), src.planes.len(), "frame {idx} planes");
        for (p, (got, want)) in out.planes.iter().zip(src.planes.iter()).enumerate() {
            assert_eq!(got.data, want.data, "frame {idx} plane {p} diverged");
        }
        packets.push(pkt);
    }
    packets
}

#[test]
fn v0v1_gray8_encodes_and_round_trips_through_trait() {
    let (w, h) = (23u32, 17u32);
    let params = v0v1_params(PixelFormat::Gray8, w, h);
    let frames: Vec<VideoFrame> = (0..3)
        .map(|f| VideoFrame {
            pts: Some(f as i64),
            planes: vec![plane8(f * 7 + 1, w, h)],
        })
        .collect();
    let packets = assert_v0v1_trait_round_trip(&params, &frames);

    // The keyframe's inline §4.4 prologue must parse as version 1 with the
    // synthesised Parameters and the default §4.1 Set (context_count 666).
    let prologue =
        parse_v0v1_frame_prologue(&packets[0].data).expect("keyframe carries inline Parameters");
    assert_eq!(prologue.record.bits_per_raw_sample, 8);
    assert!(!prologue.record.chroma_planes);
    assert!(!prologue.record.extra_plane);
    assert_eq!(prologue.quant_table_set.context_count, 666);
    // A non-keyframe carries no inline Parameters.
    assert!(matches!(
        parse_v0v1_frame_prologue(&packets[1].data),
        Err(oxideav_ffv1::Error::NonKeyframeHasNoInFrameParameters)
    ));
}

#[test]
fn v0v1_yuv420p_encodes_and_round_trips_through_trait() {
    let (w, h) = (16u32, 12u32);
    let params = v0v1_params(PixelFormat::Yuv420P, w, h);
    let frames: Vec<VideoFrame> = (0..2)
        .map(|f| VideoFrame {
            pts: Some(f as i64),
            planes: vec![
                plane8(f * 11 + 3, w, h),
                plane8(f * 13 + 5, w / 2, h / 2),
                plane8(f * 17 + 7, w / 2, h / 2),
            ],
        })
        .collect();
    assert_v0v1_trait_round_trip(&params, &frames);
}

#[test]
fn v0v1_yuv422p10_encodes_and_round_trips_through_trait() {
    // 10-bit 4:2:2 — a depth v0 could not express (no bits field); the
    // synthesised version-1 record carries it via the §4.4
    // `bits_per_raw_sample` symbol.
    let (w, h) = (14u32, 9u32);
    let params = v0v1_params(PixelFormat::Yuv422P10Le, w, h);
    let frames = vec![VideoFrame {
        pts: Some(0),
        planes: vec![
            plane_le(3, w, h, 10),
            plane_le(5, w.div_ceil(2), h, 10),
            plane_le(7, w.div_ceil(2), h, 10),
        ],
    }];
    assert_v0v1_trait_round_trip(&params, &frames);
}

#[test]
fn v0v1_gbrp12_encodes_and_round_trips_through_trait() {
    // Planar RGB through the v0/v1 path: the frame arrives in framework
    // G, B, R order, the encoder reorders it into the RFC 9043 §3.7
    // R, G, B coded layout, and the decoder reorders it back.
    let (w, h) = (8u32, 6u32);
    let params = v0v1_params(PixelFormat::Gbrp12Le, w, h);
    let frames: Vec<VideoFrame> = (0..2)
        .map(|f| VideoFrame {
            pts: Some(f as i64),
            planes: vec![
                plane_le(f * 3 + 1, w, h, 12),
                plane_le(f * 5 + 2, w, h, 12),
                plane_le(f * 7 + 4, w, h, 12),
            ],
        })
        .collect();
    let packets = assert_v0v1_trait_round_trip(&params, &frames);
    let prologue =
        parse_v0v1_frame_prologue(&packets[0].data).expect("keyframe carries inline Parameters");
    assert_eq!(prologue.record.bits_per_raw_sample, 12);
}

#[test]
fn unmappable_pixel_format_is_a_diagnosable_error() {
    // A packed format has no FFV1 §4.2 Parameters mapping: the registry
    // must refuse to build the encoder with a clear error, not hand back
    // an unconfigured encoder that fails later.
    let params = v0v1_params(PixelFormat::Rgb24, 8, 8);
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    assert!(
        ctx.codecs.first_encoder(&params).is_err(),
        "no encoder must be constructed for an unmappable pixel format"
    );
}

#[test]
fn missing_pixel_format_leaves_encoder_unconfigured() {
    // Dims but neither extradata nor pixel_format: the encoder builds
    // (deferred configuration) but send_frame diagnoses the gap.
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(8);
    params.height = Some(8);
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx
        .codecs
        .first_encoder(&params)
        .expect("encoder construction is deferred, not refused");
    let frame = VideoFrame {
        pts: None,
        planes: vec![plane8(1, 8, 8)],
    };
    assert!(
        enc.send_frame(&Frame::Video(frame)).is_err(),
        "sending a frame without configuration must error"
    );
}
