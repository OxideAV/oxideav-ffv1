//! End-to-end v0/v1 decode through the [`oxideav_core::Decoder`] trait
//! (round 342).
//!
//! RFC 9043 §4.3.3 / §4.4: versions 0 and 1 carry **no** Configuration
//! Record — their §4.2 Parameters are inline in each keyframe Frame. A
//! v0/v1 container therefore supplies `CodecParameters` with frame
//! dimensions but an **empty** `extradata`. The registry decoder detects
//! that shape and parses the §4.4 prologue off the first keyframe packet,
//! caching the record + single §4.1 Quantization Table Set for later
//! non-keyframes.
//!
//! These tests build a v0/v1 stream with `encode_frame_v0v1` /
//! `encode_frame_v0v1_inter`, feed the Frames as `Packet`s through the
//! trait, and verify the recovered `VideoFrame` planes are bit-exact.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, RuntimeContext, TimeBase};
use oxideav_ffv1::{
    encode_frame_v0v1, encode_frame_v0v1_inter, parse_quantization_table_sets,
    parse_v0v1_frame_prologue, register, ColorspaceType, DecodedFrame, DecodedFramePlane,
    Ffv1ConfigurationRecord, Ffv1Version, QuantizationTableSet, CODEC_ID_STR,
};

const V3_DEFAULT_EXTRADATA: &[u8] = &[
    0x56, 0x00, 0x30, 0x9c, 0x75, 0xdf, 0xf4, 0x60, 0xb4, 0x3a, 0x42, 0xd7, 0xd4, 0xd6, 0x86, 0x2f,
    0x74, 0x92, 0x4a, 0x72, 0xe6, 0x12, 0x9b, 0xf9, 0x2f, 0xba, 0xd1, 0x40, 0x0f, 0x89, 0xac, 0x8f,
    0xc7, 0x82, 0x07, 0xee, 0xbc, 0x31, 0x7c, 0xf5, 0x29, 0x2b,
];

fn real_qts() -> QuantizationTableSet {
    parse_quantization_table_sets(V3_DEFAULT_EXTRADATA)
        .unwrap()
        .quant_table_sets[0]
        .clone()
}

fn v0v1_record(version: Ffv1Version, chroma: bool) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version,
        micro_version: None,
        coder_type: 1,
        state_transition_delta: [0i32; 256],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: 8,
        chroma_planes: chroma,
        log2_h_chroma_subsample: if chroma { 1 } else { 0 },
        log2_v_chroma_subsample: if chroma { 1 } else { 0 },
        extra_plane: false,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: None,
        ec: None,
        intra: None,
        initial_state_delta: None,
    }
}

fn synth(seed: u64, n: usize) -> Vec<i32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z & 0xFF) as i32
        })
        .collect()
}

fn gray_frame(w: u32, h: u32, seed: u64) -> DecodedFrame {
    DecodedFrame {
        planes: vec![DecodedFramePlane {
            plane_index: 0,
            width: w,
            height: h,
            samples: synth(seed, (w * h) as usize),
        }],
        width: w,
        height: h,
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

fn v0v1_params(w: u32, h: u32) -> CodecParameters {
    // No extradata — the versions-0/1 carriage shape.
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params
}

#[test]
fn registry_decodes_v1_keyframe_no_extradata() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let cr = v0v1_record(Ffv1Version::V1, false);
    let qts = real_qts();
    let frame = gray_frame(16, 12, 0xD00D);
    let bytes = encode_frame_v0v1(&frame, &cr, &qts).expect("encode v1 keyframe");

    let params = v0v1_params(16, 12);
    let mut dec = ctx
        .codecs
        .first_decoder(&params)
        .expect("registry builds a v0/v1 ffv1 decoder from dims alone");

    let pkt = Packet::new(0, TimeBase::new(1, 1), bytes);
    dec.send_packet(&pkt).expect("send_packet");
    let video = match dec.receive_frame().expect("receive_frame") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    assert_eq!(video.planes.len(), 1, "luma only");
    // One byte per Sample at 8-bit, tight stride.
    assert_eq!(video.planes[0].stride, 16);
    assert_eq!(video.planes[0].data.len(), 16 * 12);
    let want: Vec<u8> = frame.planes[0].samples.iter().map(|&s| s as u8).collect();
    assert_eq!(video.planes[0].data, want, "luma must be bit-exact");
}

#[test]
fn registry_decodes_v1_keyframe_then_non_keyframe() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let cr = v0v1_record(Ffv1Version::V1, true);
    let qts = real_qts();

    // Keyframe establishes the inline Parameters.
    let kf = {
        let mut f = gray_frame(16, 16, 0x1111);
        // Add chroma planes (YUV420).
        f.planes.push(DecodedFramePlane {
            plane_index: 1,
            width: 8,
            height: 8,
            samples: synth(0x2222, 64),
        });
        f.planes.push(DecodedFramePlane {
            plane_index: 2,
            width: 8,
            height: 8,
            samples: synth(0x3333, 64),
        });
        f
    };
    let kf_bytes = encode_frame_v0v1(&kf, &cr, &qts).expect("encode keyframe");
    let prologue = parse_v0v1_frame_prologue(&kf_bytes).expect("parse prologue");

    // Non-keyframe reuses the keyframe's config.
    let nkf = {
        let mut f = gray_frame(16, 16, 0x4444);
        f.planes.push(DecodedFramePlane {
            plane_index: 1,
            width: 8,
            height: 8,
            samples: synth(0x5555, 64),
        });
        f.planes.push(DecodedFramePlane {
            plane_index: 2,
            width: 8,
            height: 8,
            samples: synth(0x6666, 64),
        });
        f
    };
    let nkf_bytes = encode_frame_v0v1_inter(&nkf, &prologue.record, &prologue.quant_table_set)
        .expect("encode nkf");

    let params = v0v1_params(16, 16);
    let mut dec = ctx.codecs.first_decoder(&params).expect("build decoder");

    // Keyframe through the trait.
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), kf_bytes))
        .expect("send kf");
    let kf_video = match dec.receive_frame().expect("recv kf") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    let want_y: Vec<u8> = kf.planes[0].samples.iter().map(|&s| s as u8).collect();
    assert_eq!(kf_video.planes[0].data, want_y, "keyframe luma bit-exact");

    // Non-keyframe through the trait (inherits the cached config).
    dec.send_packet(&Packet::new(1, TimeBase::new(1, 1), nkf_bytes))
        .expect("send nkf");
    let nkf_video = match dec.receive_frame().expect("recv nkf") {
        Frame::Video(v) => v,
        other => panic!("expected video, got {other:?}"),
    };
    let want_y2: Vec<u8> = nkf.planes[0].samples.iter().map(|&s| s as u8).collect();
    assert_eq!(
        nkf_video.planes[0].data, want_y2,
        "non-keyframe luma bit-exact"
    );
    let want_u2: Vec<u8> = nkf.planes[1].samples.iter().map(|&s| s as u8).collect();
    assert_eq!(
        nkf_video.planes[1].data, want_u2,
        "non-keyframe Cb bit-exact"
    );
}

#[test]
fn registry_v0v1_non_keyframe_before_keyframe_errors() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let cr = v0v1_record(Ffv1Version::V1, false);
    let qts = real_qts();
    let frame = gray_frame(8, 8, 0x7777);
    // A non-keyframe Frame as the very first packet — no cached config.
    let nkf_bytes = encode_frame_v0v1_inter(&frame, &cr, &qts).expect("encode nkf");

    let params = v0v1_params(8, 8);
    let mut dec = ctx.codecs.first_decoder(&params).expect("build decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), nkf_bytes))
        .expect("send");
    assert!(
        dec.receive_frame().is_err(),
        "a v0/v1 non-keyframe arriving before any keyframe must error"
    );
}
