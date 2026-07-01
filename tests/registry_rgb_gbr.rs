//! Round 382 — RGB / JPEG 2000 RCT (`colorspace_type == 1`) through the
//! `oxideav-core` `Encoder` / `Decoder` trait surface, exercising the §4.2
//! pixel-format mapping to the planar `Gbr` family plus the plane reorder
//! the registry applies at the trait boundary.
//!
//! RFC 9043 §3.7 recovers colour Planes in **R, G, B (, A)** order, while
//! oxideav-core's planar RGB formats (`Gbrp*Le` / `Gbrap*Le`) store
//! **G, B, R (, A)**. The registry advertises the matching `Gbr*` format
//! and reorders Planes on both sides, so a producer that hands the encoder
//! a `Gbr`-ordered `VideoFrame` gets that exact frame back from the
//! decoder — the round trip is the identity on the framework plane order,
//! not just on the internal R, G, B buffers.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{
    decode_frame_rgb, encode_configuration_record_with_quant_tables, register, ColorspaceType,
    Ffv1ConfigurationRecord, Ffv1Version, FramePixelDimensions, QuantizationTableSet, CODEC_ID_STR,
    NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

/// A v3 RGB Configuration Record (§4.2.5 fixes RGB at 4:4:4). `bits`
/// selects the depth; `extra` adds the §4.2.10 alpha Plane.
fn rgb_record(bits: u32, extra: bool) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1, // range coder — no §3.8.2 run-mode subtleties
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::Rgb,
        bits_per_raw_sample: bits,
        chroma_planes: true,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: extra,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: Some(1),
        ec: Some(0),
        intra: Some(false),
        initial_state_delta: None,
    }
}

/// The all-zero Quantization Table Set — every neighbour configuration
/// routes to context 0 (a single context). It is the minimal §4.1
/// wire-serializable table (`context_count == 1`), so it round-trips
/// through [`encode_configuration_record_with_quant_tables`] cleanly.
fn zero_qts() -> QuantizationTableSet {
    QuantizationTableSet {
        tables: [[0i32; 256]; NUM_QUANT_SUBTABLES],
        context_count: 1,
    }
}

/// Build `CodecParameters` (extradata = §4.2 Configuration Record) for an
/// RGB stream of dimensions `w`×`h`.
fn rgb_params(bits: u32, extra: bool, w: u32, h: u32) -> CodecParameters {
    let record = rgb_record(bits, extra);
    let qts = vec![zero_qts()];
    let extradata = encode_configuration_record_with_quant_tables(&record, &qts)
        .expect("RGB Configuration Record encodes");
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.extradata = extradata;
    params
}

/// Pack a channel's `w*h` Samples into a tight little-endian `VideoPlane`
/// (two bytes per Sample — every depth here is > 8-bit).
fn plane_le(samples: &[u16], w: u32) -> VideoPlane {
    let mut data = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        data.extend_from_slice(&s.to_le_bytes());
    }
    VideoPlane {
        stride: w as usize * 2,
        data,
    }
}

/// Assert an RGB `VideoFrame` in framework `Gbr` plane order round-trips
/// bit-exactly through the registry encoder + decoder, and that the format
/// the encoder advertises is `want_pf`.
fn assert_gbr_round_trip(bits: u32, extra: bool, w: u32, h: u32, want_pf: PixelFormat) {
    let mask = (1u32 << bits) - 1;
    let n = (w * h) as usize;
    // G, B, R (, A) channels — the framework `Gbr` order. Distinct
    // per-channel patterns so a mis-ordered plane is caught.
    let g: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 37 + 1) & mask) as u16)
        .collect();
    let b: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 53 + 7) & mask) as u16)
        .collect();
    let r: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 71 + 3) & mask) as u16)
        .collect();
    let a: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 29 + 5) & mask) as u16)
        .collect();

    let mut planes = vec![plane_le(&g, w), plane_le(&b, w), plane_le(&r, w)];
    if extra {
        planes.push(plane_le(&a, w));
    }
    let src = VideoFrame {
        pts: Some(11),
        planes,
    };

    let params = rgb_params(bits, extra, w, h);

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let mut enc = ctx
        .codecs
        .first_encoder(&params)
        .expect("registry builds an ffv1 RGB encoder");
    assert_eq!(
        enc.output_params().pixel_format,
        Some(want_pf),
        "encoder advertises the §4.2-derived planar Gbr format"
    );

    enc.send_frame(&Frame::Video(src.clone()))
        .expect("RGB frame encodes through the trait");
    let pkt = enc.receive_packet().expect("one packet per RGB frame");

    let mut dec = ctx
        .codecs
        .first_decoder(&params)
        .expect("registry builds an ffv1 RGB decoder");
    dec.send_packet(&pkt).expect("decoder accepts the packet");
    let Frame::Video(out) = dec.receive_frame().expect("decoder emits a frame") else {
        panic!("expected a video frame");
    };

    assert_eq!(out.planes.len(), src.planes.len(), "plane count preserved");
    assert_eq!(out.pts, Some(11), "PTS propagates");
    for (i, (got, want)) in out.planes.iter().zip(src.planes.iter()).enumerate() {
        assert_eq!(
            got.data, want.data,
            "plane {i} diverged — Gbr plane order not preserved through the trait",
        );
    }
}

#[test]
fn rgb_gbrp12_round_trips_through_trait_in_gbr_order() {
    assert_gbr_round_trip(12, false, 6, 5, PixelFormat::Gbrp12Le);
}

#[test]
fn rgb_gbrp10_round_trips_through_trait_in_gbr_order() {
    assert_gbr_round_trip(10, false, 7, 4, PixelFormat::Gbrp10Le);
}

#[test]
fn rgb_gbrp14_round_trips_through_trait_in_gbr_order() {
    assert_gbr_round_trip(14, false, 5, 5, PixelFormat::Gbrp14Le);
}

#[test]
fn rgba_gbrap12_round_trips_through_trait_in_gbr_order() {
    assert_gbr_round_trip(12, true, 6, 4, PixelFormat::Gbrap12Le);
}

#[test]
fn rgba_gbrap10_round_trips_through_trait_in_gbr_order() {
    assert_gbr_round_trip(10, true, 4, 6, PixelFormat::Gbrap10Le);
}

/// A multi-Frame RGB inter stream round-trips through the trait surface:
/// the first Frame is a §4.4 keyframe, later Frames are non-keyframes
/// carrying §3.8.1.3 per-context range-coder state, and every Frame comes
/// back bit-exact in framework `Gbr` plane order. Exercises the RGB carry
/// path (`decode_frame_rgb_with_carry` / `encode_frame_rgb_with_carry`)
/// through the registry, with the plane reorder applied per Frame.
#[test]
fn multi_frame_rgb_inter_stream_round_trips_in_gbr_order() {
    let (bits, w, h) = (12u32, 6u32, 5u32);
    let mask = (1u32 << bits) - 1;
    let n = (w * h) as usize;
    let params = rgb_params(bits, false, w, h);

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx.codecs.first_encoder(&params).expect("RGB encoder");
    let mut dec = ctx.codecs.first_decoder(&params).expect("RGB decoder");

    // Three distinct Frames (varying per-frame offset `f`).
    let frames: Vec<VideoFrame> = (0..3u32)
        .map(|f| {
            let g: Vec<u16> = (0..n)
                .map(|i| ((i as u32 * 37 + f * 13 + 1) & mask) as u16)
                .collect();
            let b: Vec<u16> = (0..n)
                .map(|i| ((i as u32 * 53 + f * 17 + 7) & mask) as u16)
                .collect();
            let r: Vec<u16> = (0..n)
                .map(|i| ((i as u32 * 71 + f * 19 + 3) & mask) as u16)
                .collect();
            VideoFrame {
                pts: Some(f as i64),
                planes: vec![plane_le(&g, w), plane_le(&b, w), plane_le(&r, w)],
            }
        })
        .collect();

    for (idx, src) in frames.iter().enumerate() {
        enc.send_frame(&Frame::Video(src.clone()))
            .expect("RGB inter frame encodes");
        let pkt = enc.receive_packet().expect("one packet per frame");
        // The first Frame is a keyframe; later Frames are non-keyframes.
        assert_eq!(
            pkt.flags.keyframe,
            idx == 0,
            "frame {idx} keyframe flag reflects §4.4"
        );
        dec.send_packet(&pkt).expect("decoder accepts inter packet");
        let Frame::Video(out) = dec.receive_frame().expect("frame") else {
            panic!("video frame");
        };
        for (p, (got, want)) in out.planes.iter().zip(src.planes.iter()).enumerate() {
            assert_eq!(
                got.data, want.data,
                "frame {idx} plane {p} diverged across the inter carry",
            );
        }
    }
}

/// Prove the plane reorder is a *real* permutation, not a symmetric no-op:
/// decode the trait-encoded packet with the direct `decode_frame_rgb` API
/// (which emits internal **R, G, B** order, RFC 9043 §3.7) and confirm the
/// encoder placed the framework's `Gbr` input channels into the correct
/// internal Planes, then confirm the trait decoder reorders them back.
#[test]
fn direct_decode_confirms_gbr_reorder_direction() {
    let (bits, w, h) = (12u32, 6u32, 5u32);
    let mask = (1u32 << bits) - 1;
    let n = (w * h) as usize;
    // Framework `Gbr` input order: plane 0 = G, plane 1 = B, plane 2 = R.
    let g: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 37 + 1) & mask) as u16)
        .collect();
    let b: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 53 + 7) & mask) as u16)
        .collect();
    let r: Vec<u16> = (0..n)
        .map(|i| ((i as u32 * 71 + 3) & mask) as u16)
        .collect();
    let src = VideoFrame {
        pts: Some(0),
        planes: vec![plane_le(&g, w), plane_le(&b, w), plane_le(&r, w)],
    };

    let params = rgb_params(bits, false, w, h);
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx.codecs.first_encoder(&params).expect("RGB encoder");
    enc.send_frame(&Frame::Video(src)).expect("encode");
    let pkt = enc.receive_packet().expect("packet");

    // Direct decode → internal R, G, B Planes.
    let record = rgb_record(bits, false);
    let qts = vec![zero_qts()];
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let direct =
        decode_frame_rgb(&pkt.data, &record, &qts, dims, false).expect("direct RGB decode");
    let r_i32: Vec<i32> = r.iter().map(|&x| x as i32).collect();
    let g_i32: Vec<i32> = g.iter().map(|&x| x as i32).collect();
    let b_i32: Vec<i32> = b.iter().map(|&x| x as i32).collect();
    // Internal plane 0 = R, 1 = G, 2 = B: the encoder mapped the framework
    // R input (VideoFrame plane 2) onto internal plane 0, etc.
    assert_eq!(
        direct.planes[0].samples, r_i32,
        "internal plane 0 must be R"
    );
    assert_eq!(
        direct.planes[1].samples, g_i32,
        "internal plane 1 must be G"
    );
    assert_eq!(
        direct.planes[2].samples, b_i32,
        "internal plane 2 must be B"
    );

    // Trait decode → framework Gbr order: plane 0 = G (internal 1), plane 1
    // = B (internal 2), plane 2 = R (internal 0).
    let mut dec = ctx.codecs.first_decoder(&params).expect("RGB decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(out) = dec.receive_frame().expect("frame") else {
        panic!("video frame");
    };
    assert_eq!(
        out.planes[0].data,
        plane_le(&g, w).data,
        "trait plane 0 = G"
    );
    assert_eq!(
        out.planes[1].data,
        plane_le(&b, w).data,
        "trait plane 1 = B"
    );
    assert_eq!(
        out.planes[2].data,
        plane_le(&r, w).data,
        "trait plane 2 = R"
    );
}
