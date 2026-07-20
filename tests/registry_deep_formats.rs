//! Round 420 — deep / alpha-carrying formats through the `oxideav-core`
//! `Encoder` / `Decoder` trait surface, driving the §4.2 pixel-format
//! mapping onto the core-0.1.31 format families:
//!
//! * **16-bit planar YUV** (`Yuv420P16Le` / `Yuv422P16Le` /
//!   `Yuv444P16Le`) — including the §3.3.1 16-bit alternate predictor;
//! * **the Yuva family** — 8-bit `Yuva422P` / `Yuva444P` and the deep
//!   4:2:2 / 4:4:4 alpha formats at 10 / 12 / 16 bits (§4.2.10
//!   `extra_plane`);
//! * **off-grid depths on deeper surfaces + the significant-bits
//!   side-channel** — 14-bit YCbCr on the 16-bit surfaces and 8-bit
//!   planar RGB / RCT on the `Gbrp*Le` 16-bit-word surfaces, with the
//!   coded §4.2.7 depth attached to every emitted frame via
//!   `VideoFrame::significant_bits`.
//!
//! Every case is a bit-exact encode → decode round trip through the
//! registry traits (FFV1 is lossless, RFC 9043 §1), checking the
//! advertised `output_params.pixel_format`, the emitted plane bytes,
//! and the attached significant-bits record.

use oxideav_core::{
    CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{
    encode_configuration_record_with_quant_tables, pixel_format_mapping_for, register,
    ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version, QuantizationTableSet, CODEC_ID_STR,
    NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

/// A v3 Configuration Record with the geometry under test.
fn record(
    cs: ColorspaceType,
    bits: u32,
    chroma: bool,
    h: u32,
    v: u32,
    extra: bool,
) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: cs,
        bits_per_raw_sample: bits,
        chroma_planes: chroma,
        log2_h_chroma_subsample: h,
        log2_v_chroma_subsample: v,
        extra_plane: extra,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: Some(1),
        ec: Some(0),
        intra: Some(false),
        initial_state_delta: None,
    }
}

/// The minimal §4.1 wire-serializable Quantization Table Set.
fn zero_qts() -> QuantizationTableSet {
    QuantizationTableSet {
        tables: [[0i32; 256]; NUM_QUANT_SUBTABLES],
        context_count: 1,
    }
}

fn params_for(cr: &Ffv1ConfigurationRecord, w: u32, h: u32) -> CodecParameters {
    let extradata = encode_configuration_record_with_quant_tables(cr, &[zero_qts()])
        .expect("Configuration Record encodes");
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.extradata = extradata;
    params
}

/// Pack `w*h`-shaped samples as the surface expects: 2-byte LE words on
/// the 16-bit-word surfaces, single bytes on the 8-bit byte surfaces.
fn plane(samples: &[u16], w: u32, wide: bool) -> VideoPlane {
    if wide {
        let mut data = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            data.extend_from_slice(&s.to_le_bytes());
        }
        VideoPlane {
            stride: w as usize * 2,
            data,
        }
    } else {
        VideoPlane {
            stride: w as usize,
            data: samples.iter().map(|&s| s as u8).collect(),
        }
    }
}

/// Deterministic per-plane test pattern at the coded depth.
fn pattern(n: usize, mul: u32, add: u32, bits: u32) -> Vec<u16> {
    let mask = (1u32 << bits) - 1;
    (0..n)
        .map(|i| ((i as u32 * mul + add) & mask) as u16)
        .collect()
}

/// Build the source `VideoFrame` for `cr` at `w`×`h` in the mapped
/// surface's plane order and word width, returning it with the plane
/// geometry used.
fn source_frame(cr: &Ffv1ConfigurationRecord, w: u32, h: u32, wide: bool) -> VideoFrame {
    let rgb = cr.colorspace_type == ColorspaceType::Rgb;
    let (cw, ch) = if cr.chroma_planes && !rgb {
        (
            w.div_ceil(1 << cr.log2_h_chroma_subsample),
            h.div_ceil(1 << cr.log2_v_chroma_subsample),
        )
    } else {
        (w, h)
    };
    let full = (w * h) as usize;
    let sub = (cw * ch) as usize;
    let bits = cr.bits_per_raw_sample;

    let mut planes = Vec::new();
    if rgb {
        // Framework `Gbr` order: G, B, R (, A) — all full resolution.
        planes.push(plane(&pattern(full, 37, 1, bits), w, wide));
        planes.push(plane(&pattern(full, 53, 7, bits), w, wide));
        planes.push(plane(&pattern(full, 71, 3, bits), w, wide));
        if cr.extra_plane {
            planes.push(plane(&pattern(full, 29, 5, bits), w, wide));
        }
    } else {
        planes.push(plane(&pattern(full, 31, 2, bits), w, wide));
        if cr.chroma_planes {
            planes.push(plane(&pattern(sub, 41, 9, bits), cw, wide));
            planes.push(plane(&pattern(sub, 59, 4, bits), cw, wide));
        }
        if cr.extra_plane {
            // §4.2.10: the extra (alpha) Plane is full resolution.
            planes.push(plane(&pattern(full, 23, 6, bits), w, wide));
        }
    }
    VideoFrame {
        pts: Some(7),
        planes,
    }
}

/// Encode → decode `cr` through the registry traits and assert: the
/// encoder advertises `want_pf`, the decoded image planes are bit-exact
/// against the source, and the decoded frame's significant-bits record
/// equals `want_sig` (None = no record attached).
fn assert_trait_round_trip(
    cr: &Ffv1ConfigurationRecord,
    w: u32,
    h: u32,
    want_pf: PixelFormat,
    want_sig: Option<&[u8]>,
) {
    let mapping = pixel_format_mapping_for(cr).expect("layout maps to a surface");
    assert_eq!(mapping.format, want_pf, "mapped storage surface");
    assert_eq!(
        mapping.significant_bits.as_deref(),
        want_sig,
        "mapping significant-bits record"
    );
    // Word width follows the surface: every `*Le` surface is 2-byte;
    // the 8-bit byte surfaces used in this file are the Yuva trio.
    let wide = !matches!(
        want_pf,
        PixelFormat::Yuva420P | PixelFormat::Yuva422P | PixelFormat::Yuva444P
    );

    let params = params_for(cr, w, h);
    let mut src = source_frame(cr, w, h, wide);
    if let Some(sig) = want_sig {
        // Producers may attach the (matching) record on input frames;
        // the encoder must accept it.
        src.set_significant_bits(sig.to_vec());
    }

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);

    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder");
    assert_eq!(
        enc.output_params().pixel_format,
        Some(want_pf),
        "encoder advertises the mapped storage surface"
    );
    enc.send_frame(&Frame::Video(src.clone())).expect("encode");
    let pkt = enc.receive_packet().expect("one packet per frame");

    let mut dec = ctx.codecs.first_decoder(&params).expect("decoder");
    dec.send_packet(&pkt).expect("send");
    let Frame::Video(out) = dec.receive_frame().expect("frame") else {
        panic!("expected a video frame");
    };

    assert_eq!(
        out.significant_bits(),
        want_sig,
        "decoded frame's significant-bits record"
    );
    assert_eq!(
        out.image_plane_count(),
        src.image_plane_count(),
        "image plane count"
    );
    for (i, (got, want)) in out
        .image_planes()
        .iter()
        .zip(src.image_planes().iter())
        .enumerate()
    {
        assert_eq!(got.data, want.data, "plane {i} diverged through the trait");
    }
}

// ─────────────────────────── 16-bit planar YUV ───────────────────────────

#[test]
fn yuv420p16_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 16, true, 1, 1, false);
    assert_trait_round_trip(&cr, 8, 6, PixelFormat::Yuv420P16Le, None);
}

#[test]
fn yuv422p16_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 16, true, 1, 0, false);
    assert_trait_round_trip(&cr, 7, 5, PixelFormat::Yuv422P16Le, None);
}

#[test]
fn yuv444p16_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 16, true, 0, 0, false);
    assert_trait_round_trip(&cr, 6, 5, PixelFormat::Yuv444P16Le, None);
}

// ────────────────────────────── Yuva family ──────────────────────────────

#[test]
fn yuva422p8_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 8, true, 1, 0, true);
    assert_trait_round_trip(&cr, 9, 5, PixelFormat::Yuva422P, None);
}

#[test]
fn yuva444p8_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 8, true, 0, 0, true);
    assert_trait_round_trip(&cr, 6, 4, PixelFormat::Yuva444P, None);
}

#[test]
fn yuva422p10_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 10, true, 1, 0, true);
    assert_trait_round_trip(&cr, 7, 6, PixelFormat::Yuva422P10Le, None);
}

#[test]
fn yuva444p12_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 12, true, 0, 0, true);
    assert_trait_round_trip(&cr, 5, 6, PixelFormat::Yuva444P12Le, None);
}

#[test]
fn yuva422p16_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 16, true, 1, 0, true);
    assert_trait_round_trip(&cr, 8, 4, PixelFormat::Yuva422P16Le, None);
}

#[test]
fn yuva444p16_round_trips_through_trait() {
    let cr = record(ColorspaceType::YCbCr, 16, true, 0, 0, true);
    assert_trait_round_trip(&cr, 6, 6, PixelFormat::Yuva444P16Le, None);
}

// ──────────── off-grid depths + significant-bits side-channel ────────────

#[test]
fn yuv444p14_rides_16bit_surface_with_significant_bits() {
    let cr = record(ColorspaceType::YCbCr, 14, true, 0, 0, false);
    assert_trait_round_trip(&cr, 6, 5, PixelFormat::Yuv444P16Le, Some(&[14, 14, 14]));
}

#[test]
fn yuv420p9_rides_10bit_surface_with_significant_bits() {
    let cr = record(ColorspaceType::YCbCr, 9, true, 1, 1, false);
    assert_trait_round_trip(&cr, 8, 6, PixelFormat::Yuv420P10Le, Some(&[9, 9, 9]));
}

#[test]
fn gray14_rides_gray16_surface_with_significant_bits() {
    let cr = record(ColorspaceType::YCbCr, 14, false, 0, 0, false);
    assert_trait_round_trip(&cr, 9, 7, PixelFormat::Gray16Le, Some(&[14]));
}

#[test]
fn yuva444p14_rides_16bit_alpha_surface_with_significant_bits() {
    let cr = record(ColorspaceType::YCbCr, 14, true, 0, 0, true);
    assert_trait_round_trip(
        &cr,
        5,
        5,
        PixelFormat::Yuva444P16Le,
        Some(&[14, 14, 14, 14]),
    );
}

#[test]
fn rgb8_rides_gbrp10_surface_with_significant_bits() {
    let cr = record(ColorspaceType::Rgb, 8, true, 0, 0, false);
    assert_trait_round_trip(&cr, 6, 5, PixelFormat::Gbrp10Le, Some(&[8, 8, 8]));
}

#[test]
fn rgba8_rides_gbrap10_surface_with_significant_bits() {
    let cr = record(ColorspaceType::Rgb, 8, true, 0, 0, true);
    assert_trait_round_trip(&cr, 5, 4, PixelFormat::Gbrap10Le, Some(&[8, 8, 8, 8]));
}

// ───────────────────────── encoder-side contracts ────────────────────────

/// A frame-attached significant-bits record that conflicts with the
/// stream's §4.2.7 depth is a diagnosable error, not silent metadata
/// loss.
#[test]
fn encoder_rejects_conflicting_significant_bits() {
    let cr = record(ColorspaceType::YCbCr, 10, true, 1, 1, false);
    let params = params_for(&cr, 8, 6);
    let mut src = source_frame(&cr, 8, 6, true);
    src.set_significant_bits(vec![8, 8, 8]); // stream is 10-bit

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder");
    let err = enc
        .send_frame(&Frame::Video(src))
        .expect_err("conflicting record must be rejected");
    assert!(
        format!("{err}").contains("significant bits"),
        "diagnostic names the conflict: {err}"
    );
}

/// A matching significant-bits record on the input frame is accepted
/// even on an exactly-mapped stream (it restates the format's depth).
#[test]
fn encoder_accepts_restated_depth_record() {
    let cr = record(ColorspaceType::YCbCr, 10, true, 1, 1, false);
    let params = params_for(&cr, 8, 6);
    let mut src = source_frame(&cr, 8, 6, true);
    src.set_significant_bits(vec![10, 10, 10]);

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder");
    enc.send_frame(&Frame::Video(src))
        .expect("matching record is accepted");
    enc.receive_packet().expect("packet");
}

// ─────────────────── v0/v1 empty-extradata trait route ───────────────────

/// The new formats also ride the empty-extradata v0/v1 route: the
/// encoder synthesises the inline §4.4 Parameters from the pixel format
/// and the stream round-trips with no Configuration Record at all.
#[test]
fn v0v1_route_round_trips_new_formats() {
    for (pf, planes, wide, bits) in [
        (PixelFormat::Yuv444P16Le, 3usize, true, 16u32),
        (PixelFormat::Yuv420P16Le, 3, true, 16),
        (PixelFormat::Yuva422P, 4, false, 8),
        (PixelFormat::Yuva444P, 4, false, 8),
        (PixelFormat::Yuva422P10Le, 4, true, 10),
        (PixelFormat::Yuva444P16Le, 4, true, 16),
    ] {
        let (w, h) = (8u32, 6u32);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(w);
        params.height = Some(h);
        params.pixel_format = Some(pf);

        // Build a source frame matching the format's geometry.
        let (hs, vs) = match pf {
            PixelFormat::Yuv444P16Le | PixelFormat::Yuva444P | PixelFormat::Yuva444P16Le => (0, 0),
            PixelFormat::Yuva422P | PixelFormat::Yuva422P10Le => (1, 0),
            _ => (1, 1),
        };
        let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
        let full = (w * h) as usize;
        let sub = (cw * ch) as usize;
        let mut fplanes = vec![plane(&pattern(full, 31, 2, bits), w, wide)];
        fplanes.push(plane(&pattern(sub, 41, 9, bits), cw, wide));
        fplanes.push(plane(&pattern(sub, 59, 4, bits), cw, wide));
        if planes == 4 {
            fplanes.push(plane(&pattern(full, 23, 6, bits), w, wide));
        }
        let src = VideoFrame {
            pts: Some(3),
            planes: fplanes,
        };

        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let mut enc = ctx.codecs.first_encoder(&params).expect("v0/v1 encoder");
        assert_eq!(enc.output_params().pixel_format, Some(pf), "{pf:?}");
        enc.send_frame(&Frame::Video(src.clone())).expect("encode");
        let pkt = enc.receive_packet().expect("packet");

        // Decode with dims but no extradata — the §4.4 prologue carries
        // the Parameters.
        let mut dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        dparams.width = Some(w);
        dparams.height = Some(h);
        let mut dec = ctx.codecs.first_decoder(&dparams).expect("v0/v1 decoder");
        dec.send_packet(&pkt).expect("send");
        let Frame::Video(out) = dec.receive_frame().expect("frame") else {
            panic!("video frame");
        };
        assert_eq!(out.significant_bits(), None, "{pf:?}: exact mapping");
        for (i, (got, want)) in out
            .image_planes()
            .iter()
            .zip(src.image_planes().iter())
            .enumerate()
        {
            assert_eq!(got.data, want.data, "{pf:?} plane {i}");
        }
    }
}

/// Multi-frame inter carry on a deep alpha format: keyframe + two
/// non-keyframes, every frame bit-exact and tagged with the §4.4 flag.
#[test]
fn multi_frame_yuva444p16_inter_stream_round_trips() {
    let cr = record(ColorspaceType::YCbCr, 16, true, 0, 0, true);
    let (w, h) = (6u32, 5u32);
    let params = params_for(&cr, w, h);

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder");
    let mut dec = ctx.codecs.first_decoder(&params).expect("decoder");

    let n = (w * h) as usize;
    for f in 0..3u32 {
        let planes = vec![
            plane(&pattern(n, 31, 2 + f * 13, 16), w, true),
            plane(&pattern(n, 41, 9 + f * 17, 16), w, true),
            plane(&pattern(n, 59, 4 + f * 19, 16), w, true),
            plane(&pattern(n, 23, 6 + f * 23, 16), w, true),
        ];
        let src = VideoFrame {
            pts: Some(f as i64),
            planes,
        };
        enc.send_frame(&Frame::Video(src.clone())).expect("encode");
        let pkt = enc.receive_packet().expect("packet");
        assert_eq!(pkt.flags.keyframe, f == 0, "frame {f} §4.4 keyframe flag");
        dec.send_packet(&pkt).expect("send");
        let Frame::Video(out) = dec.receive_frame().expect("frame") else {
            panic!("video frame");
        };
        for (i, (got, want)) in out
            .image_planes()
            .iter()
            .zip(src.image_planes().iter())
            .enumerate()
        {
            assert_eq!(got.data, want.data, "frame {f} plane {i}");
        }
    }
}
