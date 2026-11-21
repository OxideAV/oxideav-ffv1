//! Slice-parallel encode and the registry encoder options.
//!
//! * `encode_frame_with_options` emits byte-identical Frames at every
//!   thread budget (RFC 9043 §4.5: Slices are independent), keyframes
//!   and non-keyframes, both traversals, both coders.
//! * `version=3` builds a Configuration Record from the pixel format and
//!   publishes it; the stream round-trips through the registry decoder.
//! * The Slice layout election follows the granted budget, satisfies the
//!   RFC 9043 §5 Slice-size restriction, and an explicit `slices` makes
//!   the output independent of the budget.

use oxideav_core::{
    CodecId, CodecOptions, CodecParameters, ExecutionContext, Frame, Packet, PixelFormat,
    RuntimeContext, VideoFrame, VideoPlane,
};
use oxideav_ffv1::{
    CODEC_ID_STR, ColorspaceType, DecodedFrame, DecodedFramePlane, EncodeOptions,
    Ffv1ConfigurationRecord, Ffv1EncodeCarry, Ffv1SliceHeader, Ffv1Version,
    MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS, PictureStructure,
    QuantizationTableSet, encode_frame_with_options, parse_quantization_table_sets, register,
    walk_trailer_chain,
};

fn xorshift(x: &mut u32) -> u32 {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    *x
}

fn samples(w: u32, h: u32, bits: u32, seed: u32) -> Vec<i32> {
    let mask = (1i32 << bits) - 1;
    let mut rng = seed | 1;
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 2 + y * 3) as i32) << bits.saturating_sub(8);
            let v = if (y / 24) % 2 == 0 {
                base + (xorshift(&mut rng) & 7) as i32
            } else {
                77
            };
            out.push(v & mask);
        }
    }
    out
}

fn quant_tables() -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    for idx in 0..256usize {
        let d = if idx < 128 {
            idx as i32
        } else {
            idx as i32 - 256
        };
        let l = match d.unsigned_abs() {
            0 => 0,
            1..=2 => 1,
            3..=6 => 2,
            7..=14 => 3,
            15..=30 => 4,
            _ => 5,
        };
        let l = if d < 0 { -l } else { l };
        tables[0][idx] = l;
        tables[1][idx] = l * 11;
        tables[2][idx] = l * 121;
    }
    QuantizationTableSet {
        tables,
        context_count: 666,
    }
}

#[test]
fn direct_encode_is_byte_identical_at_every_budget() {
    const W: u32 = 400;
    const H: u32 = 300;
    for (rgb, bits) in [(false, 8u32), (false, 10), (true, 8), (true, 16)] {
        for coder in [0u32, 1, 2] {
            let mut deltas = [0i32; NUM_TRANSITION_DELTAS];
            if coder == 2 {
                for (i, d) in deltas.iter_mut().enumerate().skip(1).step_by(9) {
                    *d = (i % 3) as i32 - 1;
                }
            }
            let cr = Ffv1ConfigurationRecord {
                version: Ffv1Version::V3,
                micro_version: Some(4),
                coder_type: coder,
                state_transition_delta: deltas,
                colorspace_type: if rgb {
                    ColorspaceType::Rgb
                } else {
                    ColorspaceType::YCbCr
                },
                bits_per_raw_sample: bits,
                chroma_planes: true,
                log2_h_chroma_subsample: u32::from(!rgb),
                log2_v_chroma_subsample: u32::from(!rgb),
                extra_plane: false,
                num_h_slices: Some(4),
                num_v_slices: Some(3),
                quant_table_set_count: Some(1),
                ec: Some(1),
                intra: Some(false),
                initial_state_delta: None,
            };
            let headers: Vec<Ffv1SliceHeader> = (0..3u32)
                .flat_map(|sy| {
                    (0..4u32).map(move |sx| Ffv1SliceHeader {
                        slice_x: sx,
                        slice_y: sy,
                        slice_width: 1,
                        slice_height: 1,
                        quant_table_set_index_count: 2,
                        quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
                        picture_structure: PictureStructure::Progressive,
                        picture_structure_raw: 0,
                        sar_num: 0,
                        sar_den: 0,
                    })
                })
                .collect();
            let (cw, ch) = if rgb { (W, H) } else { (W / 2, H / 2) };
            let frames: Vec<DecodedFrame> = (0..3u32)
                .map(|f| DecodedFrame {
                    planes: vec![
                        DecodedFramePlane {
                            plane_index: 0,
                            width: W,
                            height: H,
                            samples: samples(W, H, bits, 100 + f),
                        },
                        DecodedFramePlane {
                            plane_index: 1,
                            width: cw,
                            height: ch,
                            samples: samples(cw, ch, bits, 200 + f),
                        },
                        DecodedFramePlane {
                            plane_index: 2,
                            width: cw,
                            height: ch,
                            samples: samples(cw, ch, bits, 300 + f),
                        },
                    ],
                    width: W,
                    height: H,
                    bits_per_raw_sample: bits,
                    colorspace: cr.colorspace_type,
                    keyframe: true,
                    slice_headers: Vec::new(),
                })
                .collect();
            let qts = [quant_tables()];
            let mut reference: Vec<Vec<u8>> = Vec::new();
            for threads in [0usize, 1, 2, 5, 12, 100] {
                let mut carry: Option<Ffv1EncodeCarry> = None;
                for (i, frame) in frames.iter().enumerate() {
                    let bytes = encode_frame_with_options(
                        frame,
                        &cr,
                        &qts,
                        &headers,
                        true,
                        i == 0,
                        &mut carry,
                        EncodeOptions::default().with_threads(threads),
                    )
                    .expect("encodes");
                    if threads == 0 {
                        reference.push(bytes);
                    } else {
                        assert!(
                            bytes == reference[i],
                            "rgb={rgb} bits={bits} coder={coder} threads={threads} frame={i}"
                        );
                    }
                }
            }
        }
    }
}

fn video_frame(w: u32, h: u32, seed: u32) -> VideoFrame {
    let plane = |pw: u32, ph: u32, s: u32| VideoPlane {
        stride: pw as usize,
        data: samples(pw, ph, 8, s).iter().map(|&v| v as u8).collect(),
    };
    VideoFrame {
        pts: Some(i64::from(seed)),
        planes: vec![
            plane(w, h, seed),
            plane(w / 2, h / 2, seed + 50),
            plane(w / 2, h / 2, seed + 90),
        ],
    }
}

/// Encode `frames` through the registry with `options` / `threads`.
fn registry_encode(
    w: u32,
    h: u32,
    options: CodecOptions,
    threads: usize,
    frames: &[VideoFrame],
) -> (Vec<u8>, Vec<Packet>) {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    params.options = options;
    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder");
    enc.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut packets = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(f.clone()))
            .expect("send_frame");
        packets.push(enc.receive_packet().expect("receive_packet"));
    }
    (enc.output_params().extradata.clone(), packets)
}

fn registry_decode(
    w: u32,
    h: u32,
    extradata: &[u8],
    packets: &[Packet],
    threads: usize,
) -> Vec<VideoFrame> {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.extradata = extradata.to_vec();
    let mut dec = ctx.codecs.first_decoder(&params).expect("decoder");
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    packets
        .iter()
        .map(|p| {
            dec.send_packet(p).expect("send_packet");
            match dec.receive_frame().expect("receive_frame") {
                Frame::Video(v) => v,
                _ => panic!("video frame expected"),
            }
        })
        .collect()
}

fn assert_round_trip(got: &[VideoFrame], want: &[VideoFrame]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        for (a, b) in g.planes.iter().zip(&w.planes) {
            assert!(a.data == b.data, "round trip must be bit-exact");
        }
    }
}

#[test]
fn version_3_option_builds_a_record_and_round_trips() {
    let (w, h) = (640u32, 360u32);
    let frames: Vec<VideoFrame> = (0..4).map(|i| video_frame(w, h, 7 + i)).collect();
    for coder in ["range", "golomb"] {
        for gop in ["0", "1", "3"] {
            let options = CodecOptions::new()
                .set("version", "3")
                .set("coder", coder)
                .set("gop", gop);
            let (extradata, packets) = registry_encode(w, h, options, 1, &frames);
            let parsed = parse_quantization_table_sets(&extradata).expect("record parses");
            assert_eq!(parsed.record.version, Ffv1Version::V3);
            assert_eq!(parsed.record.micro_version, Some(4));
            assert_eq!(
                parsed.record.coder_type,
                u32::from(coder == "range"),
                "coder={coder}"
            );
            assert_eq!(parsed.record.ec, Some(1));
            assert_eq!(parsed.record.intra, Some(gop != "3"), "gop={gop}");
            let keys: Vec<bool> = packets.iter().map(|p| p.flags.keyframe).collect();
            let want_keys = if gop == "3" {
                vec![true, false, false, true]
            } else {
                vec![true; 4]
            };
            assert_eq!(keys, want_keys, "gop={gop}");
            for threads in [1usize, 6] {
                let got = registry_decode(w, h, &extradata, &packets, threads);
                assert_round_trip(&got, &frames);
            }
        }
    }
}

#[test]
fn slice_layout_election_follows_the_budget_and_section_5() {
    let (w, h) = (640u32, 360u32);
    let frames = vec![video_frame(w, h, 1)];
    let options = || CodecOptions::new().set("version", "3");
    for (threads, want_slices) in [(1usize, 4usize), (4, 4), (6, 8), (8, 8), (16, 16)] {
        let (extradata, packets) = registry_encode(w, h, options(), threads, &frames);
        let parsed = parse_quantization_table_sets(&extradata).expect("record parses");
        // The record's raster is the same at every budget: the election
        // only groups its cells.
        assert_eq!(parsed.record.num_h_slices, Some(4));
        assert_eq!(parsed.record.num_v_slices, Some(4));
        let slices = walk_trailer_chain(&packets[0].data, true).expect("trailer chain");
        assert_eq!(slices.len(), want_slices, "threads={threads}");
        // RFC 9043 §5: slice_width * slice_height <= 16 / 4 above CIF —
        // enforced by the decoder, which must accept every layout.
        let got = registry_decode(w, h, &extradata, &packets, threads);
        assert_round_trip(&got, &frames);
    }

    // At or below CIF a single Slice is elected.
    let small = vec![video_frame(352, 288, 3)];
    let (extradata, packets) = registry_encode(352, 288, options(), 8, &small);
    assert_eq!(
        walk_trailer_chain(&packets[0].data, true)
            .expect("trailer chain")
            .len(),
        1
    );
    assert_round_trip(&registry_decode(352, 288, &extradata, &packets, 1), &small);
}

#[test]
fn explicit_slices_make_output_independent_of_the_budget() {
    let (w, h) = (640u32, 360u32);
    let frames: Vec<VideoFrame> = (0..3).map(|i| video_frame(w, h, 20 + i)).collect();
    let options = || {
        CodecOptions::new()
            .set("version", "3")
            .set("slices", "12")
            .set("gop", "2")
    };
    let (extradata, reference) = registry_encode(w, h, options(), 1, &frames);
    let parsed = parse_quantization_table_sets(&extradata).expect("record parses");
    assert_eq!(
        (parsed.record.num_h_slices, parsed.record.num_v_slices),
        (Some(4), Some(3))
    );
    for threads in [2usize, 7, 32] {
        let (extra, packets) = registry_encode(w, h, options(), threads, &frames);
        assert_eq!(extra, extradata);
        for (a, b) in packets.iter().zip(&reference) {
            assert!(a.data == b.data, "threads={threads}");
        }
    }
    assert_round_trip(&registry_decode(w, h, &extradata, &reference, 3), &frames);
}

#[test]
fn invalid_option_combinations_are_rejected() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(640);
    params.height = Some(360);
    params.pixel_format = Some(PixelFormat::Yuv420P);
    // Fewer than four Slices above CIF violates RFC 9043 §5.
    params.options = CodecOptions::new().set("version", "3").set("slices", "2");
    assert!(ctx.codecs.first_encoder(&params).is_err());
    // The encoder refuses unknown keys. The decoder leaves keys it does
    // not define alone (a stream's parameters also carry its container's
    // track records) and refuses a malformed value of its own.
    params.options = CodecOptions::new().set("nope", "1");
    assert!(ctx.codecs.first_encoder(&params).is_err());
    assert!(ctx.codecs.first_decoder(&params).is_ok());
    params.options = CodecOptions::new().set("concealment", "perhaps");
    assert!(ctx.codecs.first_decoder(&params).is_err());
}
