//! End-to-end tests for **non-keyframe coder-state carry** on the
//! decode side (RFC 9043 §3.8.1.3 / §3.8.2.5 / §5 third paragraph).
//!
//! RFC 9043 §3.8.1.3 (range coder) and §3.8.2.5 (Golomb-Rice VLC) both
//! read: "When the keyframe value (see Section 4.4) is **1**, all
//! [range coder / VLC] state variables are set to their initial state."
//! The non-keyframe (`keyframe == 0`) case is the negation: the
//! per-context coder state is **not** re-initialised — it continues from
//! the value it held at the end of the previous Frame's matching Slice.
//! §5 third paragraph keeps the Slice geometry stable across Frames, so
//! the Slices line up positionally and the carry is indexed by forward
//! Slice index.
//!
//! Because the four shipped v3 fixtures are all single-Frame keyframes
//! (`v3-frame-mt` is itself a single keyframe), this round verifies the
//! carry with a synthetic multi-Frame stream built by the in-tree
//! encoder's matching carry-aware write path
//! ([`encode_frame_range_coder_with_carry`]) and decoded back through
//! the stateful [`Ffv1DecodeSession`] (which owns the read-side carry)
//! and the carry-aware driver [`decode_frame_with_carry`]:
//!
//! * a genuine keyframe → non-keyframe round-trip reconstructs both
//!   Frames bit-exactly (the carry is *correct*);
//! * decoding the non-keyframe **without** the carry (the stateless
//!   `decode_frame`, which always re-initialises state) produces
//!   different pixels — proving the carry is *load-bearing*, not a
//!   no-op;
//! * the carry is per-Slice on a 2×2 grid;
//! * a keyframe mid-stream re-initialises state (the carry is ignored
//!   when `keyframe == 1`).

use oxideav_ffv1::{
    decode_frame, decode_frame_with_carry, encode_frame_range_coder_with_carry,
    encode_frame_with_carry, ColorspaceType, DecodeOptions, DecodedFrame, DecodedFramePlane,
    Ffv1ConfigurationRecord, Ffv1DecodeSession, Ffv1EncodeCarry, Ffv1FrameCarry, Ffv1SliceHeader,
    Ffv1Version, FramePixelDimensions, PictureStructure, QuantizationTableSet,
    MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

// -- shared fixture helpers ----------------------------------------------

fn gray_cr(num_h: u32, num_v: u32, intra: Option<bool>) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type: 1,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: 8,
        chroma_planes: false,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra,
        initial_state_delta: None,
    }
}

/// A non-trivial Quantization Table Set: a small monotone ramp on the
/// first sub-table so the §3.5 context computation lands on several
/// distinct contexts, exercising more than one per-context window (the
/// carry is per-context, so a multi-context table makes the carry
/// observable).
fn ramp_qts() -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    for (k, slot) in tables[0].iter_mut().enumerate() {
        *slot = ((k as i32) % 5) - 2;
    }
    QuantizationTableSet {
        tables,
        context_count: 64,
    }
}

/// A YCbCr Golomb-Rice (`coder_type == 0`) gray Configuration Record.
fn gray_golomb_cr(num_h: u32, num_v: u32, intra: Option<bool>) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        coder_type: 0,
        ..gray_cr(num_h, num_v, intra)
    }
}

/// A multi-context Quantization Table Set whose absolute context is
/// **never 0** (max |context| = 4 → 5 slots), so the §3.8.2.2 run mode
/// (which engages only at context 0) never triggers. That isolates the
/// §3.8.2.5 Golomb-Rice inter-Frame carry under test from the orthogonal
/// §3.8.2.2.1 run-mode-first-pixel encoder limitation, while still
/// exercising several distinct per-context VLC windows (the carry is
/// per-context, so a multi-context table makes the carry observable).
fn golomb_ramp_qts() -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    for (d, slot) in tables[0].iter_mut().enumerate() {
        let sv = if d < 128 { d as i32 } else { d as i32 - 256 };
        let mag = 1 + (sv.unsigned_abs() as i32).min(48) / 16; // 1..=4
        *slot = if sv < 0 { -mag } else { mag };
    }
    QuantizationTableSet {
        tables,
        context_count: 5,
    }
}

fn header(x: u32, y: u32, w: u32, h: u32) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x: x,
        slice_y: y,
        slice_width: w,
        slice_height: h,
        quant_table_set_index_count: 2,
        quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: PictureStructure::Unknown,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

fn gray_frame(samples: Vec<i32>, w: u32, h: u32) -> DecodedFrame {
    DecodedFrame {
        planes: vec![DecodedFramePlane {
            plane_index: 0,
            width: w,
            height: h,
            samples,
        }],
        width: w,
        height: h,
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::YCbCr,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// Deterministic pseudo-random 8-bit pixels (a xorshift walk so adjacent
/// pixels differ — a flat plane would sit in run mode and barely move
/// the coder state, hiding the carry).
fn pixels(w: u32, h: u32, seed: u32) -> Vec<i32> {
    let mut s = seed | 1;
    (0..(w * h))
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s & 0xFF) as i32
        })
        .collect()
}

// -- genuine keyframe → non-keyframe round-trip --------------------------

#[test]
fn nonkeyframe_carry_round_trips_single_slice() {
    // Encode two Frames: Frame 0 a keyframe, Frame 1 a non-keyframe
    // whose per-context coder state continues from Frame 0 (§3.8.1.3).
    // Decode both through a session that owns the read-side carry; both
    // reconstruct bit-exactly.
    let cr = gray_cr(1, 1, Some(false));
    let qts = vec![ramp_qts()];
    let (w, h) = (12u32, 9u32);
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let headers = [header(0, 0, 1, 1)];

    let frame0 = gray_frame(pixels(w, h, 0x1234), w, h);
    let frame1 = gray_frame(pixels(w, h, 0x9abc), w, h);

    let mut enc_carry: Option<Ffv1EncodeCarry> = None;
    let bytes0 = encode_frame_range_coder_with_carry(
        &frame0,
        &cr,
        &qts,
        &headers,
        true,
        true,
        &mut enc_carry,
    )
    .expect("encode keyframe");
    let bytes1 = encode_frame_range_coder_with_carry(
        &frame1,
        &cr,
        &qts,
        &headers,
        true,
        false, // non-keyframe — resumes Frame 0's coder state
        &mut enc_carry,
    )
    .expect("encode non-keyframe");

    let mut session = Ffv1DecodeSession::new(cr, qts, dims, true);
    let dec0 = session.decode_next_frame(&bytes0).expect("decode keyframe");
    let dec1 = session
        .decode_next_frame(&bytes1)
        .expect("decode non-keyframe");

    assert!(dec0.keyframe);
    assert!(!dec1.keyframe, "Frame 1 is a non-keyframe");
    assert_eq!(dec0.planes[0].samples, frame0.planes[0].samples);
    assert_eq!(
        dec1.planes[0].samples, frame1.planes[0].samples,
        "non-keyframe reconstructs bit-exactly only when the §3.8.1.3 \
         coder state carried from Frame 0"
    );
    assert_eq!(session.frames_observed(), 2);
}

#[test]
fn nonkeyframe_without_carry_decodes_differently_proving_carry_is_load_bearing() {
    // The same non-keyframe bytes, decoded through the STATELESS
    // `decode_frame` (which always re-initialises coder state to the
    // §3.8.1.3 `128` window regardless of the §4.4 keyframe value),
    // produce the WRONG pixels — the carry is the only thing that makes
    // the non-keyframe decode correctly. This is the discriminating
    // test: if the carry were a no-op, this assertion would fail.
    let cr = gray_cr(1, 1, Some(false));
    let qts = vec![ramp_qts()];
    let (w, h) = (12u32, 9u32);
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let headers = [header(0, 0, 1, 1)];

    let frame0 = gray_frame(pixels(w, h, 0x1234), w, h);
    let frame1 = gray_frame(pixels(w, h, 0x9abc), w, h);

    let mut enc_carry: Option<Ffv1EncodeCarry> = None;
    let _bytes0 = encode_frame_range_coder_with_carry(
        &frame0,
        &cr,
        &qts,
        &headers,
        true,
        true,
        &mut enc_carry,
    )
    .expect("encode keyframe");
    let bytes1 = encode_frame_range_coder_with_carry(
        &frame1,
        &cr,
        &qts,
        &headers,
        true,
        false,
        &mut enc_carry,
    )
    .expect("encode non-keyframe");

    // Stateless decode: no previous-Frame state, so the per-context
    // windows start at 128 — diverges from the encoder, which resumed
    // Frame 0's evolved windows.
    let stateless = decode_frame(&bytes1, &cr, &qts, dims, true).expect("stateless decode");
    assert_ne!(
        stateless.planes[0].samples, frame1.planes[0].samples,
        "a non-keyframe decoded WITHOUT the §3.8.1.3 carry must NOT match \
         the intended pixels — otherwise the carry is doing nothing"
    );

    // And the carry-aware driver, fed the matching previous-Frame
    // snapshot, does match. Re-run the carry-aware decode standalone to
    // confirm `decode_frame_with_carry` is the load-bearing entry point.
    let mut dec_carry: Option<Ffv1FrameCarry> = None;
    let _ = decode_frame_with_carry(
        &encode_frame_range_coder_with_carry(&frame0, &cr, &qts, &headers, true, true, &mut None)
            .unwrap(),
        &cr,
        &qts,
        dims,
        true,
        DecodeOptions::strict(),
        &mut dec_carry,
    )
    .expect("decode keyframe to seed the carry");
    let dec1 = decode_frame_with_carry(
        &bytes1,
        &cr,
        &qts,
        dims,
        true,
        DecodeOptions::strict(),
        &mut dec_carry,
    )
    .expect("decode non-keyframe with carry");
    assert_eq!(dec1.planes[0].samples, frame1.planes[0].samples);
}

// -- multi-Slice carry on a 2×2 grid -------------------------------------

#[test]
fn nonkeyframe_carry_is_per_slice_on_2x2_grid() {
    // Each of the four Slices on a 2×2 grid carries its own per-context
    // state independently across the Frame boundary. A multi-Frame
    // stream (keyframe + two non-keyframes) round-trips bit-exactly
    // through the session, which proves each Slice's carry stays in
    // lockstep with the encoder's matching Slice.
    let cr = gray_cr(2, 2, Some(false));
    let qts = vec![ramp_qts()];
    let (w, h) = (16u32, 12u32);
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let headers = [
        header(0, 0, 1, 1),
        header(1, 0, 1, 1),
        header(0, 1, 1, 1),
        header(1, 1, 1, 1),
    ];

    let frames: Vec<DecodedFrame> = [0x11u32, 0x2222, 0x33333]
        .iter()
        .map(|&s| gray_frame(pixels(w, h, s), w, h))
        .collect();

    let mut enc_carry: Option<Ffv1EncodeCarry> = None;
    let mut all_bytes = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let bytes = encode_frame_range_coder_with_carry(
            f,
            &cr,
            &qts,
            &headers,
            true,
            i == 0, // Frame 0 keyframe; the rest non-keyframes
            &mut enc_carry,
        )
        .expect("encode");
        all_bytes.push(bytes);
    }

    let mut session = Ffv1DecodeSession::new(cr, qts, dims, true);
    for (i, (bytes, f)) in all_bytes.iter().zip(frames.iter()).enumerate() {
        let dec = session.decode_next_frame(bytes).expect("decode");
        assert_eq!(dec.keyframe, i == 0);
        assert_eq!(
            dec.planes[0].samples, f.planes[0].samples,
            "Frame {i} (2x2 grid) must reconstruct bit-exactly through the \
             per-Slice §3.8.1.3 carry"
        );
        assert_eq!(dec.slice_headers.len(), 4);
    }
    assert_eq!(session.frames_observed(), 3);
}

// -- a mid-stream keyframe re-initialises state --------------------------

#[test]
fn mid_stream_keyframe_reinitialises_state_ignoring_carry() {
    // RFC 9043 §3.8.1.3: "When the keyframe value is 1, all ... state
    // variables are set to their initial state." A keyframe mid-stream
    // therefore ignores any carry. We encode Frame 1 as a keyframe and
    // confirm it decodes identically whether or not a previous Frame's
    // carry is present — i.e. the keyframe is self-contained.
    let cr = gray_cr(1, 1, Some(false));
    let qts = vec![ramp_qts()];
    let (w, h) = (10u32, 8u32);
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let headers = [header(0, 0, 1, 1)];

    let frame0 = gray_frame(pixels(w, h, 0xAAAA), w, h);
    let frame1 = gray_frame(pixels(w, h, 0xBBBB), w, h);

    // Frame 1 encoded as a KEYFRAME (keyframe = true) after Frame 0, so
    // the encoder ignores Frame 0's carry and re-initialises.
    let mut enc_carry: Option<Ffv1EncodeCarry> = None;
    let _bytes0 = encode_frame_range_coder_with_carry(
        &frame0,
        &cr,
        &qts,
        &headers,
        true,
        true,
        &mut enc_carry,
    )
    .expect("encode Frame 0");
    let bytes1_kf = encode_frame_range_coder_with_carry(
        &frame1,
        &cr,
        &qts,
        &headers,
        true,
        true,
        &mut enc_carry,
    )
    .expect("encode Frame 1 as keyframe");

    // Decode Frame 1 standalone via the stateless driver (no carry) and
    // via a session that DOES carry Frame 0's state. Both must agree,
    // because a keyframe re-initialises and ignores the carry.
    let standalone = decode_frame(&bytes1_kf, &cr, &qts, dims, true).expect("standalone");

    let mut session = Ffv1DecodeSession::new(cr.clone(), qts.clone(), dims, true);
    let bytes0_again =
        encode_frame_range_coder_with_carry(&frame0, &cr, &qts, &headers, true, true, &mut None)
            .expect("re-encode Frame 0");
    session.decode_next_frame(&bytes0_again).expect("decode F0");
    let via_session = session
        .decode_next_frame(&bytes1_kf)
        .expect("decode F1 keyframe through a carrying session");

    assert!(via_session.keyframe);
    assert_eq!(via_session.planes[0].samples, frame1.planes[0].samples);
    assert_eq!(
        via_session.planes[0].samples, standalone.planes[0].samples,
        "a keyframe decodes identically with or without a carry (§3.8.1.3 \
         re-initialises all state)"
    );
}

// -- Golomb-Rice (coder_type == 0) inter-Frame carry ---------------------
//
// RFC 9043 §3.8.2.5 is the Golomb-Rice analogue of §3.8.1.3: "When the
// keyframe value ... is 1, all [VLC] state variables are set to their
// initial state." A non-keyframe continues each per-context VLC window
// (`drift` / `error_sum` / `bias` / `count`) from the value it held at
// the end of the previous Frame's matching Slice. These tests drive the
// new write-side `encode_frame_golomb_rice_with_carry` (reached through
// the unified `encode_frame_with_carry` dispatcher) and decode back
// through the read-side `decode_frame_with_carry` / `Ffv1DecodeSession`,
// completing the "both coders, end-to-end" inter-Frame encode milestone.

#[test]
fn golomb_nonkeyframe_carry_round_trips_single_slice() {
    // Frame 0 keyframe, Frame 1 non-keyframe whose §3.8.2.5 per-context
    // VLC windows continue from Frame 0. Both reconstruct bit-exactly
    // through a session that owns the read-side carry.
    let cr = gray_golomb_cr(1, 1, Some(false));
    let qts = vec![golomb_ramp_qts()];
    let (w, h) = (12u32, 9u32);
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let headers = [header(0, 0, 1, 1)];

    let frame0 = gray_frame(pixels(w, h, 0x1234), w, h);
    let frame1 = gray_frame(pixels(w, h, 0x9abc), w, h);

    let mut enc_carry: Option<Ffv1EncodeCarry> = None;
    let bytes0 =
        encode_frame_with_carry(&frame0, &cr, &qts, &headers, true, true, &mut enc_carry).unwrap();
    let bytes1 =
        encode_frame_with_carry(&frame1, &cr, &qts, &headers, true, false, &mut enc_carry).unwrap();

    let mut session = Ffv1DecodeSession::new(cr, qts, dims, true);
    let dec0 = session.decode_next_frame(&bytes0).expect("decode keyframe");
    let dec1 = session
        .decode_next_frame(&bytes1)
        .expect("decode non-keyframe");

    assert!(dec0.keyframe);
    assert!(!dec1.keyframe);
    assert_eq!(dec0.planes[0].samples, frame0.planes[0].samples);
    assert_eq!(
        dec1.planes[0].samples, frame1.planes[0].samples,
        "Golomb-Rice non-keyframe reconstructs bit-exactly only when the \
         §3.8.2.5 VLC state carried from Frame 0"
    );
    assert_eq!(session.frames_observed(), 2);
}

#[test]
fn golomb_nonkeyframe_without_carry_decodes_differently_proving_carry_is_load_bearing() {
    // The same non-keyframe bytes through the STATELESS `decode_frame`
    // (which re-initialises the §3.8.2.5 VLC windows regardless of the
    // §4.4 keyframe value) produce the WRONG pixels — the carry is the
    // only thing that makes the Golomb-Rice non-keyframe decode correctly.
    let cr = gray_golomb_cr(1, 1, Some(false));
    let qts = vec![golomb_ramp_qts()];
    let (w, h) = (12u32, 9u32);
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let headers = [header(0, 0, 1, 1)];

    let frame0 = gray_frame(pixels(w, h, 0x1234), w, h);
    let frame1 = gray_frame(pixels(w, h, 0x9abc), w, h);

    let mut enc_carry: Option<Ffv1EncodeCarry> = None;
    let _bytes0 =
        encode_frame_with_carry(&frame0, &cr, &qts, &headers, true, true, &mut enc_carry).unwrap();
    let bytes1 =
        encode_frame_with_carry(&frame1, &cr, &qts, &headers, true, false, &mut enc_carry).unwrap();

    let stateless = decode_frame(&bytes1, &cr, &qts, dims, true).expect("stateless decode");
    assert_ne!(
        stateless.planes[0].samples, frame1.planes[0].samples,
        "a Golomb-Rice non-keyframe decoded WITHOUT the §3.8.2.5 carry must \
         NOT match the intended pixels — otherwise the carry is doing nothing"
    );

    // The carry-aware driver, seeded from the matching previous Frame,
    // does match.
    let mut dec_carry: Option<Ffv1FrameCarry> = None;
    let _ = decode_frame_with_carry(
        &encode_frame_with_carry(&frame0, &cr, &qts, &headers, true, true, &mut None).unwrap(),
        &cr,
        &qts,
        dims,
        true,
        DecodeOptions::strict(),
        &mut dec_carry,
    )
    .expect("decode keyframe to seed the carry");
    let dec1 = decode_frame_with_carry(
        &bytes1,
        &cr,
        &qts,
        dims,
        true,
        DecodeOptions::strict(),
        &mut dec_carry,
    )
    .expect("decode non-keyframe with carry");
    assert_eq!(dec1.planes[0].samples, frame1.planes[0].samples);
}

#[test]
fn golomb_nonkeyframe_carry_is_per_slice_on_2x2_grid() {
    // Each of the four Slices on a 2×2 grid carries its own §3.8.2.5
    // per-context VLC state independently across the Frame boundary. A
    // keyframe + two non-keyframes round-trip bit-exactly through the
    // session.
    let cr = gray_golomb_cr(2, 2, Some(false));
    let qts = vec![golomb_ramp_qts()];
    let (w, h) = (16u32, 12u32);
    let dims = FramePixelDimensions::new(w, h).unwrap();
    let headers = [
        header(0, 0, 1, 1),
        header(1, 0, 1, 1),
        header(0, 1, 1, 1),
        header(1, 1, 1, 1),
    ];

    let frames: Vec<DecodedFrame> = [0x11u32, 0x2222, 0x33333]
        .iter()
        .map(|&s| gray_frame(pixels(w, h, s), w, h))
        .collect();

    let mut enc_carry: Option<Ffv1EncodeCarry> = None;
    let mut all_bytes = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let bytes =
            encode_frame_with_carry(f, &cr, &qts, &headers, true, i == 0, &mut enc_carry).unwrap();
        all_bytes.push(bytes);
    }

    let mut session = Ffv1DecodeSession::new(cr, qts, dims, true);
    for (i, (bytes, f)) in all_bytes.iter().zip(frames.iter()).enumerate() {
        let dec = session.decode_next_frame(bytes).expect("decode");
        assert_eq!(dec.keyframe, i == 0);
        assert_eq!(
            dec.planes[0].samples, f.planes[0].samples,
            "Golomb Frame {i} (2x2 grid) must reconstruct bit-exactly through \
             the per-Slice §3.8.2.5 carry"
        );
        assert_eq!(dec.slice_headers.len(), 4);
    }
    assert_eq!(session.frames_observed(), 3);
}

#[test]
fn golomb_keyframe_round_trip_matches_legacy_keyframe_only_encoder() {
    // `encode_frame_with_carry(.., keyframe = true, ..)` on the
    // Golomb-Rice path must be byte-for-byte identical to the historical
    // keyframe-only `encode_frame` (the carry delegate's keyframe branch
    // ignores the carry and freshly keyframe-initialises every slot).
    use oxideav_ffv1::encode_frame;
    let cr = gray_golomb_cr(1, 1, None);
    let qts = vec![golomb_ramp_qts()];
    let (w, h) = (12u32, 9u32);
    let headers = [header(0, 0, 1, 1)];
    let frame = gray_frame(pixels(w, h, 0x55aa), w, h);

    let legacy = encode_frame(&frame, &cr, &qts, &headers, true).unwrap();
    let mut carry: Option<Ffv1EncodeCarry> = None;
    let via_carry =
        encode_frame_with_carry(&frame, &cr, &qts, &headers, true, true, &mut carry).unwrap();
    assert_eq!(
        legacy, via_carry,
        "a keyframe through the carry dispatcher is byte-identical to the \
         legacy keyframe-only encoder"
    );
}
