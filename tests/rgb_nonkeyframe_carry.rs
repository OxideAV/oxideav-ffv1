//! End-to-end tests for **non-keyframe coder-state carry** on the RGB /
//! line-major path (RFC 9043 §3.7.1 RCT + §3.8.1.3 range coder /
//! §3.8.2.5 Golomb-Rice VLC + §5 third paragraph).
//!
//! RFC 9043 §3.8.1.3 (range coder) and §3.8.2.5 (Golomb-Rice VLC) read:
//! "When the keyframe value (see Section 4.4) is **1**, all [coder]
//! state variables are set to their initial state." The non-keyframe
//! (`keyframe == 0`) case is the negation: the per-context coder state
//! is **not** re-initialised — it continues from the value it held at
//! the end of the previous Frame's matching Slice. §5 third paragraph
//! keeps the Slice geometry stable across Frames, so the carry is
//! indexed by forward Slice index.
//!
//! The read-side RGB carry (`decode_frame_rgb_with_carry`) landed in an
//! earlier round; this suite exercises the matching **write-side** carry
//! (`encode_frame_rgb_with_carry`). With both halves present a synthetic
//! multi-Frame RGB non-keyframe stream can be produced and decoded back
//! bit-exactly the way `tests/nonkeyframe_carry.rs` does for YCbCr:
//!
//! * a genuine keyframe → non-keyframe round-trip reconstructs both
//!   Frames bit-exactly (the carry is *correct*) — on all three
//!   `coder_type`s, so both the range-coder and Golomb-Rice carry
//!   channels are exercised;
//! * decoding the non-keyframe **without** the carry (the stateless
//!   `decode_frame_rgb`, which always re-initialises state) produces
//!   different pixels — proving the carry is *load-bearing*;
//! * the carry is per-Slice on a 2×2 grid;
//! * a keyframe mid-stream re-initialises state (the carry is ignored
//!   when `keyframe == 1`).
//!
//! The range-coded paths (`coder_type ∈ {1, 2}`) use a multi-context
//! Quantization Table Set so several per-context windows participate.
//! The Golomb-Rice path (`coder_type == 0`) uses a single-context table:
//! the §4.6.6-slot VLC window still evolves through the Slice and must
//! survive the Frame boundary, so the carry is genuinely exercised — and
//! a multi-context table on the Golomb-RGB path is a separate, latent
//! pre-existing issue (no in-tree Golomb-RGB test uses a non-flat
//! table), out of scope for the carry work under test here.

use oxideav_ffv1::{
    decode_frame_rgb, decode_frame_rgb_with_carry, encode_frame_rgb_with_carry, ColorspaceType,
    DecodeOptions, DecodedFrame, DecodedFramePlane, Ffv1ConfigurationRecord, Ffv1EncodeCarry,
    Ffv1FrameCarry, Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure,
    QuantizationTableSet, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

// -- shared fixture helpers ----------------------------------------------

fn rgb_cr(num_h: u32, num_v: u32, coder_type: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::Rgb,
        bits_per_raw_sample: 8,
        chroma_planes: true, // RGB always has three Planes
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(num_h),
        num_v_slices: Some(num_v),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra: Some(false),
        initial_state_delta: None,
    }
}

/// A multi-context Quantization Table Set: a small ramp on the first
/// sub-table so the §3.5 context computation lands on several distinct
/// contexts, exercising more than one per-context window (the carry is
/// per-context, so a multi-context table makes the carry observable
/// across Frames). Used on the range-coded paths.
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

/// A single-context Quantization Table Set — every neighbour
/// configuration maps to context `c`. The slot's per-context VLC window
/// still evolves through the Slice (so the carry is exercised), but the
/// table stays on the flat-context path the in-tree Golomb-RGB tests
/// already validate. Used on the Golomb-Rice path.
fn flat_qts(c: u32) -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    tables[0] = [c as i32; 256];
    QuantizationTableSet {
        tables,
        context_count: c + 1,
    }
}

/// The Quantization Table Set appropriate for a given `coder_type`:
/// multi-context for the range coder, single-context for Golomb-Rice
/// (see the module doc for why).
fn qts_for(coder_type: u32) -> Vec<QuantizationTableSet> {
    if coder_type == 0 {
        vec![flat_qts(7)]
    } else {
        vec![ramp_qts()]
    }
}

fn header(x: u32, y: u32, w: u32, h: u32) -> Ffv1SliceHeader {
    Ffv1SliceHeader {
        slice_x: x,
        slice_y: y,
        slice_width: w,
        slice_height: h,
        quant_table_set_index_count: 2, // luma slot + shared chroma slot
        quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

/// Deterministic pseudo-random 8-bit pixels (a xorshift walk so adjacent
/// pixels differ — a flat plane would sit in run mode and barely move
/// the coder state, hiding the carry). Each colour Plane gets a distinct
/// seed.
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

fn rgb_frame(w: u32, h: u32, seed: u32) -> DecodedFrame {
    DecodedFrame {
        planes: vec![
            DecodedFramePlane {
                plane_index: 0,
                width: w,
                height: h,
                samples: pixels(w, h, seed ^ 0xAAAA),
            },
            DecodedFramePlane {
                plane_index: 1,
                width: w,
                height: h,
                samples: pixels(w, h, seed ^ 0x5555),
            },
            DecodedFramePlane {
                plane_index: 2,
                width: w,
                height: h,
                samples: pixels(w, h, seed ^ 0x0F0F),
            },
        ],
        width: w,
        height: h,
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

fn assert_planes_eq(decoded: &DecodedFrame, want: &DecodedFrame, ctx: &str) {
    for (got, exp) in decoded.planes.iter().zip(want.planes.iter()) {
        assert_eq!(
            got.samples, exp.samples,
            "{ctx}: Plane {} samples diverged",
            got.plane_index
        );
    }
}

// -- genuine keyframe -> non-keyframe round-trip -------------------------

#[test]
fn rgb_nonkeyframe_carry_round_trips_all_coder_types() {
    // Encode two RGB Frames: Frame 0 a keyframe, Frame 1 a non-keyframe
    // whose per-context coder state continues from Frame 0 (§3.8.1.3 /
    // §3.8.2.5). Decode both through the carry-aware driver fed the
    // matching read-side carry; both reconstruct bit-exactly. Run on all
    // three coder_types so both carry channels (range + Golomb) are
    // covered.
    for coder_type in [0u32, 1, 2] {
        let cr = rgb_cr(1, 1, coder_type);
        let qts = qts_for(coder_type);
        let (w, h) = (10u32, 7u32);
        let dims = FramePixelDimensions::new(w, h).unwrap();
        let headers = [header(0, 0, 1, 1)];

        let frame0 = rgb_frame(w, h, 0x1234);
        let frame1 = rgb_frame(w, h, 0x9abc);

        let mut enc_carry: Option<Ffv1EncodeCarry> = None;
        let bytes0 =
            encode_frame_rgb_with_carry(&frame0, &cr, &qts, &headers, true, true, &mut enc_carry)
                .expect("encode keyframe");
        let bytes1 =
            encode_frame_rgb_with_carry(&frame1, &cr, &qts, &headers, true, false, &mut enc_carry)
                .expect("encode non-keyframe");

        let mut dec_carry: Option<Ffv1FrameCarry> = None;
        let dec0 = decode_frame_rgb_with_carry(
            &bytes0,
            &cr,
            &qts,
            dims,
            true,
            DecodeOptions::strict(),
            &mut dec_carry,
        )
        .expect("decode keyframe");
        let dec1 = decode_frame_rgb_with_carry(
            &bytes1,
            &cr,
            &qts,
            dims,
            true,
            DecodeOptions::strict(),
            &mut dec_carry,
        )
        .expect("decode non-keyframe");

        assert!(dec0.keyframe, "coder_type {coder_type}: Frame 0 keyframe");
        assert!(
            !dec1.keyframe,
            "coder_type {coder_type}: Frame 1 non-keyframe"
        );
        assert_planes_eq(&dec0, &frame0, &format!("coder_type {coder_type} f0"));
        assert_planes_eq(&dec1, &frame1, &format!("coder_type {coder_type} f1"));
    }
}

// -- carry is load-bearing -----------------------------------------------

#[test]
fn rgb_nonkeyframe_without_carry_decodes_differently() {
    // The same non-keyframe bytes decoded through the STATELESS
    // `decode_frame_rgb` (which always re-initialises coder state to its
    // §3.8.1.3 / §3.8.2.5 `128` window regardless of the §4.4 keyframe
    // value) produce the WRONG pixels — the carry is the only thing that
    // makes the non-keyframe decode correctly. If the carry were a no-op
    // this assertion would fail.
    for coder_type in [0u32, 1, 2] {
        let cr = rgb_cr(1, 1, coder_type);
        let qts = qts_for(coder_type);
        let (w, h) = (10u32, 7u32);
        let dims = FramePixelDimensions::new(w, h).unwrap();
        let headers = [header(0, 0, 1, 1)];

        let frame0 = rgb_frame(w, h, 0x1234);
        let frame1 = rgb_frame(w, h, 0x9abc);

        let mut enc_carry: Option<Ffv1EncodeCarry> = None;
        let _bytes0 =
            encode_frame_rgb_with_carry(&frame0, &cr, &qts, &headers, true, true, &mut enc_carry)
                .expect("encode keyframe");
        let bytes1 =
            encode_frame_rgb_with_carry(&frame1, &cr, &qts, &headers, true, false, &mut enc_carry)
                .expect("encode non-keyframe");

        // Stateless decode: the per-context windows start at 128, so the
        // pixels diverge from the encoder, which resumed Frame 0's
        // evolved windows.
        let stateless = decode_frame_rgb(&bytes1, &cr, &qts, dims, true).expect("stateless decode");
        let matches_all = stateless
            .planes
            .iter()
            .zip(frame1.planes.iter())
            .all(|(g, e)| g.samples == e.samples);
        assert!(
            !matches_all,
            "coder_type {coder_type}: a non-keyframe decoded WITHOUT the \
             carry must NOT match the intended pixels — otherwise the carry \
             is doing nothing"
        );
    }
}

// -- multi-Slice carry on a 2x2 grid -------------------------------------

#[test]
fn rgb_nonkeyframe_carry_is_per_slice_on_2x2_grid() {
    // Each of the four Slices on a 2×2 grid carries its own per-context
    // state independently across the Frame boundary. A multi-Frame
    // stream (keyframe + two non-keyframes) round-trips bit-exactly
    // through the carry-aware driver, which proves each Slice's carry
    // stays in lockstep with the encoder's matching Slice. Run on both
    // entropy coders.
    for coder_type in [0u32, 1] {
        let cr = rgb_cr(2, 2, coder_type);
        let qts = qts_for(coder_type);
        let (w, h) = (14u32, 10u32);
        let dims = FramePixelDimensions::new(w, h).unwrap();
        let headers = [
            header(0, 0, 1, 1),
            header(1, 0, 1, 1),
            header(0, 1, 1, 1),
            header(1, 1, 1, 1),
        ];

        let frames: Vec<DecodedFrame> = [0x11u32, 0x2222, 0x33333]
            .iter()
            .map(|&seed| rgb_frame(w, h, seed))
            .collect();

        let mut enc_carry: Option<Ffv1EncodeCarry> = None;
        let mut dec_carry: Option<Ffv1FrameCarry> = None;
        for (i, frame) in frames.iter().enumerate() {
            let keyframe = i == 0;
            let bytes = encode_frame_rgb_with_carry(
                frame,
                &cr,
                &qts,
                &headers,
                true,
                keyframe,
                &mut enc_carry,
            )
            .expect("encode frame");
            let decoded = decode_frame_rgb_with_carry(
                &bytes,
                &cr,
                &qts,
                dims,
                true,
                DecodeOptions::strict(),
                &mut dec_carry,
            )
            .expect("decode frame");
            assert_eq!(decoded.keyframe, keyframe);
            assert_planes_eq(
                &decoded,
                frame,
                &format!("coder_type {coder_type} grid frame {i}"),
            );
        }
    }
}

// -- mid-stream keyframe re-initialises (carry ignored) ------------------

#[test]
fn rgb_mid_stream_keyframe_reinitialises_state() {
    // Sequence: keyframe, non-keyframe, keyframe. The third Frame is a
    // keyframe, so it must decode correctly EVEN IF the carry holds the
    // second Frame's evolved windows (§3.8.1.3 / §3.8.2.5: a keyframe
    // re-initialises). It also matches a fully stateless decode of the
    // same keyframe bytes.
    for coder_type in [0u32, 1] {
        let cr = rgb_cr(1, 1, coder_type);
        let qts = qts_for(coder_type);
        let (w, h) = (10u32, 7u32);
        let dims = FramePixelDimensions::new(w, h).unwrap();
        let headers = [header(0, 0, 1, 1)];

        let f0 = rgb_frame(w, h, 0x1234);
        let f1 = rgb_frame(w, h, 0x9abc);
        let f2 = rgb_frame(w, h, 0xfeed);

        let mut enc_carry: Option<Ffv1EncodeCarry> = None;
        let mut dec_carry: Option<Ffv1FrameCarry> = None;

        for (frame, keyframe) in [(&f0, true), (&f1, false), (&f2, true)] {
            let bytes = encode_frame_rgb_with_carry(
                frame,
                &cr,
                &qts,
                &headers,
                true,
                keyframe,
                &mut enc_carry,
            )
            .expect("encode");
            let decoded = decode_frame_rgb_with_carry(
                &bytes,
                &cr,
                &qts,
                dims,
                true,
                DecodeOptions::strict(),
                &mut dec_carry,
            )
            .expect("decode");
            assert_eq!(decoded.keyframe, keyframe);
            assert_planes_eq(
                &decoded,
                frame,
                &format!("coder_type {coder_type} keyframe={keyframe}"),
            );
        }

        // The third (keyframe) Frame, encoded as a standalone keyframe,
        // decodes identically via the stateless driver — confirming the
        // mid-stream keyframe is genuinely independent.
        let standalone =
            encode_frame_rgb_with_carry(&f2, &cr, &qts, &headers, true, true, &mut None)
                .expect("standalone keyframe encode");
        let stateless =
            decode_frame_rgb(&standalone, &cr, &qts, dims, true).expect("stateless decode");
        assert_planes_eq(
            &stateless,
            &f2,
            &format!("coder_type {coder_type} standalone keyframe"),
        );
    }
}
