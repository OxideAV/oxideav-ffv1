//! Slice-parallel decode and damaged-Slice concealment (RFC 9043 §4.5
//! independent Slices, §4.9.2 `error_status`, §4.9.3 `slice_crc_parity`).
//!
//! * The decoded Frame and the inter-Frame state are identical at every
//!   thread budget, for both traversals and both coders.
//! * With concealment off (the default) a damaged Slice fails the Frame.
//! * With concealment on, the damaged Slice's raster rectangle is filled
//!   (grey / previous Frame), every other Slice still decodes exactly,
//!   the damage is reported, and a later non-keyframe keeps that Slice
//!   concealed until a keyframe restores its context state.
//! * No corruption of a Frame makes the concealing decoder panic.

use oxideav_ffv1::{
    ColorspaceType, ConcealedRegion, DecodeOptions, DecodedFrame, DecodedFramePlane, Error,
    Ffv1ConfigurationRecord, Ffv1DecodeSession, Ffv1EncodeCarry, Ffv1FrameCarry, Ffv1SliceHeader,
    Ffv1Version, FramePixelDimensions, MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES,
    NUM_TRANSITION_DELTAS, PictureStructure, QuantizationTableSet, SliceConcealment,
    decode_frame_rgb_with_carry, decode_frame_with_carry, encode_frame_with_carry,
    walk_trailer_chain,
};

const W: u32 = 368;
const H: u32 = 304;
const GRID: u32 = 4;

fn xorshift(x: &mut u32) -> u32 {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    *x
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

fn record(rgb: bool, coder_type: u32, bits: u32, intra: bool) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
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
        num_h_slices: Some(GRID),
        num_v_slices: Some(GRID),
        quant_table_set_count: Some(1),
        ec: Some(1),
        intra: Some(intra),
        initial_state_delta: None,
    }
}

fn headers() -> Vec<Ffv1SliceHeader> {
    let mut out = Vec::new();
    for sy in 0..GRID {
        for sx in 0..GRID {
            out.push(Ffv1SliceHeader {
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
            });
        }
    }
    out
}

fn plane(index: u8, w: u32, h: u32, bits: u32, seed: u32) -> DecodedFramePlane {
    let mask = (1i32 << bits) - 1;
    let mut rng = seed | 1;
    let mut samples = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = ((x * 3 + y * 5) as i32) << bits.saturating_sub(8);
            let noise = (xorshift(&mut rng) & 15) as i32;
            // Flat bands exercise the §3.8.2.2 run mode.
            let v = if (y / 16) % 3 == 0 { 40 } else { base + noise };
            samples.push(v & mask);
        }
    }
    DecodedFramePlane {
        plane_index: index,
        width: w,
        height: h,
        samples,
    }
}

fn source(rgb: bool, bits: u32, frame: u32) -> DecodedFrame {
    let (cw, ch) = if rgb {
        (W, H)
    } else {
        (W.div_ceil(2), H.div_ceil(2))
    };
    DecodedFrame {
        planes: vec![
            plane(0, W, H, bits, 0x1000 + frame),
            plane(1, cw, ch, bits, 0x2000 + frame),
            plane(2, cw, ch, bits, 0x3000 + frame),
        ],
        width: W,
        height: H,
        bits_per_raw_sample: bits,
        colorspace: if rgb {
            ColorspaceType::Rgb
        } else {
            ColorspaceType::YCbCr
        },
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// Encode `count` Frames; `keyframes[i]` says whether Frame `i` is one.
fn encode_stream(
    cr: &Ffv1ConfigurationRecord,
    rgb: bool,
    keyframes: &[bool],
) -> (Vec<DecodedFrame>, Vec<Vec<u8>>) {
    let qts = [quant_tables()];
    let hdr = headers();
    let mut carry: Option<Ffv1EncodeCarry> = None;
    let mut frames = Vec::new();
    let mut packets = Vec::new();
    for (i, &key) in keyframes.iter().enumerate() {
        let src = source(rgb, cr.bits_per_raw_sample, i as u32);
        let bytes = encode_frame_with_carry(&src, cr, &qts, &hdr, true, key, &mut carry)
            .expect("frame encodes");
        frames.push(src);
        packets.push(bytes);
    }
    (frames, packets)
}

fn decode(
    cr: &Ffv1ConfigurationRecord,
    bytes: &[u8],
    options: DecodeOptions,
    carry: &mut Option<Ffv1FrameCarry>,
) -> Result<DecodedFrame, Error> {
    let qts = [quant_tables()];
    let dims = FramePixelDimensions::new(W, H).expect("dims");
    match cr.colorspace_type {
        ColorspaceType::Rgb => {
            decode_frame_rgb_with_carry(bytes, cr, &qts, dims, true, options, carry)
        }
        _ => decode_frame_with_carry(bytes, cr, &qts, dims, true, options, carry),
    }
}

fn assert_same(a: &DecodedFrame, b: &DecodedFrame, what: &str) {
    assert_eq!(a.planes.len(), b.planes.len(), "{what}: plane count");
    for (pa, pb) in a.planes.iter().zip(&b.planes) {
        assert!(
            pa.samples == pb.samples,
            "{what}: plane {} differs",
            pa.plane_index
        );
    }
}

/// Pixel rectangle of raster cell (`gx`, `gy`) per the §4.6 geometry.
fn cell_rect(gx: u32, gy: u32) -> ConcealedRegionLike {
    let x0 = gx * W / GRID;
    let x1 = (gx + 1) * W / GRID;
    let y0 = gy * H / GRID;
    let y1 = (gy + 1) * H / GRID;
    ConcealedRegionLike {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConcealedRegionLike {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn like(r: &ConcealedRegion) -> ConcealedRegionLike {
    ConcealedRegionLike {
        x: r.x,
        y: r.y,
        width: r.width,
        height: r.height,
    }
}

/// Flip one byte in the middle of Slice `slice` of `bytes`.
fn damage_slice(bytes: &mut [u8], slice: usize) {
    let extents = walk_trailer_chain(bytes, true).expect("chain");
    let e = extents[slice];
    bytes[e.start + e.total_len / 2] ^= 0x5A;
}

/// Compare `got` against `want` outside `rect` (luma Plane only when
/// `luma_only`, since subsampled chroma has its own rectangle).
fn assert_exact_outside(
    got: &DecodedFrame,
    want: &DecodedFrame,
    rect: ConcealedRegionLike,
    rgb: bool,
) {
    for (p, (g, w)) in got.planes.iter().zip(&want.planes).enumerate() {
        let shift = u32::from(!rgb && p > 0);
        let (rx0, ry0) = (rect.x >> shift, rect.y >> shift);
        let rx1 = (rect.x + rect.width).div_ceil(1 << shift);
        let ry1 = (rect.y + rect.height).div_ceil(1 << shift);
        for y in 0..g.height {
            for x in 0..g.width {
                if x >= rx0 && x < rx1 && y >= ry0 && y < ry1 {
                    continue;
                }
                let i = (y * g.width + x) as usize;
                assert_eq!(g.samples[i], w.samples[i], "plane {p} at ({x},{y})");
            }
        }
    }
}

#[test]
fn every_thread_budget_decodes_identically() {
    for (rgb, bits) in [(false, 8u32), (false, 10), (true, 8), (true, 16)] {
        for coder in [0u32, 1] {
            let cr = record(rgb, coder, bits, false);
            let (frames, packets) = encode_stream(&cr, rgb, &[true, false, false, true, false]);
            let mut reference: Vec<DecodedFrame> = Vec::new();
            for threads in [0usize, 1, 2, 3, 8, 64] {
                let options = DecodeOptions::default().with_threads(threads);
                let mut carry = None;
                for (i, pkt) in packets.iter().enumerate() {
                    let got = decode(&cr, pkt, options, &mut carry).expect("decodes");
                    let what = format!("rgb={rgb} bits={bits} coder={coder} t={threads} f={i}");
                    assert_same(&got, &frames[i], &what);
                    if threads == 0 {
                        reference.push(got);
                    } else {
                        assert_same(&got, &reference[i], &what);
                        assert_eq!(got.keyframe, reference[i].keyframe);
                        assert_eq!(got.slice_headers.len(), reference[i].slice_headers.len());
                    }
                }
            }
        }
    }
}

#[test]
fn damaged_slice_fails_the_frame_by_default_at_every_budget() {
    let cr = record(false, 1, 8, true);
    let (_, mut packets) = encode_stream(&cr, false, &[true]);
    damage_slice(&mut packets[0], 9);
    damage_slice(&mut packets[0], 3);
    for threads in [1usize, 4, 16] {
        let err = decode(
            &cr,
            &packets[0],
            DecodeOptions::default().with_threads(threads),
            &mut None,
        )
        .expect_err("a damaged slice fails the frame");
        // The lowest-index damaged Slice is the one reported, whichever
        // worker reached it first.
        assert!(
            matches!(err, Error::SliceCrcMismatch { .. }),
            "threads={threads}: {err}"
        );
    }
}

#[test]
fn grey_concealment_keeps_the_other_slices() {
    for (rgb, bits) in [(false, 8u32), (true, 16)] {
        for coder in [0u32, 1] {
            let cr = record(rgb, coder, bits, true);
            let (frames, mut packets) = encode_stream(&cr, rgb, &[true]);
            // Slice 6 = raster cell (2, 1).
            damage_slice(&mut packets[0], 6);
            for threads in [1usize, 4] {
                let options = DecodeOptions::default()
                    .with_threads(threads)
                    .with_concealment(SliceConcealment::Grey);
                let mut carry = None;
                let got = decode(&cr, &packets[0], options, &mut carry).expect("concealed");
                let regions: Vec<_> = carry
                    .as_ref()
                    .expect("carry")
                    .concealed_regions()
                    .iter()
                    .map(like)
                    .collect();
                let rect = cell_rect(2, 1);
                assert_eq!(regions, vec![rect], "rgb={rgb} coder={coder}");
                assert_exact_outside(&got, &frames[0], rect, rgb);
                // Inside: mid-grey on every colour Plane.
                let grey = 1i32 << (bits - 1);
                let luma = &got.planes[0];
                for y in rect.y..rect.y + rect.height {
                    for x in rect.x..rect.x + rect.width {
                        assert_eq!(luma.samples[(y * luma.width + x) as usize], grey);
                    }
                }
                assert_eq!(got.slice_headers.len(), 15, "the damaged header is dropped");
            }
        }
    }
}

#[test]
fn previous_frame_concealment_and_state_loss_until_keyframe() {
    for coder in [0u32, 1] {
        let cr = record(false, coder, 8, false);
        // key, inter (damaged), inter (intact), key.
        let (frames, mut packets) = encode_stream(&cr, false, &[true, false, false, true]);
        damage_slice(&mut packets[1], 10); // raster cell (2, 2)
        let rect = cell_rect(2, 2);
        let options = DecodeOptions::default()
            .with_threads(3)
            .with_concealment(SliceConcealment::PreviousFrame);
        let qts = vec![quant_tables()];
        let dims = FramePixelDimensions::new(W, H).expect("dims");
        let mut session = Ffv1DecodeSession::with_options(cr.clone(), qts, dims, true, options);

        let f0 = session.decode_next_frame(&packets[0]).expect("frame 0");
        assert_same(&f0, &frames[0], "frame 0");
        assert!(session.last_concealed_regions().is_empty());

        let f1 = session.decode_next_frame(&packets[1]).expect("frame 1");
        let regions: Vec<_> = session.last_concealed_regions().iter().map(like).collect();
        assert_eq!(regions, vec![rect]);
        assert_exact_outside(&f1, &frames[1], rect, false);
        // Inside: the co-located samples of frame 0.
        let (g, prev) = (&f1.planes[0], &f0.planes[0]);
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                let i = (y * g.width + x) as usize;
                assert_eq!(g.samples[i], prev.samples[i]);
            }
        }

        // Frame 2 is intact on the wire, but its Slice 10 continues
        // context state that frame 1 never produced: still concealed.
        let f2 = session.decode_next_frame(&packets[2]).expect("frame 2");
        let regions: Vec<_> = session.last_concealed_regions().iter().map(like).collect();
        assert_eq!(regions, vec![rect], "coder={coder}");
        assert_exact_outside(&f2, &frames[2], rect, false);

        // The keyframe re-initialises every Slice.
        let f3 = session.decode_next_frame(&packets[3]).expect("frame 3");
        assert!(session.last_concealed_regions().is_empty());
        assert_same(&f3, &frames[3], "frame 3");
    }
}

#[test]
fn damaged_first_slice_conceals_the_whole_frame_unless_intra_only() {
    // Inter-capable stream: the §4.4 keyframe bit rides in Slice 0, so
    // its loss makes the Frame's state model unknown.
    let cr = record(false, 1, 8, false);
    let (_, mut packets) = encode_stream(&cr, false, &[true]);
    damage_slice(&mut packets[0], 0);
    let options = DecodeOptions::default().with_concealment(SliceConcealment::Grey);
    let mut carry = None;
    let got = decode(&cr, &packets[0], options, &mut carry).expect("concealed");
    assert_eq!(
        carry.expect("carry").concealed_regions().len(),
        (GRID * GRID) as usize
    );
    assert!(got.planes[0].samples.iter().all(|&s| s == 128));

    // Intra-only stream (§4.2.17): every Frame is a keyframe, so only
    // the damaged Slice is lost.
    let cr = record(false, 1, 8, true);
    let (frames, mut packets) = encode_stream(&cr, false, &[true]);
    damage_slice(&mut packets[0], 0);
    let mut carry = None;
    let got = decode(&cr, &packets[0], options, &mut carry).expect("concealed");
    let regions: Vec<_> = carry
        .expect("carry")
        .concealed_regions()
        .iter()
        .map(like)
        .collect();
    assert_eq!(regions, vec![cell_rect(0, 0)]);
    assert_exact_outside(&got, &frames[0], cell_rect(0, 0), false);
}

#[test]
fn concealing_decoder_never_panics_on_corruption() {
    let mut rng = 0x9E37_79B9u32;
    for (rgb, coder) in [(false, 0u32), (false, 1), (true, 0), (true, 1)] {
        let cr = record(rgb, coder, 8, false);
        let (_, packets) = encode_stream(&cr, rgb, &[true, false]);
        for round in 0..60 {
            let mut carry = None;
            for (i, clean) in packets.iter().enumerate() {
                let mut bytes = clean.clone();
                match round % 4 {
                    // Scattered byte flips.
                    0 => {
                        for _ in 0..1 + round % 7 {
                            let at = xorshift(&mut rng) as usize % bytes.len();
                            bytes[at] ^= (xorshift(&mut rng) | 1) as u8;
                        }
                    }
                    // Truncation.
                    1 => bytes.truncate(xorshift(&mut rng) as usize % bytes.len()),
                    // A zeroed span (lost sector).
                    2 => {
                        let at = xorshift(&mut rng) as usize % bytes.len();
                        let len = (xorshift(&mut rng) as usize % 4096).min(bytes.len() - at);
                        bytes[at..at + len].fill(0);
                    }
                    // Trailing garbage after the last Slice.
                    _ => bytes.extend((0..9).map(|_| xorshift(&mut rng) as u8)),
                }
                for conceal in [SliceConcealment::Grey, SliceConcealment::PreviousFrame] {
                    let options = DecodeOptions::default()
                        .with_threads(1 + round % 3)
                        .with_concealment(conceal);
                    let mut c = carry.clone();
                    let got = decode(&cr, &bytes, options, &mut c)
                        .unwrap_or_else(|e| panic!("round {round} frame {i}: {e}"));
                    assert_eq!(got.planes[0].samples.len(), (W * H) as usize);
                    if conceal == SliceConcealment::PreviousFrame {
                        carry = c;
                    }
                }
            }
        }
    }
}
