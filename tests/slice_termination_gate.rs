//! §3.8.1.1.1 range-coder termination gate
//! ([`SliceTerminationPolicy`], r411).
//!
//! RFC 9043 §3.8.1.1.1: "the end of range-coded Slices ... need to
//! terminate before the CRC at their end". This crate's encoder emits
//! the Sentinel-mode terminator on every v3 range-coded Slice since
//! r411; under `DecodeOptions::pedantic()` the decoder verifies the
//! recovered end position lands exactly on the Slice body length —
//! the same bookkeeping a conforming decoder uses to flag Slice
//! damage (and the exact class of defect the r411 black-box campaign
//! found in this crate's own pre-r411 output).

use oxideav_ffv1::{
    decode_frame_rgb_with_options, decode_frame_with_options, encode_frame, ColorspaceType,
    DecodeOptions, DecodedFrame, DecodedFramePlane, Error, Ffv1ConfigurationRecord,
    Ffv1SliceHeader, Ffv1Version, FramePixelDimensions, PictureStructure, QuantizationTableSet,
    MAX_QUANT_TABLE_SET_INDEXES, NUM_QUANT_SUBTABLES, NUM_TRANSITION_DELTAS,
};

fn gray_record(coder_type: u32) -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        version: Ffv1Version::V3,
        micro_version: Some(4),
        coder_type,
        state_transition_delta: [0; NUM_TRANSITION_DELTAS],
        colorspace_type: ColorspaceType::YCbCr,
        bits_per_raw_sample: 8,
        chroma_planes: false,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: false,
        num_h_slices: Some(1),
        num_v_slices: Some(1),
        quant_table_set_count: Some(1),
        ec: Some(0),
        intra: Some(false),
        initial_state_delta: None,
    }
}

fn rgb_record() -> Ffv1ConfigurationRecord {
    Ffv1ConfigurationRecord {
        colorspace_type: ColorspaceType::Rgb,
        chroma_planes: true,
        ..gray_record(1)
    }
}

/// Single-context §4.1 set (context 5 everywhere) — enough to drive the
/// range coder without a full cascade.
fn qts() -> QuantizationTableSet {
    let mut tables = [[0i32; 256]; NUM_QUANT_SUBTABLES];
    tables[0] = [5i32; 256];
    QuantizationTableSet {
        tables,
        context_count: 6,
    }
}

fn header(slot_count: usize, w: u32, h: u32) -> Ffv1SliceHeader {
    let _ = (w, h);
    Ffv1SliceHeader {
        slice_x: 0,
        slice_y: 0,
        slice_width: 1,
        slice_height: 1,
        quant_table_set_index_count: slot_count,
        quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
        picture_structure: PictureStructure::Progressive,
        picture_structure_raw: 0,
        sar_num: 0,
        sar_den: 0,
    }
}

fn gray_frame(w: u32, h: u32) -> DecodedFrame {
    let samples: Vec<i32> = (0..(w * h) as usize)
        .map(|i| ((i * 31) ^ (i >> 3)) as i32 & 0xFF)
        .collect();
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

fn rgb_frame(w: u32, h: u32) -> DecodedFrame {
    let planes = (0..3u8)
        .map(|p| DecodedFramePlane {
            plane_index: p,
            width: w,
            height: h,
            samples: (0..(w * h) as usize)
                .map(|i| ((i * 7 + p as usize * 13) & 0xFF) as i32)
                .collect(),
        })
        .collect();
    DecodedFrame {
        planes,
        width: w,
        height: h,
        bits_per_raw_sample: 8,
        colorspace: ColorspaceType::Rgb,
        keyframe: true,
        slice_headers: Vec::new(),
    }
}

/// Grow the single Slice's body by one byte (ec == 0 footer: a 3-byte
/// size field trails the body), producing a structurally valid Slice
/// whose range coder no longer terminates at the body end.
fn pad_single_slice_ec0(mut bytes: Vec<u8>) -> Vec<u8> {
    let len = bytes.len();
    assert!(len > 3);
    let size = u32::from(bytes[len - 3]) << 16
        | u32::from(bytes[len - 2]) << 8
        | u32::from(bytes[len - 1]);
    assert_eq!(size as usize + 3, len, "single ec=0 slice layout");
    let new_size = size + 1;
    bytes.truncate(len - 3);
    bytes.push(0); // padding byte inside the body
    bytes.push(((new_size >> 16) & 0xFF) as u8);
    bytes.push(((new_size >> 8) & 0xFF) as u8);
    bytes.push((new_size & 0xFF) as u8);
    bytes
}

#[test]
fn pedantic_accepts_conforming_range_slice_and_rejects_padding() {
    let cr = gray_record(1);
    let q = vec![qts()];
    let frame = gray_frame(24, 16);
    let hdr = header(2, 24, 16);
    let bytes = encode_frame(&frame, &cr, &q, std::slice::from_ref(&hdr), false).unwrap();
    let dims = FramePixelDimensions::new(24, 16).unwrap();

    // Conforming stream passes the pedantic gate bit-exactly.
    let dec = decode_frame_with_options(&bytes, &cr, &q, dims, false, DecodeOptions::pedantic())
        .expect("r411 encoder output terminates per §3.8.1.1.1");
    assert_eq!(dec.planes[0].samples, frame.planes[0].samples);

    // A padded body decodes under the default policy but is rejected
    // under the pedantic gate with the typed §3.8.1.1.1 error.
    let padded = pad_single_slice_ec0(bytes);
    let dec = decode_frame_with_options(&padded, &cr, &q, dims, false, DecodeOptions::strict())
        .expect("default policy tolerates the padded body");
    assert_eq!(dec.planes[0].samples, frame.planes[0].samples);
    let err = decode_frame_with_options(&padded, &cr, &q, dims, false, DecodeOptions::pedantic())
        .expect_err("pedantic gate must flag the padded body");
    assert!(
        matches!(err, Error::SliceTerminationMismatch { slice_index: 0, .. }),
        "unexpected error: {err:?}"
    );
}

#[test]
fn pedantic_gate_covers_rgb_range_slices() {
    let cr = rgb_record();
    let q = vec![qts()];
    let frame = rgb_frame(20, 12);
    let hdr = header(2, 20, 12);
    let bytes = encode_frame(&frame, &cr, &q, std::slice::from_ref(&hdr), false).unwrap();
    let dims = FramePixelDimensions::new(20, 12).unwrap();

    let dec =
        decode_frame_rgb_with_options(&bytes, &cr, &q, dims, false, DecodeOptions::pedantic())
            .expect("r411 RGB encoder output terminates per §3.8.1.1.1");
    assert_eq!(dec.planes[0].samples, frame.planes[0].samples);

    let padded = pad_single_slice_ec0(bytes);
    let err =
        decode_frame_rgb_with_options(&padded, &cr, &q, dims, false, DecodeOptions::pedantic())
            .expect_err("pedantic gate must flag the padded RGB body");
    assert!(matches!(err, Error::SliceTerminationMismatch { .. }));
}

#[test]
fn golomb_slices_are_exempt_from_the_gate() {
    // The Golomb-Rice Slice tail has no §3.8.1.1.1 terminator of its
    // own; the gate must not reject a conforming Golomb Slice.
    let cr = gray_record(0);
    let q = vec![qts()];
    let frame = gray_frame(24, 16);
    let hdr = header(2, 24, 16);
    let bytes = encode_frame(&frame, &cr, &q, std::slice::from_ref(&hdr), false).unwrap();
    let dims = FramePixelDimensions::new(24, 16).unwrap();
    let dec = decode_frame_with_options(&bytes, &cr, &q, dims, false, DecodeOptions::pedantic())
        .expect("Golomb slices bypass the range-termination gate");
    assert_eq!(dec.planes[0].samples, frame.planes[0].samples);
}
