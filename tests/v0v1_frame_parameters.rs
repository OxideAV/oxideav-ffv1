//! Versions-0/1 in-Frame `Parameters()` parse (RFC 9043 §4.4 + §4.2).
//!
//! For FFV1 versions 0 and 1 the §4.2 Parameters are **not** carried in a
//! container Configuration Record; they are range-coded inline in the
//! Frame itself, immediately after the §4.4 `keyframe` boolean
//! (`Frame( NumBytes ) { keyframe; if (keyframe && !ConfigurationRecord
//! IsPresent) Parameters(); ... }`). These tests construct such a Frame
//! prefix symbol-for-symbol from the §4.2 Figure 28 field order — only
//! the fields whose `if (version >= 3)` guard is *false* for v0/v1 are
//! present on the wire — and assert [`parse_v0v1_frame_parameters`]
//! recovers them, including the inferred single-Slice geometry
//! (`num_h_slices == num_v_slices == 1`, §4.5/§4.6) and the absent
//! `micro_version`.
//!
//! The bytes are assembled here with the published §3.8.1.2 `put_ur` /
//! `put_br` symbol primitives so the test is self-contained: it mirrors
//! the Figure 28 pseudocode directly, with no reference implementation
//! consulted.

use oxideav_ffv1::{
    parse_v0v1_frame_parameters, put_br, put_ur, ColorspaceType, Error, Ffv1Version, RangeEncoder,
    PARAMETERS_INITIAL_STATE,
};

const SYMBOL_CONTEXT_SIZE: usize = 32;

/// Description of the v0/v1 Parameters we want to encode (only the
/// fields present on the wire for v0/v1 — the §4.2 Figure 28 fields
/// whose `if (version >= 3)` guard is false).
struct V0V1Params {
    version: u32,
    coder_type: u32,
    colorspace_type: u32,
    /// Present on the wire only for `version >= 1` (§4.2.7).
    bits_per_raw_sample: Option<u32>,
    chroma_planes: bool,
    log2_h_chroma_subsample: u32,
    log2_v_chroma_subsample: u32,
    extra_plane: bool,
}

/// Build a v0/v1 keyframe Frame prefix: the §4.4 `keyframe` boolean
/// (own 1-slot state) followed by the §4.2 Figure 28 `Parameters()`
/// fields that survive the `version >= 3` guards for v0/v1.
///
/// `keyframe` lets a test exercise the §4.4 `if (keyframe ...)` guard:
/// when `false`, only the keyframe boolean is emitted (no Parameters),
/// matching a v0/v1 non-keyframe.
fn build_v0v1_frame(p: &V0V1Params, keyframe: bool) -> Vec<u8> {
    let mut re = RangeEncoder::new();

    // §4.4: keyframe boolean — "has its own initial state, set to 128".
    let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
    put_br(&mut re, &mut kf_state, keyframe);

    if keyframe {
        // §4.2: "Parameters has its own initial states, all set to 128";
        // every Parameters symbol shares the same 32-slot context window.
        let mut state = [PARAMETERS_INITIAL_STATE; 64];

        // version (ur) — always present.
        put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], p.version);
        // micro_version is `if (version >= 3)` — absent for v0/v1.
        // coder_type (ur).
        put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], p.coder_type);
        // state_transition_delta is `if (coder_type > 1)`; the tests
        // below all use coder_type <= 1, so it is absent.
        // colorspace_type (ur).
        put_ur(
            &mut re,
            &mut state[..SYMBOL_CONTEXT_SIZE],
            p.colorspace_type,
        );
        // bits_per_raw_sample is `if (version >= 1)` (§4.2.7).
        if let Some(bits) = p.bits_per_raw_sample {
            put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], bits);
        }
        // chroma_planes (br).
        put_br(&mut re, &mut state[..1], p.chroma_planes);
        // log2_h_chroma_subsample (ur).
        put_ur(
            &mut re,
            &mut state[..SYMBOL_CONTEXT_SIZE],
            p.log2_h_chroma_subsample,
        );
        // log2_v_chroma_subsample (ur).
        put_ur(
            &mut re,
            &mut state[..SYMBOL_CONTEXT_SIZE],
            p.log2_v_chroma_subsample,
        );
        // extra_plane (br).
        put_br(&mut re, &mut state[..1], p.extra_plane);
        // num_h_slices-1 / num_v_slices-1 / quant_table_set_count and the
        // §4.2.14-§4.2.17 tail are all `if (version >= 3)` — absent for
        // v0/v1. The §4.1 QuantizationTableSet cascade that follows is
        // not exercised by this Parameters-prefix parse.
    }

    re.finish()
}

#[test]
fn v1_keyframe_parameters_round_trip_ycbcr_8bit() {
    // A version-1 YCbCr 4:2:0 8-bit keyframe (chroma present, no alpha).
    let p = V0V1Params {
        version: 1,
        coder_type: 1,
        colorspace_type: 0, // YCbCr
        bits_per_raw_sample: Some(8),
        chroma_planes: true,
        log2_h_chroma_subsample: 1,
        log2_v_chroma_subsample: 1,
        extra_plane: false,
    };
    let frame = build_v0v1_frame(&p, true);

    let rec = parse_v0v1_frame_parameters(&frame).expect("v1 keyframe Parameters parse");

    assert_eq!(rec.version, Ffv1Version::V1);
    assert_eq!(rec.micro_version, None, "v0/v1 carry no micro_version");
    assert_eq!(rec.coder_type, 1);
    assert_eq!(rec.colorspace_type, ColorspaceType::YCbCr);
    assert_eq!(rec.bits_per_raw_sample, 8);
    assert!(rec.chroma_planes);
    assert_eq!(rec.log2_h_chroma_subsample, 1);
    assert_eq!(rec.log2_v_chroma_subsample, 1);
    assert!(!rec.extra_plane);
    // §4.5/§4.6: v0/v1 carry a single implied Slice — num_*_slices == 1.
    assert_eq!(rec.num_h_slices, Some(1));
    assert_eq!(rec.num_v_slices, Some(1));
    // v3-only tail fields are inferred-absent.
    assert_eq!(rec.quant_table_set_count, None);
    assert_eq!(rec.ec, None);
    assert_eq!(rec.intra, None);
}

#[test]
fn v0_keyframe_parameters_round_trip_no_bits_field() {
    // Version 0 omits `bits_per_raw_sample` on the wire (the field is
    // `if (version >= 1)`); the §4.2.7 implied value is 8.
    let p = V0V1Params {
        version: 0,
        coder_type: 0,      // Golomb-Rice
        colorspace_type: 1, // RGB / RCT
        bits_per_raw_sample: None,
        chroma_planes: true,
        log2_h_chroma_subsample: 0,
        log2_v_chroma_subsample: 0,
        extra_plane: true,
    };
    let frame = build_v0v1_frame(&p, true);

    let rec = parse_v0v1_frame_parameters(&frame).expect("v0 keyframe Parameters parse");

    assert_eq!(rec.version, Ffv1Version::V0);
    assert_eq!(rec.micro_version, None);
    assert_eq!(rec.coder_type, 0);
    assert_eq!(rec.colorspace_type, ColorspaceType::Rgb);
    // §4.2.7: version 0 implies 8-bit (field absent on the wire).
    assert_eq!(rec.bits_per_raw_sample, 8);
    assert!(rec.chroma_planes);
    assert_eq!(rec.log2_h_chroma_subsample, 0);
    assert_eq!(rec.log2_v_chroma_subsample, 0);
    assert!(rec.extra_plane);
    assert_eq!(rec.num_h_slices, Some(1));
    assert_eq!(rec.num_v_slices, Some(1));
}

#[test]
fn v0v1_non_keyframe_has_no_in_frame_parameters() {
    // §4.4: Parameters are emitted only `if (keyframe && ...)`. A
    // non-keyframe v0/v1 Frame carries just the keyframe boolean.
    let p = V0V1Params {
        version: 1,
        coder_type: 1,
        colorspace_type: 0,
        bits_per_raw_sample: Some(8),
        chroma_planes: true,
        log2_h_chroma_subsample: 1,
        log2_v_chroma_subsample: 1,
        extra_plane: false,
    };
    let frame = build_v0v1_frame(&p, false);

    let err = parse_v0v1_frame_parameters(&frame).expect_err("non-keyframe has no Parameters");
    assert!(matches!(err, Error::NonKeyframeHasNoInFrameParameters));
}

/// Build a *version-3* Parameters prefix (the full §4.2 Figure 28 prefix
/// including the v3-only `micro_version` and `num_*_slices-1` /
/// `quant_table_set_count` fields), preceded by the §4.4 keyframe
/// boolean. Used only to confirm the in-Frame parser rejects a v3 Frame
/// whose Parameters would actually parse cleanly.
fn build_v3_keyframe_prefix() -> Vec<u8> {
    let mut re = RangeEncoder::new();
    let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
    put_br(&mut re, &mut kf_state, true);

    let mut state = [PARAMETERS_INITIAL_STATE; 64];
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 3); // version
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 4); // micro_version (v3)
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 1); // coder_type
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 0); // colorspace_type
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 8); // bits_per_raw_sample
    put_br(&mut re, &mut state[..1], true); // chroma_planes
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 1); // log2_h
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 1); // log2_v
    put_br(&mut re, &mut state[..1], false); // extra_plane
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 0); // num_h_slices-1
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 0); // num_v_slices-1
    put_ur(&mut re, &mut state[..SYMBOL_CONTEXT_SIZE], 1); // quant_table_set_count
    re.finish()
}

#[test]
fn v3_version_in_frame_parameters_is_rejected() {
    // A `version >= 3` Frame stores its Parameters in the Configuration
    // Record, not inline; routing it through the in-Frame parser is an
    // error (§4.4), even when the Parameters prefix is well-formed.
    let frame = build_v3_keyframe_prefix();

    let err = parse_v0v1_frame_parameters(&frame).expect_err("v3 inline Parameters rejected");
    assert!(matches!(
        err,
        Error::InFrameParametersForbiddenForVersion(3)
    ));
}
