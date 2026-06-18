//! FFV1 Configuration Record parser (RFC 9043 §4.2, §4.3).
//!
//! The Configuration Record is the global per-stream header that
//! lives in the container's CodecPrivate / extradata area for FFV1
//! version 3 streams (§4.3). It carries the bitstream version, the
//! coder selection, optional custom state-transition deltas, the
//! colorspace identifier, the bits-per-raw-sample, the chroma layout,
//! and (for v3) slice grid + quant-table-set count.
//!
//! This module decodes the *Parameters* pseudocode of RFC 9043
//! Figure 28 up through the fields the round-1 contract requires:
//!
//! - `version`
//! - `micro_version` (when `version >= 3`)
//! - `coder_type`
//! - `state_transition_delta` (when `coder_type > 1`)
//! - `colorspace_type`
//! - `bits_per_raw_sample` (when `version >= 1`; needed structurally)
//! - `chroma_planes`            (needed structurally)
//! - `log2_h_chroma_subsample`
//! - `log2_v_chroma_subsample`
//! - `extra_plane`              (needed structurally)
//!
//! For version 3 it additionally captures `num_h_slices`,
//! `num_v_slices`, and `quant_table_set_count` because they are read
//! by the same range coder before the quant-table cascade begins,
//! and round-1 stops at that boundary.
//!
//! Quant-table decoding, initial-state-delta, `ec`, `intra`, and the
//! `configuration_record_crc_parity` validation are deferred to a
//! later round.
//!
//! The `picture_structure` value is a *slice-header* field (RFC 9043
//! §4.6.7), not part of the Configuration Record. This module exposes
//! the [`PictureStructure`] enum so callers that later parse slice
//! headers can share the type; the field is **not** decoded here.

use crate::range_coder::{RangeDecoder, PARAMETERS_INITIAL_STATE};
use crate::symbol::{get_br, get_ur, SYMBOL_CONTEXT_SIZE};
use crate::Error;

/// Length of the `state_transition_delta` array per RFC 9043 §4.2.4
/// (the loop in Figure 28 runs `for i = 1; i < 256; i++`).
///
/// Entry `[0]` is unused (the RFC leaves it implicit) and is always
/// reported as `0` in the parsed record.
pub const NUM_TRANSITION_DELTAS: usize = 256;

/// FFV1 bitstream version (RFC 9043 §4.2.1 Table 5). Version 2 is
/// reserved (experimental) and never appears in conforming streams,
/// so it is not part of the enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ffv1Version {
    /// FFV1 version 0 — no Configuration Record, parameters live in
    /// the keyframe header.
    V0,
    /// FFV1 version 1 — same, plus per-keyframe quant tables.
    V1,
    /// FFV1 version 3 — Configuration Record required; the bitstream
    /// described by this module.
    V3,
}

impl Ffv1Version {
    /// Numeric value as it appears on the wire.
    pub const fn as_u32(self) -> u32 {
        match self {
            Ffv1Version::V0 => 0,
            Ffv1Version::V1 => 1,
            Ffv1Version::V3 => 3,
        }
    }
}

/// FFV1 colorspace_type (RFC 9043 §4.2.5 Table 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorspaceType {
    /// YCbCr with no pixel transformation. Extra plane carries
    /// transparency; planes interleave plane-then-line.
    YCbCr,
    /// RGB with JPEG 2000 RCT pixel transformation. Extra plane
    /// carries transparency; planes interleave line-then-plane.
    Rgb,
}

impl ColorspaceType {
    /// Numeric value as it appears on the wire.
    pub const fn as_u32(self) -> u32 {
        match self {
            ColorspaceType::YCbCr => 0,
            ColorspaceType::Rgb => 1,
        }
    }
}

/// FFV1 picture_structure (RFC 9043 §4.6.7 Table 15).
///
/// Not decoded by this module — picture_structure lives in the slice
/// header, not the configuration record. The enum is published here
/// so slice-header parsing rounds can reuse the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureStructure {
    /// Picture structure not signalled (the "0" wire value).
    Unknown,
    /// Top field first.
    TopFieldFirst,
    /// Bottom field first.
    BottomFieldFirst,
    /// Progressive frame.
    Progressive,
}

impl PictureStructure {
    /// Map a wire value to its enum representant. Out-of-range values
    /// return [`Error::UnsupportedVersion`]; the variant carries the
    /// raw value so callers can log it.
    pub fn from_wire(v: u32) -> Result<Self, u32> {
        match v {
            0 => Ok(PictureStructure::Unknown),
            1 => Ok(PictureStructure::TopFieldFirst),
            2 => Ok(PictureStructure::BottomFieldFirst),
            3 => Ok(PictureStructure::Progressive),
            other => Err(other),
        }
    }
}

/// Parsed contents of an FFV1 Configuration Record (RFC 9043 §4.2 / §4.3).
///
/// The §4.2.15 `initial_state_delta` triple-loop is surfaced on the
/// [`Self::initial_state_delta`] field: `None` when the §4.2.14
/// `states_coded` flag is `0` for every set (the wire omits the
/// triple-loop, "initial states ... assumed to be all 128"), `Some(_)`
/// when at least one set carries the loop. The outer `Vec` is per
/// Quantization Table Set (`i` over `quant_table_set_count`); the
/// inner per-set `Option<Vec<_>>` is `None` for sets whose
/// `states_coded == 0`, `Some(deltas)` for sets whose
/// `states_coded == 1` (with `deltas.len() == context_count[i]` and
/// each inner `[i32; 32]` indexed by `k` over §4.2 `CONTEXT_SIZE`).
#[derive(Debug, Clone)]
pub struct Ffv1ConfigurationRecord {
    /// `version` from RFC 9043 §4.2.1.
    pub version: Ffv1Version,
    /// `micro_version` from RFC 9043 §4.2.2. `None` when `version < 3`
    /// (the field is absent on the wire).
    pub micro_version: Option<u32>,
    /// `coder_type` from RFC 9043 §4.2.3 (raw 0/1/2).
    pub coder_type: u32,
    /// `state_transition_delta[1..=255]` from RFC 9043 §4.2.4, when
    /// `coder_type > 1`. Otherwise filled with zeroes. Index `[0]` is
    /// always `0` (the loop in Figure 28 starts at `i = 1`).
    pub state_transition_delta: [i32; NUM_TRANSITION_DELTAS],
    /// `colorspace_type` from RFC 9043 §4.2.5.
    pub colorspace_type: ColorspaceType,
    /// `bits_per_raw_sample` from RFC 9043 §4.2.7. For `version < 1`
    /// the field is absent on the wire and the decoder MUST imply 8
    /// (§4.2.7 commentary, "use 8 by default").
    pub bits_per_raw_sample: u32,
    /// `chroma_planes` from RFC 9043 §4.2.6 (single bit on the wire,
    /// surfaced here as `bool`).
    pub chroma_planes: bool,
    /// `log2_h_chroma_subsample` from RFC 9043 §4.2.8.
    pub log2_h_chroma_subsample: u32,
    /// `log2_v_chroma_subsample` from RFC 9043 §4.2.9.
    pub log2_v_chroma_subsample: u32,
    /// `extra_plane` flag from RFC 9043 §4.2.10. Indicates whether an
    /// additional plane (alpha) is present beyond the colorspace's
    /// default.
    pub extra_plane: bool,
    /// `num_h_slices` from the v3-only block of Figure 28. `None` for
    /// version 0/1 streams where the slice grid lives in the
    /// per-frame header instead.
    pub num_h_slices: Option<u32>,
    /// `num_v_slices`, same provenance as `num_h_slices`.
    pub num_v_slices: Option<u32>,
    /// `quant_table_set_count` from the v3-only block of Figure 28.
    /// `None` when the field is absent (versions < 3).
    pub quant_table_set_count: Option<u32>,
    /// `ec` from RFC 9043 §4.2.16 (Table 13). `None` when the field is
    /// absent on the wire (versions 0/1). For v3 streams `0` means
    /// "32-bit CRC in Configuration Record only", `1` means "32-bit
    /// CRC in each Slice and the Configuration Record"; other values
    /// are reserved.
    pub ec: Option<u32>,
    /// `intra` from RFC 9043 §4.2.17 (Table 14). `None` when the field
    /// is absent on the wire (versions 0/1). `Some(false)` means
    /// keyframe may be 0 or 1; `Some(true)` means keyframe MUST be 1
    /// (intra-only stream).
    pub intra: Option<bool>,
    /// `initial_state_delta[i][j][k]` from RFC 9043 §4.2.15 (Figures 29
    /// / 30), surfacing the per-context initial-state perturbations the
    /// §4.2.14 `states_coded` flag gates on the wire.
    ///
    /// Layout: the outer `Option` is `None` when **every** Quantization
    /// Table Set's `states_coded` is `0` (the §4.2.14 default — initial
    /// states all 128 — no triple-loop on the wire). When `Some(per_set)`,
    /// `per_set.len() == quant_table_set_count` and `per_set[i]` is:
    ///
    /// * `None` for set `i` whose `states_coded == 0` (initial states
    ///   stay 128).
    /// * `Some(deltas)` for set `i` whose `states_coded == 1`: a
    ///   `Vec<[i32; INITIAL_STATE_DELTA_K]>` of length
    ///   `context_count[i]` indexed by `j` over `0..context_count[i]`
    ///   and `k` over `0..32` (§4.2 `CONTEXT_SIZE`). Each `[i32; 32]`
    ///   inner array carries the 32 signed `sr` symbols that perturb
    ///   the per-`k` initial state byte off 128 (Figures 29 / 30:
    ///   `initial_state[i][j][k] = (pred + initial_state_delta[i][j][k]) & 255`).
    ///
    /// `None` everywhere is the default the encoder emits when the
    /// codec has nothing to perturb; the parser returns `None` for the
    /// same reason (every set wrote `states_coded == 0`). When a
    /// caller wants a non-trivial table, the encoder emits the §4.2.15
    /// loop iff the corresponding `Some(deltas)` is populated with the
    /// expected shape; mismatched shapes surface as
    /// [`Error::InitialStateDeltaShapeMismatch`].
    pub initial_state_delta: Option<Vec<Option<Vec<[i32; INITIAL_STATE_DELTA_K]>>>>,
}

/// `CONTEXT_SIZE` from RFC 9043 §4.2 (right under Figure 28): the
/// per-context range-coder state width, which is also the inner
/// dimension of the §4.2.15 `initial_state_delta[i][j][k]` triple-loop
/// (`k` over `0..CONTEXT_SIZE`).
pub const INITIAL_STATE_DELTA_K: usize = 32;

/// Size of the Parameters() state buffer.
///
/// RFC 9043 §4.2 says "Parameters has its own initial states, all
/// set to 128" without specifying the buffer width. Empirically (see
/// the v3-default / v3-rgb-bgr0 / v3-grayscale fixture tests under
/// `tests/`), every Parameters symbol reads from the same 32-slot
/// context window; consecutive `ur` / `br` / `sr` symbols therefore
/// see the contexts in their post-previous-symbol state.
///
/// We allocate one context window's worth (32 slots) plus a small
/// pad. The exact size beyond 32 is irrelevant because the parser
/// never indexes past `SYMBOL_CONTEXT_SIZE`.
const PARAMETERS_STATE_LEN: usize = 64;

/// Parse a Configuration Record (RFC 9043 §4.3) from `buf`.
///
/// The slice should be the container's CodecPrivate / extradata
/// payload, including the trailing 4-byte `configuration_record_crc_parity`
/// (§4.3.2). CRC validation is *not* performed in round 1 — only the
/// Parameters block is read out.
pub fn parse_configuration_record(buf: &[u8]) -> Result<Ffv1ConfigurationRecord, Error> {
    let mut rc = RangeDecoder::new(buf)?;
    // RFC 9043 §4.2: "Parameters has its own initial states, all set
    // to 128." A fresh decoder context starts every state at 128.
    let mut state = [PARAMETERS_INITIAL_STATE; PARAMETERS_STATE_LEN];
    let record = parse_parameters(&mut rc, &mut state)?;
    // §4.2.1: "decoders SHOULD reject FFV1 bitstreams with version <= 1
    // && ConfigurationRecordIsPresent == 1." A Configuration Record is
    // by definition present here, so a v0/v1 version field is a
    // malformed container; surface it as a policy error rather than
    // silently parsing a v0/v1 prefix out of a Record that should not
    // exist. (The §4.4 in-Frame Parameters of a real v0/v1 keyframe is
    // parsed by [`parse_frame_parameters`], where no Record is present.)
    if record.version != Ffv1Version::V3 {
        return Err(Error::ConfigurationRecordForbiddenForVersion(
            record.version.as_u32(),
        ));
    }
    Ok(record)
}

/// Parse the §4.4 in-Frame `Parameters()` block of a versions-0/1 FFV1
/// keyframe (RFC 9043 §4.4 Figure: `if (keyframe && !ConfigurationRecord
/// IsPresent) Parameters()`).
///
/// For versions 0 and 1 the Parameters are **not** carried in a
/// container Configuration Record; they are range-coded inline in the
/// Frame bitstream, immediately after the §4.4 `keyframe` boolean. This
/// entry walks the same §4.2 Figure 28 `Parameters()` pseudocode as
/// [`parse_configuration_record`] but from a caller-owned range decoder
/// that has **already consumed the `keyframe` bit**, and it accepts only
/// versions 0/1 (a `version >= 3` Frame carries no in-Frame Parameters —
/// they live in the Configuration Record, so encountering one here is a
/// malformed stream).
///
/// The v3-only Figure 28 fields (`micro_version`, `num_h_slices - 1`,
/// `num_v_slices - 1`, `quant_table_set_count`) are absent for v0/v1 and
/// inferred per their §4.2 sub-sections: `quant_table_set_count` is
/// inferred 1 (§4.2.13), and the §4.6 Slice geometry is the single
/// implied Slice (`num_h_slices == num_v_slices == 1`; §4.6.1–§4.6.4 all
/// inferred, since §4.5 only emits a `SliceHeader()` for `version >= 3`).
/// The returned record therefore carries `num_h_slices == num_v_slices
/// == Some(1)` so downstream slice-geometry code (which keys off those
/// fields) sees the implied single-Slice raster.
///
/// `rc` MUST be positioned immediately after the §4.4 `keyframe` bit and
/// `state` MUST be at least [`PARAMETERS_STATE_LEN`] slots, every slot
/// initialised to [`PARAMETERS_INITIAL_STATE`] (the Frame's Parameters
/// "has its own initial states, all set to 128", §4.2).
pub(crate) fn parse_frame_parameters(
    rc: &mut RangeDecoder<'_>,
    state: &mut [u8],
) -> Result<Ffv1ConfigurationRecord, Error> {
    let mut record = parse_parameters(rc, state)?;
    // A v3 Frame stores its Parameters in the Configuration Record, not
    // inline; an in-Frame Parameters block is only emitted for v0/v1
    // (§4.4: `if (keyframe && !ConfigurationRecordIsPresent)`). Reaching
    // here with a v3 version means the caller mis-routed a v3 Frame.
    if record.version == Ffv1Version::V3 {
        return Err(Error::InFrameParametersForbiddenForVersion(
            record.version.as_u32(),
        ));
    }
    // §4.5: a `SliceHeader()` is only present for `version >= 3`; for
    // v0/v1 the Frame is the single implied Slice covering the whole
    // raster. §4.6.1–§4.6.4 infer `slice_x = slice_y = 0`,
    // `slice_width = slice_height = 1` on a 1×1 raster, so report the
    // implied `num_h_slices == num_v_slices == 1` here for the
    // slice-geometry consumers that key off these fields.
    record.num_h_slices = Some(1);
    record.num_v_slices = Some(1);
    Ok(record)
}

/// Parse the §4.2 Parameters of a versions-0/1 FFV1 **keyframe Frame**
/// directly from a raw Frame payload `buf`.
///
/// `buf` is the raw FFV1 Frame the container hands the codec (no
/// container framing). For versions 0 and 1 the Frame opens, per RFC
/// 9043 §4.4 Figure (`Frame( NumBytes )`), with the range-coded
/// `keyframe` boolean (its own initial state, 128); when that boolean is
/// `1` and no Configuration Record is present, the §4.4 pseudocode then
/// emits an inline `Parameters()` block — which this function reads.
///
/// This is the **keyframe** entry: it requires the Frame to be a
/// keyframe (a v0/v1 non-keyframe carries no inline Parameters — it
/// reuses the most recent keyframe's configuration), returning
/// [`Error::NonKeyframeHasNoInFrameParameters`] otherwise. It also
/// rejects a `version >= 3` Frame (those carry Parameters in the
/// Configuration Record, not inline; see
/// [`Error::InFrameParametersForbiddenForVersion`]).
///
/// The returned record reports the inferred v0/v1 single-Slice geometry
/// (`num_h_slices == num_v_slices == Some(1)`) and `micro_version ==
/// None`; the §4.1 Quantization Table Set cascade that follows
/// `quant_table_set_count` in Figure 28 is **not** decoded here (this
/// entry stops at the Parameters prefix, mirroring
/// [`parse_configuration_record`]). The full v0/v1 cascade + per-Frame
/// slice decode is layered on top of this parse.
pub fn parse_v0v1_frame_parameters(buf: &[u8]) -> Result<Ffv1ConfigurationRecord, Error> {
    let mut rc = RangeDecoder::new(buf)?;
    // §4.4: the Frame opens with the range-coded `keyframe` boolean,
    // which "has its own initial state, set to 128" (own 1-slot window).
    let mut kf_state = [PARAMETERS_INITIAL_STATE; 1];
    let keyframe = crate::symbol::get_br(&mut rc, &mut kf_state);
    if !keyframe {
        // §4.4: `Parameters()` is only emitted on a keyframe
        // (`if (keyframe && !ConfigurationRecordIsPresent)`). A v0/v1
        // non-keyframe inherits the prior keyframe's Parameters and
        // carries none of its own.
        return Err(Error::NonKeyframeHasNoInFrameParameters);
    }
    // §4.2: "Parameters has its own initial states, all set to 128."
    let mut state = [PARAMETERS_INITIAL_STATE; PARAMETERS_STATE_LEN];
    parse_frame_parameters(&mut rc, &mut state)
}

/// Walk the §4.2 `Parameters()` pseudocode (RFC 9043 Figure 28) from a
/// *caller-owned* range decoder + state buffer, leaving the decoder
/// positioned immediately after `quant_table_set_count`.
///
/// This is the shared core of [`parse_configuration_record`] and the
/// §4.1 quant-table cascade parser: the cascade is range-coded in the
/// **same** Parameters stream (Figure 28 calls `QuantizationTableSet(i)`
/// right after `quant_table_set_count`), so the quant-table parser must
/// resume from the very same decoder + state window rather than seeking.
///
/// `state` MUST be at least [`PARAMETERS_STATE_LEN`] bytes, every slot
/// initialised to [`PARAMETERS_INITIAL_STATE`].
pub(crate) fn parse_parameters(
    rc: &mut RangeDecoder<'_>,
    state: &mut [u8],
) -> Result<Ffv1ConfigurationRecord, Error> {
    debug_assert!(state.len() >= PARAMETERS_STATE_LEN);

    // All Parameters symbols share a single 32-slot context window
    // (see comment on `PARAMETERS_STATE_LEN`). `cursor` is constant
    // at 0; the `cursor_advance` book-keeping below mirrors the
    // pseudocode walk in RFC 9043 Figure 28 but does NOT move the
    // window — it exists purely so a future round implementing the
    // per-line context layout used by other FFV1 sections (quant
    // tables, slice header, slice content) can reuse the structure.
    let cursor = 0usize;
    const STRIDE: usize = 0;
    let mut cursor_advance: usize = 0;

    // ----- version (ur) ------------------------------------------------
    let version_raw = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    let version = match version_raw {
        0 => Ffv1Version::V0,
        1 => Ffv1Version::V1,
        3 => Ffv1Version::V3,
        other => return Err(Error::UnsupportedVersion(other)),
    };

    // ----- micro_version (ur, if version >= 3) -------------------------
    // Figure 28 guards `micro_version` behind `if (version >= 3)`. For
    // versions 0/1 the field is absent on the wire and the Parameters
    // walk continues straight to `coder_type`; `micro_version` is then
    // reported as `None`. The §4.2.1 advisory that v0/v1 streams MUST
    // NOT carry a Configuration Record is enforced by
    // [`parse_configuration_record`] — the *Configuration-Record*
    // context — not by this shared walker, which Figure 28 also drives
    // for the §4.4 in-Frame `Parameters()` of v0/v1 keyframes.
    let micro_version = if version == Ffv1Version::V3 {
        let v = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        Some(v)
    } else {
        None
    };

    // ----- coder_type (ur) ---------------------------------------------
    let coder_type = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    cursor_advance += STRIDE;
    if coder_type > 2 {
        return Err(Error::UnsupportedCoderType(coder_type));
    }

    // ----- state_transition_delta[1..256] (sr, if coder_type > 1) -----
    let mut state_transition_delta = [0i32; NUM_TRANSITION_DELTAS];
    if coder_type > 1 {
        // 255 signed symbols, one shared context window (Figure 28
        // applies `sr` repeatedly to the same state slice; the
        // adaptive transition table updates internal probabilities
        // between iterations).
        let window_start = cursor;
        for delta_slot in state_transition_delta.iter_mut().skip(1) {
            *delta_slot = crate::symbol::get_sr(
                rc,
                &mut state[window_start..window_start + SYMBOL_CONTEXT_SIZE],
            );
        }
        cursor_advance += STRIDE;
    }

    // ----- colorspace_type (ur) ---------------------------------------
    let colorspace_raw = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    cursor_advance += STRIDE;
    let colorspace_type = match colorspace_raw {
        0 => ColorspaceType::YCbCr,
        1 => ColorspaceType::Rgb,
        other => return Err(Error::UnsupportedColorspaceType(other)),
    };

    // ----- bits_per_raw_sample (ur, if version >= 1) ------------------
    // Figure 28 guards this field behind `if (version >= 1)`. For
    // version 0 it is absent on the wire and the implied value is 8 per
    // RFC 9043 §4.2.7; reading it unconditionally would consume a
    // phantom symbol and desync the rest of the v0 Parameters walk
    // (and, in a Frame, everything downstream). For version >= 1 it is
    // range-coded, and a coded 0 also means "use 8" (§4.2.7).
    let bits_per_raw_sample = if version == Ffv1Version::V0 {
        8
    } else {
        let v = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        if v == 0 {
            // §4.2.7: zero on the wire means "use 8".
            8
        } else {
            v
        }
    };

    // ----- chroma_planes (br) ------------------------------------------
    let chroma_planes = get_br(rc, &mut state[cursor..cursor + 1]);
    cursor_advance += STRIDE;

    // ----- log2_h_chroma_subsample (ur) -------------------------------
    let log2_h_chroma_subsample = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    cursor_advance += STRIDE;
    if log2_h_chroma_subsample > 4 {
        return Err(Error::InvalidChromaSubsample(log2_h_chroma_subsample));
    }

    // ----- log2_v_chroma_subsample (ur) -------------------------------
    let log2_v_chroma_subsample = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    cursor_advance += STRIDE;
    if log2_v_chroma_subsample > 4 {
        return Err(Error::InvalidChromaSubsample(log2_v_chroma_subsample));
    }

    // ----- extra_plane (br) -------------------------------------------
    let extra_plane = get_br(rc, &mut state[cursor..cursor + 1]);
    cursor_advance += STRIDE;

    // ----- v3-only: num_h_slices, num_v_slices, quant_table_set_count -
    let (num_h_slices, num_v_slices, quant_table_set_count) = if version == Ffv1Version::V3 {
        let h_minus_1 = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        let v_minus_1 = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        let qcount = get_ur(rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        (
            Some(h_minus_1.wrapping_add(1)),
            Some(v_minus_1.wrapping_add(1)),
            Some(qcount),
        )
    } else {
        (None, None, None)
    };
    // Silence "value assigned but never read" — the parser is
    // intentionally append-only and stops here.
    let _ = cursor;
    let _ = cursor_advance;

    Ok(Ffv1ConfigurationRecord {
        version,
        micro_version,
        coder_type,
        state_transition_delta,
        colorspace_type,
        bits_per_raw_sample,
        chroma_planes,
        log2_h_chroma_subsample,
        log2_v_chroma_subsample,
        extra_plane,
        num_h_slices,
        num_v_slices,
        quant_table_set_count,
        // The §4.2.14-§4.2.17 tail lives AFTER the §4.1 cascade in
        // Figure 28; this Parameters-prefix walker stops at
        // `quant_table_set_count`. The cascade-aware parser
        // (`quant_table::parse_quantization_table_sets`) is responsible
        // for resuming the same range-coder pass through the cascade
        // and then the tail, and patching ec/intra/initial_state_delta
        // into the returned record. Callers that decode only the
        // prefix see `None` on all three.
        ec: None,
        intra: None,
        initial_state_delta: None,
    })
}
