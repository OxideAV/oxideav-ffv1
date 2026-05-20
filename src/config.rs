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
/// Fields beyond round-1's scope (the quant-table cascade,
/// `initial_state_delta`, `ec`, `intra`) are intentionally omitted;
/// they will be added as the decoder lands later rounds.
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
}

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
    let version_raw = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    let version = match version_raw {
        0 => Ffv1Version::V0,
        1 => Ffv1Version::V1,
        3 => Ffv1Version::V3,
        other => return Err(Error::UnsupportedVersion(other)),
    };

    // ----- micro_version (ur, if version >= 3) -------------------------
    let micro_version = if version == Ffv1Version::V3 {
        let v = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        Some(v)
    } else {
        // For v0/v1 the Configuration Record is forbidden (§4.2.1):
        // "decoders SHOULD reject FFV1 bitstreams with version <= 1
        // && ConfigurationRecordIsPresent == 1." We surface this as a
        // policy error so misconfigured containers fail loud rather
        // than silently truncating.
        return Err(Error::ConfigurationRecordForbiddenForVersion(
            version.as_u32(),
        ));
    };

    // ----- coder_type (ur) ---------------------------------------------
    let coder_type = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
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
                &mut rc,
                &mut state[window_start..window_start + SYMBOL_CONTEXT_SIZE],
            );
        }
        cursor_advance += STRIDE;
    }

    // ----- colorspace_type (ur) ---------------------------------------
    let colorspace_raw = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    cursor_advance += STRIDE;
    let colorspace_type = match colorspace_raw {
        0 => ColorspaceType::YCbCr,
        1 => ColorspaceType::Rgb,
        other => return Err(Error::UnsupportedColorspaceType(other)),
    };

    // ----- bits_per_raw_sample (ur, if version >= 1) ------------------
    // For version == 0 the field is absent and the implied value is
    // 8 per RFC 9043 §4.2.7. For version >= 1 it is range-coded.
    let bits_per_raw_sample = {
        let v = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        if v == 0 {
            // §4.2.7: zero on the wire means "use 8".
            8
        } else {
            v
        }
    };

    // ----- chroma_planes (br) ------------------------------------------
    let chroma_planes = get_br(&mut rc, &mut state[cursor..cursor + 1]);
    cursor_advance += STRIDE;

    // ----- log2_h_chroma_subsample (ur) -------------------------------
    let log2_h_chroma_subsample = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    cursor_advance += STRIDE;
    if log2_h_chroma_subsample > 4 {
        return Err(Error::InvalidChromaSubsample(log2_h_chroma_subsample));
    }

    // ----- log2_v_chroma_subsample (ur) -------------------------------
    let log2_v_chroma_subsample = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
    cursor_advance += STRIDE;
    if log2_v_chroma_subsample > 4 {
        return Err(Error::InvalidChromaSubsample(log2_v_chroma_subsample));
    }

    // ----- extra_plane (br) -------------------------------------------
    let extra_plane = get_br(&mut rc, &mut state[cursor..cursor + 1]);
    cursor_advance += STRIDE;

    // ----- v3-only: num_h_slices, num_v_slices, quant_table_set_count -
    let (num_h_slices, num_v_slices, quant_table_set_count) = if version == Ffv1Version::V3 {
        let h_minus_1 = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        let v_minus_1 = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
        cursor_advance += STRIDE;
        let qcount = get_ur(&mut rc, &mut state[cursor..cursor + SYMBOL_CONTEXT_SIZE]);
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
    })
}
