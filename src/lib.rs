//! # oxideav-ffv1
//!
//! Pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec.
//!
//! This is the clean-room rebuild begun after the 2026-05-18 audit
//! (the prior implementation was retired for clean-room provenance
//! reasons). Round 1 landed the *Configuration Record* parser
//! (RFC 9043 §4.2 / §4.3) plus the binary range decoder and `ur` /
//! `sr` / `br` scalar-symbol primitives. Round 2 adds the *Slice
//! Header* parser (§4.6), so downstream callers can recover each
//! slice's raster geometry, per-plane quantization-table-set
//! selection, picture structure, and SAR.
//!
//! No slice **content** decoding, no pixel reconstruction, and no
//! Golomb-Rice codec are implemented yet. The public `Decoder` /
//! `Encoder` traits still return [`Error::NotImplemented`]; the crate
//! registers no codec implementation into the runtime context.
//!
//! [RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043.html

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod config;
mod range_coder;
mod slice_header;
mod symbol;

pub use config::{
    parse_configuration_record, ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version,
    PictureStructure, NUM_TRANSITION_DELTAS,
};
pub use slice_header::{parse_slice_header, Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES};

/// Errors produced by the configuration-record parser.
///
/// Slice / pixel / encode paths are not yet wired up; everything past
/// the configuration record returns [`Error::NotImplemented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Slice or pixel decoding has not been implemented in this
    /// round; the crate currently only exposes the configuration
    /// record parser.
    NotImplemented,

    /// The buffer handed to the range coder is shorter than the two
    /// seed bytes required by RFC 9043 Figure 18.
    TruncatedRangeCoder,

    /// The Configuration Record declared an FFV1 `version` value that
    /// is not described by RFC 9043 (only versions 0, 1, and 3 are).
    UnsupportedVersion(u32),

    /// The Configuration Record declared a `coder_type` outside the
    /// table in RFC 9043 §4.2.3 (only 0, 1, and 2 are defined).
    UnsupportedCoderType(u32),

    /// The Configuration Record declared a `colorspace_type` outside
    /// the table in RFC 9043 §4.2.5 (only 0 and 1 are defined).
    UnsupportedColorspaceType(u32),

    /// The Configuration Record's `log2_h_chroma_subsample` or
    /// `log2_v_chroma_subsample` exceeds the 0..=4 range implied by
    /// RFC 9043 §4.2.8 and §4.2.9.
    InvalidChromaSubsample(u32),

    /// FFV1 versions 0 and 1 must not present a Configuration Record
    /// (RFC 9043 §4.2.1 advisory: "decoders SHOULD reject FFV1
    /// bitstreams with version <= 1 && ConfigurationRecordIsPresent
    /// == 1").
    ConfigurationRecordForbiddenForVersion(u32),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => {
                f.write_str("oxideav-ffv1: only the configuration-record parser is wired up")
            }
            Error::TruncatedRangeCoder => f.write_str(
                "oxideav-ffv1: range-coded region is shorter than the two seed bytes (RFC 9043 §3.8.1.1)",
            ),
            Error::UnsupportedVersion(v) => {
                write!(f, "oxideav-ffv1: unknown FFV1 version {v} (RFC 9043 §4.2.1 lists 0/1/3)")
            }
            Error::UnsupportedCoderType(c) => {
                write!(f, "oxideav-ffv1: unknown coder_type {c} (RFC 9043 §4.2.3 lists 0/1/2)")
            }
            Error::UnsupportedColorspaceType(c) => {
                write!(f, "oxideav-ffv1: unknown colorspace_type {c} (RFC 9043 §4.2.5 lists 0/1)")
            }
            Error::InvalidChromaSubsample(s) => {
                write!(f, "oxideav-ffv1: log2_chroma_subsample {s} exceeds 4 (RFC 9043 §4.2.8/§4.2.9)")
            }
            Error::ConfigurationRecordForbiddenForVersion(v) => {
                write!(
                    f,
                    "oxideav-ffv1: FFV1 version {v} forbids a Configuration Record (RFC 9043 §4.2.1)"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// No-op codec registration. The crate registers no codec
/// implementation while only the configuration-record parser is
/// available; this stub is kept so the oxideav linkme-free dispatch
/// contract (`oxideav_core::register!`) is satisfied.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("ffv1", register);
