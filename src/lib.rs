//! # oxideav-ffv1
//!
//! Pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec.
//!
//! This is the clean-room rebuild begun after the 2026-05-18 audit
//! (the prior implementation was retired for clean-room provenance
//! reasons). Round 1 landed the *Configuration Record* parser
//! (RFC 9043 §4.2 / §4.3) plus the binary range decoder and `ur` /
//! `sr` / `br` scalar-symbol primitives. Round 2 added the *Slice
//! Header* parser (§4.6). Round 3 added the *Slice Content scaffold*
//! (§4.7 / §4.8). Round 4 wires the §4.8 `Line( p, y )` body to the
//! §3.8.2 Golomb-Rice primitives: an MSB-first [`bit_reader::BitReader`],
//! the `ur` / `sr` Golomb-Rice VLC decoder + ESC mode (§3.8.2.1.1),
//! `sign_extend` (§3.8.2.3), the per-context adaptive VLC state
//! [`VlcState`] + scalar/level decode (§3.8.2.4 / §3.8.2.4.1), the
//! `log2_run` run-mode table (§3.8.2.2.1), plus the §3.3 median
//! predictor and the §3.5 context computation. Per-row
//! `sample_difference` decode is exposed via [`decode_line`]. Round 5
//! adds the §4.1 *Quantization Table Set* cascade decode via
//! [`parse_quantization_table_sets`], which produces the
//! [`QuantTableSet`]s `decode_line` consumes directly from one
//! extradata blob. Round 6 adds the §4.3.2 *Configuration Record CRC*
//! check ([`validate_configuration_record_crc`]): the §4.9.3 generator
//! (poly `0x104C11DB7`, init 0, no inversion) run over the whole
//! extradata blob must leave a remainder of zero. Round 7 adds the
//! §4.9 *Slice Footer* parser ([`parse_slice_footer`]): it extracts
//! `slice_size` (§4.9.1), `error_status` (§4.9.2), and
//! `slice_crc_parity` (§4.9.3) from the trailing 8 bytes (`ec=1`) or
//! 3 bytes (`ec=0`) of a Slice, cross-checks the size field against
//! the buffer length, and validates the §4.9.3 whole-Slice CRC residue
//! is zero — reusing the same generator as the Configuration Record
//! CRC. Round 8 adds **per-plane pixel reconstruction** for the
//! Golomb-Rice path ([`PlaneReconstructor::reconstruct_plane`]): it
//! folds the §3.3 median predictor back into the per-pixel decode loop
//! (every Sample's §3.5 context + §3.3 prediction depend on the
//! *reconstructed* neighbours, not the raw differences), maintains the
//! §3.1 Slice border, and applies the §3.8 modular add-back
//! (`Sample = (pred + diff) mod 2^bits`, exposed standalone as
//! [`reconstruct_sample`]) to recover a full Plane as a row-major
//! `Vec<i32>`. Round 9 mirrors that for the **range-coder slice path**
//! ([`RangePlaneReconstructor::reconstruct_plane`], RFC 9043 §3.7 /
//! §3.8.1.2 / §4.8): same §3.1 border / §3.3 median / §3.8 modular
//! add-back, but each Sample's `sample_difference` comes from one
//! signed [`get_symbol`](crate::symbol)-style range-coder call rather
//! than an MSB-first Golomb-Rice VLC. The range-coder slice path
//! has **no run mode** (§3.8.2.2 is Golomb-Rice-only), uses one
//! 32-slot state window per §3.5 absolute context (all initialised to
//! 128 per §3.8.1.3), and exposes the §3.3.1 alternate 16-bit median
//! predictor through a `use_16bit_median` flag. This is the bit-engine
//! the four v3 fixtures (all `coder_type == 1`) need to reach
//! end-to-end Plane reconstruction.
//!
//! Round 10 adds the §4.9.1 **trailer-pointer chain walk**
//! ([`walk_trailer_chain`]): given a raw FFV1 frame payload + the
//! Configuration Record's `ec` flag, it walks the §4.9 Slice Footer
//! `slice_size` (`u(24)`) field backwards from the end of the frame
//! and returns one [`SliceExtent`] per Slice in forward slice-index
//! order (slice 0 first). Each extent's `frame[start .. start +
//! total_len]` is exactly the buffer
//! [`parse_slice_footer`](crate::parse_slice_footer) consumes — this is
//! the byte-range plumbing a multi-slice frame driver needs to feed
//! the existing per-Slice parsers + reconstructors. The walker reads
//! only the `u(24)` size field; per-Slice validation (CRC, slice
//! header) is left to [`parse_slice_footer`] /
//! [`parse_slice_header`] / the reconstructors as before.
//!
//! Round 137 (this round) lands the **§3.8.1 binary range *encoder*** —
//! the symmetric inverse of [`RangeDecoder`] — together with its
//! §3.8.1.2 scalar [`put_ur`] / [`put_sr`] / [`put_br`] symbol-encode
//! primitives. [`RangeEncoder`] mirrors the decoder's Figure-18/19/20
//! state machine (16-bit `range`/`low`, `(range * state) / 256`
//! splitting, one byte emitted per renormalisation) with the classic
//! delayed-byte / pending-0xFF carry technique for byte emission. The
//! scalar `put_*` family walks the same 32-slot context-window layout
//! Figure 21 reads. Every existing decode primitive
//! ([`RangeDecoder::get_rac`], [`get_ur`], [`get_sr`], [`get_br`]) now
//! has a paired encode primitive, with round-trip tests covering
//! constant / alternating / pseudo-random binary streams, the
//! `is_zero` / exponent / mantissa / sign-bit corners of `ur`/`sr`,
//! power-of-two boundaries through the saturated-exponent regime, and
//! the `i32::MIN` magnitude-overflow guard. This is the first encoder
//! primitive in the crate's master and the foundation every higher-
//! level FFV1 encoder stage (Configuration Record write, Slice Header
//! write, range-coded Slice Content write) will build on.
//!
//! Round 149 (this round) lands the **§3.8.2 Golomb-Rice content
//! encoder primitives** — the bit-coded symmetric inverses of the
//! decode-side `get_ur_golomb_esc` / `get_sr_golomb_esc` /
//! `get_vlc_symbol` / `get_vlc_symbol_level` family. A new
//! MSB-first [`BitWriter`] (the inverse of [`BitReader`]) accumulates
//! bits into an owned `Vec<u8>` and zero-pads the final partial byte
//! per the §3.8.2 "padded with zeroes" rule. On top of it,
//! [`put_ur_golomb_esc`] / [`put_sr_golomb_esc`] emit Figure 26 / 27
//! unary-prefix-plus-suffix or ESC-prefix-plus-flat-suffix codes,
//! taking the same `k` / `bits` the decoder reads. The adaptive
//! [`put_vlc_symbol`] / [`put_vlc_symbol_level`] pair then walks the
//! §3.8.2.4 / §3.8.2.4.1 sign-flip + bias inversion, choosing the same
//! `k` the decoder picks (via the shared `vlc_pick_k` helper) and
//! mutating the per-context [`VlcState`] in lockstep with the decoder
//! via the shared `vlc_update` helper. The encoder consumes the
//! same Sample-Difference residual the decoder produces, so a per-
//! context state cloned across the round trip stays bit-exactly in
//! sync. This is the per-Sample bit engine the higher-level
//! Golomb-Rice Slice Content encoder (§4.8 / §3.8.2 run mode + scalar
//! mode) will build on; this round's deliverable is the scalar /
//! level path only.
//!
//! Round 152 (this round) folds the run-mode encoder dispatch into a
//! per-row **§4.8 [`encode_line`]** — the symmetric inverse of
//! [`decode_line`]. Takes a row of signed `sample_difference` values
//! (the same values [`decode_line`] returns / writes into
//! `current_row`) plus the standard [`LineNeighborBuffers`] +
//! [`LineDecoderState`] + [`QuantTableSet`], and emits via a
//! [`BitWriter`] the bit pattern a matching [`decode_line`] recovers
//! the input row from. The encoder walks the same per-pixel state
//! machine the decoder walks — same §3.5 absolute context (with the
//! sign-flip inversion applied to the [`put_vlc_symbol`] target so the
//! decoder's post-decode flip lands on the same magnitude), same
//! run-mode predicate (`|context| == 0 && l == t == tl`), same scalar
//! / level / run-mode dispatch. The per-context [`VlcState`] entries
//! and the run-mode `run_index` / `run_mode` / `run_count` fields
//! mutate identically on both sides of the trip. Run-mode encoding
//! uses intra-row lookahead to choose between long-run "1" bits and
//! short-run "0 + l2-bit residual" with a level-coded break; the
//! §3.8.2.2 contract that Phase 3 always returns 0 (so the very first
//! run-region pixel after a `reset_run_state()` cannot encode a
//! non-zero diff) is surfaced by a `debug_assert!`.
//!
//! Round 142 lands the first **frame-level encoder
//! primitive** on top of those scalar building blocks: the
//! §4.9 Slice Footer writer ([`encode_slice_footer`] /
//! [`encode_slice_footer_with_raw_status`]), the symmetric inverse of
//! [`parse_slice_footer`]. Given a Slice body (SliceHeader +
//! SliceContent + any Golomb-Rice padding) and an `ec` flag, the
//! writer emits the §4.9 trailer: 3 bytes (`slice_size` u(24)) for
//! `ec == 0`, or 8 bytes (`slice_size` u(24) + `error_status` u(8) +
//! `slice_crc_parity` u(32)) for `ec == 1`. For the `ec == 1` path the
//! §4.9.3 parity word is solved by running the §4.9.3 generator over
//! `body || size(3) || error_status(1)` and appending its MSB-first
//! CRC — the polynomial-division property
//! `CRC(M || CRC(M)) == 0` makes the whole-Slice residue zero by
//! construction, exactly the condition the `ec == 1` parser branch
//! checks for. Encoder→parser round-trips cover every §4.9.2 Table 16
//! `error_status` value (typed + raw-byte), the `body.len() ≥ 1 << 24`
//! overflow guard, body / `error_status` bit-flip sensitivity, and the
//! `parity == CRC(pre_parity_prefix)` solver shape.
//!
//! Round 11 added the **frame-level decode driver**
//! ([`decode_frame`]): given the raw FFV1 v3 frame bytes, the parsed
//! Configuration Record (§4.2), the parsed §4.1 Quantization Table
//! Sets, the surrounding container's pixel dimensions, and the
//! Configuration Record's `ec` flag, the driver walks every per-stage
//! parser in turn — §4.9.1 trailer chain → §4.9 footer validate → §4.6
//! header parse → §4.7 plane layout → per-plane reconstruction (§3.8.2
//! Golomb-Rice for `coder_type == 0`, §3.8.1.2 range-coder for
//! `coder_type == 1 || 2`) — and stitches each Slice's per-plane
//! output into a frame-level [`DecodedFrame`] at the slice's
//! pixel-space origin. The driver's outputs are typed
//! [`DecodedFramePlane`]s in `primary_color_count` order; the
//! YCbCr / plane-major path is fully wired, the RGB / line-major path
//! surfaces [`Error::ColorspaceLayoutNotImplemented`] (the §4.7
//! row-interleaved traversal needs a row-by-row driver, follow-up
//! round). The driver delegates every byte-level decision to the
//! existing per-stage modules; its job is the plumbing.
//!
//! The public `Decoder` / `Encoder` traits still return
//! [`Error::NotImplemented`]; the crate registers no codec
//! implementation into the runtime context yet — wiring the `decode_frame`
//! driver behind the trait surface (incl. a one-call extradata-to-
//! `RuntimeContext` registration path) is a small follow-up round once
//! the Configuration Record's deferred `ec` / `intra` /
//! `initial_state_delta` fields are parsed too. The §4.8 `decode_line`
//! raw-difference entry point is retained for callers that want the
//! un-reconstructed `sample_difference` row.
//!
//! [RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043.html

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod bit_reader;
mod config;
mod crc;
mod frame;
mod frame_encode;
mod golomb_rice;
mod predictor;
mod quant_table;
mod range_coder;
mod range_encode;
mod range_reconstruct;
mod reconstruct;
mod rgb_reconstruct;
mod sample_diff;
mod slice_content;
mod slice_footer;
mod slice_header;
mod symbol;
mod trailer_chain;

pub use bit_reader::{BitReader, BitWriter};
pub use config::{
    parse_configuration_record, ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version,
    PictureStructure, NUM_TRANSITION_DELTAS,
};
pub use crc::validate_configuration_record_crc;
pub use frame::{decode_frame, DecodedFrame, DecodedFramePlane};
pub use frame_encode::{encode_frame_golomb_rice, encode_frame_range_coder};
pub use golomb_rice::{
    get_sr_golomb_esc, get_ur_golomb, get_ur_golomb_esc, get_vlc_symbol, get_vlc_symbol_level,
    put_sr_golomb_esc, put_ur_golomb_esc, put_vlc_symbol, put_vlc_symbol_level, sign_extend,
    VlcState, LOG2_RUN, VLC_STATE_INITIAL,
};
pub use predictor::{
    absolute_context, median_predict, raw_context, AbsoluteContext, NeighborSamples, QuantTableSet,
    NUM_QUANT_SUBTABLES,
};
pub use quant_table::{
    parse_quantization_table_sets, ParametersWithQuantTables, QuantizationTableSet,
    MAX_CONTEXT_INPUTS,
};
pub use range_coder::{
    build_one_state, RangeDecoder, RangeEncoder, DEFAULT_ONE_STATE, PARAMETERS_INITIAL_STATE,
};
pub use range_encode::RangePlaneEncoder;
pub use range_reconstruct::RangePlaneReconstructor;
pub use reconstruct::{reconstruct_sample, PlaneReconstructor, BORDER_LEFT, BORDER_RIGHT};
pub use rgb_reconstruct::decode_frame_rgb;
pub use sample_diff::{
    decode_line, encode_line, LineDecoderState, LineNeighborBuffers, BORDER_WIDTH,
};
pub use slice_content::{
    compute_slice_content, FramePixelDimensions, Line, LineVisit, Plane, PlaneTraversal,
    SliceContent, MAX_PRIMARY_COLOR_COUNT,
};
pub use slice_footer::{
    encode_slice_footer, encode_slice_footer_with_raw_status, parse_slice_footer, Ffv1SliceFooter,
    SliceErrorStatus, SLICE_FOOTER_LEN_EC0, SLICE_FOOTER_LEN_EC1,
};
pub use slice_header::{
    encode_slice_header, encode_slice_header_to_encoder, parse_slice_header,
    parse_slice_header_from_decoder, Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES,
};
pub use symbol::{get_br, get_sr, get_ur, put_br, put_sr, put_ur, SYMBOL_CONTEXT_SIZE};
pub use trailer_chain::{slice_footer_len, walk_trailer_chain, SliceExtent};

/// Errors produced by the configuration-record / slice-header / slice-content scaffold.
///
/// Pixel-decode paths are not yet wired up; everything past the slice
/// content scaffold returns [`Error::NotImplemented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Slice **pixel** decoding (sample-difference symbol stream) has
    /// not been implemented in this round; the crate currently only
    /// exposes the configuration record + slice header + slice content
    /// scaffold parsers.
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

    /// [`compute_slice_content`] was called with a Configuration
    /// Record whose `num_h_slices` / `num_v_slices` are absent
    /// (FFV1 versions 0 and 1). The slice raster lives in the
    /// per-keyframe header for those versions; this round handles
    /// version 3 only.
    SliceRequiresVersion3,

    /// The frame pixel dimensions handed to
    /// [`compute_slice_content`] were zero in at least one axis. FFV1
    /// frames must have non-zero width and height (RFC 9043 §6
    /// security considerations explicitly call out
    /// `frame_pixel_width * frame_pixel_height` overflow handling).
    InvalidFramePixelDimensions {
        /// Frame width as supplied by the caller.
        width: u32,
        /// Frame height as supplied by the caller.
        height: u32,
    },

    /// The parsed Slice Header addresses a raster cell outside the
    /// Configuration Record's `num_h_slices × num_v_slices` grid, OR
    /// the derived `slice_pixel_width` / `slice_pixel_height` (per
    /// RFC 9043 §4.7.3 / §4.8.2) collapses to zero. The first case is
    /// a malformed slice header; the second is a malformed
    /// `(frame_pixel_*, num_*_slices, slice_*)` combination
    /// (e.g. a frame that's narrower than the slice grid resolution
    /// would let any cell collapse).
    SliceRasterOutOfRange {
        /// `slice_x` from the Slice Header.
        slice_x: u32,
        /// `slice_y` from the Slice Header.
        slice_y: u32,
        /// `slice_width` from the Slice Header.
        slice_width: u32,
        /// `slice_height` from the Slice Header.
        slice_height: u32,
        /// `num_h_slices` from the Configuration Record.
        num_h_slices: u32,
        /// `num_v_slices` from the Configuration Record.
        num_v_slices: u32,
    },

    /// The Configuration Record declared a `quant_table_set_count`
    /// outside the valid range. RFC 9043 §4.2.13 requires
    /// `1 <= quant_table_set_count <= 8` ("MUST NOT be 0", "MUST be
    /// less than or equal to 8").
    InvalidQuantTableSetCount(u32),

    /// A §4.1 Quantization Table Set produced a `context_count` that
    /// violates the §4.1.2 bound (`1 <= context_count <= 32768`). The
    /// variant carries the offending (saturated) `context_count`.
    QuantContextCountOutOfRange(u32),

    /// A §4.1 Quantization Table run-length stream was malformed: a run
    /// would overrun the 128-entry first half, or more than 128
    /// distinct quantization levels were coded.
    MalformedQuantTable,

    /// The Configuration Record failed its §4.3.2
    /// `configuration_record_crc_parity` check: running the §4.9.3 CRC
    /// (poly `0x104C11DB7`, init 0, no inversion) over the whole
    /// extradata blob did not leave a remainder of zero. The variant
    /// carries the non-zero residue, which is `0` iff the record is
    /// intact.
    ConfigurationRecordCrcMismatch(u32),

    /// The byte range handed to [`parse_slice_footer`] is shorter than
    /// the §4.9 Slice Footer it must contain (3 bytes for `ec == 0`,
    /// 8 bytes for `ec == 1`).
    TruncatedSliceFooter,

    /// The §4.9.1 `slice_size` field disagrees with the supplied
    /// buffer length minus the footer length. A conforming Slice
    /// satisfies `slice_size == buffer_len - footer_len`; a mismatch
    /// almost always means the §4.9.1 trailer-pointer chain was
    /// mis-walked.
    SliceSizeOutOfRange {
        /// The `slice_size` value read from the footer's `u(24)`.
        field: u32,
        /// The `buffer_len - footer_len` the parser expected.
        expected: u32,
    },

    /// The Slice failed its §4.9.3 `slice_crc_parity` check: running
    /// the §4.9.3 CRC (poly `0x104C11DB7`, init 0, no inversion) over
    /// the whole Slice (footer included) did not leave a remainder of
    /// zero.
    SliceCrcMismatch {
        /// The non-zero whole-Slice CRC residue (`0` iff intact).
        residue: u32,
        /// The §4.9.3 `slice_crc_parity` value stored in the footer,
        /// surfaced so a caller can log the expected-vs-computed
        /// mismatch.
        stored_parity: u32,
    },

    /// The frame-level driver was asked to decode a colorspace whose
    /// §4.7 traversal order it does not yet implement. Currently only
    /// `colorspace_type == 0` (YCbCr, plane-major) is wired; RGB
    /// (`colorspace_type == 1`, line-major / row-interleaved between
    /// Planes) needs a row-by-row driver variant.
    ColorspaceLayoutNotImplemented,
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
            Error::SliceRequiresVersion3 => f.write_str(
                "oxideav-ffv1: compute_slice_content requires FFV1 version 3 (the v0/v1 slice grid lives in the keyframe header, not the Configuration Record)",
            ),
            Error::InvalidFramePixelDimensions { width, height } => write!(
                f,
                "oxideav-ffv1: frame pixel dimensions {width}x{height} are invalid (both axes must be non-zero per RFC 9043 §6)"
            ),
            Error::SliceRasterOutOfRange {
                slice_x,
                slice_y,
                slice_width,
                slice_height,
                num_h_slices,
                num_v_slices,
            } => write!(
                f,
                "oxideav-ffv1: slice ({slice_x},{slice_y})+{slice_width}x{slice_height} lies outside the {num_h_slices}x{num_v_slices} raster (RFC 9043 §4.6 / §4.7)"
            ),
            Error::InvalidQuantTableSetCount(c) => write!(
                f,
                "oxideav-ffv1: quant_table_set_count {c} outside 1..=8 (RFC 9043 §4.2.13)"
            ),
            Error::QuantContextCountOutOfRange(c) => write!(
                f,
                "oxideav-ffv1: quant-table context_count {c} outside 1..=32768 (RFC 9043 §4.1.2)"
            ),
            Error::MalformedQuantTable => f.write_str(
                "oxideav-ffv1: malformed §4.1 quantization-table run-length stream",
            ),
            Error::ConfigurationRecordCrcMismatch(residue) => write!(
                f,
                "oxideav-ffv1: configuration record CRC check failed, residue 0x{residue:08x} (RFC 9043 §4.3.2)"
            ),
            Error::TruncatedSliceFooter => f.write_str(
                "oxideav-ffv1: slice byte range is shorter than its §4.9 Slice Footer (3 bytes for ec=0, 8 for ec=1)",
            ),
            Error::SliceSizeOutOfRange { field, expected } => write!(
                f,
                "oxideav-ffv1: slice_size field {field} != expected {expected} (RFC 9043 §4.9.1 trailer-chain mismatch)"
            ),
            Error::SliceCrcMismatch { residue, stored_parity } => write!(
                f,
                "oxideav-ffv1: slice CRC check failed, residue 0x{residue:08x} (stored parity 0x{stored_parity:08x}, RFC 9043 §4.9.3)"
            ),
            Error::ColorspaceLayoutNotImplemented => f.write_str(
                "oxideav-ffv1: frame driver only wires the §4.7 YCbCr plane-major path; RGB (colorspace_type=1) line-major traversal not yet implemented",
            ),
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
