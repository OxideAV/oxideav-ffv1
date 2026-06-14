//! Stateful multi-Frame FFV1 decode session (RFC 9043 §4.4 / §4.2.17 /
//! §5).
//!
//! The frame-level decode drivers ([`crate::decode_frame`] /
//! [`crate::decode_frame_rgb`]) are single-Frame and stateless by
//! design, so the two RFC 9043 rules that span more than one Frame
//! live with the caller that walks Frames in coded order:
//!
//! * **§5 third paragraph** — "For each Frame with a keyframe value
//!   of 0, each Slice MUST have the same value of slice_x, slice_y,
//!   slice_width, and slice_height as a Slice in the previous Frame."
//!   The round-268 [`SliceGeometryStabilityTracker`] implements the
//!   cross-Frame walk; rounds 274 / 279 surfaced both of its
//!   `observe_frame` arguments ([`DecodedFrame::keyframe`],
//!   [`DecodedFrame::slice_headers`]) on the decode output.
//! * **§4.2.17 `intra`** (Table 14) — "keyframe MUST be 1 (keyframes
//!   only)" when the Configuration Record declares `intra == 1`
//!   ("Inferred to be 0 if not present"). A per-Frame value
//!   constrained by a per-stream Parameters field is again a
//!   stream-scope check no single-Frame driver can own.
//!
//! [`Ffv1DecodeSession`] is the driver-level stitch the round-279 row
//! queued: it owns the per-stream decode inputs (Configuration
//! Record, §4.1 Quantization Table Sets, container pixel dimensions,
//! `ec` flag, [`DecodeOptions`]) plus the cross-Frame state (the §5
//! tracker and the coded-order Frame counter), routes each Frame's
//! bytes to the right single-Frame driver on the Configuration
//! Record's §4.2.5 `colorspace_type` — mirroring the dispatch
//! [`crate::encode_frame`] performs on the write side — and applies
//! both stream-scope conformance gates to every decoded Frame.
//!
//! Both gates are **structural wire-conformance gates**, like the §5
//! second-paragraph raster-coverage and max-slice-size gates already
//! inside the single-Frame drivers: they fire under
//! [`DecodeOptions::strict`] and [`DecodeOptions::lenient`] alike.
//! The §4.9.2 / §4.9.3 policy fields govern *corruption tolerance*,
//! not conformance, and flow through to the inner driver unchanged.

use crate::config::{ColorspaceType, Ffv1ConfigurationRecord};
use crate::frame::{decode_frame_with_carry, DecodeOptions, DecodedFrame, Ffv1FrameCarry};
use crate::quant_table::QuantizationTableSet;
use crate::rgb_reconstruct::decode_frame_rgb_with_carry;
use crate::slice_content::{FramePixelDimensions, SliceGeometryStabilityTracker};
use crate::Error;

/// Stateful multi-Frame FFV1 decode session.
///
/// Construct one per coded stream with the per-stream inputs every
/// single-Frame decode shares, then call
/// [`decode_next_frame`][Self::decode_next_frame] once per Frame **in
/// coded order**. Each call:
///
/// 1. routes the Frame bytes to [`crate::decode_frame_with_options`]
///    (YCbCr / plane-major) or
///    [`crate::decode_frame_rgb_with_options`] (RGB / line-major) on
///    the Configuration Record's §4.2.5 `colorspace_type` — the same
///    dispatch [`crate::encode_frame`] performs on the write side;
/// 2. applies the §4.2.17 `intra` gate: when the Configuration Record
///    declares `intra == 1`, a decoded Frame whose §4.4 `keyframe`
///    value is 0 surfaces [`Error::NonKeyframeInIntraStream`]
///    (RFC 9043 §4.2.17 Table 14: "keyframe MUST be 1 (keyframes
///    only)");
/// 3. applies the §5 third-paragraph Slice-geometry stability gate
///    via the owned [`SliceGeometryStabilityTracker`]
///    (`observe_frame(decoded.keyframe, &decoded.slice_headers)`),
///    surfacing [`Error::SliceGeometryUnstable`] when a non-keyframe
///    re-splits the Slice raster.
///
/// A Frame that fails either stream-scope gate does **not** advance
/// the session: the tracker's previous-Frame reference and the
/// coded-order Frame counter are left untouched (the violating Frame
/// is non-conforming and must not become the §5 reference for its
/// successor — the same contract
/// [`SliceGeometryStabilityTracker::observe_frame`] keeps).
///
/// Callers that decode through their own driver plumbing (or replay
/// pre-decoded Frames) can feed the session with
/// [`observe_decoded_frame`][Self::observe_decoded_frame], which runs
/// the same two stream-scope gates without re-decoding.
#[derive(Debug, Clone)]
pub struct Ffv1DecodeSession {
    cr: Ffv1ConfigurationRecord,
    quant_table_sets: Vec<QuantizationTableSet>,
    frame_dims: FramePixelDimensions,
    ec: bool,
    options: DecodeOptions,
    tracker: SliceGeometryStabilityTracker,
    frames_observed: u64,
    /// The previous Frame's end-of-Frame per-Slice per-slot coder state
    /// (RFC 9043 §3.8.1.3 / §3.8.2.5). `None` before the first Frame.
    /// On a non-keyframe both the YCbCr / plane-major driver
    /// ([`decode_frame_with_carry`]) and the RGB / line-major driver
    /// ([`decode_frame_rgb_with_carry`]) resume the per-context windows
    /// from here instead of re-initialising them; every decoded Frame
    /// writes its fresh snapshot back. This is the cross-Frame state no
    /// stateless single-Frame driver can hold.
    carry: Option<Ffv1FrameCarry>,
}

impl Ffv1DecodeSession {
    /// Create a session with the default strict [`DecodeOptions`]
    /// (every §4.9.3 CRC residue must be zero; no Slice may declare
    /// the §4.9.2 Table 16 `Uncorrectable` status).
    ///
    /// * `cr` — the parsed §4.2 Configuration Record for the stream.
    /// * `quant_table_sets` — the parsed §4.1 Quantization Table Sets
    ///   in stream order.
    /// * `frame_dims` — the surrounding container's reported pixel
    ///   dimensions (FFV1's Configuration Record carries no frame
    ///   width / height per §4.2).
    /// * `ec` — the §4.2.16 `ec` flag (derive from `cr.ec` when the
    ///   cascade-aware parser populated it).
    pub fn new(
        cr: Ffv1ConfigurationRecord,
        quant_table_sets: Vec<QuantizationTableSet>,
        frame_dims: FramePixelDimensions,
        ec: bool,
    ) -> Self {
        Self::with_options(
            cr,
            quant_table_sets,
            frame_dims,
            ec,
            DecodeOptions::strict(),
        )
    }

    /// Create a session with explicit [`DecodeOptions`]. The options
    /// flow through to every per-Frame driver call; the session's two
    /// stream-scope gates (§4.2.17 `intra`, §5 third paragraph) are
    /// structural and policy-independent, mirroring the §5 gates the
    /// single-Frame drivers already enforce under either policy.
    pub fn with_options(
        cr: Ffv1ConfigurationRecord,
        quant_table_sets: Vec<QuantizationTableSet>,
        frame_dims: FramePixelDimensions,
        ec: bool,
        options: DecodeOptions,
    ) -> Self {
        Self {
            cr,
            quant_table_sets,
            frame_dims,
            ec,
            options,
            tracker: SliceGeometryStabilityTracker::new(),
            frames_observed: 0,
            carry: None,
        }
    }

    /// The Configuration Record the session decodes against.
    pub fn configuration_record(&self) -> &Ffv1ConfigurationRecord {
        &self.cr
    }

    /// The [`DecodeOptions`] every per-Frame driver call uses.
    pub fn options(&self) -> DecodeOptions {
        self.options
    }

    /// Number of Frames that have passed both stream-scope gates so
    /// far (decoded via [`decode_next_frame`][Self::decode_next_frame]
    /// or fed via
    /// [`observe_decoded_frame`][Self::observe_decoded_frame]). A
    /// Frame that failed a gate is not counted.
    pub fn frames_observed(&self) -> u64 {
        self.frames_observed
    }

    /// `true` once at least one conforming Frame has been observed —
    /// i.e. the §5 third-paragraph rule has a previous-Frame
    /// reference to validate the next non-keyframe against.
    pub fn has_previous_frame(&self) -> bool {
        self.tracker.has_previous_frame()
    }

    /// Decode the next Frame of the stream in coded order.
    ///
    /// Routes on the Configuration Record's §4.2.5 `colorspace_type`
    /// (`Rgb` → the RGB / line-major driver, `YCbCr` → the
    /// plane-major driver), then applies the §4.2.17 `intra` and §5
    /// third-paragraph gates to the decoded Frame. On success the
    /// Frame becomes the §5 previous-Frame reference and the decoded
    /// [`DecodedFrame`] is returned.
    ///
    /// # Errors
    ///
    /// * Any error the routed single-Frame driver surfaces
    ///   ([`crate::decode_frame_with_options`] /
    ///   [`crate::decode_frame_rgb_with_options`]).
    /// * [`Error::NonKeyframeInIntraStream`] — the Configuration
    ///   Record declares `intra == 1` but the Frame's §4.4 `keyframe`
    ///   value is 0 (RFC 9043 §4.2.17 Table 14).
    /// * [`Error::SliceGeometryUnstable`] — a non-keyframe Frame's
    ///   Slice geometry matches no Slice of the previous Frame
    ///   (RFC 9043 §5 third paragraph).
    ///
    /// On a gate failure the session state (tracker reference, Frame
    /// counter) is unchanged.
    pub fn decode_next_frame(&mut self, frame_bytes: &[u8]) -> Result<DecodedFrame, Error> {
        match self.cr.colorspace_type {
            ColorspaceType::YCbCr => {
                // YCbCr / plane-major: carry the §3.8.1.3 / §3.8.2.5
                // per-context coder state across non-keyframes. Decode
                // into a working copy of the carry so a post-decode
                // conformance-gate failure leaves the session's carry
                // (and tracker / counter) untouched — the same
                // "violating Frame does not advance the session"
                // contract the §4.2.17 / §5 gates keep.
                let mut working_carry = self.carry.clone();
                let decoded = decode_frame_with_carry(
                    frame_bytes,
                    &self.cr,
                    &self.quant_table_sets,
                    self.frame_dims,
                    self.ec,
                    self.options,
                    &mut working_carry,
                )?;
                self.observe_decoded_frame(&decoded)?;
                // Both stream-scope gates passed — commit the Frame's
                // end-of-Frame coder-state snapshot for the next
                // non-keyframe to resume.
                self.carry = working_carry;
                Ok(decoded)
            }
            ColorspaceType::Rgb => {
                // RGB / line-major: thread the §3.8.1.3 / §3.8.2.5
                // per-context coder state across non-keyframes, the same
                // way the YCbCr branch does. Decode into a working copy
                // of the carry so a post-decode conformance-gate failure
                // leaves the session's carry (and tracker / counter)
                // untouched — the "violating Frame does not advance the
                // session" contract the §4.2.17 / §5 gates keep.
                let mut working_carry = self.carry.clone();
                let decoded = decode_frame_rgb_with_carry(
                    frame_bytes,
                    &self.cr,
                    &self.quant_table_sets,
                    self.frame_dims,
                    self.ec,
                    self.options,
                    &mut working_carry,
                )?;
                self.observe_decoded_frame(&decoded)?;
                // Both stream-scope gates passed — commit the Frame's
                // end-of-Frame coder-state snapshot for the next
                // non-keyframe to resume.
                self.carry = working_carry;
                Ok(decoded)
            }
        }
    }

    /// Run the two stream-scope conformance gates on an
    /// already-decoded Frame, in coded order, without re-decoding.
    ///
    /// This is the seam for callers that drive the single-Frame
    /// decoders themselves (custom container plumbing, replay of
    /// cached decodes) but still want the session's §4.2.17 / §5
    /// cross-Frame enforcement. Reads exactly the two fields rounds
    /// 274 / 279 surfaced: [`DecodedFrame::keyframe`] and
    /// [`DecodedFrame::slice_headers`].
    ///
    /// On success the Frame becomes the §5 previous-Frame reference
    /// and the Frame counter advances. On failure the session state
    /// is unchanged (the violating Frame never becomes the reference
    /// for its successor).
    ///
    /// # Errors
    ///
    /// * [`Error::NonKeyframeInIntraStream`] — `intra == 1` on the
    ///   Configuration Record but `frame.keyframe == false`
    ///   (RFC 9043 §4.2.17 Table 14: "keyframe MUST be 1 (keyframes
    ///   only)").
    /// * [`Error::SliceGeometryUnstable`] — RFC 9043 §5 third
    ///   paragraph, via
    ///   [`SliceGeometryStabilityTracker::observe_frame`].
    pub fn observe_decoded_frame(&mut self, frame: &DecodedFrame) -> Result<(), Error> {
        // RFC 9043 §4.2.17 Table 14: `intra == 1` → "keyframe MUST be
        // 1 (keyframes only)". `intra` is "Inferred to be 0 if not
        // present", so an absent field (`None`, versions 0/1 or a
        // prefix-only parse) carries no constraint, same as
        // `Some(false)`.
        if self.cr.intra == Some(true) && !frame.keyframe {
            return Err(Error::NonKeyframeInIntraStream {
                frame_index: self.frames_observed,
            });
        }

        // RFC 9043 §5 third paragraph. The tracker leaves its
        // previous-Frame reference untouched on Err, so a violating
        // Frame never becomes the reference for its successor.
        self.tracker
            .observe_frame(frame.keyframe, &frame.slice_headers)?;

        self.frames_observed += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Ffv1Version;
    use crate::config::NUM_TRANSITION_DELTAS;
    use crate::frame::DecodedFramePlane;
    use crate::slice_header::{Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES};
    use crate::PictureStructure;

    fn gray_cr(intra: Option<bool>, num_h: u32, num_v: u32) -> Ffv1ConfigurationRecord {
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

    fn qts() -> Vec<QuantizationTableSet> {
        let mut tables = [[0i32; 256]; crate::predictor::NUM_QUANT_SUBTABLES];
        tables[0] = [5i32; 256];
        vec![QuantizationTableSet {
            tables,
            context_count: 6,
        }]
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

    fn frame_with(keyframe: bool, headers: Vec<Ffv1SliceHeader>) -> DecodedFrame {
        DecodedFrame {
            planes: vec![DecodedFramePlane {
                plane_index: 0,
                width: 4,
                height: 2,
                samples: vec![0; 8],
            }],
            width: 4,
            height: 2,
            bits_per_raw_sample: 8,
            colorspace: ColorspaceType::YCbCr,
            keyframe,
            slice_headers: headers,
        }
    }

    fn session(intra: Option<bool>) -> Ffv1DecodeSession {
        Ffv1DecodeSession::new(
            gray_cr(intra, 2, 2),
            qts(),
            FramePixelDimensions::new(4, 2).unwrap(),
            true,
        )
    }

    #[test]
    fn intra_one_rejects_non_keyframe_and_leaves_state_untouched() {
        let mut s = session(Some(true));
        s.observe_decoded_frame(&frame_with(true, vec![header(0, 0, 2, 2)]))
            .expect("keyframe conforms to intra == 1");
        assert_eq!(s.frames_observed(), 1);

        // Same geometry as the previous Frame, so only the §4.2.17
        // gate can fire — proving the intra gate runs before (and
        // independently of) the §5 geometry gate.
        let err = s
            .observe_decoded_frame(&frame_with(false, vec![header(0, 0, 2, 2)]))
            .expect_err("intra == 1 forbids keyframe == 0 (RFC 9043 §4.2.17 Table 14)");
        assert_eq!(err, Error::NonKeyframeInIntraStream { frame_index: 1 });
        assert_eq!(s.frames_observed(), 1, "violating Frame is not counted");

        // The stream continues fine on the next keyframe.
        s.observe_decoded_frame(&frame_with(true, vec![header(0, 0, 2, 2)]))
            .expect("subsequent keyframe conforms");
        assert_eq!(s.frames_observed(), 2);
    }

    #[test]
    fn intra_zero_or_absent_admits_non_keyframes() {
        for intra in [None, Some(false)] {
            let mut s = session(intra);
            s.observe_decoded_frame(&frame_with(true, vec![header(0, 0, 2, 2)]))
                .expect("keyframe opens the stream");
            s.observe_decoded_frame(&frame_with(false, vec![header(0, 0, 2, 2)]))
                .expect("intra inferred 0 / declared 0 admits keyframe == 0");
            assert_eq!(s.frames_observed(), 2);
        }
    }

    #[test]
    fn geometry_gate_fires_and_reference_survives_violation() {
        let mut s = session(Some(false));
        s.observe_decoded_frame(&frame_with(true, vec![header(0, 0, 2, 2)]))
            .expect("keyframe opens the stream");

        // A non-keyframe re-split violates §5 third paragraph.
        let err = s
            .observe_decoded_frame(&frame_with(
                false,
                vec![header(0, 0, 1, 2), header(1, 0, 1, 2)],
            ))
            .expect_err("re-split non-keyframe violates §5");
        assert!(matches!(
            err,
            Error::SliceGeometryUnstable { slice_index: 0, .. }
        ));
        assert_eq!(s.frames_observed(), 1);

        // The previous-Frame reference is still the keyframe's
        // geometry: a conforming non-keyframe passes.
        s.observe_decoded_frame(&frame_with(false, vec![header(0, 0, 2, 2)]))
            .expect("original geometry still conforms");
        assert_eq!(s.frames_observed(), 2);
    }

    #[test]
    fn accessors_report_session_state() {
        let s = session(None);
        assert_eq!(s.frames_observed(), 0);
        assert!(!s.has_previous_frame());
        assert_eq!(s.options(), DecodeOptions::strict());
        assert_eq!(s.configuration_record().coder_type, 1);

        let lenient = Ffv1DecodeSession::with_options(
            gray_cr(None, 1, 1),
            qts(),
            FramePixelDimensions::new(4, 2).unwrap(),
            true,
            DecodeOptions::lenient(),
        );
        assert_eq!(lenient.options(), DecodeOptions::lenient());
    }
}
