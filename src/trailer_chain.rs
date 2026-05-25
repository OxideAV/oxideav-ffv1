//! FFV1 §4.9.1 trailer-pointer chain walk.
//!
//! A v3 FFV1 frame is a concatenation of N Slices. There is no
//! per-frame index or sentinel marking slice boundaries on the wire —
//! every Slice ends with the §4.9 Slice Footer whose first field is a
//! `u(24)` `slice_size` reporting the *body* length (SliceHeader +
//! SliceContent + Golomb-Rice padding, **excluding** the footer).
//!
//! Locating the slice boundaries inside a raw frame payload therefore
//! requires walking the chain backwards from the end of the frame:
//!
//! ```text
//! cursor = frame.len()
//! while cursor > 0:
//!     slice_size = read_u24_be(frame[cursor - footer_len .. cursor - footer_len + 3])
//!     whole_slice_len = slice_size + footer_len
//!     slice_starts_at = cursor - whole_slice_len
//!     emit Slice(start = slice_starts_at, len = whole_slice_len)
//!     cursor = slice_starts_at
//! ```
//!
//! The footer length is 8 bytes when the Configuration Record carries
//! `ec == 1` (the §4.9 `if (ec)` branch) and 3 bytes when `ec == 0`. The
//! emitted records appear in reverse order (slice N-1 first, slice 0
//! last) because the chain is walked backwards; this module reverses
//! before returning so the caller sees slice 0 at index 0 — matching
//! the natural reading order of the Slice Header's `slice_x` /
//! `slice_y` raster coordinates.
//!
//! ## Why this lives in its own module
//!
//! The chain walk is the single piece of plumbing that turns a raw
//! per-frame byte buffer into the input array a frame-level Slice
//! decode driver consumes. It depends only on §4.9.1 (the `u(24)`
//! `slice_size` field) and the footer length convention; it does NOT
//! parse the SliceHeader, the SliceContent, the range coder, the
//! Golomb-Rice coder, or the §4.9.3 Slice CRC. Those validations are
//! the job of [`parse_slice_footer`](crate::parse_slice_footer),
//! [`parse_slice_header`](crate::parse_slice_header), and the per-plane
//! reconstructors that follow it. Keeping the chain walk a
//! tiny pure-byte-arithmetic function lets a fuzz harness exercise it
//! independently and lets the frame driver delegate the per-Slice
//! validation to the existing parsers.
//!
//! ## Round-126 contract
//!
//! `walk_trailer_chain(frame, ec)` returns one [`SliceExtent`] per
//! Slice. Each extent carries `start` (byte offset inside `frame`) and
//! `total_len` (whole-Slice length including the footer); the
//! corresponding range `frame[start .. start + total_len]` is exactly
//! the buffer [`parse_slice_footer`](crate::parse_slice_footer) expects.
//! Both the chain accounting and the §4.9.1 size field are
//! cross-checked against the buffer length; any inconsistency surfaces
//! as a typed error and the chain walk aborts rather than reporting
//! partial results.

use crate::slice_footer::{SLICE_FOOTER_LEN_EC0, SLICE_FOOTER_LEN_EC1};
use crate::Error;

/// Byte extent of one Slice inside a frame's raw payload.
///
/// `start` is the offset (in bytes, from the start of the frame
/// payload) at which the Slice's SliceHeader begins. `total_len`
/// includes both the body (SliceHeader + SliceContent +
/// Golomb-Rice padding) and the 3- / 8-byte Slice Footer — i.e. the
/// byte range `frame[start .. start + total_len]` is the whole-Slice
/// buffer [`parse_slice_footer`](crate::parse_slice_footer) consumes.
///
/// `total_len` equals `slice_size + footer_len`, where `slice_size` is
/// the §4.9.1 field decoded from the trailer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceExtent {
    /// Byte offset (in the frame payload) where the Slice begins.
    pub start: usize,
    /// Total byte length of the Slice (body + footer).
    pub total_len: usize,
}

impl SliceExtent {
    /// One-past-the-last byte offset of this Slice inside the frame.
    pub fn end(self) -> usize {
        self.start + self.total_len
    }
}

/// Footer length helper: 8 bytes when `ec` is set, 3 bytes otherwise.
///
/// Exposed alongside the chain walker so a caller that pre-computes the
/// per-slice byte ranges (e.g. to lay them out in a `Vec<&[u8]>`) does
/// not have to re-derive the constant.
pub fn slice_footer_len(ec: bool) -> usize {
    if ec {
        SLICE_FOOTER_LEN_EC1
    } else {
        SLICE_FOOTER_LEN_EC0
    }
}

/// Walk the §4.9.1 trailer-pointer chain backwards through `frame`,
/// returning one [`SliceExtent`] per Slice in **forward** order
/// (`extents[0]` is slice 0, `extents.last()` is the highest-indexed
/// slice).
///
/// `ec` selects the footer length (8 bytes for `ec == 1`, 3 bytes for
/// `ec == 0`); it MUST match the Configuration Record's
/// `error_correction` (`ec`) field.
///
/// # Algorithm
///
/// At each iteration the cursor sits one byte past the end of the
/// next-to-discover Slice. The slice's footer ends at the cursor; the
/// `u(24)` `slice_size` field is read from the first three bytes of the
/// footer (offsets `cursor - footer_len .. cursor - footer_len + 3`).
/// The whole Slice occupies `[cursor - slice_size - footer_len .. cursor]`.
/// The cursor advances backwards by that amount and the loop continues
/// until the cursor reaches `0` (no bytes remain).
///
/// # Returns
///
/// The discovered extents in **slice-index order** (slice 0 first). For
/// a single-slice frame the returned vector has one entry whose
/// `start == 0` and `total_len == frame.len()`. For a `num_h_slices *
/// num_v_slices`-sliced frame, `extents.len()` equals the slice count
/// declared in the Configuration Record — but the chain walk does NOT
/// re-derive this from the slice-grid metadata; it relies purely on the
/// §4.9.1 trailer pointers because that is what the wire format
/// actually provides.
///
/// # Errors
///
/// * [`Error::TruncatedSliceFooter`] — `frame.len()` is shorter than a
///   single footer, OR a §4.9.1 `slice_size` field announces a body
///   longer than the bytes remaining in front of the footer (i.e. the
///   chain walk would underflow the cursor).
///
/// The walk is greedy and aborts on the first error; partial results
/// are NOT returned, because the §4.9.1 chain is a tightly-coupled
/// sequence — one mis-read `slice_size` shifts every preceding
/// boundary, so partial answers would be actively misleading.
pub fn walk_trailer_chain(frame: &[u8], ec: bool) -> Result<Vec<SliceExtent>, Error> {
    let footer_len = slice_footer_len(ec);
    if frame.len() < footer_len {
        return Err(Error::TruncatedSliceFooter);
    }

    // Discovered in reverse (slice N-1 first); we reverse before
    // returning so the caller sees slice 0 at index 0.
    let mut rev_extents: Vec<SliceExtent> = Vec::new();
    let mut cursor: usize = frame.len();

    while cursor > 0 {
        // A Slice must contain at least its footer.
        if cursor < footer_len {
            return Err(Error::TruncatedSliceFooter);
        }

        // The §4.9.1 `slice_size` field is the first three bytes of
        // the footer (`u(24)` big-endian).
        let size_off = cursor - footer_len;
        let s0 = frame[size_off];
        let s1 = frame[size_off + 1];
        let s2 = frame[size_off + 2];
        let slice_size = (u32::from(s0) << 16) | (u32::from(s1) << 8) | u32::from(s2);
        let slice_size = slice_size as usize;

        // The whole Slice (body + footer) occupies these bytes.
        let total_len = slice_size + footer_len;
        if total_len > cursor {
            // The declared body length plus footer would underflow the
            // cursor — a malformed §4.9.1 field or an `ec` mismatch
            // (e.g. trying to read a 3-byte footer from an `ec=1`
            // frame). Either way the only honest signal is "this
            // chain doesn't parse".
            return Err(Error::TruncatedSliceFooter);
        }
        let start = cursor - total_len;

        rev_extents.push(SliceExtent { start, total_len });
        cursor = start;
    }

    rev_extents.reverse();
    Ok(rev_extents)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a synthetic Slice byte sequence whose footer announces
    /// `body_size` as its `slice_size`. The body bytes are filled with
    /// `body_fill`; the footer is zero-padded (slice_size + zero
    /// error_status + zero parity for `ec=1`). The whole-Slice CRC is
    /// NOT solved for — the chain walker doesn't read the parity.
    fn synth_slice(body_size: usize, body_fill: u8, ec: bool) -> Vec<u8> {
        let footer_len = slice_footer_len(ec);
        let mut buf = vec![body_fill; body_size];
        // size field (u(24) big-endian)
        buf.push(((body_size >> 16) & 0xFF) as u8);
        buf.push(((body_size >> 8) & 0xFF) as u8);
        buf.push((body_size & 0xFF) as u8);
        if ec {
            // error_status + 4-byte parity (irrelevant for the walker).
            buf.push(0);
            buf.extend_from_slice(&[0u8; 4]);
        }
        debug_assert_eq!(buf.len(), body_size + footer_len);
        buf
    }

    // ----- ec=1, single-slice ------------------------------------------------

    #[test]
    fn single_slice_ec1_round_trip() {
        let body_size = 100usize;
        let slice = synth_slice(body_size, 0xAA, true);
        let extents = walk_trailer_chain(&slice, true).unwrap();
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].start, 0);
        assert_eq!(extents[0].total_len, body_size + 8);
        assert_eq!(extents[0].end(), slice.len());
    }

    // ----- ec=0, single-slice ------------------------------------------------

    #[test]
    fn single_slice_ec0_round_trip() {
        let body_size = 50usize;
        let slice = synth_slice(body_size, 0x55, false);
        let extents = walk_trailer_chain(&slice, false).unwrap();
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].start, 0);
        assert_eq!(extents[0].total_len, body_size + 3);
    }

    // ----- multi-slice forward ordering --------------------------------------

    #[test]
    fn four_slices_ec1_emitted_in_forward_order() {
        // Encode 4 Slices end-to-end with body sizes 10, 20, 30, 40 bytes.
        // After the chain walk, extents[0] must point at the first body
        // (size 10), extents[3] at the last body (size 40).
        let mut frame = Vec::new();
        let mut starts = Vec::new();
        let mut totals = Vec::new();
        for (i, body_size) in [10usize, 20, 30, 40].into_iter().enumerate() {
            starts.push(frame.len());
            let s = synth_slice(body_size, 0x10 + i as u8, true);
            totals.push(s.len());
            frame.extend_from_slice(&s);
        }
        let extents = walk_trailer_chain(&frame, true).unwrap();
        assert_eq!(extents.len(), 4);
        for (i, ext) in extents.iter().enumerate() {
            assert_eq!(ext.start, starts[i], "slice {i} start");
            assert_eq!(ext.total_len, totals[i], "slice {i} total_len");
        }
        // The walk must have covered the whole frame: last extent ends
        // at frame.len().
        assert_eq!(extents.last().unwrap().end(), frame.len());
    }

    #[test]
    fn four_slices_ec0_emitted_in_forward_order() {
        let mut frame = Vec::new();
        let mut starts = Vec::new();
        for (i, body_size) in [5usize, 7, 11, 13].into_iter().enumerate() {
            starts.push(frame.len());
            let s = synth_slice(body_size, 0x20 + i as u8, false);
            frame.extend_from_slice(&s);
        }
        let extents = walk_trailer_chain(&frame, false).unwrap();
        assert_eq!(extents.len(), 4);
        for (i, ext) in extents.iter().enumerate() {
            assert_eq!(ext.start, starts[i], "ec=0 slice {i} start");
        }
        assert_eq!(extents.last().unwrap().end(), frame.len());
    }

    // ----- error paths -------------------------------------------------------

    #[test]
    fn empty_frame_rejected_for_ec1() {
        assert!(matches!(
            walk_trailer_chain(&[], true),
            Err(Error::TruncatedSliceFooter)
        ));
    }

    #[test]
    fn empty_frame_rejected_for_ec0() {
        assert!(matches!(
            walk_trailer_chain(&[], false),
            Err(Error::TruncatedSliceFooter)
        ));
    }

    #[test]
    fn frame_shorter_than_footer_rejected_ec1() {
        // 5 bytes < 8-byte ec=1 footer.
        let frame = [0u8; 5];
        assert!(matches!(
            walk_trailer_chain(&frame, true),
            Err(Error::TruncatedSliceFooter)
        ));
    }

    #[test]
    fn frame_shorter_than_footer_rejected_ec0() {
        // 2 bytes < 3-byte ec=0 footer.
        let frame = [0u8; 2];
        assert!(matches!(
            walk_trailer_chain(&frame, false),
            Err(Error::TruncatedSliceFooter)
        ));
    }

    #[test]
    fn declared_slice_size_overruns_frame() {
        // ec=1, but the trailing u(24) reports a body size of 999, far
        // greater than the 16-byte frame. The walker must surface
        // TruncatedSliceFooter rather than producing a nonsense extent.
        let mut frame = vec![0u8; 16];
        let size_off = frame.len() - 8;
        // size_field = 999 = 0x0003E7 → bytes 0x00 0x03 0xE7.
        frame[size_off] = 0;
        frame[size_off + 1] = 0x03;
        frame[size_off + 2] = 0xE7;
        assert!(matches!(
            walk_trailer_chain(&frame, true),
            Err(Error::TruncatedSliceFooter)
        ));
    }

    #[test]
    fn declared_slice_size_exactly_at_boundary_ok() {
        // ec=1, frame = body(0) + footer(8) = 8 bytes; size_field = 0.
        // This is the degenerate "empty-body single-slice" case the
        // §4.9 footer parser also accepts — the walker emits one
        // extent of length 8 starting at offset 0.
        let mut frame = vec![0u8; 8];
        // size_field = 0 → already zero.
        frame[3] = 0; // error_status
        let extents = walk_trailer_chain(&frame, true).unwrap();
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].start, 0);
        assert_eq!(extents[0].total_len, 8);
    }

    // ----- ec mismatch caught by structural check ----------------------------

    #[test]
    fn ec_mismatch_walk_fails_when_3_byte_footer_misread_as_8_byte() {
        // Build an ec=0 single-slice frame (body 4 bytes + 3-byte
        // footer = 7 bytes). Calling the walker with ec=true tries to
        // read an 8-byte footer; 7 < 8 → TruncatedSliceFooter.
        let frame = synth_slice(4, 0xCC, false);
        assert_eq!(frame.len(), 7);
        assert!(matches!(
            walk_trailer_chain(&frame, true),
            Err(Error::TruncatedSliceFooter)
        ));
    }

    // ----- forward-order helper: end() -------------------------------------

    #[test]
    fn slice_extent_end_is_start_plus_total_len() {
        let e = SliceExtent {
            start: 17,
            total_len: 42,
        };
        assert_eq!(e.end(), 59);
    }

    // ----- non-greedy: chain walk visits every byte --------------------------

    #[test]
    fn chain_walk_covers_every_byte_of_frame() {
        // Build a 5-slice frame; the union of all extents must equal
        // [0, frame.len()) with no gaps and no overlaps.
        let mut frame = Vec::new();
        for (i, body_size) in [3usize, 19, 7, 200, 42].into_iter().enumerate() {
            let s = synth_slice(body_size, 0x30 + i as u8, true);
            frame.extend_from_slice(&s);
        }
        let extents = walk_trailer_chain(&frame, true).unwrap();
        assert_eq!(extents.len(), 5);
        let mut expected_start = 0usize;
        for ext in &extents {
            assert_eq!(ext.start, expected_start);
            expected_start += ext.total_len;
        }
        assert_eq!(expected_start, frame.len());
    }

    // ----- many-slice walk does not blow the stack ---------------------------

    #[test]
    fn many_small_slices_walked_iteratively() {
        // 64 1-byte-body slices, ec=1 → 64 * 9 = 576 bytes total.
        let mut frame = Vec::new();
        for i in 0..64u8 {
            frame.extend_from_slice(&synth_slice(1, i, true));
        }
        let extents = walk_trailer_chain(&frame, true).unwrap();
        assert_eq!(extents.len(), 64);
        for (i, ext) in extents.iter().enumerate() {
            assert_eq!(ext.start, i * 9);
            assert_eq!(ext.total_len, 9);
        }
    }
}
