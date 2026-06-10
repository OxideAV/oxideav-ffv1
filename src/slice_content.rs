//! FFV1 Slice Content scaffold (RFC 9043 §4.7 / §4.8).
//!
//! This module materializes the *structural* layout of a Slice's
//! content: the per-plane pixel grid, the per-line container, and the
//! traversal order — without decoding any `sample_difference` symbols.
//! Pixel decode (the `sd`-typed sample-difference stream that fills
//! each `Line`) is the next round; round 3's job is to give that round
//! a typed `SliceContent { planes: Vec<Plane> }` to fill in.
//!
//! Per RFC 9043 §4.7 pseudocode, a Slice Content is one of two
//! interleavings:
//!
//! ```text
//! SliceContent( ) {
//!     if (colorspace_type == 0) {              // YCbCr: Plane then Line
//!         for (p = 0; p < primary_color_count; p++) {
//!             for (y = 0; y < plane_pixel_height[ p ]; y++) {
//!                 Line(p, y)
//!             }
//!         }
//!     } else if (colorspace_type == 1) {       // RGB:   Line then Plane
//!         for (y = 0; y < slice_pixel_height; y++) {
//!             for (p = 0; p < primary_color_count; p++) {
//!                 Line(p, y)
//!             }
//!         }
//!     }
//! }
//! ```
//!
//! and §4.7.1 defines `primary_color_count` as:
//!
//! ```text
//! 1 + (chroma_planes ? 2 : 0) + (extra_plane ? 1 : 0)
//! ```
//!
//! The per-plane width/height come from §4.7.2 / §4.8.1: chroma planes
//! (`p == 1 || p == 2`) are subsampled by
//! `1 << log2_{h,v}_chroma_subsample` when `chroma_planes == 1`; every
//! other plane runs at the slice's full pixel width/height.
//!
//! Slice geometry (the pixel-space rectangle each Slice covers) is
//! itself derived from the Slice Header's raster coordinates (`slice_x`,
//! `slice_y`, `slice_width`, `slice_height`), the per-stream raster
//! shape (`num_h_slices`, `num_v_slices`), and the *frame* pixel
//! dimensions:
//!
//! - §4.8.3: `slice_pixel_x  = floor(slice_x  * frame_pixel_width  / num_h_slices)`
//! - §4.7.4: `slice_pixel_y  = floor(slice_y  * frame_pixel_height / num_v_slices)`
//! - §4.8.2: `slice_pixel_width  = floor((slice_x + slice_width)  * frame_pixel_width  / num_h_slices) - slice_pixel_x`
//! - §4.7.3: `slice_pixel_height = floor((slice_y + slice_height) * frame_pixel_height / num_v_slices) - slice_pixel_y`
//!
//! Spec gap: RFC 9043 §4.7.3 prints the right-hand side as
//! `slice_pixel_height` instead of `frame_pixel_height`. This is a
//! documentation typo — the §4.8.2 sibling formula reads
//! `frame_pixel_width`, the §4.7.4 definition of `slice_pixel_y`
//! requires `frame_pixel_height` to be the per-stream constant, and
//! using `slice_pixel_height` on the RHS of its own definition is
//! self-referential. We use `frame_pixel_height` (the unambiguous
//! reading) and the v3-default/v3-grayscale/v3-rgb-bgr0 fixtures
//! confirm the chosen per-plane pixel sizes match the trace files'
//! `PLANE w=… h=…` ground truth bit-exactly.
//!
//! Frame pixel dimensions are NOT part of the Configuration Record
//! (RFC 9043 §4.2 lists no `width` / `height` fields) — they come from
//! the surrounding container (Matroska `PixelWidth` / `PixelHeight`,
//! AVI `biWidth` / `biHeight`, MP4 `tkhd` `width` / `height`, etc.).
//! The caller therefore supplies them as an explicit
//! [`FramePixelDimensions`] argument.

use crate::config::{ColorspaceType, Ffv1ConfigurationRecord};
use crate::slice_header::{Ffv1SliceHeader, MAX_QUANT_TABLE_SET_INDEXES};
use crate::Error;

/// Frame-level pixel dimensions, supplied by the container.
///
/// FFV1's Configuration Record does NOT carry frame width / height
/// (see RFC 9043 §4.2 Parameters — neither field is listed); they are
/// part of the per-container track header (Matroska `PixelWidth` /
/// `PixelHeight`, AVI BITMAPINFOHEADER `biWidth` / `biHeight`, MP4
/// `tkhd`, etc.). The slice-pixel-grid math in §4.7.3 / §4.7.4 /
/// §4.8.2 / §4.8.3 needs both, so callers pass them in explicitly.
///
/// Constructing a zero-dimension value is rejected by
/// [`compute_slice_content`] (a `0 * anything / N` raster cell would
/// trivially collapse and the resulting `SliceContent` would be
/// indistinguishable from a missing one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePixelDimensions {
    /// Frame width in pixels (the unsubsampled luma width when
    /// chroma_planes is set; the single-plane width when it isn't).
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl FramePixelDimensions {
    /// Construct a [`FramePixelDimensions`] from width and height.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidFramePixelDimensions`] if either
    /// dimension is zero.
    pub fn new(width: u32, height: u32) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidFramePixelDimensions { width, height });
        }
        Ok(Self { width, height })
    }
}

/// Maximum number of planes per Slice (RFC 9043 §4.7.1).
///
/// `primary_color_count = 1 + (chroma_planes ? 2 : 0) + (extra_plane ? 1 : 0)`,
/// so the result is in `1..=4` inclusive — 1 for grayscale, 2 for
/// grayscale + alpha, 3 for YUV / RGB, 4 for YUV/RGB + alpha.
pub const MAX_PRIMARY_COLOR_COUNT: usize = 4;

/// A single Line of a Plane (RFC 9043 §4.8).
///
/// In the round-3 scaffold a Line is the typed *container* for one
/// row of sample differences; the actual `sd` symbol stream (which
/// would populate the inner samples) is not decoded yet. The Line
/// carries its plane index and y position so a future round's pixel
/// decoder can route each `sample_difference[p][y][x]` to the right
/// slot without recomputing the iteration order.
///
/// The plane's pixel width is held on the parent [`Plane`] (and is
/// constant across all of that plane's lines); it is intentionally
/// NOT duplicated per-Line to keep the per-Line struct cache-friendly
/// and to avoid drift between the plane-level dimensioning and the
/// per-row width when round 4 lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Line {
    /// `p` from §4.7 / §4.8 — the plane index this line belongs to
    /// (`0..primary_color_count`).
    pub plane_index: u8,
    /// `y` from §4.7 / §4.8 — the row index inside its plane
    /// (`0..plane_pixel_height[p]`).
    pub y: u32,
}

/// A single Plane of a Slice (RFC 9043 §2.1 "Plane" + §4.7.2 /
/// §4.8.1 dimensions).
///
/// The scaffold's `Plane` records its pixel rectangle (`width`,
/// `height`) and an empty `lines` vector pre-sized to `height`.
/// `lines[y]` is the typed home of row `y`'s `sample_difference`
/// stream — populated by round 4's pixel-decode pass; round 3 leaves
/// the row identity (`plane_index` + `y`) populated and the actual
/// sample storage absent.
#[derive(Debug, Clone)]
pub struct Plane {
    /// `p` from §4.7 / §4.8 — the plane's index inside its slice.
    pub plane_index: u8,
    /// `plane_pixel_width[p]` (RFC 9043 §4.8.1) — the per-plane row
    /// length in pixels.
    pub width: u32,
    /// `plane_pixel_height[p]` (RFC 9043 §4.7.2) — the per-plane
    /// column length in pixels.
    pub height: u32,
    /// One entry per row of the plane; `lines[y].y == y`. Empty on
    /// the round-3 scaffold; populated by future pixel-decode rounds.
    ///
    /// Storage is allocated up front (`Vec::with_capacity(height)`)
    /// and filled with placeholder [`Line`] records carrying their
    /// plane / y identity so a consumer can pre-walk the structure
    /// before any samples are decoded.
    pub lines: Vec<Line>,
}

/// Slice Content scaffold (RFC 9043 §4.7).
///
/// `planes` length equals `primary_color_count` per §4.7.1; the
/// iteration order recorded in `traversal` mirrors the spec
/// pseudocode (Plane-major for `colorspace_type == 0`, Line-major
/// for `colorspace_type == 1`).
#[derive(Debug, Clone)]
pub struct SliceContent {
    /// `primary_color_count` planes (RFC 9043 §4.7.1).
    pub planes: Vec<Plane>,
    /// `slice_pixel_x` (RFC 9043 §4.8.3) — pixel-space top-left x.
    pub slice_pixel_x: u32,
    /// `slice_pixel_y` (RFC 9043 §4.7.4) — pixel-space top-left y.
    pub slice_pixel_y: u32,
    /// `slice_pixel_width` (RFC 9043 §4.8.2) — pixel-space width.
    pub slice_pixel_width: u32,
    /// `slice_pixel_height` (RFC 9043 §4.7.3, with the documented
    /// `frame_pixel_height` reading) — pixel-space height.
    pub slice_pixel_height: u32,
    /// Iteration order used to walk `Line(p, y)` calls, mirroring
    /// §4.7 SliceContent() pseudocode.
    pub traversal: PlaneTraversal,
}

/// Iteration order for Line decoding inside a SliceContent
/// (RFC 9043 §4.7 pseudocode).
///
/// - [`PlaneTraversal::PlaneMajor`] — outer `for p`, inner `for y`.
///   Used when `colorspace_type == 0` (YCbCr).
/// - [`PlaneTraversal::LineMajor`] — outer `for y`, inner `for p`.
///   Used when `colorspace_type == 1` (RGB).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneTraversal {
    /// `for (p) for (y) Line(p, y)` — YCbCr / `colorspace_type == 0`.
    PlaneMajor,
    /// `for (y) for (p) Line(p, y)` — RGB / `colorspace_type == 1`.
    LineMajor,
}

impl PlaneTraversal {
    /// Map the configuration-record colorspace_type to the iteration
    /// order RFC 9043 §4.7 mandates.
    pub fn for_colorspace(cs: ColorspaceType) -> Self {
        match cs {
            ColorspaceType::YCbCr => Self::PlaneMajor,
            ColorspaceType::Rgb => Self::LineMajor,
        }
    }
}

/// Per-plane traversal step: `(plane_index, y_in_plane)`.
///
/// Round 3 only needs the visit-order primitive; round 4 will feed
/// each step into the `sample_difference` decode loop of RFC 9043
/// §4.8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineVisit {
    /// The plane being visited at this step.
    pub plane_index: u8,
    /// The row within that plane.
    pub y: u32,
}

/// Compute `primary_color_count` per RFC 9043 §4.7.1.
///
/// Result is in `1..=4`. Returns `usize` because it is immediately
/// used as a vector length.
fn primary_color_count(cr: &Ffv1ConfigurationRecord) -> usize {
    1 + usize::from(cr.chroma_planes) * 2 + usize::from(cr.extra_plane)
}

/// Compute `slice_pixel_x` (RFC 9043 §4.8.3) and `slice_pixel_y`
/// (RFC 9043 §4.7.4) plus their per-slice widths/heights
/// (§4.8.2 / §4.7.3) for one slice on the configured raster.
///
/// `frame` provides the per-stream frame dimensions; `header`
/// provides the raster cell.
///
/// Returns `(slice_pixel_x, slice_pixel_y, slice_pixel_width,
/// slice_pixel_height)`.
fn slice_pixel_rect(
    header: &Ffv1SliceHeader,
    num_h_slices: u32,
    num_v_slices: u32,
    frame: FramePixelDimensions,
) -> (u32, u32, u32, u32) {
    // 32×32 -> 64-bit math is enough; FFV1 frames are bounded by
    // 2^32 pixels on either dimension well below the multiplication
    // overflow boundary in u64. We stay in u64 to keep the math
    // explicit and crash-free under spec-legal but extreme inputs.
    let nx = u64::from(num_h_slices);
    let ny = u64::from(num_v_slices);
    let fw = u64::from(frame.width);
    let fh = u64::from(frame.height);
    let sx = u64::from(header.slice_x);
    let sy = u64::from(header.slice_y);
    let sw = u64::from(header.slice_width);
    let sh = u64::from(header.slice_height);

    // §4.8.3: slice_pixel_x = floor(slice_x * frame_pixel_width / num_h_slices)
    let pixel_x = sx * fw / nx;
    // §4.7.4: slice_pixel_y = floor(slice_y * frame_pixel_height / num_v_slices)
    let pixel_y = sy * fh / ny;
    // §4.8.2: slice_pixel_width  = floor((slice_x + slice_width) * frame_pixel_width  / num_h_slices) - slice_pixel_x
    let pixel_w = (sx + sw) * fw / nx - pixel_x;
    // §4.7.3 (with frame_pixel_height applied for the typo'd RHS;
    // see module-level doc): slice_pixel_height = floor((slice_y + slice_height) * frame_pixel_height / num_v_slices) - slice_pixel_y
    let pixel_h = (sy + sh) * fh / ny - pixel_y;

    (
        pixel_x as u32,
        pixel_y as u32,
        pixel_w as u32,
        pixel_h as u32,
    )
}

/// Compute `plane_pixel_width[p]` (RFC 9043 §4.8.1).
///
/// Chroma planes (`p == 1 || p == 2`) get the slice's pixel width
/// shifted by `log2_h_chroma_subsample`; every other plane runs at
/// the full slice width. The §4.8.1 `ceil(... / (1 << shift))` form
/// is rewritten as `(w + (1<<shift) - 1) >> shift` to avoid a u32
/// division when the shift is small (the spec's algebra is identical
/// because the result of `ceil(a/b)` for nonnegative integers equals
/// `(a + b - 1) / b`).
fn plane_pixel_width(slice_pixel_width: u32, plane_index: u8, cr: &Ffv1ConfigurationRecord) -> u32 {
    if cr.chroma_planes && (plane_index == 1 || plane_index == 2) {
        let shift = cr.log2_h_chroma_subsample;
        let denom = 1u32.checked_shl(shift).expect(
            "RFC 9043 §4.2.8 caps log2_h_chroma_subsample at 4; checked in parse_configuration_record",
        );
        // ceil(slice_pixel_width / denom) = (slice_pixel_width + denom - 1) / denom
        slice_pixel_width.saturating_add(denom - 1) / denom
    } else {
        slice_pixel_width
    }
}

/// Compute `plane_pixel_height[p]` (RFC 9043 §4.7.2).
///
/// Mirrors [`plane_pixel_width`] over the vertical subsample.
fn plane_pixel_height(
    slice_pixel_height: u32,
    plane_index: u8,
    cr: &Ffv1ConfigurationRecord,
) -> u32 {
    if cr.chroma_planes && (plane_index == 1 || plane_index == 2) {
        let shift = cr.log2_v_chroma_subsample;
        let denom = 1u32.checked_shl(shift).expect(
            "RFC 9043 §4.2.9 caps log2_v_chroma_subsample at 4; checked in parse_configuration_record",
        );
        slice_pixel_height.saturating_add(denom - 1) / denom
    } else {
        slice_pixel_height
    }
}

/// Build the typed [`SliceContent`] scaffold for one slice.
///
/// `header` is the parsed Slice Header (round 2). `cr` is the parsed
/// Configuration Record (round 1). `frame` carries the per-stream
/// pixel dimensions the surrounding container reported (FFV1's
/// Configuration Record does NOT contain them — see §4.2 Parameters).
///
/// The result is a `SliceContent` whose `planes[i]` matches RFC 9043
/// §4.7.1's `i`-th `primary_color_count` entry, with per-plane
/// dimensions per §4.7.2 / §4.8.1 and the appropriate
/// `PlaneTraversal` for the configured colorspace.
///
/// **Pixel decode is not performed.** Each `Plane.lines[y]` is a
/// stub carrying its `(plane_index, y)` identity so round 4 can
/// drive the `sample_difference` decode without recomputing
/// iteration order.
///
/// # Errors
///
/// * [`Error::SliceRequiresVersion3`] — the configuration record's
///   `num_h_slices` / `num_v_slices` are `None` (versions 0/1 carry
///   the slice grid inside the keyframe header, not the
///   Configuration Record — this round only handles version 3).
/// * [`Error::InvalidFramePixelDimensions`] — `frame.width` or
///   `frame.height` is zero.
/// * [`Error::SliceRasterOutOfRange`] — the parsed Slice Header
///   addresses a raster cell outside the configured
///   `num_h_slices` × `num_v_slices` grid, OR a derived
///   `slice_pixel_width` / `slice_pixel_height` collapses to zero.
pub fn compute_slice_content(
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    frame: FramePixelDimensions,
) -> Result<SliceContent, Error> {
    let (num_h_slices, num_v_slices) = match (cr.num_h_slices, cr.num_v_slices) {
        (Some(h), Some(v)) => (h, v),
        _ => return Err(Error::SliceRequiresVersion3),
    };
    if num_h_slices == 0 || num_v_slices == 0 {
        return Err(Error::SliceRasterOutOfRange {
            slice_x: header.slice_x,
            slice_y: header.slice_y,
            slice_width: header.slice_width,
            slice_height: header.slice_height,
            num_h_slices,
            num_v_slices,
        });
    }
    // Raster-bounds check: §4.6.1 / §4.6.2 / §4.6.3 / §4.6.4 admit
    // any `slice_x + slice_width <= num_h_slices` and similarly for
    // y; anything else addresses a cell off the configured grid.
    // `map_or(true, ...)` keeps the `Option::None` (overflow) case as
    // "out of range" without `is_none_or` (MSRV 1.82 vs ours 1.80).
    let x_oob = header
        .slice_x
        .checked_add(header.slice_width)
        .map_or(true, |sum| sum > num_h_slices);
    let y_oob = header
        .slice_y
        .checked_add(header.slice_height)
        .map_or(true, |sum| sum > num_v_slices);
    if x_oob || y_oob {
        return Err(Error::SliceRasterOutOfRange {
            slice_x: header.slice_x,
            slice_y: header.slice_y,
            slice_width: header.slice_width,
            slice_height: header.slice_height,
            num_h_slices,
            num_v_slices,
        });
    }

    let (pixel_x, pixel_y, pixel_w, pixel_h) =
        slice_pixel_rect(header, num_h_slices, num_v_slices, frame);
    if pixel_w == 0 || pixel_h == 0 {
        return Err(Error::SliceRasterOutOfRange {
            slice_x: header.slice_x,
            slice_y: header.slice_y,
            slice_width: header.slice_width,
            slice_height: header.slice_height,
            num_h_slices,
            num_v_slices,
        });
    }

    let count = primary_color_count(cr);
    debug_assert!(count <= MAX_PRIMARY_COLOR_COUNT);
    // Compile-time check that the slice-header's quant-index array
    // never out-runs the slice-content's plane array.
    const _: () = assert!(MAX_QUANT_TABLE_SET_INDEXES <= MAX_PRIMARY_COLOR_COUNT);

    let mut planes = Vec::with_capacity(count);
    for p in 0..count {
        let p_u8 = p as u8;
        let width = plane_pixel_width(pixel_w, p_u8, cr);
        let height = plane_pixel_height(pixel_h, p_u8, cr);
        let mut lines = Vec::with_capacity(height as usize);
        for y in 0..height {
            lines.push(Line {
                plane_index: p_u8,
                y,
            });
        }
        planes.push(Plane {
            plane_index: p_u8,
            width,
            height,
            lines,
        });
    }

    Ok(SliceContent {
        planes,
        slice_pixel_x: pixel_x,
        slice_pixel_y: pixel_y,
        slice_pixel_width: pixel_w,
        slice_pixel_height: pixel_h,
        traversal: PlaneTraversal::for_colorspace(cr.colorspace_type),
    })
}

/// RFC 9043 §5 "Restrictions" max-slice-size threshold (in pixels).
///
/// The spec phrases the trigger as
/// `frame_pixel_width * frame_pixel_height > 101376`, where 101376 =
/// 352 × 288 = a CIF frame. Above this size the §5 restriction
///
/// > slice_width * slice_height MUST be less or equal to
/// > num_h_slices * num_v_slices / 4
///
/// applies on every Slice in the Frame; at or below it the
/// restriction is silent (any raster footprint is admissible).
pub const SECTION_5_MAX_SLICE_AREA_THRESHOLD: u64 = 101_376;

/// RFC 9043 §5 "Restrictions" — per-Slice raster footprint cap on
/// version-3 streams whose frame area exceeds
/// [`SECTION_5_MAX_SLICE_AREA_THRESHOLD`].
///
/// The §5 restriction is:
///
/// > To ensure that fast multithreaded decoding is possible, starting
/// > with version 3 and if frame_pixel_width * frame_pixel_height is
/// > more than 101376, slice_width * slice_height MUST be less or
/// > equal to num_h_slices * num_v_slices / 4.
///
/// In raster-cell units (which is what the Slice Header carries —
/// `slice_width` / `slice_height` are §4.6.3 / §4.6.4 cell counts on
/// the `num_h_slices × num_v_slices` grid), the restriction caps each
/// Slice at one quarter of the raster, i.e. a slice can cover at most
/// `floor(num_h_slices * num_v_slices / 4)` raster cells. This is the
/// per-Frame multithreading floor: at four equal Slices the floor is
/// exactly the raster size; smaller Slices satisfy it by construction;
/// a single Slice covering the whole raster violates it whenever the
/// Frame is above the threshold.
///
/// This function exposes the check as a pure structural validator on
/// `(header, cr, frame_dims)`. The frame-level decode drivers
/// ([`crate::decode_frame`] / [`crate::decode_frame_rgb`]) invoke it
/// per-Slice after the §4.6 Slice Header parse so a §5 violation
/// aborts the frame before any pixel reconstruction touches the
/// offending Slice.
///
/// # Returns
///
/// * `Ok(())` — either the §5 trigger does not apply (versions 0/1,
///   or `frame_pixel_width * frame_pixel_height <=
///   SECTION_5_MAX_SLICE_AREA_THRESHOLD`), or it applies and the
///   per-Slice footprint satisfies it.
/// * [`Error::SliceMaxSizeExceeded`] — the trigger applies and
///   `slice_width * slice_height > num_h_slices * num_v_slices / 4`.
///
/// Other §5 restrictions (no-gap / no-overlap raster coverage on a
/// per-Frame basis; per-non-keyframe Slice stability across Frames)
/// are out of scope for this round — they require multi-Slice / multi-
/// Frame state and are tracked separately.
///
/// # Errors
///
/// * [`Error::SliceRequiresVersion3`] when the Configuration Record's
///   `num_h_slices` / `num_v_slices` are absent (FFV1 v0 / v1). The §5
///   restriction is only normative on v3 and the validator has no
///   way to evaluate the inequality without the grid shape; surface
///   this as the same error every other §4 / §5 grid-dependent helper
///   uses for that branch.
/// * [`Error::InvalidFramePixelDimensions`] when `frame_dims` is zero
///   in either axis. The §5 trigger inequality
///   `frame_pixel_width * frame_pixel_height > 101376` is ill-defined
///   on a zero-area Frame; reject for the same reason
///   [`compute_slice_content`] rejects zero-area Frames.
pub fn validate_slice_max_size_restriction(
    header: &Ffv1SliceHeader,
    cr: &Ffv1ConfigurationRecord,
    frame_dims: FramePixelDimensions,
) -> Result<(), Error> {
    // §5 "starting with version 3" — earlier versions are unconstrained
    // (and the Configuration Record carries no slice grid for them).
    // Use the same surface as every other §4 grid-dependent helper:
    // missing num_h_slices / num_v_slices → SliceRequiresVersion3.
    let (num_h_slices, num_v_slices) = match (cr.num_h_slices, cr.num_v_slices) {
        (Some(h), Some(v)) => (h, v),
        _ => return Err(Error::SliceRequiresVersion3),
    };
    if frame_dims.width == 0 || frame_dims.height == 0 {
        return Err(Error::InvalidFramePixelDimensions {
            width: frame_dims.width,
            height: frame_dims.height,
        });
    }

    // §5 trigger: `frame_pixel_width * frame_pixel_height > 101376`.
    // Compute in u64 so the multiplication can't overflow at u32
    // dimensions (matches the §6 security-considerations advisory).
    let frame_area: u64 = u64::from(frame_dims.width) * u64::from(frame_dims.height);
    if frame_area <= SECTION_5_MAX_SLICE_AREA_THRESHOLD {
        // Restriction silent — any raster footprint is admissible
        // (including a 1-Slice frame).
        return Ok(());
    }

    // §5 cap: `slice_width * slice_height <= num_h_slices * num_v_slices / 4`.
    // The spec spells this in raster-cell units. The right-hand side
    // is integer division — the floor of (num_h*num_v) / 4. The
    // canonical 2×2 slice raster gives `num_h*num_v/4 = 1`, so each
    // Slice is capped at exactly one cell on every above-threshold
    // 2×2 Frame; the canonical 4×4 raster gives `4`, so each Slice
    // can cover at most four cells; etc. The "fast multithreading"
    // motive in the spec text is the per-thread Slice quota at four
    // workers — every Slice fits in at most one quarter of the raster
    // so four threads can each take one Slice in lockstep.
    let raster_cells: u64 = u64::from(num_h_slices) * u64::from(num_v_slices);
    let slice_cap: u64 = raster_cells / 4;
    let slice_cells: u64 = u64::from(header.slice_width) * u64::from(header.slice_height);
    if slice_cells > slice_cap {
        return Err(Error::SliceMaxSizeExceeded {
            slice_width: header.slice_width,
            slice_height: header.slice_height,
            num_h_slices,
            num_v_slices,
            frame_pixel_width: frame_dims.width,
            frame_pixel_height: frame_dims.height,
        });
    }
    Ok(())
}

/// RFC 9043 §5 "Restrictions" — per-Frame Slice raster coverage validator.
///
/// The §5 second paragraph states:
///
/// > For each Frame, each position in the Slice raster MUST be filled by
/// > one and only one Slice of the Frame (no missing Slice position and
/// > no Slice overlapping).
///
/// This validator takes the set of Slice Headers parsed off a single
/// Frame's `Vec<SliceExtent>` (forward order, slice 0 first) plus the
/// surrounding Configuration Record (for the `num_h_slices ×
/// num_v_slices` grid shape) and proves the union of every Slice's
/// raster footprint exactly tiles the grid: every cell is claimed by
/// **at least one** Slice (no gaps) and every cell is claimed by **at
/// most one** Slice (no overlaps).
///
/// The check is a pure structural primitive — no range coder, no pixel
/// buffer, no frame bytes touched. The §5 cap on per-Slice size is
/// orthogonal: it lives in [`validate_slice_max_size_restriction`].
/// A conforming Frame satisfies both.
///
/// # Algorithm
///
/// Each Slice Header carries `slice_x` / `slice_y` (top-left raster
/// cell, §4.6.1 / §4.6.2) and `slice_width` / `slice_height` (raster
/// cell counts, §4.6.3 / §4.6.4). For each Slice we paint its
/// `slice_width × slice_height` raster cells with the Slice's forward
/// index. The first cell already painted by an earlier Slice surfaces
/// `SliceRasterOverlap` (with both colliding Slice indices). After
/// painting every Slice, the first unpainted cell surfaces
/// `SliceRasterUncovered`. Coverage is therefore strictly equivalent
/// to "every cell painted exactly once".
///
/// The §4.6 per-Slice raster-bounds check (`slice_x + slice_width <=
/// num_h_slices`) is the existing job of
/// [`compute_slice_content`] / [`Error::SliceRasterOutOfRange`]; this
/// validator surfaces the same error variant when an individual Slice
/// addresses cells off the grid so the §5 walk does not need a
/// secondary "this cell index is out of bounds" surface.
///
/// # Returns
///
/// * `Ok(())` — every raster cell of the `num_h_slices × num_v_slices`
///   grid is claimed by exactly one of the supplied Slice Headers.
/// * [`Error::SliceRasterOverlap`] — two distinct Slices both claim the
///   same raster cell. Carries both Slice indices in forward order
///   (`first_slice_index < second_slice_index`) and the offending
///   raster coordinate.
/// * [`Error::SliceRasterUncovered`] — at least one raster cell is
///   not claimed by any Slice. Carries the first uncovered cell in
///   row-major scan order (`y` outer, `x` inner) so a caller can log
///   the canonical missing position.
/// * [`Error::SliceRasterOutOfRange`] — an individual Slice's
///   `(slice_x, slice_y, slice_width, slice_height)` would address
///   cells outside the configured `num_h_slices × num_v_slices` grid
///   (or `num_h_slices` / `num_v_slices` is zero). Mirrors the surface
///   [`compute_slice_content`] uses for the same condition.
/// * [`Error::SliceRequiresVersion3`] — the Configuration Record's
///   `num_h_slices` / `num_v_slices` are absent (FFV1 v0 / v1). The
///   §5 paragraph is normative on every Frame, but the v0 / v1 slice
///   grid lives in the per-keyframe header rather than the
///   Configuration Record so this helper has nothing to validate
///   against. The wider §5 max-size validator surfaces the same
///   error for the same reason.
///
/// # Errors
///
/// See "Returns" above.
pub fn validate_slice_raster_coverage(
    headers: &[Ffv1SliceHeader],
    cr: &Ffv1ConfigurationRecord,
) -> Result<(), Error> {
    let (num_h_slices, num_v_slices) = match (cr.num_h_slices, cr.num_v_slices) {
        (Some(h), Some(v)) => (h, v),
        _ => return Err(Error::SliceRequiresVersion3),
    };
    if num_h_slices == 0 || num_v_slices == 0 {
        // The grid shape is degenerate — no cell can be addressed. Use
        // the same SliceRasterOutOfRange surface compute_slice_content
        // uses for the same condition. With no slices the call is
        // trivially "0 cells painted of 0", but a zero grid is itself
        // malformed; report against the first supplied header (or a
        // synthesised zero header when no slices were passed).
        let probe = headers.first().cloned().unwrap_or(Ffv1SliceHeader {
            slice_x: 0,
            slice_y: 0,
            slice_width: 0,
            slice_height: 0,
            quant_table_set_index_count: 0,
            quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
            picture_structure: crate::config::PictureStructure::Progressive,
            picture_structure_raw: 3,
            sar_num: 0,
            sar_den: 0,
        });
        return Err(Error::SliceRasterOutOfRange {
            slice_x: probe.slice_x,
            slice_y: probe.slice_y,
            slice_width: probe.slice_width,
            slice_height: probe.slice_height,
            num_h_slices,
            num_v_slices,
        });
    }

    // Paint each Slice's raster footprint, recording the forward Slice
    // index that claimed each cell. A second paint of the same cell is
    // an overlap; an unpainted cell at the end of the walk is a gap.
    //
    // The grid count is bounded by num_h_slices * num_v_slices, both
    // u32; do the product in usize on 64-bit hosts (overflow-safe), and
    // bail to SliceRasterOutOfRange when an explicit u32 multiplication
    // overflow occurs (defensive — the §4 parser's existing range
    // checks make this branch unreachable on a conforming
    // Configuration Record).
    let raster_cells_u32 = match num_h_slices.checked_mul(num_v_slices) {
        Some(c) => c,
        None => {
            let probe = headers.first().cloned().unwrap_or(Ffv1SliceHeader {
                slice_x: 0,
                slice_y: 0,
                slice_width: 0,
                slice_height: 0,
                quant_table_set_index_count: 0,
                quant_table_set_index: [0; MAX_QUANT_TABLE_SET_INDEXES],
                picture_structure: crate::config::PictureStructure::Progressive,
                picture_structure_raw: 3,
                sar_num: 0,
                sar_den: 0,
            });
            return Err(Error::SliceRasterOutOfRange {
                slice_x: probe.slice_x,
                slice_y: probe.slice_y,
                slice_width: probe.slice_width,
                slice_height: probe.slice_height,
                num_h_slices,
                num_v_slices,
            });
        }
    };
    let mut painted: Vec<Option<u32>> = vec![None; raster_cells_u32 as usize];

    for (slice_index, header) in headers.iter().enumerate() {
        // Per-Slice bounds check. Mirrors compute_slice_content so the
        // §5 walk and the §4.7 layout pass surface identically on a
        // malformed Slice.
        let x_oob = header
            .slice_x
            .checked_add(header.slice_width)
            .map_or(true, |sum| sum > num_h_slices);
        let y_oob = header
            .slice_y
            .checked_add(header.slice_height)
            .map_or(true, |sum| sum > num_v_slices);
        if x_oob || y_oob {
            return Err(Error::SliceRasterOutOfRange {
                slice_x: header.slice_x,
                slice_y: header.slice_y,
                slice_width: header.slice_width,
                slice_height: header.slice_height,
                num_h_slices,
                num_v_slices,
            });
        }

        // Paint every cell the Slice claims; collide on the first
        // overlap so the error carries the lowest forward-index Slice
        // pair that conflicts.
        for cy in header.slice_y..(header.slice_y + header.slice_height) {
            for cx in header.slice_x..(header.slice_x + header.slice_width) {
                let idx = (cy as usize) * (num_h_slices as usize) + (cx as usize);
                if let Some(prior) = painted[idx] {
                    return Err(Error::SliceRasterOverlap {
                        x: cx,
                        y: cy,
                        first_slice_index: prior,
                        second_slice_index: slice_index as u32,
                    });
                }
                painted[idx] = Some(slice_index as u32);
            }
        }
    }

    // Find the first unpainted cell in row-major scan order so the
    // gap diagnostic is deterministic.
    for cy in 0..num_v_slices {
        for cx in 0..num_h_slices {
            let idx = (cy as usize) * (num_h_slices as usize) + (cx as usize);
            if painted[idx].is_none() {
                return Err(Error::SliceRasterUncovered { x: cx, y: cy });
            }
        }
    }

    Ok(())
}

/// The §5 third-paragraph geometry quadruple — `(slice_x, slice_y,
/// slice_width, slice_height)` per RFC 9043 §4.6.1-§4.6.4. Private:
/// the stability rule names exactly these four Slice Header fields
/// and no others.
type SliceGeometry = (u32, u32, u32, u32);

fn slice_geometry(header: &Ffv1SliceHeader) -> SliceGeometry {
    (
        header.slice_x,
        header.slice_y,
        header.slice_width,
        header.slice_height,
    )
}

/// Shared matcher for [`validate_slice_geometry_stability`] and
/// [`SliceGeometryStabilityTracker::observe_frame`]: every current
/// Slice's §4.6.1-§4.6.4 geometry quadruple must appear among the
/// previous Frame's quadruples ("as a Slice in the previous Frame" —
/// an existence check, not a same-forward-index check; §5 imposes no
/// ordering on the match).
fn validate_geometry_against(
    previous: &[SliceGeometry],
    current_headers: &[Ffv1SliceHeader],
) -> Result<(), Error> {
    for (slice_index, header) in current_headers.iter().enumerate() {
        let geometry = slice_geometry(header);
        if !previous.contains(&geometry) {
            return Err(Error::SliceGeometryUnstable {
                slice_index: slice_index as u32,
                slice_x: geometry.0,
                slice_y: geometry.1,
                slice_width: geometry.2,
                slice_height: geometry.3,
            });
        }
    }
    Ok(())
}

/// RFC 9043 §5 "Restrictions" — non-keyframe Slice-geometry stability
/// validator.
///
/// The §5 third paragraph states:
///
/// > For each Frame with a keyframe value of 0, each Slice MUST have
/// > the same value of slice_x, slice_y, slice_width, and
/// > slice_height as a Slice in the previous Frame.
///
/// This is the only §5 restriction that spans more than one Frame, so
/// — unlike [`validate_slice_raster_coverage`] and
/// [`validate_slice_max_size_restriction`], which a single-Frame
/// decode driver can enforce inline — its inputs are the Slice
/// Headers of **two consecutive Frames**. The caller applies it only
/// when the current Frame's §4.4 `keyframe` value is 0 (on a keyframe
/// the rule does not bind; §3.8.1.3 / §3.8.2.5 re-initialise all
/// coder state instead). For a sequenced multi-Frame walk, prefer
/// [`SliceGeometryStabilityTracker`], which folds the keyframe gating
/// and the previous-Frame bookkeeping into one `observe_frame` call
/// per Frame.
///
/// # Semantics
///
/// For each current-Frame Slice in forward (trailer-chain) order, the
/// §4.6.1-§4.6.4 quadruple `(slice_x, slice_y, slice_width,
/// slice_height)` must equal the quadruple of **some** Slice of the
/// previous Frame — §5 says "as a Slice in the previous Frame", an
/// existence requirement, so a permuted Slice order across Frames is
/// conforming. No other Slice Header field participates: §5 names
/// exactly the four geometry fields, so `quant_table_set_index`,
/// `picture_structure`, and SAR may all change Frame-to-Frame.
/// Combined with the §5 second-paragraph partition rule (both Frames
/// tile the same `num_h_slices × num_v_slices` raster exactly), the
/// existence check makes the two Frames' geometry sets equal; this
/// validator checks the literal third-paragraph direction only and
/// leaves the partition rule to [`validate_slice_raster_coverage`].
///
/// # Returns
///
/// * `Ok(())` — every current Slice's geometry quadruple appears in
///   the previous Frame (vacuously true for an empty current Frame).
/// * [`Error::SliceGeometryUnstable`] — carries the forward index and
///   the geometry quadruple of the **first** unmatched current Slice
///   so the diagnostic is deterministic.
///
/// # Errors
///
/// See "Returns" above.
pub fn validate_slice_geometry_stability(
    previous_headers: &[Ffv1SliceHeader],
    current_headers: &[Ffv1SliceHeader],
) -> Result<(), Error> {
    let previous: Vec<SliceGeometry> = previous_headers.iter().map(slice_geometry).collect();
    validate_geometry_against(&previous, current_headers)
}

/// Stateful multi-Frame driver for the RFC 9043 §5 third-paragraph
/// non-keyframe Slice-geometry stability rule.
///
/// The frame-level decode drivers ([`crate::decode_frame`] /
/// [`crate::decode_frame_rgb`]) are single-Frame and stateless, so
/// the cross-Frame §5 rule lives with the caller that walks Frames in
/// coded order. Feed each Frame's §4.4 `keyframe` value plus its
/// forward-ordered Slice Headers to [`Self::observe_frame`]:
///
/// * `keyframe == true` — the rule does not bind (§5 restricts only
///   Frames "with a keyframe value of 0"); the Frame's geometry is
///   recorded as the new previous-Frame reference and `Ok(())` is
///   returned.
/// * `keyframe == false` — the Frame's Slices are validated against
///   the **immediately preceding** Frame's geometry via the same
///   matcher as [`validate_slice_geometry_stability`]. On success the
///   Frame becomes the new previous-Frame reference. A non-keyframe
///   observed before any Frame at all is validated against the empty
///   set — there is no "previous Frame" whose Slices could match, so
///   any Slice surfaces [`Error::SliceGeometryUnstable`] (a stream
///   cannot meaningfully open on a non-keyframe anyway: §3.8.1.3 /
///   §3.8.2.5 only initialise the coder state when `keyframe` is 1).
///
/// On `Err` the tracker's previous-Frame reference is left untouched:
/// the violating Frame is non-conforming and does not become the
/// reference for its successor.
#[derive(Debug, Clone, Default)]
pub struct SliceGeometryStabilityTracker {
    /// Geometry quadruples of the most recent conforming Frame, or
    /// `None` before the first observed Frame.
    previous: Option<Vec<SliceGeometry>>,
}

impl SliceGeometryStabilityTracker {
    /// Create a tracker with no previous Frame observed.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` once at least one conforming Frame has been observed.
    pub fn has_previous_frame(&self) -> bool {
        self.previous.is_some()
    }

    /// Observe one Frame in coded order. See the type-level doc for
    /// the keyframe / non-keyframe semantics.
    ///
    /// # Errors
    ///
    /// [`Error::SliceGeometryUnstable`] when `keyframe == false` and
    /// a Slice's §4.6.1-§4.6.4 geometry quadruple matches no Slice of
    /// the previous Frame (RFC 9043 §5 third paragraph).
    pub fn observe_frame(
        &mut self,
        keyframe: bool,
        headers: &[Ffv1SliceHeader],
    ) -> Result<(), Error> {
        if !keyframe {
            let empty: [SliceGeometry; 0] = [];
            let previous: &[SliceGeometry] = self.previous.as_deref().unwrap_or(&empty);
            validate_geometry_against(previous, headers)?;
        }
        self.previous = Some(headers.iter().map(slice_geometry).collect());
        Ok(())
    }
}

impl SliceContent {
    /// Number of planes (`primary_color_count` per RFC 9043 §4.7.1).
    pub fn primary_color_count(&self) -> usize {
        self.planes.len()
    }

    /// Total `Line(p, y)` calls the §4.7 pseudocode makes for this
    /// slice — sum of `plane_pixel_height[p]` across all planes.
    pub fn line_count(&self) -> usize {
        self.planes.iter().map(|p| p.lines.len()).sum()
    }

    /// Walk the slice's `Line(p, y)` calls in the order RFC 9043 §4.7
    /// mandates, returning each `(plane_index, y)` pair.
    ///
    /// For `PlaneMajor` (YCbCr) the iteration is plane-then-row:
    /// `(0,0)..(0,h_0-1), (1,0)..(1,h_1-1), …`. Note that planes
    /// often have different heights when chroma is subsampled, so
    /// the outer plane index advances after exhausting each plane's
    /// own row count.
    ///
    /// For `LineMajor` (RGB) the iteration is row-then-plane:
    /// `(0,0), (1,0), …, (P-1,0), (0,1), (1,1), …`. The RFC's
    /// pseudocode uses `slice_pixel_height` as the outer bound for
    /// the LineMajor path, which is well-defined precisely because
    /// the RGB colorspace forbids chroma subsampling
    /// (`colorspace_type == 1 && (log2_*_chroma_subsample != 0)` is
    /// outside the spec per §4.2.5) — so every plane shares the
    /// same `plane_pixel_height` and the inner loop is safe.
    pub fn line_visits(&self) -> Vec<LineVisit> {
        let mut out = Vec::with_capacity(self.line_count());
        match self.traversal {
            PlaneTraversal::PlaneMajor => {
                for plane in &self.planes {
                    for line in &plane.lines {
                        out.push(LineVisit {
                            plane_index: plane.plane_index,
                            y: line.y,
                        });
                    }
                }
            }
            PlaneTraversal::LineMajor => {
                // The RGB path is constrained by §4.2.5 to have
                // chroma_planes==1 and zero chroma subsample, so
                // every plane has the same `plane_pixel_height`. Use
                // the first plane's height as the outer bound; if
                // somehow a malformed input slipped through (no
                // primary planes) the loop is a no-op.
                let outer_h = self.planes.first().map(|p| p.lines.len()).unwrap_or(0);
                for y in 0..outer_h {
                    for plane in &self.planes {
                        out.push(LineVisit {
                            plane_index: plane.plane_index,
                            // y is bounded by `outer_h`, the shared
                            // plane height. `plane.lines[y]` exists
                            // for every plane on the RGB path; we
                            // index defensively in case a future
                            // round introduces a malformed plane.
                            y: plane.lines.get(y).map(|l| l.y).unwrap_or(y as u32),
                        });
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ColorspaceType, Ffv1ConfigurationRecord, Ffv1Version};

    fn cr(
        chroma_planes: bool,
        log2_hs: u32,
        log2_vs: u32,
        extra_plane: bool,
        colorspace: ColorspaceType,
    ) -> Ffv1ConfigurationRecord {
        Ffv1ConfigurationRecord {
            version: Ffv1Version::V3,
            micro_version: Some(4),
            coder_type: 1,
            state_transition_delta: [0; crate::config::NUM_TRANSITION_DELTAS],
            colorspace_type: colorspace,
            bits_per_raw_sample: 8,
            chroma_planes,
            log2_h_chroma_subsample: log2_hs,
            log2_v_chroma_subsample: log2_vs,
            extra_plane,
            num_h_slices: Some(2),
            num_v_slices: Some(2),
            quant_table_set_count: Some(2),
            ec: Some(0),
            intra: Some(false),
            initial_state_delta: None,
        }
    }

    fn header(slice_x: u32, slice_y: u32, w: u32, h: u32) -> Ffv1SliceHeader {
        Ffv1SliceHeader {
            slice_x,
            slice_y,
            slice_width: w,
            slice_height: h,
            quant_table_set_index_count: 2,
            quant_table_set_index: [0, 0, 0],
            picture_structure: crate::config::PictureStructure::Progressive,
            picture_structure_raw: 3,
            sar_num: 0,
            sar_den: 0,
        }
    }

    #[test]
    fn primary_color_count_table() {
        // Y only.
        assert_eq!(
            primary_color_count(&cr(false, 0, 0, false, ColorspaceType::YCbCr)),
            1
        );
        // Y + A.
        assert_eq!(
            primary_color_count(&cr(false, 0, 0, true, ColorspaceType::YCbCr)),
            2
        );
        // YUV.
        assert_eq!(
            primary_color_count(&cr(true, 1, 1, false, ColorspaceType::YCbCr)),
            3
        );
        // YUVA.
        assert_eq!(
            primary_color_count(&cr(true, 1, 1, true, ColorspaceType::YCbCr)),
            4
        );
        // RGB (chroma_planes=1, no subsample).
        assert_eq!(
            primary_color_count(&cr(true, 0, 0, false, ColorspaceType::Rgb)),
            3
        );
        // RGBA.
        assert_eq!(
            primary_color_count(&cr(true, 0, 0, true, ColorspaceType::Rgb)),
            4
        );
    }

    #[test]
    fn plane_pixel_dims_no_subsample() {
        let c = cr(true, 0, 0, false, ColorspaceType::Rgb);
        assert_eq!(plane_pixel_width(32, 0, &c), 32);
        assert_eq!(plane_pixel_width(32, 1, &c), 32);
        assert_eq!(plane_pixel_width(32, 2, &c), 32);
        assert_eq!(plane_pixel_height(24, 0, &c), 24);
        assert_eq!(plane_pixel_height(24, 1, &c), 24);
        assert_eq!(plane_pixel_height(24, 2, &c), 24);
    }

    #[test]
    fn plane_pixel_dims_yuv420() {
        let c = cr(true, 1, 1, false, ColorspaceType::YCbCr);
        assert_eq!(plane_pixel_width(64, 0, &c), 64);
        assert_eq!(plane_pixel_width(64, 1, &c), 32);
        assert_eq!(plane_pixel_width(64, 2, &c), 32);
        assert_eq!(plane_pixel_height(48, 0, &c), 48);
        assert_eq!(plane_pixel_height(48, 1, &c), 24);
        assert_eq!(plane_pixel_height(48, 2, &c), 24);
    }

    #[test]
    fn plane_pixel_dims_yuv420_odd_widths_round_up() {
        // 63 with 4:2:0 → ceil(63/2) = 32 for chroma planes (not 31).
        let c = cr(true, 1, 1, false, ColorspaceType::YCbCr);
        assert_eq!(plane_pixel_width(63, 0, &c), 63);
        assert_eq!(plane_pixel_width(63, 1, &c), 32);
        assert_eq!(plane_pixel_height(47, 2, &c), 24);
    }

    #[test]
    fn slice_pixel_rect_v3_default_2x2_128x96() {
        // v3-default fixture: 128×96 frame, 2×2 raster, every cell
        // becomes 64×48 pixels per the trace's PLANE w=64 h=48. We
        // don't need the Configuration Record here; only the
        // num_h/v_slices and frame dimensions feed slice_pixel_rect.
        let frame = FramePixelDimensions::new(128, 96).unwrap();
        // Cell (0,0).
        let h = header(0, 0, 1, 1);
        let (x, y, w, ph) = slice_pixel_rect(&h, 2, 2, frame);
        assert_eq!((x, y, w, ph), (0, 0, 64, 48));
        // Cell (1,0).
        let h = header(1, 0, 1, 1);
        let (x, y, w, ph) = slice_pixel_rect(&h, 2, 2, frame);
        assert_eq!((x, y, w, ph), (64, 0, 64, 48));
        // Cell (0,1).
        let h = header(0, 1, 1, 1);
        let (x, y, w, ph) = slice_pixel_rect(&h, 2, 2, frame);
        assert_eq!((x, y, w, ph), (0, 48, 64, 48));
        // Cell (1,1).
        let h = header(1, 1, 1, 1);
        let (x, y, w, ph) = slice_pixel_rect(&h, 2, 2, frame);
        assert_eq!((x, y, w, ph), (64, 48, 64, 48));
    }

    #[test]
    fn traversal_for_colorspace() {
        assert_eq!(
            PlaneTraversal::for_colorspace(ColorspaceType::YCbCr),
            PlaneTraversal::PlaneMajor
        );
        assert_eq!(
            PlaneTraversal::for_colorspace(ColorspaceType::Rgb),
            PlaneTraversal::LineMajor
        );
    }

    #[test]
    fn rejects_zero_frame_dimensions() {
        assert!(matches!(
            FramePixelDimensions::new(0, 100),
            Err(Error::InvalidFramePixelDimensions {
                width: 0,
                height: 100
            })
        ));
        assert!(matches!(
            FramePixelDimensions::new(100, 0),
            Err(Error::InvalidFramePixelDimensions {
                width: 100,
                height: 0
            })
        ));
    }

    #[test]
    fn rejects_slice_off_raster() {
        let c = cr(true, 1, 1, false, ColorspaceType::YCbCr);
        let frame = FramePixelDimensions::new(128, 96).unwrap();
        // 2×2 raster but slice_x + slice_width = 3 → off the grid.
        let h = header(2, 0, 1, 1);
        assert!(matches!(
            compute_slice_content(&h, &c, frame),
            Err(Error::SliceRasterOutOfRange { .. })
        ));
        // slice_x + slice_width = 2 + 1 = 3 again, slightly different.
        let h = header(0, 2, 1, 1);
        assert!(matches!(
            compute_slice_content(&h, &c, frame),
            Err(Error::SliceRasterOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_v0_v1_config_record() {
        let mut c = cr(true, 1, 1, false, ColorspaceType::YCbCr);
        c.num_h_slices = None;
        c.num_v_slices = None;
        let frame = FramePixelDimensions::new(128, 96).unwrap();
        let h = header(0, 0, 1, 1);
        assert!(matches!(
            compute_slice_content(&h, &c, frame),
            Err(Error::SliceRequiresVersion3)
        ));
    }

    #[test]
    fn yuv_slice_content_plane_shape() {
        let c = cr(true, 1, 1, false, ColorspaceType::YCbCr);
        let frame = FramePixelDimensions::new(128, 96).unwrap();
        let h = header(0, 0, 1, 1);
        let sc = compute_slice_content(&h, &c, frame).unwrap();
        assert_eq!(sc.primary_color_count(), 3);
        // Plane 0 = Y, 64×48.
        assert_eq!(sc.planes[0].plane_index, 0);
        assert_eq!(sc.planes[0].width, 64);
        assert_eq!(sc.planes[0].height, 48);
        assert_eq!(sc.planes[0].lines.len(), 48);
        // Plane 1 = U, 32×24.
        assert_eq!(sc.planes[1].width, 32);
        assert_eq!(sc.planes[1].height, 24);
        assert_eq!(sc.planes[1].lines.len(), 24);
        // Plane 2 = V, 32×24.
        assert_eq!(sc.planes[2].width, 32);
        assert_eq!(sc.planes[2].height, 24);
        // YCbCr is plane-major.
        assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);
        // 48 + 24 + 24 = 96 line visits per RFC §4.7 pseudocode.
        assert_eq!(sc.line_count(), 96);
        let visits = sc.line_visits();
        assert_eq!(visits.len(), 96);
        // First visit is (Y, y=0), last is (V, y=23).
        assert_eq!(
            visits[0],
            LineVisit {
                plane_index: 0,
                y: 0
            }
        );
        assert_eq!(
            visits[visits.len() - 1],
            LineVisit {
                plane_index: 2,
                y: 23
            }
        );
        // After the Y plane finishes (48 rows) the next visit is
        // (U, y=0).
        assert_eq!(
            visits[48],
            LineVisit {
                plane_index: 1,
                y: 0
            }
        );
    }

    #[test]
    fn rgb_slice_content_line_major_interleave() {
        let c = cr(true, 0, 0, false, ColorspaceType::Rgb);
        let frame = FramePixelDimensions::new(64, 48).unwrap();
        let h = header(0, 0, 1, 1);
        let sc = compute_slice_content(&h, &c, frame).unwrap();
        assert_eq!(sc.primary_color_count(), 3);
        // 64 / 2 = 32, 48 / 2 = 24.
        assert_eq!(sc.slice_pixel_width, 32);
        assert_eq!(sc.slice_pixel_height, 24);
        for plane in &sc.planes {
            assert_eq!(plane.width, 32);
            assert_eq!(plane.height, 24);
        }
        assert_eq!(sc.traversal, PlaneTraversal::LineMajor);
        let visits = sc.line_visits();
        // 24 rows × 3 planes = 72 line visits.
        assert_eq!(visits.len(), 72);
        // First three visits are (0,0)(1,0)(2,0).
        assert_eq!(
            visits[0],
            LineVisit {
                plane_index: 0,
                y: 0
            }
        );
        assert_eq!(
            visits[1],
            LineVisit {
                plane_index: 1,
                y: 0
            }
        );
        assert_eq!(
            visits[2],
            LineVisit {
                plane_index: 2,
                y: 0
            }
        );
        // Then row y=1: (0,1)(1,1)(2,1).
        assert_eq!(
            visits[3],
            LineVisit {
                plane_index: 0,
                y: 1
            }
        );
        assert_eq!(
            visits[5],
            LineVisit {
                plane_index: 2,
                y: 1
            }
        );
    }

    #[test]
    fn grayscale_slice_content_single_plane() {
        let c = cr(false, 0, 0, false, ColorspaceType::YCbCr);
        let frame = FramePixelDimensions::new(64, 48).unwrap();
        let h = header(0, 0, 1, 1);
        let sc = compute_slice_content(&h, &c, frame).unwrap();
        assert_eq!(sc.primary_color_count(), 1);
        assert_eq!(sc.planes[0].width, 32);
        assert_eq!(sc.planes[0].height, 24);
        // Grayscale also reports PlaneMajor (YCbCr-class).
        assert_eq!(sc.traversal, PlaneTraversal::PlaneMajor);
        assert_eq!(sc.line_count(), 24);
    }

    #[test]
    fn yuva_extra_plane_added() {
        let c = cr(true, 1, 1, true, ColorspaceType::YCbCr);
        let frame = FramePixelDimensions::new(128, 96).unwrap();
        let h = header(0, 0, 1, 1);
        let sc = compute_slice_content(&h, &c, frame).unwrap();
        // primary_color_count = 1 + 2 + 1 = 4.
        assert_eq!(sc.primary_color_count(), 4);
        // The extra plane (alpha, index 3) runs at FULL slice size,
        // not chroma-subsampled.
        assert_eq!(sc.planes[3].plane_index, 3);
        assert_eq!(sc.planes[3].width, 64);
        assert_eq!(sc.planes[3].height, 48);
    }

    #[test]
    fn line_records_carry_their_identity() {
        let c = cr(true, 1, 1, false, ColorspaceType::YCbCr);
        let frame = FramePixelDimensions::new(128, 96).unwrap();
        let h = header(1, 1, 1, 1);
        let sc = compute_slice_content(&h, &c, frame).unwrap();
        // Each plane's lines should have plane_index==plane.plane_index
        // and y == its row index.
        for plane in &sc.planes {
            for (y, line) in plane.lines.iter().enumerate() {
                assert_eq!(line.plane_index, plane.plane_index);
                assert_eq!(line.y as usize, y);
            }
        }
        // And the bottom-right slice cell anchors at (64, 48).
        assert_eq!(sc.slice_pixel_x, 64);
        assert_eq!(sc.slice_pixel_y, 48);
    }

    // ----- RFC 9043 §5 "Restrictions" — per-Slice max-size gate ----

    /// Helper: build a v3 config record with caller-controlled
    /// `num_h_slices` / `num_v_slices`.
    fn cr_v3_grid(num_h_slices: u32, num_v_slices: u32) -> Ffv1ConfigurationRecord {
        let mut c = cr(false, 0, 0, false, ColorspaceType::YCbCr);
        c.num_h_slices = Some(num_h_slices);
        c.num_v_slices = Some(num_v_slices);
        c
    }

    #[test]
    fn section_5_threshold_matches_cif() {
        // 352 × 288 = 101376 = SECTION_5_MAX_SLICE_AREA_THRESHOLD.
        // The §5 text says "more than 101376", so a Frame *at* the
        // threshold falls in the unrestricted regime.
        assert_eq!(SECTION_5_MAX_SLICE_AREA_THRESHOLD, 352 * 288);
        assert_eq!(SECTION_5_MAX_SLICE_AREA_THRESHOLD, 101_376);
    }

    #[test]
    fn section_5_below_threshold_admits_any_footprint() {
        // 128 × 96 = 12288, well below 101376 — §5 is silent. A 1×1
        // Slice covering the entire 1×1 raster (the maximum possible
        // footprint at this grid) must validate clean.
        let c = cr_v3_grid(1, 1);
        let frame = FramePixelDimensions::new(128, 96).unwrap();
        let h = header(0, 0, 1, 1);
        assert!(validate_slice_max_size_restriction(&h, &c, frame).is_ok());

        // And on a 2×2 raster the full-raster (2×2) Slice still
        // validates clean below the threshold — the §5 cap of
        // 2*2/4 = 1 would reject `slice_w*slice_h = 4` above the
        // threshold, but not below.
        let c2 = cr_v3_grid(2, 2);
        let h2 = header(0, 0, 2, 2);
        assert!(validate_slice_max_size_restriction(&h2, &c2, frame).is_ok());
    }

    #[test]
    fn section_5_at_threshold_admits_any_footprint() {
        // 352 × 288 = 101376 — exactly at the threshold. §5 reads
        // "more than 101376", so the inequality is strict; at the
        // threshold any Slice footprint passes.
        let c = cr_v3_grid(2, 2);
        let frame = FramePixelDimensions::new(352, 288).unwrap();
        // 2*2 raster cells with a single Slice covering all four —
        // the maximum possible footprint. Cap would be 1 above the
        // threshold; at the threshold it's silent.
        let h = header(0, 0, 2, 2);
        assert!(validate_slice_max_size_restriction(&h, &c, frame).is_ok());
    }

    #[test]
    fn section_5_above_threshold_caps_at_quarter_raster_2x2() {
        // 353 × 288 = 101664 > 101376. On a 2×2 raster the §5 cap
        // is `num_h * num_v / 4 = 1`. A 1×1 Slice satisfies it (==1);
        // a 1×2 Slice (covering 2 cells) violates it (2>1); a 2×2
        // Slice (covering 4 cells) violates more strongly.
        let c = cr_v3_grid(2, 2);
        let frame = FramePixelDimensions::new(353, 288).unwrap();

        // 1×1: passes — slice_w*slice_h = 1 == cap = 1.
        let h_ok = header(0, 0, 1, 1);
        assert!(validate_slice_max_size_restriction(&h_ok, &c, frame).is_ok());

        // 1×2: fails.
        let h_v = header(0, 0, 1, 2);
        match validate_slice_max_size_restriction(&h_v, &c, frame) {
            Err(Error::SliceMaxSizeExceeded {
                slice_width,
                slice_height,
                num_h_slices,
                num_v_slices,
                frame_pixel_width,
                frame_pixel_height,
            }) => {
                assert_eq!(slice_width, 1);
                assert_eq!(slice_height, 2);
                assert_eq!(num_h_slices, 2);
                assert_eq!(num_v_slices, 2);
                assert_eq!(frame_pixel_width, 353);
                assert_eq!(frame_pixel_height, 288);
            }
            other => panic!("expected SliceMaxSizeExceeded, got {other:?}"),
        }

        // 2×2 (whole raster): fails.
        let h_full = header(0, 0, 2, 2);
        assert!(matches!(
            validate_slice_max_size_restriction(&h_full, &c, frame),
            Err(Error::SliceMaxSizeExceeded { .. })
        ));
    }

    #[test]
    fn section_5_above_threshold_4x4_cap_is_four() {
        // 4×4 raster → cap = 16/4 = 4 cells per Slice. So a 2×2 Slice
        // (4 cells) is admissible; a 1×5 Slice would exceed 4 but
        // also exceed the raster — that's a §4.6 raster-bounds error
        // rather than §5. The realistic §5 violation at a 4×4 raster
        // is a Slice that covers more than four cells on the grid,
        // e.g. 3×2 = 6 or 4×4 = 16.
        let c = cr_v3_grid(4, 4);
        let frame = FramePixelDimensions::new(400, 300).unwrap(); // > 101376

        // 2×2 = 4 cells: passes (== cap).
        let h_eq = header(0, 0, 2, 2);
        assert!(validate_slice_max_size_restriction(&h_eq, &c, frame).is_ok());

        // 3×2 = 6 cells: fails (> 4).
        let h_over = header(0, 0, 3, 2);
        assert!(matches!(
            validate_slice_max_size_restriction(&h_over, &c, frame),
            Err(Error::SliceMaxSizeExceeded { .. })
        ));

        // 4×4 = 16 cells (whole raster, single-Slice frame): fails.
        let h_full = header(0, 0, 4, 4);
        assert!(matches!(
            validate_slice_max_size_restriction(&h_full, &c, frame),
            Err(Error::SliceMaxSizeExceeded { .. })
        ));
    }

    #[test]
    fn section_5_integer_division_uses_floor() {
        // The §5 cap is integer division. With a 3×3 raster, the
        // expected cap is `9 / 4 = 2` (floor). A 1×1 Slice (1 cell)
        // passes; a 1×2 Slice (2 cells) is at the cap and passes;
        // a 3×1 Slice (3 cells) fails.
        let c = cr_v3_grid(3, 3);
        let frame = FramePixelDimensions::new(400, 300).unwrap(); // > 101376
        let h_one = header(0, 0, 1, 1);
        let h_two = header(0, 0, 1, 2);
        let h_three = header(0, 0, 3, 1);
        assert!(validate_slice_max_size_restriction(&h_one, &c, frame).is_ok());
        assert!(validate_slice_max_size_restriction(&h_two, &c, frame).is_ok());
        assert!(matches!(
            validate_slice_max_size_restriction(&h_three, &c, frame),
            Err(Error::SliceMaxSizeExceeded { .. })
        ));
    }

    #[test]
    fn section_5_rejects_v0_v1_with_slice_requires_version3() {
        // The Configuration Record's `num_h_slices` / `num_v_slices`
        // are None for v0 / v1; the §5 inequality has no grid shape
        // to evaluate against. Surface as
        // `SliceRequiresVersion3` (the same error every other §4 / §5
        // grid-dependent helper uses for the v0 / v1 branch).
        let mut c = cr(false, 0, 0, false, ColorspaceType::YCbCr);
        c.num_h_slices = None;
        c.num_v_slices = None;
        let frame = FramePixelDimensions::new(400, 300).unwrap();
        let h = header(0, 0, 1, 1);
        assert!(matches!(
            validate_slice_max_size_restriction(&h, &c, frame),
            Err(Error::SliceRequiresVersion3)
        ));
    }

    #[test]
    fn section_5_rejects_zero_frame_dimensions() {
        // The §5 trigger inequality `frame_area > 101376` is ill-
        // defined on a zero-area Frame; reject for the same reason
        // `compute_slice_content` rejects zero-area Frames. The
        // `FramePixelDimensions::new` constructor catches `(0, ..)` /
        // `(.., 0)` up front, but the struct's fields are `pub` so a
        // caller (e.g. a future deserialiser that initialises the
        // record from raw container bytes) could still hand the
        // validator a zero-area dimensions struct. We exercise the
        // validator's defensive branch by building one directly.
        let c = cr_v3_grid(2, 2);
        let h = header(0, 0, 1, 1);

        let zero_w = FramePixelDimensions {
            width: 0,
            height: 100,
        };
        assert!(matches!(
            validate_slice_max_size_restriction(&h, &c, zero_w),
            Err(Error::InvalidFramePixelDimensions {
                width: 0,
                height: 100
            })
        ));

        let zero_h = FramePixelDimensions {
            width: 100,
            height: 0,
        };
        assert!(matches!(
            validate_slice_max_size_restriction(&h, &c, zero_h),
            Err(Error::InvalidFramePixelDimensions {
                width: 100,
                height: 0
            })
        ));
    }

    // ---- RFC 9043 §5 second paragraph — raster-coverage validator ----

    #[test]
    fn raster_coverage_single_slice_full_1x1_grid() {
        // 1×1 raster: a single Slice claiming the lone cell is the
        // canonical conformant Frame. Validates clean.
        let c = cr_v3_grid(1, 1);
        let headers = vec![header(0, 0, 1, 1)];
        assert!(validate_slice_raster_coverage(&headers, &c).is_ok());
    }

    #[test]
    fn raster_coverage_2x2_grid_four_unit_slices() {
        // Canonical 2×2 raster with four 1×1 Slices: bit-exact tiling
        // of the v3-default 128×96 fixture's layout. Validates clean.
        let c = cr_v3_grid(2, 2);
        let headers = vec![
            header(0, 0, 1, 1),
            header(1, 0, 1, 1),
            header(0, 1, 1, 1),
            header(1, 1, 1, 1),
        ];
        assert!(validate_slice_raster_coverage(&headers, &c).is_ok());
    }

    #[test]
    fn raster_coverage_4x4_grid_four_2x2_slices() {
        // 4×4 raster split into four 2×2 Slices is the canonical
        // §5-admissible above-threshold tiling (each Slice ≤
        // raster/4). Coverage is exact; validates clean.
        let c = cr_v3_grid(4, 4);
        let headers = vec![
            header(0, 0, 2, 2),
            header(2, 0, 2, 2),
            header(0, 2, 2, 2),
            header(2, 2, 2, 2),
        ];
        assert!(validate_slice_raster_coverage(&headers, &c).is_ok());
    }

    #[test]
    fn raster_coverage_rejects_gap_first_cell() {
        // 2×2 raster but only three Slices supplied (missing the
        // bottom-right cell). The validator surfaces SliceRasterUncovered
        // pointing at the first uncovered cell in row-major order
        // (here (1,1) since rows 0..1 are fully painted).
        let c = cr_v3_grid(2, 2);
        let headers = vec![
            header(0, 0, 1, 1),
            header(1, 0, 1, 1),
            header(0, 1, 1, 1),
            // missing: header(1, 1, 1, 1)
        ];
        match validate_slice_raster_coverage(&headers, &c) {
            Err(Error::SliceRasterUncovered { x: 1, y: 1 }) => {}
            other => panic!("expected SliceRasterUncovered((1,1)), got {other:?}"),
        }
    }

    #[test]
    fn raster_coverage_rejects_gap_no_slices() {
        // No Slices at all on a 2×2 raster: the first uncovered cell
        // in row-major order is (0,0).
        let c = cr_v3_grid(2, 2);
        let headers: Vec<Ffv1SliceHeader> = Vec::new();
        match validate_slice_raster_coverage(&headers, &c) {
            Err(Error::SliceRasterUncovered { x: 0, y: 0 }) => {}
            other => panic!("expected SliceRasterUncovered((0,0)), got {other:?}"),
        }
    }

    #[test]
    fn raster_coverage_rejects_overlap_full_overlap() {
        // Two Slices claim exactly the same single cell. The validator
        // surfaces SliceRasterOverlap with the first-painted index
        // (0) and the colliding index (1).
        let c = cr_v3_grid(2, 2);
        let headers = vec![
            header(0, 0, 1, 1),
            header(0, 0, 1, 1), // collides with slice 0 at (0,0)
            header(1, 0, 1, 1),
            header(0, 1, 1, 2),
        ];
        match validate_slice_raster_coverage(&headers, &c) {
            Err(Error::SliceRasterOverlap {
                x: 0,
                y: 0,
                first_slice_index: 0,
                second_slice_index: 1,
            }) => {}
            other => panic!("expected SliceRasterOverlap at (0,0) between 0/1, got {other:?}"),
        }
    }

    #[test]
    fn raster_coverage_rejects_overlap_partial_overlap() {
        // A 2×1 Slice (cells (0,0) and (1,0)) followed by a 1×1 Slice
        // at (1,0) collides on the (1,0) cell. The validator detects
        // the overlap on the second paint and records the indices in
        // forward order.
        let c = cr_v3_grid(2, 2);
        let headers = vec![
            header(0, 0, 2, 1), // cells (0,0) and (1,0)
            header(1, 0, 1, 1), // collides with slice 0 at (1,0)
            header(0, 1, 2, 1),
        ];
        match validate_slice_raster_coverage(&headers, &c) {
            Err(Error::SliceRasterOverlap {
                x: 1,
                y: 0,
                first_slice_index: 0,
                second_slice_index: 1,
            }) => {}
            other => panic!("expected SliceRasterOverlap at (1,0) between 0/1, got {other:?}"),
        }
    }

    #[test]
    fn raster_coverage_rejects_overlap_before_uncovered_check() {
        // An overlap and an uncovered cell co-exist. The validator
        // surfaces the overlap first because the per-Slice paint loop
        // runs before the row-major scan. This pins ordering so the
        // diagnostic chain is deterministic.
        let c = cr_v3_grid(2, 2);
        let headers = vec![
            header(0, 0, 1, 1),
            header(0, 0, 1, 1), // overlap at (0,0)
                                // (1,0), (0,1), (1,1) are gaps
        ];
        assert!(matches!(
            validate_slice_raster_coverage(&headers, &c),
            Err(Error::SliceRasterOverlap { .. })
        ));
    }

    #[test]
    fn raster_coverage_rejects_off_raster_slice_with_out_of_range() {
        // A Slice whose footprint exits the grid surfaces
        // SliceRasterOutOfRange (the same surface compute_slice_content
        // uses for the same condition). The grid is 2×2; a Slice
        // claiming `slice_x + slice_width = 3` is off the raster.
        let c = cr_v3_grid(2, 2);
        let headers = vec![header(0, 0, 3, 1)];
        assert!(matches!(
            validate_slice_raster_coverage(&headers, &c),
            Err(Error::SliceRasterOutOfRange { .. })
        ));
    }

    #[test]
    fn raster_coverage_rejects_v0_v1_config_record() {
        // The Configuration Record's `num_h_slices` / `num_v_slices`
        // are absent for v0 / v1; surface as SliceRequiresVersion3
        // (same surface every other §4 / §5 grid-dependent helper
        // uses for that branch).
        let mut c = cr_v3_grid(2, 2);
        c.num_h_slices = None;
        c.num_v_slices = None;
        let headers = vec![header(0, 0, 1, 1)];
        assert!(matches!(
            validate_slice_raster_coverage(&headers, &c),
            Err(Error::SliceRequiresVersion3)
        ));
    }

    #[test]
    fn raster_coverage_3x3_admissible_5_slice_tiling() {
        // Non-rectangular tiling of a 3×3 raster: a 2×2 Slice at
        // (0,0) plus four 1×1 Slices around it tile the grid
        // exactly. The five-Slice coverage is a stress test for the
        // walk's order-independence — the row-major scan should find
        // no gap even though Slice 0 paints the top-left quadrant
        // out of cell order.
        //
        // Layout (digits = forward Slice index):
        //   0 0 1
        //   0 0 2
        //   3 4 5
        // Note: this is not a §5-cap-compliant Frame above 101376
        // pixels (the 2×2 Slice violates the cap = 9/4 = 2), but
        // raster coverage is an orthogonal property — this
        // validator only checks the partition rule, not the
        // per-Slice size cap.
        let c = cr_v3_grid(3, 3);
        let headers = vec![
            header(0, 0, 2, 2), // top-left 2x2
            header(2, 0, 1, 1), // (2,0)
            header(2, 1, 1, 1), // (2,1)
            header(0, 2, 1, 1), // (0,2)
            header(1, 2, 1, 1), // (1,2)
            header(2, 2, 1, 1), // (2,2)
        ];
        assert!(validate_slice_raster_coverage(&headers, &c).is_ok());
    }

    #[test]
    fn raster_coverage_3x3_gap_diagnoses_first_uncovered_cell_in_row_major_order() {
        // Same layout as above but with the (2,1) Slice removed —
        // (2,1) is the first uncovered cell in row-major order
        // because (2,0) is still claimed by Slice 1.
        let c = cr_v3_grid(3, 3);
        let headers = vec![
            header(0, 0, 2, 2),
            header(2, 0, 1, 1),
            // missing header(2, 1, 1, 1)
            header(0, 2, 1, 1),
            header(1, 2, 1, 1),
            header(2, 2, 1, 1),
        ];
        match validate_slice_raster_coverage(&headers, &c) {
            Err(Error::SliceRasterUncovered { x: 2, y: 1 }) => {}
            other => panic!("expected SliceRasterUncovered((2,1)), got {other:?}"),
        }
    }

    // ---- RFC 9043 §5 third paragraph: non-keyframe Slice-geometry
    // ---- stability (`validate_slice_geometry_stability` +
    // ---- `SliceGeometryStabilityTracker`).

    #[test]
    fn geometry_stability_identical_partition_passes() {
        // The canonical 2×2 four-cell partition repeated verbatim
        // across two Frames — the §5 third-paragraph happy path.
        let frame = vec![
            header(0, 0, 1, 1),
            header(1, 0, 1, 1),
            header(0, 1, 1, 1),
            header(1, 1, 1, 1),
        ];
        assert!(validate_slice_geometry_stability(&frame, &frame).is_ok());
    }

    #[test]
    fn geometry_stability_permuted_order_passes() {
        // §5 reads "as a Slice in the previous Frame" — an existence
        // requirement. A permuted forward order across Frames keeps
        // the same geometry set and must pass.
        let previous = vec![
            header(0, 0, 1, 1),
            header(1, 0, 1, 1),
            header(0, 1, 1, 1),
            header(1, 1, 1, 1),
        ];
        let mut current = previous.clone();
        current.reverse();
        assert!(validate_slice_geometry_stability(&previous, &current).is_ok());
    }

    #[test]
    fn geometry_stability_ignores_non_geometry_fields() {
        // §5 names exactly slice_x / slice_y / slice_width /
        // slice_height; the other §4.6 header fields may change
        // Frame-to-Frame without violating the rule.
        let previous = vec![header(0, 0, 2, 2)];
        let mut changed = header(0, 0, 2, 2);
        changed.quant_table_set_index = [1, 1, 0];
        changed.picture_structure = crate::config::PictureStructure::TopFieldFirst;
        changed.picture_structure_raw = 1;
        changed.sar_num = 4;
        changed.sar_den = 3;
        assert!(validate_slice_geometry_stability(&previous, &[changed]).is_ok());
    }

    #[test]
    fn geometry_stability_changed_split_diagnoses_first_unmatched_slice() {
        // Previous Frame: one 2×1 Slice. Current Frame: two 1×1
        // Slices. Neither current quadruple appears in the previous
        // Frame; the diagnostic pins the *first* (forward index 0).
        let previous = vec![header(0, 0, 2, 1)];
        let current = vec![header(0, 0, 1, 1), header(1, 0, 1, 1)];
        match validate_slice_geometry_stability(&previous, &current) {
            Err(Error::SliceGeometryUnstable {
                slice_index: 0,
                slice_x: 0,
                slice_y: 0,
                slice_width: 1,
                slice_height: 1,
            }) => {}
            other => panic!("expected SliceGeometryUnstable(slice 0, (0,0) 1x1), got {other:?}"),
        }
    }

    #[test]
    fn geometry_stability_first_unmatched_index_is_deterministic() {
        // Slice 0 matches; slice 1 grew one raster cell taller. The
        // diagnostic carries forward index 1 and its quadruple.
        let previous = vec![header(0, 0, 1, 1), header(1, 0, 1, 1)];
        let current = vec![header(0, 0, 1, 1), header(1, 0, 1, 2)];
        match validate_slice_geometry_stability(&previous, &current) {
            Err(Error::SliceGeometryUnstable {
                slice_index: 1,
                slice_x: 1,
                slice_y: 0,
                slice_width: 1,
                slice_height: 2,
            }) => {}
            other => panic!("expected SliceGeometryUnstable(slice 1, (1,0) 1x2), got {other:?}"),
        }
    }

    #[test]
    fn geometry_stability_empty_current_is_vacuously_ok() {
        // "each Slice MUST ..." over zero Slices binds nothing. (The
        // §5 second-paragraph partition rule rejects a slice-less
        // Frame separately — the two validators are orthogonal.)
        let previous = vec![header(0, 0, 2, 2)];
        assert!(validate_slice_geometry_stability(&previous, &[]).is_ok());
    }

    #[test]
    fn geometry_stability_empty_previous_rejects_first_slice() {
        // No previous-Frame Slice exists, so no current Slice can
        // match one.
        let current = vec![header(0, 0, 2, 2)];
        match validate_slice_geometry_stability(&[], &current) {
            Err(Error::SliceGeometryUnstable {
                slice_index: 0,
                slice_x: 0,
                slice_y: 0,
                slice_width: 2,
                slice_height: 2,
            }) => {}
            other => panic!("expected SliceGeometryUnstable(slice 0, (0,0) 2x2), got {other:?}"),
        }
    }

    #[test]
    fn geometry_tracker_keyframe_opens_stream_without_check() {
        // §5 restricts only Frames "with a keyframe value of 0"; the
        // opening keyframe records its geometry and passes.
        let mut tracker = SliceGeometryStabilityTracker::new();
        assert!(!tracker.has_previous_frame());
        assert!(tracker.observe_frame(true, &[header(0, 0, 2, 2)]).is_ok());
        assert!(tracker.has_previous_frame());
    }

    #[test]
    fn geometry_tracker_non_keyframe_first_frame_with_slices_rejects() {
        // A non-keyframe before any observed Frame validates against
        // the empty set: there is no previous Frame whose Slices
        // could match (§3.8.1.3 / §3.8.2.5 only initialise coder
        // state on keyframe == 1, so such a stream is malformed
        // regardless).
        let mut tracker = SliceGeometryStabilityTracker::new();
        match tracker.observe_frame(false, &[header(0, 0, 1, 1)]) {
            Err(Error::SliceGeometryUnstable { slice_index: 0, .. }) => {}
            other => panic!("expected SliceGeometryUnstable(slice 0), got {other:?}"),
        }
        // The violating Frame did not become the reference.
        assert!(!tracker.has_previous_frame());
    }

    #[test]
    fn geometry_tracker_stable_sequence_tracks_immediately_previous_frame() {
        // §5 binds against "the previous Frame" — the immediately
        // preceding one, not the last keyframe. After a mid-stream
        // keyframe re-tiles the raster, a following non-keyframe
        // must match the NEW geometry; the old one now fails.
        let geometry_a = vec![header(0, 0, 2, 2)];
        let geometry_b = vec![header(0, 0, 1, 2), header(1, 0, 1, 2)];
        let mut tracker = SliceGeometryStabilityTracker::new();
        assert!(tracker.observe_frame(true, &geometry_a).is_ok());
        assert!(tracker.observe_frame(false, &geometry_a).is_ok());
        // A keyframe may change the tiling freely.
        assert!(tracker.observe_frame(true, &geometry_b).is_ok());
        assert!(tracker.observe_frame(false, &geometry_b).is_ok());
        // ... but geometry_a no longer matches the previous Frame.
        match tracker.observe_frame(false, &geometry_a) {
            Err(Error::SliceGeometryUnstable {
                slice_index: 0,
                slice_x: 0,
                slice_y: 0,
                slice_width: 2,
                slice_height: 2,
            }) => {}
            other => panic!("expected SliceGeometryUnstable(slice 0, (0,0) 2x2), got {other:?}"),
        }
    }

    #[test]
    fn geometry_tracker_error_leaves_previous_reference_untouched() {
        // A non-conforming Frame must not become the reference for
        // its successor: after the geometry_b rejection the original
        // geometry_a still validates.
        let geometry_a = vec![header(0, 0, 2, 2)];
        let geometry_b = vec![header(0, 0, 1, 2), header(1, 0, 1, 2)];
        let mut tracker = SliceGeometryStabilityTracker::new();
        assert!(tracker.observe_frame(true, &geometry_a).is_ok());
        assert!(tracker.observe_frame(false, &geometry_b).is_err());
        assert!(tracker.observe_frame(false, &geometry_a).is_ok());
    }

    #[test]
    fn geometry_tracker_default_matches_new() {
        // `Default` and `new()` agree: both start with no previous
        // Frame and reject an opening non-keyframe identically.
        let mut from_default = SliceGeometryStabilityTracker::default();
        let mut from_new = SliceGeometryStabilityTracker::new();
        assert!(!from_default.has_previous_frame());
        assert!(!from_new.has_previous_frame());
        assert_eq!(
            from_default
                .observe_frame(false, &[header(0, 0, 1, 1)])
                .is_err(),
            from_new
                .observe_frame(false, &[header(0, 0, 1, 1)])
                .is_err()
        );
    }
}
