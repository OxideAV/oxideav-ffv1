# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Native deep/RGB format remap onto core 0.1.33 (r430).**
  `pixel_format_for` now maps the layouts oxideav-core 0.1.33 added
  native variants for: 8-bit planar RGB → `Gbrp8` (one byte per
  Sample), 16-bit planar RGB / RGBA → `Gbrp16Le` / `Gbrap16Le`, and
  deep 4:2:0 + alpha at 10 / 12 / 16 bits → `Yuva420P10Le` /
  `Yuva420P12Le` / `Yuva420P16Le`. The corresponding
  significant-bits surface detours are retired: those six layouts now
  carry no side-channel record, and the registry `Decoder` packs
  8-bit RGB planes one byte per Sample (previously 2-byte `Gbrp10Le`
  words + `[8, 8, 8]`). `record_for_pixel_format` (the v0/v1
  trait-encoder synthesis) and the `Gbr` plane-reorder cover the six
  new formats, so each round-trips through the §4.2 mapping identity
  and the trait surface. Surface mappings newly REACHABLE through the
  added variants: 15-bit RGB / RGBA → `Gbrp16Le` / `Gbrap16Le` +
  `[15, …]`, and off-grid deep 4:2:0 + alpha (9 / 11 / 13 / 14 /
  15-bit) → the deep `Yuva420P*Le` surfaces. Surface mappings kept
  (no native variant exists): odd-depth 9 / 11 / 13-bit RGB and
  8-bit RGBA on `Gbrap10Le`. Still honestly unmapped: planar
  gray + alpha, deep 4:1:1, sub-8-bit RGB, reserved shifts. The
  reference / non-uniform corpus pins are unchanged (only the
  reported `PixelFormat` label and record moved); the
  `registry_roundtrip` fuzz table grows 32 → 38 mapped formats.

### Added

- **Docs-staged corpus-C conformance suite (r434).**
  `tests/staged_corpus_c.rs` decodes the 13 reference-encoded streams
  staged under `docs/video/ffv1/fixtures/` (8 `nonuniform-*` §4.8
  floor-division grid pins on odd frame dimensions + 5 `deep-*`
  colour pins: 9-bit YUVA 4:2:0 / 4:4:4, 14-bit gray / YCbCr 4:4:4 /
  RGB + alpha) straight from the staged bytes — full SHA-256 + byte
  count pins on every `input.mkv` / `expected.raw`, a container-layer
  EBML walk to extract the Configuration Record (including the
  VfW-compat track shape whose private data prepends a 40-byte
  BITMAPINFOHEADER with biCompression `FFV1`) and the coded Frames,
  then bit-exact decode against `expected.raw` through BOTH the
  direct `decode_frame*_with_carry` API (under
  `DecodeOptions::pedantic()`) and the framework
  `oxideav_core::Decoder` trait (mapping surface + significant-bits
  record asserted per stream: `Gray16Le`+`[14]`,
  `Yuv444P16Le`+`[14,14,14]`, `Yuva420P10Le`/`Yuva444P10Le`+`[9,…]`,
  native `Gbrap14Le`). All 13 streams (26 Frames, keyframe + carried
  non-keyframe each) decode bit-exactly with zero divergences. The
  suite is gated on docs presence (standalone-crate CI passes
  vacuously; no stream bytes are copied into this repository).
  Geometry note pinned by the tests: six of the eight `nonuniform-*`
  fixtures are genuinely non-uniform; the `3x3`-over-`99x75` pair
  divides evenly (99 = 3 × 33, 75 = 3 × 25), so its §4.8 floor
  divisions produce equal extents despite the fixture name.

- **Whole-loop encode pins at the six newly-native formats (r430).**
  `tests/registry_native_pins.rs` drives the registry `Encoder` on the
  v0/v1 empty-extradata route at `Gbrp8`, `Gbrp16Le`, `Gbrap16Le` and
  the deep `Yuva420P10Le` / `Yuva420P12Le` / `Yuva420P16Le` — streams
  the trait encoder could not synthesise before the remap — pinning
  each packet's FNV-1a-64 (keyframe + two carried non-keyframes per
  format, 9×7 odd-dimension frames, full-depth content) and asserting
  the inline §4.4 Parameters, the §4.4 keyframe flags, and the
  bit-exact trait decode-back with the native label and no
  significant-bits record.

- **Deep-format §4.2 pixel-format mapping (r420).** `pixel_format_for`
  now maps the layouts oxideav-core 0.1.31 added exact variants for:
  16-bit planar YUV (`Yuv420P16Le` / `Yuv422P16Le` / `Yuv444P16Le`),
  the 8-bit alpha trio (`Yuva422P` / `Yuva444P` alongside the existing
  `Yuva420P`), and the deep alpha-carrying 4:2:2 / 4:4:4 family at
  10 / 12 / 16 bits (`Yuva422P10Le` … `Yuva444P16Le`).
  `record_for_pixel_format` (the v0/v1 trait-encoder synthesis) covers
  the same additions, so every new format round-trips through the §4.2
  mapping identity.

- **Side-channel-aware mapping: `pixel_format_mapping_for` /
  `Ffv1PixelFormatMapping` (r420).** Maps the §4.2 layouts whose exact
  depth has no named framework variant onto the smallest named storage
  surface that holds it, paired with the per-plane significant-bits
  record `oxideav_core::VideoFrame` (0.1.31) carries for exactly this
  purpose: 9 / 11 / 13 / 14 / 15-bit YCbCr (gray, 4:2:0 / 4:2:2 /
  4:4:4, ± alpha at 4:2:2 / 4:4:4) ride the 10 / 12 / 16-bit surfaces;
  sub-8-bit YCbCr rides the 8-bit byte surfaces; and 8 / 9 / 11 /
  13-bit planar RGB / RCT (± alpha) rides the `Gbrp*Le` / `Gbrap*Le`
  16-bit-word surfaces — closing the 8-bit planar-RGB mapping gap.
  15 / 16-bit planar RGB, deep 4:2:0-plus-alpha, planar gray + alpha,
  deep 4:1:1 and reserved shifts stay honestly unmapped.

- **Non-uniform §4.8 slice-grid + deep-format reference corpus — 13
  reference-encoded keyframe+inter streams (r420).** Generated
  black-box (reference toolchain; RIFF/AVI container walk for packet
  extraction) and inlined with per-frame SHA-256 pins in
  `tests/data/nonuniform_deep_fixtures.rs`, driven by
  `tests/nonuniform_deep_reference_decode.rs`. Eight `nonuni-*` streams
  put the §4.8.2 / §4.8.3 floor divisions on grids the frame dimensions
  do NOT divide evenly (odd dims — 61×47, 97×65, 99×75 — where the
  format admits them; 4:2:0-safe even shapes with indivisible slice
  counts otherwise), covering 2×2 / 3×2 / 3×3 grids × range +
  Golomb-Rice × YCbCr / gray / YUVA / RGB-RCT × 8/10/16-bit; the tests
  recompute the §4.8 geometry from the decoded §4.6 Slice Headers and
  assert every grid is genuinely non-uniform, tiles the raster exactly,
  and decodes bit-exact under `DecodeOptions::pedantic()`. Five
  `deep-*` streams pin the deep Yuva family (4:2:2/4:4:4 at 10/12/16
  bits) and the off-grid 9 / 14-bit depths against reference bytes.
  Every stream also decodes bit-exactly through the framework
  `Decoder` trait with its mapping surface + significant-bits record
  asserted.

- **External-conformance corpus grown 28 → 34 streams (r420).** Six
  new self-encoded cells validated black-box against the reference
  decoder (bit-exact, zero warnings; procedure + results in
  `tests/external_conformance_notes.md`): the deep Yuva family
  (`yuva422p10le` / `yuva444p12le` / `yuva444p16le`), the off-grid
  14-bit and 9-bit YCbCr depths, and a 16-bit 4:4:4 stream on a
  non-uniform §4.8 3×2 grid over odd 63×47 dimensions. The 28
  pre-existing pins revalidated unchanged in the same run.

- **Registry traits wired onto the deep-format mapping (r420).** The
  framework `Decoder` attaches the mapping's significant-bits record to
  every emitted `VideoFrame` and packs planes at the mapped surface's
  word width; the framework `Encoder` advertises the mapped surface on
  `output_params.pixel_format`, reads input planes at the surface word
  width, skips frame side-channel entries via
  `VideoFrame::image_planes`, and rejects (with a typed diagnostic) an
  attached significant-bits record that conflicts with the stream's
  §4.2.7 depth. 19 new trait-surface round-trip tests
  (`tests/registry_deep_formats.rs`) cover 16-bit YUV, the full Yuva
  family, 9/14-bit YCbCr on deeper surfaces, and 8-bit planar RGB /
  RGBA on the `Gbrp10Le` / `Gbrap10Le` surfaces, plus the v0/v1
  empty-extradata route and a deep-alpha inter-carry stream.

### Changed

- **8-bit RGB / RCT streams now map through the registry traits** (were
  previously unmapped: `pixel_format` stayed unset and planes crossed
  the trait boundary in internal R, G, B byte order). They now ride the
  `Gbrp10Le` / `Gbrap10Le` storage surfaces: 2-byte little-endian
  words, framework G, B, R (, A) plane order, and a `[8, 8, …]`
  significant-bits record on every decoded frame. Callers that fed the
  trait encoder one-byte R, G, B planes for an 8-bit RGB stream must
  switch to the advertised surface layout; the direct `decode_frame*` /
  `encode_frame*` API is unchanged.

- **Inter-frame reference-decode corpus — 16 reference-encoded
  keyframe+inter streams (r416).** Staged under
  `docs/video/ffv1/fixtures/inter-*` (generation command, keyframe
  pattern, and SHA-256 per fixture) and driven by
  `tests/reference_inter_decode.rs` from inlined packets
  (`tests/data/reference_inter_fixtures.rs`, per-frame SHA-256 pins):
  every stream is one §4.4 keyframe plus carried non-keyframes, decoded
  bit-exactly through the §3.8.1.3 / §3.8.2.5 carry drivers under
  `DecodeOptions::pedantic()`, with each Frame's §4.4 keyframe flag
  matching the reference toolchain's report. Coverage: v0/v1 inline
  Parameters (range + Golomb-Rice) and v3 across 8/10/12/16-bit,
  4:2:0/4:2:2/4:4:4/gray/RGB/RGBA, 2×2 slices, `-context 1`,
  Golomb-Rice, a reference-encoded `coder_type == 2` custom-table
  stream (the §3.8.1.6 decode path was previously validated only
  against this crate's own encoder), a mid-stream-keyframe stream
  (`-g 2` — carry re-initialisation on a later keyframe), and a
  `-slicecrc 0` stream. This answers the long-open "reference-encoded
  inter fixtures" ask by generating the corpus black-box per the
  standing fixture-staging ruling.

- **Registry-level §4.2.16 `ec` resolution on the first Frame (r416).**
  Black-box finding (module doc of `tests/reference_inter_decode.rs`):
  the current reference encoder's Configuration Record tail does not
  read back under the RFC 9043 Figure 28 layout its own parser accepts
  — re-authoring a fixture record with this crate's Figure 28 writer
  (`ec == 1`) decodes the reference packets bit-exact while `ec == 0`
  fails hard (so the parsed `ec` is honoured), yet the reference
  writer's own two-set records parse to non-physical tail values (`ec`
  up to 7; `intra == 1` on streams carrying non-keyframes). A
  `-slicecrc 0` stream can therefore misdeclare `ec != 0` and previously
  failed entirely ("slice byte range is shorter than its §4.9 Slice
  Footer"). The registry `Decoder` now treats the record-derived `ec`
  as a hypothesis until the first Frame decodes: one retry with the
  opposite §4.9 footer shape, locking whichever hypothesis yields a
  fully-validated Frame (§4.9.1 trailer chain + §4.9 size cross-check +
  §4.9.3 CRC residue + §5 raster coverage all still gate it).
  Truthfully-declared records decode on the first attempt and never
  retry; the direct `decode_frame*` API is unchanged. Covered by
  `tests/registry_inter_ec_resilience.rs` against the real misdeclared
  fixture.

### Added

- **Mid-stream-keyframe self-encoded conformance stream (r416).**
  `v3-yuv420p8-range-kf2` joins the external-conformance corpus (now 28
  streams): keyframes at Frames 0 and 2 emitted through
  `encode_frame_with_carry`, so the §3.8.1.3 per-context state
  re-initialises mid-stream and the following non-keyframe carries from
  the NEW keyframe. The reference decoder reports the §4.4 pattern
  `1 0 1 0` and decodes the stream bit-exactly — the encode-side mirror
  of the reference-encoded `inter-v3-yuv420p8-range-g2` decode fixture.

### Changed

- **External-conformance matrix completed: 27/27 (r416).** The one
  cell r411 left as a validator limitation — `v0-yuv420p8-custom`
  (version 0, `coder_type == 2`) — was root-caused with black-box
  delta probes: the reference decoder rejects a v0/v1 inline-Parameters
  custom table containing **any** zero transition (`invalid state
  transition 0`), including the all-zero-delta table that equals the
  §3.8.1.5 default (whose entries `1..=8` / `249..=255` are zero),
  while its version-3 Configuration Record path accepts the same delta
  block. Both table shapes are RFC-conforming — Figure 28 places no
  version condition on the §4.2.4 block and the zeroed states are
  unreachable from the §3.8.1.3 initial state 128 — so the corpus now
  exercises encoder freedom in the interoperable direction:
  `custom_transition_deltas()` lifts every zero-default entry to the
  self-loop `i`, making every transmitted transition nonzero. With
  that table, **all 27 corpus streams decode bit-exactly in the
  external reference decoder with zero warnings.** Byte impact is
  confined to the two custom streams (the v3 one only in its extradata
  blob; the lifted entries are never visited while coding); probe
  matrix and updated results in
  `tests/external_conformance_notes.md`.

### Added

- **Externally-validated encoder conformance corpus** (r411):
  `tests/external_conformance.rs` pins (SHA-256 per packet + per
  extradata blob) a 27-stream self-encoded corpus over versions 0/1/3 ×
  all three §4.2.3 coders × gray/YUV/YUVA/RGB/RGBA × 8/10/12/14/16-bit
  × single/2×2/non-uniform-3×2 slice grids × ec 0/1 × §4.2.17 intra,
  each stream a keyframe plus carried non-keyframes (up to a 4-frame
  chain), with bit-exact self round-trip. 26/27 decode bit-exactly in
  the black-box external reference decoder (zero warnings); the one exception (v0 +
  `coder_type == 2`) is RFC-conforming per Figure 28 but unimplemented
  by the validator. Generation + validation procedure and per-stream
  results in `tests/external_conformance_notes.md`; setting
  `FFV1_CONFORMANCE_EXPORT_DIR` exports the corpus for out-of-tree
  re-validation.

- **v0/v1 inter-Frame coder-state carry** (r411): RFC 9043 §3.8.1.3 /
  §3.8.2.5 re-initialise the per-context coder state only "when the
  keyframe value is 1" — on every FFV1 version — so a conforming v0/v1
  non-keyframe *continues* the previous Frame's state over the implied
  single Slice, exactly as v3 carries it per Slice. New carry-aware
  entry points `decode_frame_v0v1_with_carry` /
  `decode_frame_v0v1_inter_with_carry` and `encode_frame_v0v1_with_carry`
  / `encode_frame_v0v1_inter_with_carry` implement that (validated
  bit-exact against reference-encoded v0/v1 keyframe + inter streams,
  both coders, and self-encoded streams now decode bit-exact in the
  external reference decoder). The registry `Decoder` / `Encoder` v0/v1
  routes now use the carry variants, so multi-Frame v0/v1 trait streams
  are conforming. The historical stateless `decode_frame_v0v1_inter` /
  `encode_frame_v0v1_inter` remain unchanged as the degenerate no-carry
  pair (fresh states each Frame — self-consistent but NOT what a
  conforming decoder expects); their docs now say so.

- **§3.8.1.1.1 termination gate** (`DecodeOptions::pedantic()` /
  `SliceTerminationPolicy`, r411): opt-in decode-side verification that
  every v3 range-coded Slice body ends with the Sentinel-mode
  terminator at exactly the §4.9.1 body length — the bookkeeping a
  conforming decoder uses to flag Slice damage, and the read-side
  mirror of this round's encoder termination fix. A mismatch surfaces
  the typed `Error::SliceTerminationMismatch`; the default
  `SliceTerminationPolicy::Accept` keeps the historical behaviour (and
  keeps pre-r411 self-encoded archives decodable). Golomb-Rice Slices
  are exempt (their byte-aligned tail has no §3.8.1.1.1 terminator of
  its own).

### Fixed

- **§3.8.2 Golomb-Rice sign-flip edge fold** (r411, found by the
  `roundtrip` fuzz target minutes after the corpus landed): a §3.5
  sign-flipped context whose folded Sample Difference is exactly
  `-2^(bits-1)` negates to `+2^(bits-1)` — one past the top of the §3.8
  `bits`-wide signed window — and the Golomb-Rice symbol coder's suffix
  arithmetic wraps it to a different value on the wire. Since §3.8
  Sample reconstruction is modular, `±2^(bits-1)` code the same Sample:
  the encoder now folds the edge back to `-2^(bits-1)` before emission
  (`fold_coded_diff`), on both the scalar and the §3.8.2.4.1 level-break
  paths. Six Golomb corpus streams changed bytes and were re-validated
  bit-exact against the external reference decoder before re-pinning;
  regression gate: `tests/golomb_sign_flip_fold.rs` (the exact fuzz
  artifact, a 47×14 v1 RGBA Golomb frame in 9-bit coded RCT space). The
  range-coder paths are unaffected (their symbols are not
  width-bounded).
- **§3.8.1.1.1 range-coder termination on every v3 Slice** (r411,
  found by black-box external-reference-decoder validation of
  self-encoded streams): a v3 Slice's range-coded region now ends with
  the Sentinel-mode terminator — the discarded state-129 symbol plus a
  don't-care boundary byte — at BOTH termination points RFC 9043
  §3.8.1.1.1 names: the end of a range-coded Slice (before its §4.9
  footer) and the Slice-Header → Golomb-Rice content switch. The
  emitted body reads identically in Sentinel mode (one-byte over-read)
  and Closed mode (zero-fill), and a conforming decoder's end-position
  bookkeeping lands exactly on the Slice length. Previously the encoder
  flushed with a bare `finish()`, leaving every range Slice one byte
  longer than a conforming decoder consumes — tolerated on keyframes
  but treated as Slice damage on non-keyframes, which silently
  concealed (previous-Frame copy) every carried inter Frame. The v3
  decoder now also recovers the Golomb-content boundary by *reading*
  the sentinel (`RangeDecoder::terminate_sentinel`) instead of trusting
  the raw cursor, fixing bit-exact decode of reference-encoded v3
  Golomb-Rice streams whose sentinel renormalisation crosses a byte
  boundary.
- **Slice-scoped §3.8.2.2.1 run state on the §4.7 line-major RGB
  Golomb-Rice path** (r411, same black-box campaign): `run_index` is
  "reset to zero for each Plane and Slice", and on the line-major RGB
  interleave the Slice is the governing scope — ONE run triple evolves
  across the whole `for y { for p { Line(p, y) } }` walk, shared by
  every Plane (`run_mode` / `run_count` stay per-Line). The previous
  per-Plane split round-tripped self-consistently but desynchronised
  against conforming streams; reference-encoded v3 RGB Golomb streams
  (keyframe + inter) now decode bit-exactly, and self-encoded ones
  externally validate. Applies to v3 and v0/v1 RGB drivers on both the
  encode and decode sides.

### Added

- APPLICATION of the §4.2.15 explicit initial states (RFC 9043 Figures
  29/30): `reconstruct_initial_states()` folds the transmitted deltas
  through the Figure 29 predictor chain + Figure 30 modular
  reconstruction, and every keyframe-initialised per-§4.6.6-slot
  range-coder window (YCbCr + RGB, decode + encode) now seeds from the
  reconstructed states instead of the §3.8.1.3 all-128 default when
  `states_coded == 1`. FFmpeg-interop padding rows (`j >=
  context_count`) parse but never seed a live context. Golomb-Rice
  (`coder_type == 0`) §3.8.2.5 VLC state is unaffected (§4.2.15 is
  "the initial range coder state"). Degenerate reconstructed states
  (0, and the default-table band feeding into it via
  `one_state[1..=8] == 0`) previously zeroed the range on a 1-bit and
  spun the encoder's renormalisation loop forever while growing its
  output unboundedly; two cold-path guards now close every entry into
  state 0 with no per-bit cost: coder construction sanitizes
  transitions-into-0 out of the active table (self-loops, unreachable
  by valid streams — this also covers hostile `coder_type == 2`
  custom tables), and the §4.2.15 seed boundary clamps explicit
  state-0 seeds to 1, identically on both sides.
  r390 black-box probes: the reference decoder parses the triple-loop
  identically but applies it through a non-RFC context labelling, so
  non-zero explicit states are self-interoperable only (all-zero
  deltas remain fully reference-compatible); divergence documented on
  `reconstruct_initial_states`.

- `states-coded-1` conformance gate: parse + bit-exact frame decode of the
  hand-authored RFC 9043 §4.2.14/§4.2.15 `states_coded == 1` fixture
  (docs commit bb7e387), plus a parse→re-encode→re-parse round-trip of the
  coded record.
- `QuantizationTableSet::len_counts()` (recover §4.1 `len_count` from the
  decoded tables) and `QuantizationTableSet::initial_state_row_count()`
  (the §4.2.15 row count actually consumed on the wire — the
  FFmpeg-interop counts 942/645 for the pinned `[6,6,6,1,1]` /
  `[5,5,5,1,1]` shapes, RFC `context_count` fallback otherwise).
- `CONFIGURATION_RECORD_CRC_PARITY_LEN` (§4.3.2 parity word length).

### Fixed

- §4.2.15 `initial_state_delta` symbol-coding layout on both parse and
  encode: the delta block uses ONE dedicated 32-slot window freshly
  initialised to 128 (fixture-pinned byte-exactly), not the shared
  adapted Parameters window, and iterates the FFmpeg-interop row count
  rather than the §4.1 `context_count`. Streams with a coded triple-loop
  previously desynchronised at `ec`/`intra`.

## [0.0.9](https://github.com/OxideAV/oxideav-ffv1/compare/v0.0.8...v0.0.9) - 2026-07-03

### Other

- BENCHMARKS.md/CHANGELOG/README — fold in the lazy-context step + final golomb encode numbers
- lazy §3.5 context in the Golomb encode_line run loop — golomb encode -2.7..-5.7%
- BENCHMARKS.md — round-386 matrix results, profiles, optimization log
- neighbour-carry in the per-row §3.3/§3.5 stencil loops — cumulative decode -13..-19%
- §3.8.2 bit-engine fast paths — golomb decode -13%, golomb encode -22..-25%
- fixed 32-slot context windows through the §3.8.1.2 symbol coder — range decode -5..-9%
- slicing-by-8 §4.9.3 CRC — golomb decode -5.5%/-12.9%, byte-identical outputs
- Criterion bench harness across decode+encode × coder × depth × colorspace × slices
- encoder byte-exactness pins across the r386 bench matrix
- fix §3.8.2.2 run-mode encoder desync on multi-context quant tables
- CHANGELOG — note ColorspaceLayoutNotImplemented Display change
- README — Status headline covers versions 0/1/3 + both API surfaces
- reconcile stale round-1 scaffold docs with the shipped surface
- rustfmt the fuzz package (registry_roundtrip + pre-existing drift)
- registry_roundtrip fuzz target — trait-surface lossless identity
- encode versions 0/1 through the framework Encoder trait
- retire stale RunModeFirstPixelNonZero docs + nonzero-first-pixel v0/v1 proof tests
- README — RGB/RCT maps to planar Gbr formats through the trait
- multi-frame RGB inter-carry round-trip through the trait
- RGB/RCT end-to-end round-trip tests through the framework trait
- map RGB / RCT to planar Gbr framework PixelFormats (10/12/14-bit)
- fix reconstruct_sample debug-build overflow panic on adversarial bit depths
- fix §3.8.1.1.1 Sentinel-mode boundary corruption + add roundtrip fuzz target
- README — record r374 9/12/15/16-bit + non-uniform-grid coverage
- non-uniform slice-grid round-trips (§4.8 floor division)
- RGB (RCT) exception-boundary + 16-bit general round-trips
- v0/v1 YCbCr 9 / 12 / 16-bit chroma-Frame round-trips
- v3 YCbCr 9 / 12 / 16-bit chroma-Frame round-trips (range coder)
- cargo-fuzz harness — decode / parse panic-freedom (4 targets)
- fix two v0/v1 RGB decode OOB panics on non-conforming Records
- neutralize black-box-validator naming in r361 fixture prose
- reference-fixture corpus — v3-frame-mt 256x192 16-Slice (13→14)
- reference-fixture corpus — v3-rgb-bgr0 RGB-no-alpha (12→13)
- reference-fixture corpus — multi-Slice v3-default (2x2) + 4x4 (10→12)
- reference-fixture decode corpus — v3-grayscale + v3-yuv444p16 (8→10)
- §3.8.1.1.1 Sentinel-mode range→Golomb handoff + v0 golomb fixture
- §3.8.2.2 Golomb-Rice run-mode as a per-Line ctx-0 decode loop
- reference-fixture decode corpus — 7 end-to-end bit-exact tests vs expected.raw
- decode + encode v0/v1 coder_type 2 (custom state-transition table)
- add in-place state-transition-table swap to the range coder
- versions 0/1 decode through the framework Decoder trait (round 342)
- versions 0/1 RGB / RCT (colorspace_type 1) decode + encode (round 342)
- versions 0/1 Golomb-Rice (coder_type 0) encode round-trip (round 342)
- versions 0/1 single-Slice YCbCr decode + encode end-to-end (round 342)
- framework Encoder emits inter-Frame (non-keyframe) streams end-to-end
- YCbCr Golomb-Rice inter-Frame encode carry + unified encode_frame_with_carry dispatcher
- §4.4 in-Frame Parameters parse for versions 0/1 (round 333)
- surface §4.2-derived PixelFormat on the framework encoder output_params
- wire the oxideav_core::Encoder trait (slice-grid derivation)
- Wire FFV1 decoder behind oxideav_core::Decoder + register codec (round 317)
- refresh to current status, drop per-round changelog cruft

### Added

- **Criterion bench harness + BENCHMARKS.md (round 386, depth mode).**
  `benches/decode.rs` / `benches/encode.rs` over a shared 12-scenario
  matrix (coder 0/1/2 × 8/10/16-bit × YCbCr-4:2:0/RGB × 1/4/16-slice
  grids at 320×240), all inputs synthesised in-bench on a realistic
  666-context Quantization Table Set; throughput normalised to
  raw-sample bytes. `tests/optimization_pins.rs` pins an FNV-1a-64 hash
  of the encoder's output for every scenario plus the lossless
  decode-back invariant, so any hot-path change that flips a single
  output byte fails CI. Results, profiles, and the optimization log
  live in `BENCHMARKS.md`.

### Changed

- **Hot-path performance (round 386) — outputs byte-identical
  throughout** (encoder pins + reference fixture corpus green at every
  step): slicing-by-8 §4.9.3 CRC (eight compile-time tables, eight
  independent loads per 8-byte block; the CRC gate fell from ~8% to
  ~1.6% of a Golomb decode); fixed 32-slot `[u8; 32]` context windows
  through the §3.8.1.2 scalar symbol coder (no per-slot bounds checks)
  with `#[inline]` on `RangeDecoder::get_rac`/`refill`; §3.8.2
  bit-engine fast paths (32-bit-word `BitReader` refill, the §3.8.2.1
  unary prefix decoded via one 12-bit peek + leading-zero count, bulk
  `BitWriter::put_bits`); neighbour-carry in the per-row §3.3/§3.5
  stencil loops (only `tr` + `tt` loaded per Sample); lazy §3.5
  context in the Golomb `encode_line` run loop (in-run Samples never
  consult it). Net vs the round-386 baseline (aarch64 macOS): decode
  −10…−20% across the matrix (Golomb 8-bit 127 → 156 MiB/s, range
  8-bit 66 → 82 MiB/s), encode −4…−29% (Golomb 16-bit 1.443 →
  1.031 ms). Two measured-and-reverted experiments are documented in
  BENCHMARKS.md (encoder-side `put_rac` inlining: +39% on 16-bit
  range encode; decoder-side lazy context: +1.1%).

### Fixed

- **§3.8.2.2 run-mode encoder desync on multi-context Quantization Table
  Sets (round 386).** The Golomb-Rice `encode_line` run scanner ended a
  run at the first nonzero-context Sample (a "predicate break" that does
  not exist on the decode side — RFC 9043 §3.8.2.2 leaves run mode only
  "as soon as a nonzero difference is found", and the decoder's
  `run_count` countdown never re-evaluates the context). On any table
  whose §3.5 context genuinely varies with the neighbours (every
  realistic table, including the registry's default 666-context set), a
  long-run `1` bit could silently claim a zero difference for a Sample
  whose actual difference was nonzero — a *lossy* encode at every
  flat-region → textured-region boundary; the decoder reconstructed a
  repeat of the previous row. Additionally the §3.8.2.4.1 level-coded
  break Sample was encoded against `state.vlc[0]` instead of the
  breaking Sample's own §3.5 context window (the decoder reads
  `state.vlc[abs_ctx.index]`, which need not be 0 because the break can
  land on a nonzero-context Sample). Both defects were invisible to the
  zero-/single-context tables the unit suites use. Covered by
  `tests/golomb_run_mode_multicontext.rs` (v3 gray/RGB, 8/16-bit, v0/v1
  inline-Parameters — all Golomb drivers share `encode_line`).

### Added

- **Versions 0/1 encoding through the framework `Encoder` trait
  (round 382).** RFC 9043 §4.3.3 / §4.4: v0/v1 streams carry no §4.2
  Configuration Record — their Parameters ride inline in each keyframe
  Frame. The registry encoder now accepts the same configuration shape the
  registry *decoder* already accepted: `CodecParameters` with **empty
  extradata** plus a `pixel_format` and dimensions. It synthesises a
  version-1 record from the pixel format (`record_for_pixel_format`, the
  exact inverse of `pixel_format_for` over its mapped range — verified by
  a round-trip unit test across all 21 mapped formats), installs a
  §4.1-constructed default Quantization Table Set (11 symmetric levels on
  the three §3.5 Figure 5 primary differences, flat on the two
  second-order ones; scale chain 11³ = 1331 → `context_count == 666`), and
  emits the first Frame as a §4.4 keyframe (inline Parameters + Set) and
  later Frames as non-keyframes. `output_params.extradata` stays empty so
  a muxer writes no CodecPrivate. An unmappable pixel format is a
  diagnosable construction error. Covered by
  `tests/registry_v0v1_encoder.rs`: gray8 / yuv420p / yuv422p10 / planar
  RGB (`Gbrp12Le`, with the G,B,R ⇄ R,G,B reorder) multi-frame trait
  round-trips, inline-prologue shape assertions, and the two
  misconfiguration paths.

- **`registry_roundtrip` fuzz target — trait-surface lossless identity
  (round 382).** A sixth cargo-fuzz harness lifts the `roundtrip`
  contract onto the framework trait: a well-formed `VideoFrame` (one of
  the 21 mapped `PixelFormat`s, bounded dims, depth-masked samples)
  encodes through the registry `oxideav_core::Encoder` and decodes back
  through the registry `oxideav_core::Decoder`, asserting bit-exact plane
  bytes over a keyframe + non-keyframe pair — putting the §4.2
  pixel-format mapping and its inverse, the §4.1 default Quantization
  Table Set wire round-trip, the LE plane packing, the planar-`Gbr*`
  plane reorder, and the Frame sequencing under `decode(encode(x)) == x`.
  ~100k executions clean on the first run.

- **Nonzero-first-pixel v0/v1 Golomb proof tests + retired-doc
  reconciliation (round 382).** Two `tests/v0v1_roundtrip.rs` tests force
  a non-zero Sample Difference onto the run-region first Sample (gray +
  yuv420 across all Planes) and round-trip bit-exact via the §3.8.2.4.1
  zero-length short run. Stale prose describing the retired
  `RunModeFirstPixelNonZero` rejection (Error-variant doc, `encode_line`
  Errors section, `encode_frame_v0v1` Errors list, lib.rs module doc,
  README) is reconciled; the never-constructed variant is kept for API
  stability and documented as retired.

- **RGB / RCT is now first-class through the framework `Decoder` /
  `Encoder` trait — §4.2 pixel-format mapping to the planar `Gbr` family
  (round 382).** `pixel_format_for` previously returned `None` for every
  `colorspace_type == 1` (RGB / JPEG 2000 RCT) stream, so RGB frames
  decoded through the registry carried no advertised `PixelFormat` and a
  transcode could not label them. They now map to oxideav-core's planar
  RGB formats: 10 / 12 / 14-bit RGB → `Gbrp10Le` / `Gbrp12Le` /
  `Gbrp14Le`, and with the §4.2.10 alpha `extra_plane` →
  `Gbrap10Le` / `Gbrap12Le` / `Gbrap14Le`. The RFC 9043 §3.7 RCT recovers
  Planes in **R, G, B (, A)** order while the `Gbr*` formats store
  **G, B, R (, A)**, so the registry's plane converters now reorder Planes
  at the trait boundary (`gbr_plane_order` on decode, its exact inverse
  `gbr_input_order` on encode) — the advertised format and the emitted /
  consumed plane order agree by construction, verified by a
  mutual-inverse unit test. 8-bit and 16-bit planar RGB (no `Gbrp`
  variant — the framework's 8/16-bit RGB formats are packed, not planar)
  and odd depths stay honestly `None`. YCbCr and the unmapped RGB depths
  are unaffected (native `DecodedFrame` plane order, identity permutation).

### Changed

- **`Error::ColorspaceLayoutNotImplemented` Display text (round 382).**
  The message claimed RGB line-major traversal was "not yet implemented";
  both layouts have long been implemented in sibling drivers. It now says
  the colorspace was routed to the wrong §4.7 frame driver and names the
  correct entry points (`decode_frame`/`encode_frame` for YCbCr,
  `decode_frame_rgb`/`encode_frame_rgb` for RGB). Part of a wider
  stale-doc reconciliation (frame.rs scope notes, config.rs /
  slice_header.rs module docs, the retired `NotImplemented` variant doc,
  and the fuzz workflow comment).

### Fixed

- **`reconstruct_sample` debug-build overflow panic on adversarial bit
  depths (round 377).** The §3.8 modular-reduction mask was computed as
  `(1i32 << bits) - 1` after clamping `bits` to `1..=31`. At `bits == 31`,
  `1i32 << 31` is `i32::MIN`, so the `- 1` underflowed `i32` and panicked
  in a debug build — a panic-freedom violation reachable from a fuzzed
  Configuration Record (caught by the scheduled `registry_decode` fuzz
  target). The mask is now computed in `u32` (`(1u32 << bits) - 1`), exact
  and panic-free for every clamp value; covered by
  `reconstruct_sample_adversarial_bits_do_not_overflow`.
- **§3.8.1.1.1 Sentinel-mode boundary corruption on the v0/v1 Golomb-Rice
  path (round 377).** `RangeEncoder::terminate_sentinel` flushed the final
  `low` register without rounding its low byte to zero, so for some
  prologue byte-alignments the decoder's mandatory one-byte over-read past
  the boundary (RFC 9043 §3.8.1.1.1) landed on the *first appended
  Golomb-Rice byte* and let it change the last §4.1 sub-table symbol's
  `low < range` decision. The recovered Quantization Table Set's
  `context_count` then came out wrong and the v0/v1 self round-trip
  diverged. First surfaced (via the new `roundtrip` fuzz target) on 16-bit
  RGB — where the §3.7 RCT coded width `bits + 1 == 17` shifts the prologue
  length into the failing alignment — but the bug was alignment-driven, not
  depth-driven. The fix rounds `low` up to the next `0x100` boundary when
  that still lies inside the live `range`, making the over-read byte a true
  don't-care; it preserves bit-exact decode of the reference
  `v0-yuv420-golomb-rice` fixture (Sentinel-encoded externally) and is
  covered by a `RangeEncoder::terminate_sentinel` symbol-count sweep plus a
  depth × dimension v0/v1 Golomb round-trip matrix (`tests/v0v1_roundtrip.rs`).

### Added

- **Encode → decode round-trip fuzz target (round 377).** A fifth
  `fuzz/fuzz_targets/roundtrip.rs` cargo-fuzz harness inverts the existing
  decode-only attack surface: it builds a well-formed `DecodedFrame` +
  matching `Ffv1ConfigurationRecord` / §4.1 Quantization Table Set / §4.6
  Slice Header from the attacker bytes, encodes it, decodes the result, and
  asserts the recovered Planes are bit-exact (FFV1 is lossless, RFC 9043
  §1: `decode(encode(x)) == x`). It sweeps the version (0/1/3) ×
  `coder_type` (0/1/2) × colorspace (YCbCr/RGB) × bit-depth (8/9/10/12/16)
  cross product; it found the Sentinel-mode boundary bug above on the first
  run and now passes 320k+ executions clean.

- **v3 YCbCr 9 / 12 / 16-bit chroma-Frame round-trips (round 374).** The
  `tests/chroma_encode_frame.rs` suite previously covered the v3 range-coded
  chroma drivers only at 8-bit and 10-bit. Three new end-to-end
  `encode_frame` → `decode_frame` round-trips close the higher-depth gap on
  the range coder (`coder_type == 1`):
  - `range_yuv420_9bit_single_slice` — 9-bit 4:2:0, the smallest depth
    above the 8-bit byte boundary (RFC 9043 §4.2.3 restricts only
    Golomb-Rice to `bits_per_raw_sample <= 8`; the range coder carries
    9-bit Samples unrestricted). Exercises the §3.8 modular wrap at a
    `0 .. 512` window.
  - `range_yuv444_12bit_single_slice` — 12-bit 4:4:4, three full-resolution
    Planes through the ordinary §3.3 median predictor at a `0 .. 4096`
    window.
  - `range_yuv420_16bit_predictor_exception_single_slice` — 16-bit 4:2:0
    with Samples spanning the full `0 .. 65536` range, the first
    chroma-Frame round-trip exercising the RFC 9043 §3.3.1 exception
    predictor (`median(left16s, top16s, left16s + top16s - diag16s)` with
    two's-complement 16-bit reinterpretation), which §3.3.1 mandates for
    `colorspace_type == 0 && bits == 16 && coder_type ∈ {1, 2}`. The
    exception was previously verified only by a predictor unit test, not a
    full Frame round-trip.
- **v0/v1 YCbCr 9 / 12 / 16-bit chroma-Frame round-trips (round 374).** The
  versions-0/1 inline-Parameters round-trip suite covered chroma only at
  8-bit (grayscale reached 10/16-bit). Three new
  `encode_frame_v0v1` → `decode_frame_v0v1` tests exercise the
  single-implied-Slice driver across the higher depths on the range coder:
  9-bit 4:2:0, 12-bit 4:4:4, and 16-bit 4:2:0. The 16-bit case is the first
  v0/v1 chroma-Frame round-trip driving the RFC 9043 §3.3.1 exception
  predictor across all three Planes (prior v0/v1 16-bit coverage was
  grayscale only).
- **RGB (RCT) exception-boundary + 16-bit general round-trips (round 374).**
  The v3 RGB / JPEG 2000 RCT round-trip suite exercised the §3.7.2.1
  exception only at 10-bit. Three new `encode_frame_rgb` → `decode_frame_rgb`
  tests bracket the exception window and cross out of it:
  - `..._9bit_exception` — the lower boundary of the §3.7.2.1 window
    (9..=15, extra_plane == 0); RCT coding width bits + 1 == 10.
  - `..._15bit_exception` — the upper boundary; RCT coding width 16, the
    widest still inside the exception range.
  - `..._16bit_general` — 16-bit RGB, *outside* the exception window, so the
    general Figure 6 / 7 RCT applies (per the §3.7.2.1 Background note,
    16-bit RCT carries no GBR/BGR Plane swap); RCT coding width 17, the
    widest the RGB path reaches. No prior test exercised 16-bit RGB.
- **Non-uniform slice-grid round-trips (§4.8 floor division) (round 374).**
  The prior multi-Slice tests all used evenly-divisible Frame dimensions, so
  every Slice was the same size. Two new tests drive a 3×2 Slice grid over a
  7×5 4:4:4 Frame where neither dimension divides evenly: the RFC 9043
  §4.8.2 / §4.8.3 (and §4.7.3 / §4.7.4) floor-division Slice positioning then
  yields Slice widths 2 / 2 / 3 and heights 2 / 3 — every Slice a different
  size. `range_yuv444_3x2_non_uniform_slice_grid` and its
  `golomb_..._3x2_...` mirror verify the floor-division carve matches
  bit-exactly between encode and decode on both the range and Golomb-Rice
  paths.
- **cargo-fuzz harness — decode / parse panic-freedom (round 368).** Added
  a `fuzz/` cargo-fuzz package with four libFuzzer targets driving
  attacker-controlled bytes through the crate's public parse / decode
  surface, plus a scheduled `.github/workflows/fuzz.yml` (daily, 30-minute
  budget split across the four targets via the org-level
  `crate-fuzz.yml@master` reusable workflow):
  - `config_record` — the §4.2 Configuration Record parse
    (`parse_configuration_record`) + the §4.1 Quantization Table Set
    cascade (`parse_quantization_table_sets`); every Parameter field and
    quant-table delta is the §3.8.1 range coder reading attacker bytes.
  - `decode_frame` — the v3 YCbCr (`decode_frame`) and RGB
    (`decode_frame_rgb`) pipelines, with the attacker controlling the
    Configuration Record bytes, the coded Frame bytes, and the frame
    dimensions (bounded to keep allocation finite); reaches the §4.6 /
    §4.7 / §4.9 header / content / footer walk, the §4.9.1 trailer chain,
    and the §3.3 / §3.5 / §3.7 / §3.8 reconstruction.
  - `decode_v0v1` — the versions-0/1 inline-Parameters decode
    (`decode_frame_v0v1`, RFC 9043 §4.4 prologue on one resumed
    range-coder pass).
  - `registry_decode` — the realistic end-to-end container surface:
    `CodecParameters` (§4.3.3 extradata + dims) plus a coded `Packet`
    driven through the registry-installed `oxideav_core::Decoder` trait
    (`send_packet` / `receive_frame`), covering both the v3 and the
    empty-extradata v0/v1 routing, with two Packets per input to reach the
    §3.8.1.3 / §3.8.2.5 cross-Frame coder-state carry.

  The contract under test is panic-freedom on every input shape (no
  out-of-bounds index, no debug-build arithmetic overflow, no `unwrap` on
  an attacker-forced `None` / `Err`); a malformed stream must surface a
  typed `Error`, never a panic. The initial campaign surfaced two
  index-out-of-bounds panics on the v0/v1 RGB decode path (both fixed —
  see _Fixed_ below); after the fixes, extended runs (millions of
  iterations across the four targets under AddressSanitizer) found no
  further panics. Clean-room: every target links only this crate's public
  API plus `oxideav-core`'s public surface; no external decoder, library,
  or oracle.
- **`Error::RgbRecordMissingChromaPlanes` (round 368).** New typed error
  surfaced when a Frame declares RGB (`colorspace_type == 1`) but its
  §4.2.6 `chroma_planes` flag is `0`, leaving the derived
  `primary_color_count` (§4.7.1) below the three R / G / B Planes the
  §3.7.1 inverse RCT requires. RGB always carries the three colour Planes
  (§4.2.5), so such a Record is non-conforming; the single-Frame RGB
  drivers now reject it with this error instead of indexing past the
  Plane vector.
- **Reference-fixture decode corpus: `v3-frame-mt` (round 361).** Added
  the 14th end-to-end bit-exact reference fixture: an 8-bit YUV 4:2:0
  256×192 stream with a 4×4 = 16-Slice grid and per-Slice CRC. Larger
  than the other multi-Slice fixtures (256×192 vs 128×96), it exercises
  the §5 slice-grid partition + §4.9.1 trailer chain at a non-trivial
  Slice geometry (each luma Slice 64×48). The reference-decode corpus now
  covers every fixture under `docs/video/ffv1/fixtures/` except the
  version-2 stream (`v2-multislice-2x2`), which FFV1 reserves as
  experimental and never emits in conforming bitstreams.
- **Reference-fixture decode corpus: `v3-rgb-bgr0` (round 361).** Added
  the 13th end-to-end bit-exact reference fixture: an 8-bit packed BGR0
  RGB stream (`colorspace_type == 1`, JPEG 2000 RCT, `transparency == 0`
  -> 3 Planes), exercising the RGB driver (`decode_frame_rgb`) with **no**
  alpha Plane — distinct from the four-Plane `v3-rgba`. The R/G/B Planes
  were unpacked from the reference bgr0 `expected.raw`.
- **Reference-fixture decode corpus: `v3-default` (2×2) +
  `v3-multislice-4x4` (round 361).** Extended
  `tests/reference_fixture_decode.rs` to 12 end-to-end bit-exact
  reference fixtures with the first two **multi-Slice** streams.
  `v3-default` is the canonical 4-Slice (2×2 grid) YUV 4:2:0 fixture and
  `v3-multislice-4x4` is the 16-Slice (4×4 grid) maximum-default-slices
  fixture; both validate the §5 slice-grid raster partition, the §4.9.1
  trailer chain, per-Slice §4.6 Header parse and §4.9 footer validation,
  and per-Slice §4.9.3 CRC end-to-end against the reference decoder's
  `expected.raw`. Frames + extradata extracted black-box from the
  fixtures' Matroska containers.
- **Reference-fixture decode corpus: `v3-grayscale` + `v3-yuv444p16`
  (round 361).** Extended `tests/reference_fixture_decode.rs` from 8 to
  10 end-to-end bit-exact reference fixtures. `v3-grayscale` is an 8-bit
  single-plane luma-only stream (`chroma_planes == 0`,
  `plane_count == 1`) that exercises the no-chroma path through the v3
  YCbCr driver; `v3-yuv444p16` is a 16-bit YUV 4:4:4 stream
  (`bits_per_raw_sample == 16`, no subsampling) that exercises the
  full-precision 16-bit sample path. Both fixture frames + extradata
  were extracted black-box from the `docs/video/ffv1/fixtures/*/input.mkv`
  Matroska containers and decode bit-exact against the reference
  decoder's `expected.raw`.
- **§3.8.2 Golomb-Rice run-mode decode loop + Sentinel-mode handoff
  (round 357).** Rewrote the §3.8.2.2 run mode as a per-Line state
  machine governed solely by the absolute context being 0: a context-0
  Sample enters run mode, the §3.8.2.2.1 prefix selects a long run
  (`1 << log2_run[run_index]` zeros, `run_index` growing when the run
  fits the remaining Line width per the `x + run_count <= w` guard) or a
  short run (residual zero count then a level-coded break, §3.8.2.4.1).
  The switch from the range-coded versions-0/1 inline Parameters to the
  byte-aligned Golomb-Rice Slice Content now uses **Sentinel mode**
  (RFC 9043 §3.8.1.1.1): `RangeEncoder::terminate_sentinel` writes a
  discarded state-129 terminator and `RangeDecoder::terminate_sentinel`
  recovers the byte boundary. Together these let the reference
  `v0-yuv420-golomb-rice` fixture decode bit-exact (new test in
  `tests/reference_fixture_decode.rs`), the eighth fixture in the
  reference corpus and the first Golomb-Rice reference stream.

### Changed

- A nonzero Sample Difference at the first Sample of a Golomb-Rice run
  region is now **encodable** as a §3.8.2.2.1 zero-length short run
  (immediate level-coded break, §3.8.2.4.1). The `encode_frame*` Golomb
  path no longer returns `Error::RunModeFirstPixelNonZero` (the variant
  is retained for API compatibility but is never produced); the
  affected round-trip tests now assert bit-exact reconstruction instead
  of the former error.

- **Reference-fixture decode corpus** (round 350) — seven new end-to-end
  tests (`tests/reference_fixture_decode.rs`) decode each fixture's coded
  Frame and assert the reconstructed Planes are bit-exact against the
  reference decoder's `expected.raw`: `v3-flat-color` (8-bit YUV 4:2:0
  low-entropy run/zero range paths), `v3-yuv422p10` (10-bit 4:2:2),
  `v3-yuv420p12` (12-bit 4:2:0), `v3-rgba` (`transparency == 1`,
  four-Plane RGB + alpha over the JPEG 2000 RCT), `v3-context-1` (the
  large `-context 1` Quantization Table Set), `v0-yuv420-rangecoder`
  (FFV1 version 0 inline Parameters), and `v1-single-slice` (version 1,
  128×96). Fixture Frames are extracted black-box from each `input.mkv` /
  `input.avi` (Matroska / AVI parsing is independent of the FFV1
  bitstream) and inlined with their `expected.raw` in
  `tests/data/reference_fixtures.rs`. This lifts reference-validated
  decode coverage from 4 v3 fixtures to 11 fixtures spanning 8/10/12-bit
  depths, 4:2:0 / 4:2:2 chroma, RGBA, the large context model, and the
  v0/v1 single-stream range coder.

- **FFV1 versions 0 / 1 `coder_type == 2` (custom state-transition table)
  single-stream decode + encode** (round 347) — closes the last v0/v1 coder
  gap. RFC 9043 §4.4 / §4.2.4 / §3.8.1.6: unlike v3 — where the §4.2
  Parameters live in a separate §4.3 Configuration Record range-coder pass
  and each §4.5 Slice opens a fresh range coder already seeded with the
  §3.8.1.6 custom table — v0/v1 carries the Parameters inline and shares one
  continuous range-coder pass with the §4.7 Slice Content. The §4.2.4
  `state_transition_delta` (and the keyframe boolean + Parameters that
  precede them) are therefore read with the §3.8.1.5 *default* table (a
  custom table cannot apply to the symbols that define it); once the deltas
  are known the live coder swaps onto the §3.8.1.6 custom table — at the
  Parameters → Slice-Content boundary — via the new
  `RangeDecoder::set_one_state` / `RangeEncoder::set_one_state`, which
  replace the active transition table in place while preserving the
  byte-window state (`low` / `range` / cursor). A non-keyframe (no inline
  Parameters) is seeded with the custom table from the start, exactly as the
  v3 driver seeds each Slice. The resolution is pinned against the v3
  driver's own table handling in `frame.rs`, not guessed. `decode_frame_v0v1`
  / `encode_frame_v0v1` and their `_inter` siblings now cover `coder_type ==
  2` on both colour layouts (YCbCr plane-major and RGB line-major RCT,
  including alpha and the §3.3.1 16-bit alternate median predictor), and a
  multi-Frame v0/v1 `coder_type == 2` stream decodes bit-exact end-to-end
  through the framework `Decoder` trait. Test count: 592 total (+13: 11 in
  `v0v1_roundtrip`, 1 framework-trait multi-frame in `registry_v0v1_decoder`,
  1 mid-pass-swap unit test in `range_coder`; −1 obsolete
  `encode_rejects_coder_type_2`).

- **FFV1 versions 0 / 1 single-Slice YCbCr decode + encode** (round 342) —
  the first end-to-end v0/v1 path, closing the README's "Versions 0 / 1
  are not yet decodable end-to-end" limitation for the YCbCr / range-coder
  case. New `frame_v0v1` module: `decode_frame_v0v1` reconstructs a v0/v1
  keyframe Frame (the §4.4 inline §4.2 Parameters + the single §4.1
  Quantization Table Set + the implied single §4.7 Slice Content, with no
  §4.6 Slice Header, no §4.9 Slice Footer, and no §4.9.1 trailer chain);
  `decode_frame_v0v1_inter` decodes a v0/v1 non-keyframe (which inherits
  the keyframe's inline Parameters). `encode_frame_v0v1` /
  `encode_frame_v0v1_inter` are the symmetric write side. New
  `quant_table::parse_v0v1_frame_prologue` parses the §4.4 `keyframe` +
  §4.2 Parameters + single §4.1 cascade off one resumed range-coder pass
  and hands back the live decoder positioned at the Slice Content;
  `config_encode::encode_v0v1_frame_prologue` is its inverse. Self
  round-trip is bit-exact lossless across versions 0 and 1, gray /
  YUV420 / YUV444 / YUVA420 (alpha), 8 / 10 / 16-bit depths (16-bit
  exercises the §3.3.1 alternate median predictor), and degenerate 1×1 /
  1×N / N×1 rasters. The §4.7 RGB / line-major (`colorspace_type == 1`)
  and `coder_type == 2` (custom state-transition table, whose
  mid-Parameters table-ordering is unpinned by the RFC for the
  single-stream v0/v1 case) v0/v1 paths surface explicit errors and are
  tracked as follow-ups.

- **FFV1 versions 0 / 1 decode through the framework `Decoder` trait**
  (round 342) — wires the v0/v1 path into the registry `Decoder`. RFC 9043
  §4.3.3 / §4.4: v0/v1 carry no Configuration Record (their §4.2
  Parameters are inline in each keyframe Frame), so a v0/v1 container
  supplies `CodecParameters` with frame dimensions but empty `extradata`.
  `make_decoder` now detects that shape and builds a v0/v1-mode decoder
  that parses the §4.4 prologue off the first keyframe packet (caching the
  record + single §4.1 Quantization Table Set) and decodes via
  `decode_frame_v0v1`; later non-keyframes inherit the cached config via
  `decode_frame_v0v1_inter`. `reset` invalidates the cache (the next
  packet after a seek must re-supply the inline Parameters). A v0/v1
  non-keyframe arriving before any keyframe is a diagnosable error. Test
  count: 580 total, was 577 (+3 in `tests/registry_v0v1_decoder.rs`:
  keyframe-only, keyframe-then-non-keyframe, and the non-keyframe-first
  error path, all through the `Decoder` trait surface).

- **FFV1 versions 0 / 1 RGB / RCT (`colorspace_type == 1`) decode +
  encode** (round 342) — extends the v0/v1 path to the §4.7 line-major
  JPEG 2000 RCT layout. `decode_frame_v0v1` / `decode_frame_v0v1_inter`
  now reconstruct an RGB v0/v1 Frame (the §4.7 `for y { for p { Line(p,
  y) } }` interleave keeping each Plane's entropy + border state alive,
  then the §3.7.1 inverse RCT), reusing the v3 RGB per-Plane line-state +
  inverse-RCT machinery over the implied single Slice. `encode_frame_v0v1`
  / `encode_frame_v0v1_inter` emit the symmetric RGB write side by reusing
  the v3 RGB per-Slice content encoders with the §4.4 inline-Parameters
  prologue substituted for the §4.6 Slice Header and the §4.9 footer
  dropped. RGB v0/v1 round-trips bit-exactly for `coder_type == 1` (range
  default) across 8 / 10-bit and with / without the alpha plane; the
  `coder_type == 0` (Golomb) RGB encode path is wired but, because the
  forward RCT lifts the Cb / Cr corner to the §3.7.2 offset, the §3.8.2.2
  `RunModeFirstPixelNonZero` constraint makes a synthetic round-trip
  fixture hard to construct (decode accepts it). Test count: 577 total,
  was 574 (RGB range round-trips replace the now-obsolete RGB-rejection
  test).

- **FFV1 versions 0 / 1 Golomb-Rice (`coder_type == 0`) encode** (round
  342) — extends `encode_frame_v0v1` / `encode_frame_v0v1_inter` to the
  §3.8.2 Golomb-Rice path: the §4.4 prologue is range-coded and
  byte-aligned, then the implied single §4.7 Slice Content is appended as
  a Golomb-Rice bit stream (reusing the v3 single-Slice content encoder
  over the whole-Frame implied Slice). The decode side already accepted
  `coder_type == 0`, so v0/v1 Golomb now round-trips bit-exactly (gray /
  YUV420 / YUVA420, keyframe + non-keyframe), inheriting the documented
  §3.8.2.2 `RunModeFirstPixelNonZero` limitation shared with the v3
  Golomb encoder. Test count: 574 total, was 551 (+23 in
  `tests/v0v1_roundtrip.rs`).

- **Framework `Encoder` emits inter-Frame (non-keyframe) streams** (round
  338) — closes the README's "the framework encoder always emits
  keyframes" limitation, completing the encode-side end-to-end inter-Frame
  milestone through the `oxideav_core::Encoder` trait surface. The
  registry encoder (`Ffv1FrameEncoder`) now holds an `Ffv1EncodeCarry` and
  a first-Frame flag: the first `send_frame` of a stream emits a §4.4
  keyframe (re-initialising all §3.8.1.3 / §3.8.2.5 per-context coder
  state) and every later `send_frame` emits a non-keyframe whose
  per-context state continues from the previous Frame via
  `encode_frame_with_carry`, mirroring the framework `Decoder`'s existing
  cross-packet carry. The `Packet`'s keyframe flag now reflects the actual
  §4.4 value (was hard-coded `true`). The §4.2.17 `intra` flag (Table 14:
  `intra == 1` → "keyframe MUST be 1") is honoured: an `intra == 1`
  Configuration Record forces every coded Frame to a keyframe, so the
  encoder never produces a stream the decoder's §4.2.17 intra gate would
  reject. With both halves wired, a multi-Frame inter FFV1 stream now
  round-trips end-to-end through the trait surface (registry encoder →
  registry decoder), reconstructing every Frame bit-exactly. Test count:
  551 total, was 549 (+2 in `tests/registry_encoder.rs`:
  `multi_frame_inter_stream_round_trips_through_trait_surface` — a
  three-Frame keyframe-then-non-keyframes round-trip that also proves the
  non-keyframe payload differs from a standalone keyframe encode of the
  same Frame — and `intra_one_configuration_forces_keyframe_only_output`,
  building an `intra == 1` Configuration Record off the v3-default parse
  and asserting all coded Frames are keyframes).

- **Non-keyframe coder-state carry on the YCbCr Golomb-Rice *encode*
  path + unified `encode_frame_with_carry` dispatcher** (round 338) —
  closes the last missing coder on the write side of the RFC 9043
  §3.8.1.3 (range) / §3.8.2.5 (Golomb-Rice VLC) inter-Frame coder-state
  carry. The decode side already carried both coders across non-keyframes
  for both colorspaces (`decode_frame_with_carry` /
  `decode_frame_rgb_with_carry`, threaded by `Ffv1DecodeSession` and the
  framework `Decoder`); the encode side carried the YCbCr **range** coder
  (`encode_frame_range_coder_with_carry`) and both RGB coders
  (`encode_frame_rgb_with_carry`) but **not** the YCbCr Golomb-Rice path —
  so a `coder_type == 0` YCbCr multi-Frame inter stream could be decoded
  but not produced. New public
  `encode_frame_golomb_rice_with_carry(..., keyframe: bool, &mut
  Option<Ffv1EncodeCarry>)`: on a non-keyframe each Slice's
  per-§4.6.6-slot VLC window (`drift` / `error_sum` / `bias` / `count`)
  resumes from the previous Frame's matching Slice's `LineDecoderState`
  instead of the §3.8.2.5 keyframe-init values; on a keyframe every slot
  starts fresh and the carry is ignored; on return the `golomb_slices`
  channel of `Ffv1EncodeCarry` holds this Frame's end-of-Frame snapshot.
  The §3.8.2.2.1 run-mode triple stays per-Plane (reset unconditionally)
  and is not carried, exactly as on the read side.
  `encode_frame_golomb_rice` now delegates with `keyframe = true` + a
  `None` carry, byte-for-byte unchanged. A new public
  `encode_frame_with_carry(..., keyframe, carry)` dispatches on §4.2.5
  `colorspace_type` + §4.2.3 `coder_type` to the three carry-aware
  drivers, mirroring `encode_frame`'s keyframe-only dispatch — the
  symmetric write-side mirror of `decode_frame_with_carry` /
  `decode_frame_rgb_with_carry`. Test count: 549 total, was 545 (+4 in
  `tests/nonkeyframe_carry.rs`: a Golomb single-Slice keyframe →
  non-keyframe round-trip, a load-bearing discriminator (stateless
  `decode_frame` of the same non-keyframe bytes produces different
  pixels), a per-Slice 2×2-grid carry across three Frames, and a
  keyframe-parity check that `encode_frame_with_carry(.., true, ..)` is
  byte-identical to the legacy `encode_frame`). The multi-context
  `golomb_ramp_qts` keeps the absolute context off 0 so the §3.8.2.2
  run mode never engages, isolating the carry from the orthogonal
  run-mode-first-pixel encoder limitation.

- **§4.4 in-Frame `Parameters()` parse for FFV1 versions 0 and 1**
  (round 333) — a new public `parse_v0v1_frame_parameters(&[u8]) ->
  Result<Ffv1ConfigurationRecord, Error>` reads the §4.2 Parameters
  block that versions 0 and 1 carry **inline in the Frame** (RFC 9043
  §4.4: `Frame( NumBytes ) { keyframe; if (keyframe &&
  !ConfigurationRecordIsPresent) Parameters(); ... }`) rather than in a
  container Configuration Record. It consumes the §4.4 `keyframe`
  boolean (own initial state 128), then walks the §4.2 Figure 28
  `Parameters()` fields whose `if (version >= 3)` guards are false for
  v0/v1, inferring the v3-only fields (`micro_version = None`,
  `quant_table_set_count` §4.2.13 = 1) and reporting the §4.5/§4.6
  single implied-Slice geometry (`num_h_slices == num_v_slices ==
  Some(1)`, since §4.5 emits a `SliceHeader()` only for `version >= 3`).
  It rejects a §4.4 non-keyframe (`Error::NonKeyframeHasNoInFrame
  Parameters` — a v0/v1 non-keyframe inherits the prior keyframe's
  config and carries no inline Parameters) and a misrouted `version >=
  3` Frame (`Error::InFrameParametersForbiddenForVersion`). This is the
  first piece of the v0/v1 decode path, which previously hard-rejected
  all v0/v1 streams. Two supporting changes to the shared §4.2
  `parse_parameters` walker: (1) it now infers `micro_version = None`
  for v0/v1 instead of erroring — the §4.2.1 "no Configuration Record
  for v0/v1" advisory moved into `parse_configuration_record` (the
  Record-present context) where it belongs; (2) it now honours the
  Figure 28 `if (version >= 1)` guard on `bits_per_raw_sample`, so a
  version-0 Frame (which omits the field — §4.2.7 implies 8) no longer
  consumes a phantom symbol and desyncs the rest of the Parameters
  walk. Test count: 545 total, was 541 (+4 in
  `tests/v0v1_frame_parameters.rs`: a v1 YCbCr-8-bit and a v0
  RGB-no-bits-field round-trip built symbol-for-symbol from the §4.2
  field order, plus the non-keyframe and v3-rejection guards).

- **§4.2-derived `PixelFormat` surfaced on the framework encoder's
  `output_params`** (round 327) — a new public
  `pixel_format_for(&Ffv1ConfigurationRecord) -> Option<PixelFormat>`
  reads the RFC 9043 §4.2 Parameters that fix a Frame's plane geometry
  (§4.2.5 `colorspace_type`, §4.2.6 `chroma_planes`, §4.2.7
  `bits_per_raw_sample`, §4.2.8 / §4.2.9 `log2_*_chroma_subsample`,
  §4.2.10 `extra_plane`) and maps them to the exact `oxideav_core`
  `PixelFormat` the decoder's plane packing produces — `Gray8` /
  `Gray10Le` / `Gray12Le` / `Gray16Le` for luma-only YCbCr; `Yuv420P` /
  `Yuv422P` / `Yuv444P` / `Yuv411P` and their 10/12-bit `*Le` siblings
  for chroma YCbCr keyed on the §4.2.8 / §4.2.9 subsample shift pair;
  and `Yuva420P` for the one 8-bit 4:2:0-plus-alpha shape the framework
  enum carries. It returns `None` — rather than a near-miss variant —
  for every §4.2 layout with no exact, plane-order-and-packing-faithful
  framework variant: RGB / JPEG 2000 RCT (the §3.7.1 driver emits **R,
  G, B** plane order, which the framework's G, B, R `Gbrp*Le` family
  does not match; §4.2.5 fixes RGB at 4:4:4), 16-bit YUV (the
  v3-yuv444p16 corpus shape), any subsampled-plus-alpha YUV, planar
  gray-plus-alpha, and reserved subsample shifts. `make_encoder` now
  populates `output_params.pixel_format` from this helper when an exact
  variant exists (overriding a caller's pre-set guess) and leaves any
  caller-supplied value untouched otherwise, so a downstream muxer
  reading `Encoder::output_params` gets the correct format label. The
  helper is re-exported from the crate root and closes the previously
  dangling `[pixel_format_for]` doc reference in `registry.rs`. Test
  count: 541 total, was 534 (+7: 5 lib unit tests in
  `src/registry.rs::tests` covering the grayscale / YUV / extra-plane /
  RGB / unrepresented-combo mapping table, plus 2 integration tests in
  `tests/registry_encoder.rs` —
  `encoder_surfaces_section_4_2_pixel_format` (direct derivation off the
  parsed v3-default record + the encoder's override-the-guess
  precedence) and a `register_installs_encoder` assertion that
  `output_params.pixel_format == Some(Yuv420P)`).

- **Framework encoder registration** (round 324) — `register` now also
  installs an `ffv1` `oxideav_core::Encoder` alongside the decoder, so
  the registry advertises both directions (`with_encode()` capability +
  `encoder(make_encoder)` factory). The encoder reuses the same
  `CodecParameters` the decoder consumes (the §4.2 Configuration Record
  in `extradata`, validated against the §4.3.2 Record CRC, plus the
  frame `width` / `height`), and — closing the README's headline
  encode-side follow-up — **derives the §4.6 Slice Header grid from the
  Configuration Record's §4.2.11 / §4.2.12 `num_h_slices ×
  num_v_slices`**, one Slice per raster cell (`slice_width ==
  slice_height == 1`), defaulting to a single 1×1 Slice when the v3-only
  fields are absent. `Encoder::send_frame` converts an incoming
  `Frame::Video` to the internal `DecodedFrame` (the inverse of the
  decoder's plane packing — one byte per Sample at ≤ 8-bit depth, two
  little-endian bytes otherwise, with chroma planes at the §4.2.8 /
  §4.2.9 subsampled dimensions), then emits one coded keyframe per Frame
  (FFV1 is intra-only) via `encode_frame`, propagating the input PTS.
  `output_params` carries the §4.2 Configuration Record back out for a
  downstream muxer. A new `tests/registry_encoder.rs` encodes the
  `v3-default` reference pixels through the trait surface (registry
  factory → `send_frame` → `receive_packet`) across the fixture's 2×2
  slice grid and round-trips them back through the `Decoder` bit-exactly,
  plus drain-contract / unconfigured / wrong-plane-count guards. The
  historical direct `encode_frame*` API is retained unchanged.

- **Framework decoder registration** (round 317) — `register` now
  installs an `ffv1` `oxideav_core::Decoder` into the runtime registry
  instead of being a no-op. The decoder reads the §4.2 Configuration
  Record from `CodecParameters::extradata` (RFC 9043 §4.3.3), validates
  the §4.3.2 Record CRC, parses the §4.1 Quantization Table Set
  cascade, then routes each `Packet` on the §4.2.5 `colorspace_type` to
  the plane-major (YCbCr) or line-major (RGB) frame driver, threading
  the §3.8.1.3 / §3.8.2.5 per-context coder state across non-keyframes
  and emitting an `oxideav_core::VideoFrame` (one byte per Sample at
  ≤ 8-bit depth, two little-endian bytes otherwise). Registration
  claims the two RFC 9043 §4.3.3 container tags — the AVI / VfW FourCC
  `FFV1` (§4.3.3.1) and the Matroska Codec ID `V_FFV1` (§4.3.3.4). The
  §4.2.16 per-Slice CRC footer presence is derived as `ec != 0`
  (Table 13). A new `tests/registry_decoder.rs` decodes the
  `v3-default` fixture through the trait surface (registry factory →
  `send_packet` → `receive_frame`) bit-exactly against `expected.raw`.
  The historical direct API (`decode_frame*` / `encode_frame*` /
  per-stage parsers) is retained unchanged (dual-API convention).
  `CODEC_ID_STR` / `register_codecs` are newly exported.

- **Run-mode "first Sample" encodability gate on the Golomb-Rice encode
  path** (round 309) — RFC 9043 §3.8.2.2 / §3.8.2.4.1. The §3.8.2.2 run
  state machine begins every run with a `0` Sample Difference (Phase 3
  emits a long-run "1" — Sample Difference 0 — or a short run that
  returns 0 for the current Sample and level-codes the break on the
  *next* Sample), so a non-zero `sample_difference` at the **first**
  Sample of a run region (absolute context 0 with `l == t == tl`,
  immediately after a run-state reset) has no Golomb-Rice encoding —
  there is no preceding zero-run Sample to carry the short-run prefix.
  The shared §4.8 content encoder (`encode_line`) previously hit a
  `debug_assert!` here — a no-op in release builds, where it silently
  emitted a corrupt stream. It now returns the new typed
  `Error::RunModeFirstPixelNonZero { x }` (the Sample index within the
  Line), propagated through both Golomb-Rice frame drivers
  (`encode_frame` YCbCr / plane-major and `encode_frame_rgb` RGB /
  line-major on `coder_type == 0`). `encode_line`'s signature changes
  from `()` to `Result<(), Error>`. Such a pixel field never appears in
  a stream a conforming FFV1 decoder produced (every run begins with a
  `0` Sample Difference); it can only arise from caller pixel data the
  active Quantization Table Set routes into run mode at the first run
  Sample with a non-zero residual, and the range coder (`coder_type ∈
  {1, 2}`, no run mode) carries the same pixels without restriction. 5
  new tests (523 total, was 518): 1 lib unit test
  (`encode_line_rejects_non_zero_first_run_sample` in
  `src/sample_diff.rs::tests`) plus 4 integration tests in
  `tests/run_mode_first_pixel.rs` — the gate fires on a single Slice and
  on a 2×2 grid (top-left Slice), the surgical companion
  (zero-then-non-zero run = Case B) still round-trips bit-exactly, and
  the exact rejected frame round-trips on the range coder.

### Fixed

- **v0/v1 RGB decode no longer panics on a non-conforming Plane count
  (round 368, fuzz finding).** The `fuzz/decode_v0v1` harness surfaced two
  index-out-of-bounds panics on the versions-0/1 RGB
  (`colorspace_type == 1`) decode path driven by malformed §4.4
  inline-Parameters Records:
  - A Record selecting RGB with `chroma_planes == 0` derived
    `primary_color_count < 3`, so the §3.7.1 inverse-RCT blit
    (`apply_inverse_rct_and_blit`) indexed `plane_states[1]` /
    `plane_states[2]` past the end of the (too-short) Plane vector. Fixed
    by rejecting the Record up front with the new
    `Error::RgbRecordMissingChromaPlanes` (RGB always carries three R/G/B
    Planes, §4.2.5), in both the v0/v1 and the v3 RGB drivers.
  - A Record decoding the chroma / alpha Planes at a *different* size than
    luma let the blit's `src = y * y_plane.width + x` index run off a
    smaller `cb_plane` / `cr_plane` / `alpha_plane` buffer. Fixed by
    bounding the blit to the common region of all participating Planes and
    indexing each Plane with its own width (a no-op for conforming 4:4:4
    RGB, where every Plane shares luma's dimensions).

  Both inputs are pinned as regression tests in
  `tests/fuzz_regressions.rs`.
- **§3.5 context routing on the Golomb-Rice *encode* path now matches the
  production decoder** (round 308) — the shared §4.8 Golomb-Rice content
  encoder (`encode_line`, used by both the YCbCr / plane-major
  `encode_frame` and the RGB / line-major `encode_frame_rgb` on
  `coder_type == 0`) previously evaluated the RFC 9043 §3.5 context (and
  the §3.8.2.2 run-region predicate) from the per-pixel *Sample
  Difference* values it pre-filled into its `current_row` buffer, whereas
  the production decoder (`PlaneReconstructor::reconstruct_row`) evaluates
  them from the reconstructed *Sample* neighbours (`l = cur[idx-1]`,
  `ll = cur[idx-2]`). The two agreed only for a *single-context*
  Quantization Table Set, where the routed context is constant regardless
  of the neighbour values; any genuinely multi-context table desynced the
  §3.5 routing between encode and decode and the frame failed to
  round-trip. `encode_line` now pre-fills `current_row` with this Line's
  reconstructed *Samples* (`pred = median(l, t, tl)`, `Sample =
  reconstruct_sample(pred, diff, bits)`), so both the per-pixel context
  and the run-mode lookahead read Samples — matching the decoder
  bit-for-bit. Single-context streams (every shipped fixture + every
  prior round-trip test) are byte-for-byte unchanged. 6 new multi-context
  round-trip tests (3 in `tests/chroma_encode_frame.rs`, 3 in
  `tests/rgb_encode_frame.rs`) cover the YCbCr and RGB Golomb paths with a
  genuine multi-context table on both single-slice and 2×2-grid frames
  (plus range-coder parity on the same table). Known remaining limitation:
  a non-zero Sample Difference at the *first* run-region pixel after a
  state reset (context 0 with `l == t == tl`) is not representable under
  the per-call §3.8.2.2.1 run state machine — the multi-context tests use
  tables whose absolute context is never 0, isolating this §3.5 fix from
  that orthogonal run-mode-encoder follow-up.

### Added

- **Non-keyframe coder-state carry on the RGB / line-major *encode* path**
  (round 301) — the symmetric inverse of the round-294
  `decode_frame_rgb_with_carry`, completing the RFC 9043 §3.8.1.3 (range
  coder) / §3.8.2.5 (Golomb-Rice VLC) inter-Frame coder-state carry on
  the RGB path. New `encode_frame_rgb_with_carry(..., keyframe: bool,
  &mut Option<Ffv1EncodeCarry>)`: on a non-keyframe each Slice's
  per-§4.6.6-slot coder window resumes from the previous Frame's matching
  Slice; on a keyframe every slot starts fresh and the carry is ignored;
  on return the channel holds this Frame's end-of-Frame snapshot.
  `Ffv1EncodeCarry` grew a `golomb_slices` channel alongside its existing
  range-coder channel so the RGB driver populates the range channel for
  `coder_type ∈ {1, 2}` and the Golomb-Rice channel for `coder_type ==
  0`, mirroring the read-side `Ffv1FrameCarry`. The §3.8.2.2.1 run-mode
  triple stays per-Plane (reset unconditionally) and is not carried.
  `encode_frame_rgb` delegates to the new variant with `keyframe = true`
  + a `None` carry, byte-for-byte unchanged. With both halves present a
  synthetic multi-Frame RGB non-keyframe stream now round-trips
  end-to-end (4 new integration tests in
  `tests/rgb_nonkeyframe_carry.rs`).

- **Non-keyframe coder-state carry on the RGB / line-major decode path**
  (round 294) — extends the round-286 §3.8.1.3 / §3.8.2.5 inter-Frame
  carry to the `colorspace_type == 1` (RGB / JPEG 2000 RCT) driver. New
  `decode_frame_rgb_with_carry(..., &mut Option<Ffv1FrameCarry>)` is the
  RGB analogue of `decode_frame_with_carry`: on a non-keyframe it resumes
  each Slice's per-§4.6.6-slot range / Golomb-Rice window from the
  supplied `Ffv1FrameCarry` instead of the `128`-initialised window, and
  writes the Frame's end-of-Frame snapshot back. The §3.8.2.2.1 run-mode
  triple stays per-Plane (reset unconditionally, keyframe or not) — only
  the slot-level per-context window carries. `Ffv1DecodeSession` now
  threads its carry through the RGB branch too (previously RGB
  re-initialised coder state every Frame, correct only for all-keyframe
  streams). `decode_frame_rgb_with_options` is unchanged behaviourally
  (delegates with a `None` carry). The write-side RGB carry remains a
  follow-up (the in-tree RGB encoder writes `keyframe = 1`).

- **Non-keyframe coder-state carry on the decode side** (round 286) —
  RFC 9043 §3.8.1.3 (range coder) and §3.8.2.5 (Golomb-Rice VLC) state
  that the per-context coder state is re-initialised "When the keyframe
  value ... is 1"; the negation is that a non-keyframe (`keyframe == 0`)
  **continues** the per-context state from the value it held at the end
  of the previous Frame's matching Slice. §5 third paragraph keeps the
  Slice geometry stable across Frames, so the carry is indexed by forward
  Slice index. New public surface:

  - `Ffv1FrameCarry` — opaque per-Slice / per-§4.6.6-slot end-of-Frame
    coder-state snapshot (read side). `decode_frame_with_carry(...,
    &mut Option<Ffv1FrameCarry>)` is the inter-Frame YCbCr / plane-major
    driver: on a non-keyframe it resumes each Slice's per-slot windows
    from the supplied carry instead of the §3.8.1.3 / §3.8.2.5
    `128`-initialised window, and writes the current Frame's snapshot
    back on return. `decode_frame` / `decode_frame_with_options` delegate
    with no carry (the historical standalone-keyframe behaviour,
    unchanged).
  - `Ffv1EncodeCarry` + `encode_frame_range_coder_with_carry(...,
    keyframe: bool, &mut Option<Ffv1EncodeCarry>)` — the symmetric
    write-side mirror: the §4.4 `keyframe` boolean is now caller-chosen
    and the per-context state resumes across non-keyframes, so a genuine
    multi-Frame non-keyframe stream can be produced and round-tripped.
    `encode_frame_range_coder` delegates with `keyframe = true`, no
    carry.
  - `Ffv1DecodeSession` now owns an `Ffv1FrameCarry` across the coded
    Frame sequence: `decode_next_frame` threads it through the YCbCr
    driver, committing the snapshot only after both stream-scope
    conformance gates pass (a gate failure leaves the carry, tracker, and
    Frame counter untouched). The RGB / line-major driver does not yet
    thread the carry — a follow-up; it stays correct for all-keyframe RGB
    streams.

  The §3.8.2.2.1 run-mode triple (`run_index` / `run_mode` /
  `run_count`) is **not** part of the carry — it is reset per Plane /
  Slice unconditionally, keyframe or not. 4 new integration tests in
  `tests/nonkeyframe_carry.rs` (505 total, was 501): a single-Slice
  keyframe → non-keyframe round-trip that reconstructs both Frames
  bit-exactly; a discriminating test proving the carry is load-bearing
  (the same non-keyframe bytes decoded through the stateless
  `decode_frame` produce *different* pixels); per-Slice carry on a 2×2
  grid across three Frames; and a mid-stream keyframe that re-initialises
  state (the carry is ignored when `keyframe == 1`).

- **Stateful multi-Frame decode session `Ffv1DecodeSession`**
  (round 283) — closes the round-279 follow-up ("a stateful
  multi-Frame decode session object that owns the tracker
  (driver-level stitch)") and completes the RFC 9043 §5
  third-paragraph arc (rounds 268 → 274 → 279 → 283). The new
  `src/decode_session.rs` session owns the per-stream decode inputs
  (Configuration Record, §4.1 Quantization Table Sets,
  `FramePixelDimensions`, `ec`, `DecodeOptions`) plus the cross-Frame
  state no stateless single-Frame driver can hold (the round-268
  [`SliceGeometryStabilityTracker`] and a coded-order Frame counter).
  Public surface:

  - `Ffv1DecodeSession::new(cr, quant_table_sets, frame_dims, ec)` /
    `with_options(..., options)` — one session per coded stream;
    `new` defaults to `DecodeOptions::strict()`.
  - `decode_next_frame(&mut self, frame_bytes) -> Result<DecodedFrame,
    Error>` — routes on the §4.2.5 `colorspace_type` to
    `decode_frame_with_options` (YCbCr / plane-major) or
    `decode_frame_rgb_with_options` (RGB / line-major), the same
    dispatch [`encode_frame`] performs on the write side, then applies
    two stream-scope conformance gates: the **§4.2.17 `intra` gate**
    (Table 14 — `intra == 1` means "keyframe MUST be 1 (keyframes
    only)"; "Inferred to be 0 if not present", so `None` /
    `Some(false)` carry no constraint) surfacing the new
    `Error::NonKeyframeInIntraStream { frame_index }`, and the **§5
    third-paragraph geometry-stability gate** via
    `tracker.observe_frame(decoded.keyframe, &decoded.slice_headers)`
    — exactly the two `DecodedFrame` fields rounds 274 / 279
    surfaced.
  - `observe_decoded_frame(&mut self, &DecodedFrame)` — the same two
    gates on replayed / externally-decoded Frames without
    re-decoding.
  - Accessors `frames_observed()`, `has_previous_frame()`,
    `options()`, `configuration_record()`.

  Both stream-scope gates are structural wire-conformance gates and
  fire under `strict()` and `lenient()` alike, mirroring the
  policy-independence of the in-driver §5 raster-coverage /
  max-slice-size gates; the §4.9.2 / §4.9.3 policies flow through to
  the routed driver unchanged. A Frame that fails either gate never
  advances the session (tracker reference + counter untouched), so a
  violating Frame cannot become the §5 reference for its successor.
  Test count: 501 total, was 492 (+9: 4 lib unit tests in
  `src/decode_session.rs::tests` —
  `intra_one_rejects_non_keyframe_and_leaves_state_untouched`,
  `intra_zero_or_absent_admits_non_keyframes`,
  `geometry_gate_fires_and_reference_survives_violation`,
  `accessors_report_session_state`; plus 5 integration tests in
  `tests/decode_session.rs` —
  `session_ycbcr_multi_frame_decode_is_bit_exact_against_stateless_driver`
  (three-Frame coded stream, per-Frame bit-exact against
  `decode_frame`), `session_rgb_routes_line_major_driver`,
  `session_options_flow_through_to_the_routed_driver` (a §4.9.2
  `Uncorrectable` rewrite aborts a strict session without advancing
  it and decodes bit-exactly on a lenient one),
  `session_intra_gate_rejects_replayed_non_keyframe`, and
  `session_geometry_gate_rejects_replayed_resplit_non_keyframe`
  pinning `SliceGeometryUnstable` on forward index 1). Remaining
  follow-ups on this arc: true non-keyframe *decoding* (carrying
  §3.8.1.3 / §3.8.2.5 per-context coder state across Frames when
  `keyframe == 0` — the session is the object that would own that
  state) and the `Decoder` trait / registry stitch.

- **§4.6 Slice Headers surfaced on `DecodedFrame`** (round 279) —
  closes the round-274 follow-up ("a public decode→Slice-Header
  surface — the *other* argument the tracker needs — remains a
  follow-up"). Both frame-level decode drivers
  (`decode_frame_with_options` YCbCr / plane-major in `src/frame.rs`,
  `decode_frame_rgb_with_options` RGB / line-major in
  `src/rgb_reconstruct.rs`) already parse every §4.6 Slice Header in
  the round-260 pass-1 preamble that feeds the §5 second-paragraph
  raster-coverage gate, then dropped the parsed headers after
  validation. The new public
  `DecodedFrame::slice_headers: Vec<Ffv1SliceHeader>` field carries
  them instead — in forward Slice-index order (slice 0 first, the
  §4.9.1 trailer-chain order both drivers walk) at zero additional
  parse cost. Paired with the round-274 `keyframe` field this
  completes the decode-output side of the RFC 9043 §5 third-paragraph
  rule ("For each Frame with a keyframe value of 0, each Slice MUST
  have the same value of slice_x, slice_y, slice_width, and
  slice_height as a Slice in the previous Frame"): a caller walks
  Frames in coded order and feeds
  `tracker.observe_frame(decoded.keyframe, &decoded.slice_headers)`
  into a [`SliceGeometryStabilityTracker`] with both arguments read
  straight off the decode — no out-of-band header re-parse. A
  zero-Slice Frame carries an empty vector; on the encode side the
  field is ignored (the encoders take Slice geometry from the
  caller-supplied header slice, never from `DecodedFrame`). Test
  count: 492 total, was 487 (+5: 4 end-to-end tests in the new
  `tests/decoded_slice_headers.rs` —
  `ycbcr_decode_surfaces_forward_ordered_slice_headers`
  (2×2-grid encode → decode, field-for-field §4.6 header round-trip
  in forward order),
  `rgb_decode_surfaces_forward_ordered_slice_headers` (RGB /
  line-major mirror),
  `ycbcr_strict_and_lenient_surface_identical_headers` (the pass-1
  parse runs before the §4.9.2 / §4.9.3 gates and is
  policy-independent), and
  `decoded_pair_drives_section5_stability_tracker_end_to_end` (a
  two-stream tracker drive built purely from decode outputs that
  pins `Error::SliceGeometryUnstable` on forward index 1 — the
  `(1, 0, 1, 1)` Slice — when a 2×2 partition follows a
  single-Slice Frame); plus
  `decode_v3_default_surfaces_trace_ordered_slice_headers` in
  `tests/frame_driver.rs`, asserting the v3-default fixture surfaces
  the reference trace's four-Slice 2×2 geometry — raster quadruples
  `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)`, each 1×1,
  `quant_table_set_index_count == 2` — in trailer-chain forward
  order). The `src/frame.rs::tests` tracker unit test now reads both
  `observe_frame` arguments off the `DecodedFrame` fields. Remaining
  follow-up on this arc: a stateful multi-Frame decode session
  object that owns the tracker (driver-level stitch).

- **§4.4 `keyframe` value surfaced on `DecodedFrame`** (round 274) —
  closes the round-268 follow-up ("both drivers currently consume the
  boolean off Slice 0 and discard it"). RFC 9043 §4.4 opens each Frame
  with a single range-coded `keyframe` boolean (its own state, initial
  value 128) at the very start of the first Slice's range-coded
  region. Both frame-level decode drivers
  (`decode_frame_with_options` YCbCr / plane-major in `src/frame.rs`,
  `decode_frame_rgb_with_options` RGB / line-major in
  `src/rgb_reconstruct.rs`) now capture that value into the new public
  `DecodedFrame::keyframe: bool` field instead of binding it to
  `_keyframe` and dropping it. A zero-Slice Frame (degenerate, no §4.4
  boolean) defaults to `true` — vacuously self-contained. Surfacing
  the value lets a caller drive the §5 third-paragraph multi-Frame
  [`SliceGeometryStabilityTracker::observe_frame`] (which takes
  `keyframe` as its first argument) across a coded Frame sequence and
  enforce the §4.2.18 `intra` keyframe constraint — the
  single-Frame/stateless drivers cannot run that check themselves.
  Test count: 487 total, was 485 (+2 lib unit tests in
  `src/frame.rs::tests`:
  `decoded_frame_carries_keyframe_field`,
  `decoded_frame_keyframe_drives_section5_stability_tracker`; plus
  three existing `tests/frame_encode_dispatch.rs` round-trip tests now
  also assert `decoded.keyframe` end-to-end across the Golomb-Rice,
  range-coder, and RGB / line-major paths). The encoder writes
  `keyframe = true` for every Frame (intra-only codec), so the
  round-trips confirm the surfacing wiring against a real
  encode→decode. A public decode→Slice-Header surface (the *other*
  argument the tracker needs) plus driver-level tracker wiring across
  a Frame sequence remain a follow-up.

- **§5 "Restrictions" non-keyframe Slice-geometry stability
  validator** (round 268) — lands the last of the three RFC 9043 §5
  restrictions, queued by rounds 249 / 257 as "requires multi-Frame
  state". The §5 third paragraph states: "For each Frame with a
  keyframe value of 0, each Slice MUST have the same value of
  slice_x, slice_y, slice_width, and slice_height as a Slice in the
  previous Frame." Two new public surfaces in `src/slice_content.rs`:

  - `validate_slice_geometry_stability(previous_headers,
    current_headers) -> Result<(), Error>` — pure structural
    primitive over the Slice Headers of two consecutive Frames. For
    each current-Frame Slice in forward (trailer-chain) order, the
    §4.6.1-§4.6.4 quadruple `(slice_x, slice_y, slice_width,
    slice_height)` must equal the quadruple of *some* Slice of the
    previous Frame ("as a Slice in the previous Frame" is an
    existence requirement — a permuted forward order across Frames
    conforms, and no other §4.6 field participates: a Slice may
    change `quant_table_set_index` / `picture_structure` / SAR
    Frame-to-Frame). The first unmatched Slice surfaces the new
    `Error::SliceGeometryUnstable { slice_index, slice_x, slice_y,
    slice_width, slice_height }` so the diagnostic is deterministic.
  - `SliceGeometryStabilityTracker` — stateful multi-Frame driver
    for the same rule (`observe_frame(keyframe, headers)` once per
    Frame in coded order). A keyframe records its geometry as the
    new previous-Frame reference without any check (§5 restricts
    only Frames "with a keyframe value of 0"; §3.8.1.3 / §3.8.2.5
    re-initialise all coder state there); a non-keyframe validates
    against the *immediately preceding* Frame — not the last
    keyframe — and becomes the reference on success. A non-keyframe
    observed before any Frame validates against the empty set (no
    previous Frame exists whose Slices could match); a violating
    Frame never becomes the reference for its successor.

  The frame-level decode drivers are single-Frame and stateless, so
  driver-level wiring is queued as a follow-up: it first needs the
  §4.4 `keyframe` value surfaced off the decode (both drivers
  currently consume the boolean off Slice 0 and discard it). Test
  count: 485 total, was 473 (+12 lib unit tests in
  `src/slice_content.rs::tests`:
  `geometry_stability_identical_partition_passes`,
  `geometry_stability_permuted_order_passes`,
  `geometry_stability_ignores_non_geometry_fields`,
  `geometry_stability_changed_split_diagnoses_first_unmatched_slice`,
  `geometry_stability_first_unmatched_index_is_deterministic`,
  `geometry_stability_empty_current_is_vacuously_ok`,
  `geometry_stability_empty_previous_rejects_first_slice`,
  `geometry_tracker_keyframe_opens_stream_without_check`,
  `geometry_tracker_non_keyframe_first_frame_with_slices_rejects`,
  `geometry_tracker_stable_sequence_tracks_immediately_previous_frame`,
  `geometry_tracker_error_leaves_previous_reference_untouched`,
  `geometry_tracker_default_matches_new`).

- **§5 "Restrictions" per-Frame Slice raster-coverage gate on the
  frame-level decode drivers** (round 260) — closes the round-257
  follow-up by folding
  [`validate_slice_raster_coverage`] into both
  `decode_frame_with_options` (YCbCr / plane-major) and
  `decode_frame_rgb_with_options` (RGB / line-major) as a two-pass
  collect-then-validate preamble. The new `pub(crate)` helper
  `collect_slice_headers_for_raster_validation` mirrors the pass-2
  inline preamble (per Slice it strips the §4.9 trailer, seeds a
  `RangeDecoder` over the body bytes, consumes the §4.4 `keyframe`
  boolean on Slice 0, and calls `parse_slice_header_from_decoder`)
  and returns the forward-ordered headers
  [`validate_slice_raster_coverage`] consumes. Pass-2 reconstructs
  the per-Slice `RangeDecoder` cursors over the same body bytes,
  so the §3.8.1 byte-positional decode state matches the pass-1
  Header reads bit-for-bit. Behaviour: a §5 partition violation —
  two Slices addressing the same raster cell
  (`Error::SliceRasterOverlap`) or any raster cell left unclaimed
  (`Error::SliceRasterUncovered`) — aborts the Frame decode
  **before any per-Slice pixel reconstruction starts**, so a
  conforming §5 violation cannot corrupt the per-Plane output
  buffers. The §5 gate is structural and orthogonal to the §4.9.3
  CRC / §4.9.2 `error_status` policies; both
  `DecodeOptions::strict()` and `DecodeOptions::lenient()` surface
  the violation, mirroring the round-249 max-slice-size gate's
  policy-independence contract. The per-Slice raster-bounds check
  (`slice_x + slice_width <= num_h_slices` etc.) and the v0 / v1
  absent-grid branch reuse the existing surfaces
  (`Error::SliceRasterOutOfRange`, `Error::SliceRequiresVersion3`)
  so the §5 walk and `compute_slice_content` agree on malformed-
  Slice diagnostics. Test count: 473 total, was 463 (+10
  integration tests in `tests/section_5_raster_coverage.rs` for the
  positive 1×1 + 2×2 partitions on both drivers, the deterministic
  overlap and gap diagnostics on a 2×1 grid, and the lenient-still-
  aborts policy-independence assertions). Three pre-existing
  `tests/frame_driver.rs` tests
  (`decode_v3_grayscale_single_slice_produces_one_plane`,
  `decode_v3_grayscale_is_bit_exact_against_expected_raw`,
  `decode_v3_rgb_bgr0_runs_line_major_pipeline`,
  `decode_v3_rgb_bgr0_slice0_is_bit_exact_against_expected_raw`)
  fed single-slice fragments from multi-slice fixtures and were
  updated to clone the Configuration Record onto a §5-conforming
  1×1 grid — the slice's wire fields (`slice_x = 0`,
  `slice_width = 1`, etc.) are identical on a 2×2 and 1×1 grid so
  the §3.8.1 range-coder Header state evolves identically, and the
  pre-260 64×48 framing simply placed the slice's pixel rectangle
  at the top-left quadrant of a larger output buffer (the bit-exact
  assertions now walk the slice's intrinsic 32×24 region directly).

- **§5 "Restrictions" per-Frame Slice raster-coverage validator**
  (round 257) — wires RFC 9043 §5 second paragraph ("For each Frame,
  each position in the Slice raster MUST be filled by one and only
  one Slice of the Frame (no missing Slice position and no Slice
  overlapping)") as a pure structural primitive. The new
  `validate_slice_raster_coverage(headers, cr) -> Result<(), Error>`
  takes the forward-ordered Slice Headers parsed off a single
  Frame plus the surrounding Configuration Record and proves the
  union of every Slice's `slice_width × slice_height` raster
  footprint exactly tiles the `num_h_slices × num_v_slices` grid:
  every cell claimed by at least one Slice (no gaps) and at most
  one Slice (no overlaps). Two new error variants surface the two
  ways §5 can fail: `Error::SliceRasterOverlap { x, y,
  first_slice_index, second_slice_index }` and
  `Error::SliceRasterUncovered { x, y }`. The overlap detector
  surfaces the lowest forward-index pair on the first colliding
  paint; the gap detector surfaces the first uncovered cell in
  row-major scan order so the diagnostic is deterministic. Per-Slice
  raster bounds (`slice_x + slice_width <= num_h_slices`) and the
  v0 / v1 absent-grid branch reuse the existing surfaces
  (`Error::SliceRasterOutOfRange`, `Error::SliceRequiresVersion3`)
  for consistency with `compute_slice_content` and
  `validate_slice_max_size_restriction`. The validator is pure-
  structural — no range coder, no pixel buffer, no frame bytes
  touched — and orthogonal to the round-249 §5 per-Slice size cap;
  a conforming Frame satisfies both. 12 new lib unit tests cover
  the canonical 1×1 / 2×2-unit / 4×4-quartile tilings, an
  irregular 3×3 5-Slice partition, the no-Slices empty-Frame gap
  case, partial / full overlap cases with pinned diagnostic
  ordering (overlap surfaces before gap), the off-raster
  `SliceRasterOutOfRange` passthrough, and the v0 / v1
  `SliceRequiresVersion3` branch. Frame-driver wiring (two-pass
  collect-then-validate over the slice loop) is queued for a
  follow-up round.

- **§5 "Restrictions" max-slice-size gate on the frame decoders**
  (round 249) — wires RFC 9043 §5's per-Slice raster-footprint cap
  ("starting with version 3 and if `frame_pixel_width *
  frame_pixel_height` is more than 101376, `slice_width *
  slice_height` MUST be less or equal to `num_h_slices *
  num_v_slices / 4`") into the frame-level decode drivers. The cap
  is the four-way-parallel-decoding floor: above the 101376-pixel
  CIF trigger every Slice fits in at most one quarter of the raster
  so four threads can each take one Slice in lockstep. New surfaces:

  - `validate_slice_max_size_restriction(header, cr, frame_dims) ->
    Result<(), Error>` in `src/slice_content.rs` — pure structural
    validator. Returns `Ok(())` on v0 / v1 (silent — no slice grid
    in the Configuration Record) and on v3 below or at the §5
    trigger; surfaces `Error::SliceMaxSizeExceeded { slice_width,
    slice_height, num_h_slices, num_v_slices, frame_pixel_width,
    frame_pixel_height }` on a v3 violation above the trigger.
  - `SECTION_5_MAX_SLICE_AREA_THRESHOLD: u64 = 101_376` — exported
    constant that documents the CIF trigger and ties the validator's
    arithmetic to the spec value.
  - `Error::SliceMaxSizeExceeded` — new structural error variant
    carrying the offending Slice's raster footprint, the
    Configuration Record's `num_h_slices` / `num_v_slices`, and the
    Frame's pixel dimensions so a caller can log the §5 violation.

  Wiring: both frame-level decode drivers
  (`frame.rs::decode_frame_with_options` on the YCbCr / plane-major
  path, `rgb_reconstruct.rs::decode_frame_rgb_with_options` on the
  RGB / line-major path) invoke the validator immediately after
  `parse_slice_header_from_decoder` and before
  `compute_slice_content`, so a §5 violation aborts the Frame
  before any per-Plane reconstructor touches the offending Slice's
  body. The legacy `decode_frame` / `decode_frame_rgb` entry points
  delegate to the options-aware variants, so they pick up the §5
  gate automatically.

  Behaviour: the §5 gate is structural and independent of the
  §4.9.2 `slice_error_status_policy` / §4.9.3 `slice_crc_policy`
  fields — `lenient()` decodes still abort on `SliceMaxSizeExceeded`
  because the §5 cap is a wire-conformance error, not a corruption
  signal a partial-recovery decode should swallow.

  Tests: 8 new unit tests in `src/slice_content.rs::tests`
  (`section_5_threshold_matches_cif`,
  `section_5_below_threshold_admits_any_footprint`,
  `section_5_at_threshold_admits_any_footprint`,
  `section_5_above_threshold_caps_at_quarter_raster_2x2`,
  `section_5_above_threshold_4x4_cap_is_four`,
  `section_5_integer_division_uses_floor`,
  `section_5_rejects_v0_v1_with_slice_requires_version3`,
  `section_5_rejects_zero_frame_dimensions`) plus 6 end-to-end
  integration tests in `tests/section_5_max_slice_size.rs`
  (`frame_area_constants_line_up_with_spec`,
  `ycbcr_small_frame_below_threshold_admits_full_raster_slice`,
  `ycbcr_above_threshold_violating_raster_aborts_decode`,
  `ycbcr_above_threshold_admissible_raster_round_trips`,
  `rgb_above_threshold_violating_raster_aborts_decode`,
  `rgb_above_threshold_admissible_raster_round_trips`) driving the
  full encode → decode pipeline on both drivers. Test count: 451
  total, was 437 (+14: 8 lib + 6 integration).

  Other §5 restrictions (no-gap / no-overlap raster coverage per
  Frame; non-keyframe Slice-stability invariant across Frames)
  require multi-Slice / multi-Frame state and are queued for a
  follow-up round.

- **§4.9.2 `error_status` Table 16 policy gate on the frame
  decoders** (round 244) — extends `DecodeOptions` with a second
  independent integrity gate that mirrors the round-238
  `slice_crc_policy` over the §4.9.2 `error_status` field. The new
  `SliceErrorStatusPolicy` enum has the same Reject / Accept shape:
  `Reject` (default) aborts the frame decode via the new
  `Error::SliceErrorStatus { slice_index, status }` whenever a
  per-Slice footer declares the Table 16 `Uncorrectable` (`2`)
  status; `Accept` is the opt-in lenient mode that lets the per-
  Slice pixel reconstruction run best-effort. `Correctable` (`1`)
  and `Reserved` (`>=3`) are not rejection targets per the policy
  doc — `Reject` lets them through, because the §4.9.3 CRC residue
  is the stronger fixity signal for those wire values. The gate
  is independent of `slice_crc_policy`, so a caller can mix the two
  (e.g. `SliceCrcPolicy::Accept` + `SliceErrorStatusPolicy::Reject`
  to tolerate residue mismatches but still abort on the
  encoder-declared `Uncorrectable`); `DecodeOptions::strict()` /
  `lenient()` set both gates the same way for convenience.

  The wiring lives in `frame.rs::decode_frame_with_options` (the
  YCbCr / plane-major driver) and `rgb_reconstruct.rs::decode_frame_rgb_with_options`
  (the RGB / line-major driver) — both now surface the parsed
  `Ffv1SliceFooter` (instead of binding it to `_footer`) and check
  the §4.9.2 typed status before any per-Slice pixel reconstruction
  touches the body. The legacy `decode_frame` / `decode_frame_rgb`
  entry points are unchanged: they delegate to the options-aware
  functions with the default `DecodeOptions::strict()`, so every
  prior caller picks up the new `Uncorrectable` abort path
  automatically (every shipped fixture and every prior in-tree
  encode writes `NoError`, so the strict default does not change
  any existing test's observable behaviour).

  Eight new end-to-end integration tests in
  `tests/decode_options_error_status_gate.rs` cover the matrix:
  `ycbcr_gate_reject_default_aborts_on_uncorrectable_status`
  (Reject path surfaces `Error::SliceErrorStatus { slice_index: 0,
  status: 2 }`, with default / strict / legacy entry points all
  agreeing); `ycbcr_gate_accept_partial_decode_returns_bit_exact_frame`
  (Accept path returns a bit-exact `DecodedFrame` — body bytes are
  intact, only the §4.9.2 byte + re-solved parity change, so the
  per-Sample reconstruction reproduces the original input);
  `ycbcr_gate_clean_status_all_policies_match_legacy_bit_exact`
  (regression — clean `NoError` still produces bit-exact decode
  under every policy); `ycbcr_gate_correctable_status_passes_under_reject`
  (`Correctable == 1` is not a rejection target under Reject);
  `ycbcr_gate_reserved_status_passes_under_reject` (reserved-range
  byte `0xAB` passes Reject); `ycbcr_gate_independent_of_crc_policy_field`
  (mixed-policy invariant — Reject on the §4.9.2 gate fires even
  with `SliceCrcPolicy::Accept`, and `SliceErrorStatusPolicy::Accept`
  passes when residue is zero by construction);
  `rgb_gate_reject_aborts_accept_passes_with_bit_exact_decode` and
  `rgb_gate_correctable_status_passes_under_reject` mirror the
  YCbCr Reject / Accept / Correctable assertions on the RGB /
  line-major driver. The fabricator
  `rewrite_single_slice_error_status` rebuilds the §4.9 footer with
  `encode_slice_footer_with_raw_status` — the same solver the
  encoder uses for every clean Slice — so the §4.9.3 CRC residue
  stays zero by construction and the test isolates the §4.9.2 gate
  under test.

- **§4.2.15 `initial_state_delta` triple-loop surfaced on the
  Configuration Record + encoder learns the `states_coded == 1`
  branch** (round 241) — closes the round-236 follow-up that
  consumed the §4.2.15 deltas off the wire without storing them.
  The new `Ffv1ConfigurationRecord::initial_state_delta` field is
  an `Option<Vec<Option<Vec<[i32; INITIAL_STATE_DELTA_K]>>>>` (with
  `INITIAL_STATE_DELTA_K == 32` from §4.2 `CONTEXT_SIZE`): `None`
  when every Quantization Table Set wrote `states_coded == 0` on
  the wire (the §4.2.14 default — "initial states ... assumed to be
  all 128"), `Some(per_set)` when at least one set carries the
  loop. Each `per_set[i]` is `None` for sets whose
  `states_coded == 0` and `Some(deltas)` for sets whose
  `states_coded == 1`, with `deltas.len() == context_count[i]` and
  each inner `[i32; 32]` carrying the 32 signed `sr` symbols
  indexed by `k` (Figures 29 / 30: `initial_state[i][j][k] =
  (pred + initial_state_delta[i][j][k]) & 255`). The parser
  (`quant_table::parse_parameters_tail`) now populates the field
  instead of discarding the symbols; the encoder
  (`config_encode::encode_parameters_tail`) emits
  `states_coded == 1` + the matching triple-loop iff the field is
  populated with the right shape, falling back to the
  `states_coded == 0` default otherwise. A new
  `Error::InitialStateDeltaShapeMismatch { set_index,
  expected_context_count, actual_context_count }` rejects
  caller-supplied per-set vectors whose length disagrees with the
  §4.1 cascade's `context_count[i]` up-front so the encoder never
  produces a desynchronised wire stream.

  Seven new tests (429 total, was 422) in
  `src/config_encode.rs::tests`:
  `round_trip_initial_state_delta_zero_row_single_set` (one set
  with `states_coded == 1` and an all-zero row, exercising the
  loop emission against a benign payload);
  `round_trip_initial_state_delta_nontrivial_row_single_set`
  (mixed-sign small magnitudes on the 32 `sr` symbols);
  `round_trip_initial_state_delta_mixed_sets_states_coded`
  (two-set record where set 0 codes a loop and set 1 stays at the
  §4.2.14 default — the per-set `Option` distinguishes them on
  round-trip and ec/intra still land correctly after the per-set
  tail);
  `round_trip_initial_state_delta_all_unset_stays_none` (sanity
  inverse — the explicit `None` traverses encode + parse cleanly);
  `initial_state_delta_shape_mismatch_rejected` (the
  context-count guard fires before any wire bytes are emitted);
  `round_trip_initial_state_delta_preserves_signed_extremes`
  (`i32::MIN` / `i32::MAX` survive the `sr` round-trip per the
  symbol-level tests); and
  `round_trip_initial_state_delta_encoder_is_deterministic`
  (back-to-back encodes of the populated field produce
  byte-identical blobs). Each round-trip asserts
  `validate_configuration_record_crc(&blob) == Ok(())` so the
  §4.3.2 parity word is verified to be solved against the new wire
  footprint.
- **§4.9.3 per-Slice CRC validation gate on the frame decoders**
  (round 238) — adds a `DecodeOptions { slice_crc_policy }` struct
  plus options-aware `decode_frame_with_options` /
  `decode_frame_rgb_with_options` entry points. The new
  `SliceCrcPolicy::Reject` (default) preserves the historical
  abort-on-mismatch behaviour every prior round shipped;
  `SliceCrcPolicy::Accept` is the opt-in partial-recovery mode that
  tolerates a non-zero §4.9.3 whole-Slice CRC residue so the
  per-Slice reconstructors run best-effort on damaged input. The
  underlying `parse_slice_footer_with_options(buf, ec, policy)` exposes
  the same gate at the parser level, and `Ffv1SliceFooter` gains a
  `crc_residue: Option<u32>` field that always carries the §4.9.3
  residue on `ec == 1` (`Some(0)` on a clean Slice; `Some(non_zero)`
  when `Accept` was used). Structural failures
  (`TruncatedSliceFooter`, `SliceSizeOutOfRange`) stay policy-
  independent. Three new `src/slice_footer.rs::tests` unit tests
  (`options_clean_slice_both_policies_agree_with_legacy`,
  `options_corrupted_body_accept_returns_residue_reject_errors`,
  `options_ec0_policy_irrelevant_no_residue`) and four new
  `tests/decode_options_crc_gate.rs` end-to-end integration tests
  (`ycbcr_gate_reject_default_aborts_on_crc_failure`,
  `ycbcr_gate_accept_partial_decode_returns_structurally_valid_frame`,
  `ycbcr_gate_clean_slice_both_policies_match_legacy_bit_exact`,
  `rgb_gate_reject_aborts_accept_partial_decodes`) cover the
  encode → corrupt → decode round-trip on both the YCbCr / plane-
  major and RGB / line-major drivers, asserting that the lenient
  path returns a structurally valid `DecodedFrame` (right plane
  count + dimensions, every Sample in the §3.8 modular range) and
  that the strict path's `Error::SliceCrcMismatch` carries both the
  non-zero residue and the on-wire §4.9.3 parity for diagnostics.
  The legacy `decode_frame` / `decode_frame_rgb` / `parse_slice_footer`
  entry points are unchanged — they delegate to the new
  options-aware functions with `SliceCrcPolicy::Reject`, so every
  prior caller keeps the same behaviour bit-for-bit.
- **§4.2.14-§4.2.17 Parameters tail on the encoder + parser**
  (round 236) — `encode_configuration_record_with_quant_tables` now
  emits the `version >= 3` post-cascade block of Figure 28
  (`states_coded` per Set, `ec`, `intra`) on the same resumed range
  coder + shared 32-slot Parameters state buffer the prefix and
  cascade share. The companion `parse_quantization_table_sets`
  consumes the same tail: the `states_coded` `br` per Set drives an
  optional §4.2.15 `initial_state_delta[i][j][k]` triple-loop
  (`context_count[i] * CONTEXT_SIZE` signed deltas), then `ec` (`ur`)
  and `intra` (`ur`) close the block. Two new fields appear on
  `Ffv1ConfigurationRecord`: `ec: Option<u32>` and
  `intra: Option<bool>` — both `None` when the field is absent on the
  wire (versions 0/1, or callers that only invoked
  `parse_configuration_record` rather than the cascade-aware parser).
  The encoder always writes `states_coded = 0` (the §4.2.14 default —
  "initial states ... assumed to be all 128") for every Set, omitting
  the §4.2.15 triple-loop; on the parse side `states_coded == 1` is
  honoured (the deltas are consumed off the wire) but the per-byte
  delta values are not yet stored on the record. Five new tests in
  `src/config_encode.rs::tests`
  (`round_trip_tail_default_ec_intra`,
  `round_trip_tail_ec_one_intra_true`,
  `round_trip_tail_none_defaults_to_zero`,
  `round_trip_tail_multi_set_states_coded_zero`,
  `round_trip_tail_with_state_transition_delta`) exercise the new
  symbol-for-symbol round-trip across single-Set / multi-Set /
  `None`-defaults / `coder_type == 2`. The pre-existing
  `tests/fixture_config_encode.rs` structural round-trip across the
  four v3 corpus extradata blobs continues to pass — the encoder
  faithfully re-emits whatever `ec` / `intra` the parser produced
  from the corpus, so the parse-encode-parse triangle closes
  regardless of which §4.2.14 sub-path the original FFV1 encoder
  exercised.
- **§4.6.6 per-slot VLC sharing on the Golomb-Rice RGB driver**
  (round 227) — extends the §4.6.6 per-slot state-buffer rule to
  the `coder_type == 0` branch of [`encode_frame_rgb`] /
  [`decode_frame_rgb`], closing the only remaining slot-keying
  gap the round-220 row called out as a follow-up. On the
  Golomb-Rice path the per-context entropy state has two
  components with distinct lifetimes — the per-context VLC
  window (`drift` / `error_sum` / `bias` / `count`) keyed by
  §4.6.6 slot (Planes routed to the same slot share one window)
  and the §3.8.2.2.1 run-mode triple
  (`run_index` / `run_mode` / `run_count`) keyed per Plane. The
  driver now allocates one
  [`crate::sample_diff::LineDecoderState`] per slot (encoder) /
  [`crate::reconstruct::PlaneEntropyState`] per slot (decoder),
  lazily on first touch so the §3.8.2.5 keyframe-init contract
  holds, and a saved-run-triple snapshot per Plane that is
  loaded into the slot state at the start of every row encode /
  decode and saved back at the end. Two new `pub(crate)`
  accessors on `PlaneEntropyState` (`save_run_state` /
  `load_run_state`) encapsulate the swap; the encoder reaches
  `LineDecoderState`'s already-`pub` run-mode fields directly.
  No public surface changes. Encoder and decoder mutate the
  per-slot VLC window in lockstep so all 18 prior
  `tests/rgb_encode_frame.rs` round-trip tests stay green
  byte-for-byte. 5 new tests in `tests/rgb_encode_frame.rs`
  (`_golomb_rice_high_entropy_chroma_planes`,
  `_golomb_rice_distinct_per_slot_qts_indexes`,
  `_golomb_rice_extra_plane_distinct_slot`,
  `_golomb_rice_run_mode_dominates_per_plane`,
  `_golomb_rice_2x2_slice_grid_with_alpha`) exercise the
  slot-key distinctions the prior per-Plane allocation could
  not — distinct per-slot `context_count`, alpha-Plane slot
  independence, per-Plane run-triple reset under flat content,
  and 2×2 slice grid + extra-plane combination. Lib tests: 272
  (unchanged); integration: +5 (405 → 410 total). The §4.6.6
  per-slot state-buffer rule is now uniform across all four
  driver branches (`coder_type ∈ {0, 1, 2}` × `colorspace_type
  ∈ {YCbCr, RGB}`).

### Fixed

- **RGB / line-major Cr-Plane reconstruction divergence on the
  range-coded path** (round 220) — RFC 9043 §4.6.6 ("Quantization
  Table Set index ... **and the initial states**") keys the per-
  context entropy state by the §4.6.6 *slot*, so two Planes that
  map to the same slot (Cb + Cr on every `chroma_planes` Slice)
  must share one persistent `RangePlaneState` across the §4.7
  line-major interleave. Round 214 applied that rule to
  [`decode_frame`] / [`encode_frame_range_coder`]; round 220 extends
  it to the RGB / `colorspace_type == 1` line-major driver
  ([`decode_frame_rgb`]) and its [`encode_frame_rgb`] mirror on the
  range-coded (`coder_type ∈ {1, 2}`) path. Prior to this round
  `decode_frame_rgb` allocated a fresh `RangePlaneState` per Plane;
  the second Plane to touch the chroma slot (Cr) silently re-
  keyframe-initialised its per-context window instead of continuing
  Cb's evolution, observable as the documented Cr-Plane divergence
  on `v3-rgb-bgr0` slice 0 (Y + Cb decoded bit-exactly against the
  reference, Cr did not). The fix lifts the per-context state out
  of `PlaneLineState` into a per-`quant_table_set_index_count`-slot
  `Vec<Option<RangePlaneState>>` owned by the driver, lazily filled
  on first touch of each slot so the §3.8.1.3 keyframe-init
  contract still holds; the symmetric change lands on the
  encoder side (per-slot `RangePlaneEncoderState`). The Golomb-Rice
  (`coder_type == 0`) RGB path keeps its per-Plane
  `PlaneEntropyState` for now — line-major slot-sharing on the
  Golomb path additionally needs the §3.8.2.2.1 run-mode triple
  split to remain per-Plane while the per-context VLC fields share
  across the slot; the only shipped v3 RGB fixture
  (`v3-rgb-bgr0`) is `coder_type == 1`, so the split is queued as a
  follow-up. New regression test
  `decode_v3_rgb_bgr0_slice0_is_bit_exact_against_expected_raw` in
  `tests/frame_driver.rs` decodes the v3-rgb-bgr0 slice-0 byte
  payload (the top-left 32×24 region of the 64×48 frame) through
  [`decode_frame_rgb`] and asserts every R / G / B Sample matches
  the reference decoder's `expected.raw` byte-for-byte (2 304
  entries — 768 per Plane); the test was red before the fix on the
  Cr-derived colour Plane and green after. All 18 existing
  `tests/rgb_encode_frame.rs` encoder→decoder round-trip tests
  stay green because the encoder and decoder shift in lockstep.
  Lib tests: 272 (unchanged); integration: +1 test
  (404 → 405 total).
- **YCbCr Cr-plane reconstruction divergence on
  `quant_table_set_index = [0, 0]` layouts** (round 214) — RFC 9043
  §4.6.6 reads "`quant_table_set_index` indicates the Quantization
  Table Set index to select the Quantization Table Set **and the
  initial states** for the Slice Content", so the per-context state
  buffer the §3.8.1 / §3.8.2 entropy coder reads is keyed by the
  §4.6.6 *slot* (§4.6.5 `quant_table_set_index_count = 1 +
  (chroma_planes || version <= 3 ? 1 : 0) + (extra_plane ? 1 : 0)` —
  luma slot, chroma slot, optional extra-plane slot), not by the
  Plane and not by the resolved Quantization Table Set. The chroma
  slot is shared by Cb and Cr in the §4.7 plane-then-line traversal:
  both Planes feed the same persistent per-context state, with only
  the §3.8.2.2.1 run-mode triple resetting per Plane.
  [`decode_frame`] previously allocated a fresh per-context state
  inside every `PlaneReconstructor::reconstruct_plane` /
  `RangePlaneReconstructor::reconstruct_plane` call, so on
  `v3-default`'s `[0, 0]` layout (both slots resolve to Quantization
  Table Set 0) Cr — the second Plane to touch the chroma slot — read
  its symbols against a wrongly keyframe-initialised state instead
  of continuing Cb's evolution. Observable as 3066 / 3072 Cr Samples
  diverging from the reference `expected.raw`; Y and Cb decoded
  bit-exactly throughout. Fixed by adding `pub(crate)`
  `_with_state` variants on the three per-Plane reconstruct /
  encode entry points
  (`RangePlaneReconstructor::reconstruct_plane_with_state`,
  `PlaneReconstructor::reconstruct_plane_with_state`,
  `RangePlaneEncoder::encode_plane_with_state`) and threading one
  `&mut state` per `quant_table_set_index_count` slot through the
  per-Plane loops in [`decode_frame`] /
  [`encode_frame_range_coder`] / `encode_slice_content_golomb`. The
  legacy `reconstruct_plane` / `encode_plane` entry points are
  preserved as one-liner shims for external callers. New regression
  test `decode_v3_default_is_bit_exact_against_expected_raw` in
  `tests/frame_driver.rs` decodes the full v3-default frame and
  asserts every Sample of every reconstructed Plane (Y 128×96 + Cb
  64×48 + Cr 64×48 = 18 432 entries) matches the inlined
  `docs/video/ffv1/fixtures/v3-default/expected.raw` byte-for-byte;
  the test was red before the fix (Y / Cb green, all 3072 Cr Samples
  diverging) and green after. All 19 existing test groups (404
  tests, was 386) stay green — encoder→decoder round-trips on every
  prior path keep passing because the encoder side shifts in
  lockstep with the decoder.
- **Multi-Plane Golomb-Rice (`coder_type == 0`) `decode_frame` bit-stream
  cursor** (round 208) — Plane 1 (Cb), Plane 2 (Cr), and Plane 3 (alpha)
  silently re-read Plane 0's bytes from offset zero on
  `chroma_planes == true` YCbCr Slices. The per-Plane decode loop in
  [`decode_frame`] was re-constructing a fresh [`BitReader`] from
  `body[rc.position()..]` *inside* the loop, dropping the bit-stream
  cursor between Planes; the §3.8.2.2.1 per-Plane state reset
  (`PlaneEntropyState::new(...)` + `reset_run_state()`) only resets the
  VLC contexts + run-mode state, NOT the bit-stream cursor. The fix is
  one shared [`BitReader`] constructed *outside* the per-Plane loop on
  the `coder_type == 0` arm. The bug was dormant because every prior
  Golomb-Rice round-trip test targeted either a single-Plane grayscale
  frame OR the RGB / line-major driver (which runs its own per-row
  bit-reader plumbing). 14 new round-trip tests in
  `tests/chroma_encode_frame.rs` cover the
  `(coder_type ∈ {0, 1, 2}) × (4:4:4 / 4:2:2 / 4:2:0) × (extra_plane ∈
  {true, false})` matrix that `encode_frame` reaches; the Golomb-Rice
  cases (`golomb_yuv*`) were red before the fix and green after.

### Added

- **YCbCr chroma-planes `encode_frame` → `decode_frame` round-trip
  coverage** (round 208) — `tests/chroma_encode_frame.rs` (14 tests).
  Covers every `(coder_type ∈ {0, 1, 2}) × chroma-subsample-shape ×
  extra_plane` shape the public [`encode_frame`] dispatcher routes to:
  4:4:4 single-slice 8-bit Golomb-Rice + range-coder; 4:2:2 single-slice
  8-bit on both; 4:2:0 single-slice 8-bit on both; 4:2:0 2×2 slice grid
  8-bit on both; 4:4:4 + extra (alpha) Plane 8-bit on both; 4:2:0
  single-slice 10-bit on the range coder; the all-zero
  `state_transition_delta` `coder_type == 2` byte-equality with
  `coder_type == 1` on 4:2:0; `ec == 0` 3-byte footer on a 4:2:0 frame;
  distinct per-Plane-category Quantization Table Sets routed via
  `quant_table_set_index = [0, 1]` on 4:2:0. Each test asserts every
  Plane's `samples` matches the input bit-exactly after the round-trip;
  a wrong per-Plane width/height, wrong chroma origin
  (`plane_origin`), wrong quant-set routing (`quant_index_slot`), or
  wrong bit-stream cursor handoff on either side surfaces as a
  Plane-divergence assertion.

- **§4.2 Parameters + §4.1 Quantization Table Set cascade encoder**
  (round 202) — `encode_configuration_record_with_quant_tables` is the
  symmetric inverse of `parse_quantization_table_sets`: given an
  `Ffv1ConfigurationRecord` plus a `&[QuantizationTableSet]` it emits
  the §4.3 extradata byte stream and appends a §4.3.2
  `configuration_record_crc_parity` word solved so the whole-blob
  §4.9.3 CRC residue is zero. The encoder mirrors the parser's walk
  symbol-for-symbol: §4.2 Parameters prefix (version → micro_version →
  coder_type → optional §4.2.4 `state_transition_delta[1..=255]` `sr`
  loop when `coder_type > 1` → colorspace → bits_per_raw_sample →
  chroma_planes → log2_*_chroma_subsample → extra_plane → v3
  num_h_slices_minus_1 / num_v_slices_minus_1 / quant_table_set_count)
  followed by the §4.1 cascade (`quant_table_set_count` Sets, each
  five sub-tables; per-table state-window reset to 128 mirroring the
  decoder's empirical reset granularity; arithmetic coder continues
  uninterrupted). The §4.1 quantization-table inversion derives the
  `len - 1` run-length stream from each input table's first-half
  values, asserting each successive group equals `scale * v` for `v =
  0, 1, 2, …` (otherwise `Error::MalformedQuantTable`); the §4.1
  sign-flipped second-half reflection is validated as a precondition
  on the input table. The §4.2.14 / §4.2.15 / §4.2.16 / §4.2.17
  Parameters tail (`states_coded`, `initial_state_delta`, `ec`,
  `intra`) is intentionally NOT emitted — it stays blocked on the
  #904 DOCS-GAP, exactly like `parse_quantization_table_sets` stops
  at the same boundary. A produced blob therefore round-trips through
  `parse_quantization_table_sets` to an equal record + cascade, but
  is not byte-identical to a corpus fixture's CodecPrivate (corpus
  extradata carries the §4.2.14+ tail). A typed-wrapper convenience
  `encode_parameters_with_quant_tables(parsed)` is provided for
  callers holding a parsed `ParametersWithQuantTables`. 20 new tests
  (292 total in the lib, was 258; 14 `config_encode::tests`: minimal
  v3 round-trip + CRC residue zero; 8 rejection paths covering
  non-v3 version, `coder_type > 2`, `chroma_subsample > 4`, empty /
  >8 cascade, declared-count mismatch, broken sign-reflection,
  non-zero `table[0]`, fictitious `context_count`; two-Sets round
  trip; `coder_type == 2` round trip with sparse signed
  `state_transition_delta`; wrapper-vs-direct API equality; encoder
  determinism), plus 6 fixture integration tests in
  `tests/fixture_config_encode.rs` (round-tripping the corpus
  extradata of `v3-default` / `v3-grayscale` / `v3-rgb-bgr0` /
  `v3-yuv444p16` through parse → encode → re-parse with
  field-for-field record equality, every sub-table equality, and
  re-encoded-blob §4.3.2 CRC residue zero; an output-size sanity
  check; and wrapper API parity across all four fixtures).
- **Unified `encode_frame` dispatch helper** (round 196) — consolidates
  the three specialised frame encoders behind one entry point that
  routes on the [`Ffv1ConfigurationRecord`]'s §4.2.5 `colorspace_type`
  and §4.2.3 `coder_type`: `colorspace_type == Rgb` → `encode_frame_rgb`
  (which itself splits `coder_type == 0` Golomb-Rice vs `1 | 2` range
  coder internally); `colorspace_type == YCbCr` with `coder_type == 0` →
  `encode_frame_golomb_rice`; `coder_type == 1 | 2` →
  `encode_frame_range_coder`; any `coder_type > 2` →
  `Error::UnsupportedCoderType`. This is the symmetric counterpart to
  the routing [`decode_frame`] already performs on the read side, so
  callers no longer replicate the entropy-coder/colorspace switch at
  every call site. New `tests/frame_encode_dispatch.rs` (6 tests)
  asserts each combination is byte-identical to the delegate it should
  reach and round-trips through the matching decoder.
- **Golomb-Rice RGB / JPEG 2000 RCT frame encode** ([`encode_frame_rgb`]
  with `coder_type == 0`, round 193) — closes the `coder_type == 0`
  branch of the round-190 RGB encoder. The keyframe bit (slice 0) and
  the §4.6 SliceHeader still go through a per-Slice [`RangeEncoder`];
  [`RangeEncoder::finish()`] lands on the byte boundary the decoder
  finds with `consumed = rc.position()`, and the §4.8 SliceContent then
  writes through a [`BitWriter`] tail driven by [`encode_line`]. A new
  `PlaneLineGolombEncodeState` mirrors the decoder's `PlaneLineState`
  on the Golomb-Rice arm: each Plane carries its own
  [`LineDecoderState`] (per-context VLC window + run state) and §3.1
  border row buffers (`BORDER_WIDTH`-padded, matching [`encode_line`])
  across the §4.7 line-major traversal `for y { for p { Line(p, y) }
  }`. A new private `sample_diffs_for_row_coded` derives per-row
  signed `sample_difference` values in the §3.8 RCT coded-Sample
  space (`bits = bits_per_raw_sample + 1`), reusing the same §3.3
  median + §3.8 modular-wrap convention as the YCbCr Golomb encoder.
  The §4.4 keyframe bit, §4.6 SliceHeader
  (`encode_slice_header_to_encoder`), forward-RCT
  (`forward_rct_for_slice`), and §4.9 SliceFooter
  (`encode_slice_footer`, with §4.9.3 CRC parity solved by
  construction) all re-use the existing per-stage primitives. 6 new
  positive round-trip tests in `tests/rgb_encode_frame.rs` (18 total,
  was 12; 363 total in the crate, was 357): single-slice 8-bit
  general-formula, flat-RGB-plane (run-mode dominated), 8-bit + alpha
  plane, 10-bit §3.7.2.1 exception (Figure 8 forward / Figure 9
  inverse), 2×2 slice grid, and `ec == 0` (3-byte footer). Every
  round-trip closes via [`decode_frame_rgb`] and asserts bit-for-bit
  Plane equality (R, G, B Samples, and alpha when present).
  [`encode_frame_rgb`] now accepts `coder_type ∈ {0, 1, 2}`; values
  outside that range still surface [`Error::UnsupportedCoderType`].

## [0.0.7](https://github.com/OxideAV/oxideav-ffv1/releases/tag/v0.0.7) - 2026-05-30

### Other

- §4.7 RGB / JPEG 2000 RCT frame encoder (round 190)
- coder_type==2 alt state-transition table (round 179)
- range-coded SliceContent encoder (round 164)
- frame-level Golomb-Rice + YCbCr encoder (round 159)
- §4.8 Golomb-Rice run-mode + scalar encode_line (round 152)
- §3.8.2 Golomb-Rice content encoder primitives (round 149)
- §4.6 Slice Header encoder (symmetric inverse of parse_slice_header)
- §4.9 Slice Footer encoder (first frame-level encoder primitive)
- §3.8.1 binary range encoder + §3.8.1.2 scalar put_ur/put_sr/put_br (round 137)
- end-to-end Golomb-Rice full-frame slice-assembly tests (round 136)
- inline grayscale ground-truth, drop workspace-docs include_bytes
- §4.4 keyframe field + §3.7.2 RGB line-major decode driver (round 12)
- round-129 end-to-end decode driver
- §4.9.1 trailer-pointer chain walk per RFC 9043 (round 10)
- per-plane range-coder pixel reconstruction per RFC 9043 §3.7/§3.8.1.2/§4.8
- per-plane Golomb-Rice pixel reconstruction per RFC 9043 §3.1/§3.3/§3.8
- Slice Footer parser per RFC 9043 §4.9
- Configuration Record CRC validation per RFC 9043 §4.3.2 / §4.9.3
- Quantization Table Set cascade decode per RFC 9043 §4.1
- §3.8.2 Golomb-Rice + §3.3/§3.5 predictor & context + decode_line
- Slice Content scaffold per RFC 9043 §4.7 / §4.8
- Slice Header parser per RFC 9043 §4.6
- Configuration Record parser per RFC 9043 §4.2/§4.3
- orphan rebuild: clean-room scaffold post 2026-05-18 audit

### Added

- **RGB / JPEG 2000 RCT frame encoder** ([`encode_frame_rgb`], round 190) —
  symmetric inverse of [`decode_frame_rgb`] for the `coder_type == 1 || 2`
  (range-coded) path. Given an `R / G / B` (and optional alpha) Plane
  [`DecodedFrame`], it applies the §3.7.1 *forward* RCT (general Figure 6
  or the §3.7.2.1 exception when `9 <= bits_per_raw_sample <= 15 &&
  !extra_plane`) with the §3.7.2 positive offset on Cb/Cr, then walks the
  §4.7 line-major traversal (`for y { for p { Line(p, y) } }`) emitting
  per-Sample range-coded `sample_difference` values via the existing
  [`RangePlaneEncoder::encode_row`] under a single per-Slice
  [`RangeEncoder`] cursor (header + content share the same cursor on the
  range-coded path, mirroring [`decode_frame_rgb`]). Per-Plane state
  (`RangePlaneEncoderState` + §3.1 border buffers) is held in a new
  `PlaneLineEncodeState` symmetric to the decoder's `PlaneLineState` and
  stepped one row per Plane each outer-`y` iteration so the
  non-contiguous per-Plane Lines stay byte-for-byte in sync with the
  matching decoder. The §4.4 keyframe bit, §4.6 SliceHeader, and §4.9
  SliceFooter (with §4.9.3 CRC parity solved by construction) all reuse
  the existing per-stage encoder primitives. `coder_type == 0`
  (Golomb-Rice RGB encode) surfaces [`Error::UnsupportedCoderType`] for
  now — it's a follow-up round because it needs a symmetric
  `PlaneReconstructor::encode_row` that does not yet exist. 12 new tests
  in `tests/rgb_encode_frame.rs` (357 total, was 345): single-slice
  8-bit / 10-bit general-formula round trips, 8-bit + alpha plane,
  flat-RGB plane, 2×2 slice grid, `ec == 0` (3-byte footer),
  `coder_type == 2` with a uniform +1 transition delta, determinism
  across calls, plus negative-path coverage for V0 / YCbCr config /
  `coder_type == 0` / zero frame dimensions. Every round-trip closes via
  [`decode_frame_rgb`] and asserts bit-for-bit Plane equality (R, G, B
  Samples, and alpha when present).

- **`coder_type == 2` (alternative state-transition table) wired through
  the range-coded decode + encode drivers** (round 179) — RFC 9043
  §3.8.1.4 Figure 22 / §3.8.1.6. A new public
  [`build_one_state`]`(deltas)` helper layers the Configuration Record's
  `state_transition_delta[1..=255]` onto [`DEFAULT_ONE_STATE`]
  (`one_state[i] = default_state_transition[i] +
  state_transition_delta[i]`, modulo 256). Three call sites pick up the
  derived table when `coder_type == 2`: [`decode_frame`] (YCbCr /
  plane-major), [`decode_frame_rgb`] (RGB / line-major), and
  [`encode_frame_range_coder`] (the round-164 range-coded encode
  driver). Previously `decode_frame` accepted `coder_type == 2` but
  silently fell back to the default table (a bug for any §3.8.1.6
  fixture — none ship today but the path is now provably exercised),
  and `encode_frame_range_coder` rejected `coder_type == 2` outright
  with [`Error::UnsupportedCoderType`]. Now the encoder and the
  matching decoder both pick `build_one_state(&cr.state_transition_delta)`
  and the per-bit state transitions, per-Sample state windows, and
  recovered Plane samples all line up bit-for-bit on the round trip.
  The §3.3.1 16-bit alt-median predicate
  (`coder_type == 1 || coder_type == 2`) gates identically on both
  sides. 8 new tests (345 total, was 337): 4 unit tests in
  `range_coder::tests` (all-zero delta returns [`DEFAULT_ONE_STATE`];
  uniform +1 delta shifts every entry; uniform -2 delta subtracts
  modularly with the expected u8 wrap; a per-symbol-independent-context
  encoder→decoder round-trip exercising every transition the derived
  table covers) and 4 unit tests in `frame_encode::tests` (8-bit /
  10-bit / 2×2 slice-grid round-trips through [`decode_frame`] with a
  sparse +1/-1 delta pattern; plus a regression that pins
  zero-delta `coder_type == 2` to byte-for-byte equal output with
  `coder_type == 1` so a future leak of the derived table into a code
  path that should still use the default would fail loud). The
  `range_rejects_non_range_coder` test was renamed
  `range_rejects_golomb_rice_coder_type` and now only asserts the
  rejection of `coder_type ∈ {0, 3, 7, 255}` — `coder_type == 2` joined
  the accepted-set.

- **Range-coded SliceContent encoder** ([`RangePlaneEncoder`] +
  [`encode_frame_range_coder`], round 164) — symmetric inverse of
  [`RangePlaneReconstructor::reconstruct_plane`] and of the
  `coder_type == 1` + `colorspace_type == 0` branch of
  [`decode_frame`]. The per-Plane encoder folds §3.1 border buffers +
  §3.3 (or §3.3.1 alt) median predictor + §3.5 `absolute_context`
  sign-flip inversion + §3.8.1.3 per-context 32-slot state windows
  initialised to 128 into a per-Sample loop that calls [`put_sr`]
  against the same per-context window the matching decoder-side
  [`get_sr`] call reads. The signed `diff = sample - pred` is folded
  into the §3.8 half-modulus `[-2^(bits-1), 2^(bits-1))` so the
  decoder's `reconstruct_sample(pred, diff, bits)` recovers the
  input Sample exactly. The frame-level
  [`encode_frame_range_coder`] driver keeps the §4.6 SliceHeader and
  the §4.8 SliceContent on a **single** [`RangeEncoder`] cursor
  (no byte-alignment step on the range-coded path, §4.5); the §4.4
  `keyframe` boolean is emitted into slice 0 only against a
  separate init-128 state. `coder_type == 2` (per-context arithmetic
  transition-table variant) reuses the same loop with a swapped
  `one_state` table and stays a follow-up; RGB / line-major on the
  range-coded path likewise stays a follow-up. 30 new tests (337
  total, was 307): 16 unit tests in `range_encode::tests`
  (state-window initialisation + isolation; `normalise_diff`
  invariants; six per-Plane round trips through
  `RangePlaneReconstructor`; multi-plane decoder-cursor sharing;
  encoder determinism; multi-context QTS), 8 unit tests in
  `frame_encode::tests` (single-slice 8/10-bit + ec=0 + 2×2 slice
  grid round trips; determinism; error paths), 6 integration tests
  in `tests/range_encode_frame.rs` (public-API end-to-end including
  flat-Plane zero-diff edge case).

- **Frame-level Golomb-Rice encoder** ([`encode_frame_golomb_rice`],
  round 159) — the symmetric inverse of the `coder_type == 0` +
  `colorspace_type == 0` (Golomb-Rice / YCbCr / plane-major) branch
  of [`decode_frame`]. Composes [`encode_slice_header_to_encoder`]
  (§4.6), [`encode_line`] (§4.8 / §3.8.2), and
  [`encode_slice_footer`] (§4.9) into a per-Slice pipeline driven by
  the §4.4 keyframe boolean (slice 0 only, range-coded against its
  own init-128 state) + [`RangeEncoder::finish`] byte-align +
  plane-major §4.7 traversal with a fresh per-Plane
  [`LineDecoderState`] (per §3.8.2.2.1). Per-row `sample_difference`
  is derived from the §3.3 median predictor + §3.8 modular wrap with
  the same `prev`/`prev_prev` rotation and §3.1 left-border seed
  that [`PlaneReconstructor`] uses, so the encoder and the matching
  decoder mutate every per-context VLC state and every run-mode
  field identically across the round trip. The concatenated Slice
  byte stream is the FFV1 v3 frame payload that [`decode_frame`]
  consumes. RGB / `colorspace_type == 1` surfaces
  [`Error::ColorspaceLayoutNotImplemented`]; range-coded
  SliceContent (`coder_type == 1 || 2`) surfaces
  [`Error::UnsupportedCoderType`]; both are explicit follow-ups
  (their decode-side counterparts are also follow-ups). Validation
  is the primary correctness check: every test ends by feeding the
  encoder's output back into [`decode_frame`] and asserting the
  reconstructed `DecodedFrame.planes[*].samples` match the input
  pixel buffer bit-exactly. 14 new tests (307 total, was 293) in
  `frame_encode::tests`: single-slice 8-bit + 10-bit grayscale
  round-trip; **2×2 slice grid** (the assembly assertion — every
  slice must land in its correct pixel quadrant); 1×3 vertical
  stack (catches `slice_pixel_y` / row-stride faults); an `ec=0`
  3-byte-footer round-trip; determinism (two encodes of the same
  input yield byte-identical buffers); and five error paths
  ([`Error::SliceRequiresVersion3`] for v0/v1,
  [`Error::ColorspaceLayoutNotImplemented`] for RGB,
  [`Error::UnsupportedCoderType`] for range-coded,
  [`Error::InvalidQuantTableSetCount`] for an out-of-range
  `quant_table_set_index`, [`Error::SliceSizeOutOfRange`] for a
  header with `slice_width == 0`), plus three helper-coverage
  tests for `sample_diffs_for_row` / `quant_index_slot` /
  `plane_origin`. The chroma-subsampling math is in place
  (per-Plane origin shift and per-Plane qts routing both mirror the
  decode-side helpers in `frame.rs`); fixture coverage for a
  chroma-subsampled frame is a future round.

- **§4.8 Golomb-Rice run-mode + scalar `encode_line`** ([`encode_line`],
  round 152) — the symmetric inverse of [`decode_line`]. Takes a row of
  signed `sample_difference` values (the same values [`decode_line`]
  returns / writes back into `current_row`) plus the standard
  [`LineNeighborBuffers`] + [`LineDecoderState`] + [`QuantTableSet`],
  and emits via a [`BitWriter`] the bit pattern a matching
  [`decode_line`] call recovers the input row from. The encoder walks
  the same per-pixel state machine the decoder walks — same §3.5
  absolute context (with sign-flip inversion on the put_vlc_symbol
  target), same run-mode predicate (`|context| == 0 && l == t == tl`),
  same scalar / level / run-mode dispatch — so the per-context
  [`VlcState`] entries and the run-mode `run_index` / `run_mode` /
  `run_count` fields mutate identically on both sides of the trip and
  the post-trip state windows match symbol-for-symbol. Run-mode
  encoding uses intra-row lookahead to choose between long-run "1"
  bits (consume `1 << log2_run[run_index]` consecutive zeros) and
  short-run "0 + l2-bit residual" with a level-coded break: if a
  non-zero diff in run-region is reachable inside the row, short-run
  with `rc = zero_run - 1` sets `run_mode = 2` for the level-coded
  follow-up; otherwise long-run consumes a full unit. The first
  run-region pixel after a `reset_run_state()` cannot encode a
  non-zero diff (the decoder's Phase 3 always returns 0); this is the
  §3.8.2.2 contract and a `debug_assert!` surfaces it. 12 new tests in
  `sample_diff::tests` (293 total, was 281): scalar-only path,
  negative-context sign-flip path, all-zero run-mode (long-run unary),
  zeros-then-break (short-run + level), short-run with one-zero +
  level break, two-zeros + level break, mixed scalar/run via predicate
  changes, higher-bit-depth (16-bit ESC), multi-row continuity across
  the row boundary (state straddles per §3.8.2.2.1), empty-row
  no-bits, and a strict per-context VLC state lockstep check. Round
  trip is the primary correctness assertion (the encoder's output is
  decoded back through [`decode_line`] and the row + state are
  asserted to match exactly).
- **§3.8.2 Golomb-Rice content encoder primitives** ([`BitWriter`],
  [`put_ur_golomb_esc`], [`put_sr_golomb_esc`], [`put_vlc_symbol`],
  [`put_vlc_symbol_level`], round 149) — the bit-coded symmetric
  inverses of the decode-side `get_ur_golomb_esc` / `get_sr_golomb_esc`
  / `get_vlc_symbol` / `get_vlc_symbol_level` family. [`BitWriter`] is
  the MSB-first inverse of the existing [`BitReader`]: bits are
  accumulated into a `u64` accumulator, full bytes commit to an owned
  `Vec<u8>`, and [`BitWriter::finish`] zero-pads the final partial byte
  per the RFC 9043 §3.8.2 "padded with zeroes" rule (so the §4.9 Slice
  Footer can begin byte-aligned the way the existing parser expects).
  [`put_ur_golomb_esc`] emits the Figure 26 non-ESC unary-prefix-plus-
  `k`-bit-suffix encoding when `value >> k < 12`, or the ESC twelve-zero-
  prefix plus flat `bits`-wide field (value − 11) otherwise.
  [`put_sr_golomb_esc`] folds signed values onto unsigned via the
  Figure 27 interleave (`0, -1, 1, -2, 2, …` → `0, 1, 2, 3, 4, …`),
  handling `i32::MIN` through `unsigned_abs` so the magnitude doubling
  never overflows. [`put_vlc_symbol`] is the §3.8.2.4 adaptive scalar
  encoder: it picks the same `k` the decoder will (via a shared
  `vlc_pick_k` helper), inverts the sign-flip-and-bias transformation
  (`v = target - bias; v_raw = flip ? -1 - v : v`), emits the signed
  Golomb-Rice code, and updates the per-context [`VlcState`] via the
  shared `vlc_update` helper so the encoder and decoder state windows
  drift in lockstep. [`put_vlc_symbol_level`] is the §3.8.2.4.1 level-
  coded variant for the first non-zero sample after a run-mode run
  breaks (inverts the decoder's `if diff >= 0 { diff += 1 }` shift).
  All four primitives are quant-table-independent and therefore
  unit-testable in isolation; they are the per-Sample bit engine the
  higher-level §4.8 Golomb-Rice Slice Content encoder (with run mode
  + scalar mode + level mode dispatch) will build on. 22 new tests
  (281 total, was 259): 6 [`BitWriter`] tests in `bit_reader::tests`
  (byte-aligned MSB-first emission, bit-at-a-time emission, cross-
  byte-boundary writes that re-decode through a fresh [`BitReader`],
  partial-byte zero padding, 32-bit-wide values, a 100+-bit deterministic
  bit-run round-trip, partial-state reporting via `bits_buffered`), plus
  16 [`golomb_rice`] tests covering: `put_ur_golomb_esc` non-ESC values
  at every `k ∈ 0..=4`, the non-ESC ↔ ESC `prefix == 12` boundary
  at every `k ∈ 0..=5` (both sides), zero at every `k ∈ 0..=8`, a
  byte-image check against RFC 9043 §3.8.2.1.3 Table 3 row for the
  ESC value 139; `put_sr_golomb_esc` paired-sign round trips at
  `k = 0` and `k = 2`, large magnitudes through ESC at `k ∈ {0, 1, 3, 5}`,
  the `i32::MIN` magnitude guard; `put_vlc_symbol` zero-only, alternating
  signs, a 500-zero constant run, count-rescale crossings at the
  `count == 128` boundary, a 500-symbol xorshift Sample-Difference
  stream, the wider `bits = 16` path, plus a strict step-by-step
  state-lockstep test that snapshots the encoder state after each
  symbol and asserts the decoder reproduces it exactly; and
  `put_vlc_symbol_level` paired-sign round trips at `bits = 8`.
- **§4.6 Slice Header encoder** ([`encode_slice_header`] /
  [`encode_slice_header_to_encoder`], round 146) — the symmetric
  inverse of [`parse_slice_header`] / [`parse_slice_header_from_decoder`].
  Walks the Figure-in-§4.6 fields in the same order — `slice_x`,
  `slice_y`, `slice_width - 1`, `slice_height - 1`, the §4.6.5-derived
  `quant_table_set_index[i]` loop, `picture_structure_raw`, `sar_num`,
  `sar_den` — each one a [`put_ur`] against the shared 32-slot context
  window §4.6 places at the start of the Slice's range-coded region
  (same `PARAMETERS_INITIAL_STATE = 128` seed the decoder uses). The
  `_to_encoder` variant chains directly into a caller-owned
  [`RangeEncoder`] for the `coder_type >= 1` Slices where SliceHeader
  and SliceContent share one range coder cursor (the encode-side
  mirror of the existing [`parse_slice_header_from_decoder`]); the
  freestanding [`encode_slice_header`] returns the standalone byte
  region for `coder_type == 0` Slices and standalone testing.
  `slice_width == 0` / `slice_height == 0` is rejected
  ([`Error::SliceSizeOutOfRange`]) — the wire field is
  `slice_width - 1`, so 0 would underflow the round-trip and a 0-pixel
  Slice has no §4.7 layout to match anyway. A header whose
  `quant_table_set_index_count` field disagrees with what the
  Configuration Record's §4.6.5 derivation produces is also rejected
  — emitting a different number of `ur` symbols than the decoder's
  matching loop reads would desync every subsequent field. 17 new
  tests (259 total, was 242): 14 unit tests in `slice_header::tests`
  covering per-field round trips, encoder determinism, both
  rejection paths, and the chained-vs-freestanding API equivalence,
  plus 3 integration tests in `tests/fixture_slice_header.rs` that
  parse the real corpus fixtures (`v3-default` slices 0–3,
  `v3-grayscale` slice 0, `v3-rgb-bgr0` slice 0), re-encode the parsed
  `Ffv1SliceHeader`, and assert the re-encoded bytes re-parse to the
  same struct — the encoder reproduces every field of the corpus's
  SliceHeaders symbol-for-symbol on the shared context window.

- **§4.9 Slice Footer encoder** ([`encode_slice_footer`] /
  [`encode_slice_footer_with_raw_status`], round 142) — the symmetric
  inverse of [`parse_slice_footer`] and the first frame-level FFV1
  encoder primitive shipping in `src/`. Given a Slice body
  (SliceHeader + SliceContent + any Golomb-Rice padding) and an `ec`
  flag, appends the §4.9 trailer: 3 bytes (`slice_size` u(24)) for
  `ec == 0`, or 8 bytes (`slice_size` u(24) + `error_status` u(8) +
  `slice_crc_parity` u(32)) for `ec == 1`. The §4.9.3 parity solver
  runs the §4.9.3 generator (poly `0x104C11DB7`, init 0, no inversion,
  MSB-first) over the prefix `body || size(3) || error_status(1)` and
  appends the resulting 32-bit CRC as the trailing parity word — the
  polynomial-division property `CRC(M || CRC(M)) == 0` drives the
  whole-Slice CRC residue to zero by construction, exactly the
  condition the `ec == 1` branch of [`parse_slice_footer`] checks for.
  The typed [`SliceErrorStatus`] (NoError / Correctable / Uncorrectable
  / Reserved) round-trips against the §4.9.2 Table 16 wire byte;
  callers needing a specific reserved value (3..=255) reach for the
  `_with_raw_status` variant. The §4.9.1 `u(24)` `slice_size` overflow
  (`body.len() >= 1 << 24`) is surfaced as
  [`Error::SliceSizeOutOfRange`] with `expected: 1 << 24`. 11 new
  unit tests in `slice_footer::tests` (211 total, was 200) cover
  encoder→parser round-trips for both `ec` branches, every typed +
  reserved-raw `error_status` value, the overflow + upper-fit
  boundaries, deterministic parity, parity-mixing under body and
  `error_status` bit-flips, and the
  `parity == ffv1_crc32(body || size || error_status)` solver shape.
- **§3.8.1 binary range *encoder*** ([`RangeEncoder`], round 137) — the
  symmetric inverse of the existing [`RangeDecoder`]. Mirrors the
  decoder's RFC 9043 Figure-18/19/20 state machine (16-bit
  `range` / `low`, `(range * state) / 256` split, one byte emitted per
  renormalisation) with the classic delayed-byte / pending-0xFF carry
  technique for byte emission: the most-recently-emitted byte is
  cached so a later renorm can fold a carry into it, and runs of
  `0xFF` bytes (which the carry would propagate through) are deferred
  until a non-`0xFF` byte commits whether the carry happened. The
  state-transition tables (`ONE_STATE` / derived `ZERO_STATE`) are
  identical to the decoder's, so the encoded byte stream re-decodes
  through a fresh [`RangeDecoder`] to the exact bit sequence the
  encoder consumed. New public API: [`RangeEncoder::new`] /
  [`RangeEncoder::with_one_state`] / [`RangeEncoder::put_rac`] /
  [`RangeEncoder::finish`]. The previously-test-only encoder in
  `tests/frame_assembly_golomb.rs` is now a duplicate of this
  primitive (kept local to the test for now; a follow-up round can
  delete it after the test switches to the public API).
- **§3.8.1.2 scalar `put_ur` / `put_sr` / `put_br` symbol encoders** —
  symmetric inverses of [`get_ur`] / [`get_sr`] / [`get_br`]. Each
  walks the 32-slot context-window layout RFC 9043 Figure 21 reads in
  the same order: an `is_zero` bit at offset 0 (early-exit when the
  value is zero); a unary exponent terminated by a single 0-bit using
  offsets `1..=10` with saturation at index 9; the MSB-first mantissa
  using offsets `22..=31` with saturation at index 9; and (for `sr`) a
  sign bit at offsets `11..=21` with saturation at index 10. The
  encoder side handles the `i32::MIN` magnitude-overflow corner by
  promoting to `i64` for the negation step — mirroring the decoder's
  `-(a as i64) as i32` cast. Now exported alongside the decoder
  counterparts as [`get_ur`] / [`get_sr`] / [`get_br`] / [`put_ur`] /
  [`put_sr`] / [`put_br`] / [`SYMBOL_CONTEXT_SIZE`].
- 21 new tests cover the round trips: 6 [`RangeEncoder`] round-trips
  through [`RangeDecoder`] (constant-zero / constant-one streams that
  exercise both arithmetic branches and the high-side carry path,
  alternating bits, deterministic pseudo-random bits, a per-symbol
  independent-context regime, and the "empty-stream still produces a
  decodeable buffer" guard) + 10 scalar round-trips through the same
  decoder (`put_ur` zero / small / power-of-two-boundary / saturated-
  exponent / mixed-pseudo-random / per-symbol-independent-context;
  `put_sr` zero-and-signed-pairs / int-boundary / mixed signed
  pseudo-random; `put_br` alternating). Every round trip asserts both
  the value sequence *and* the post-trip context-window state, so any
  asymmetry between the decoder's state mutation and the encoder's
  would surface immediately.
- This is the first encoder primitive in the crate's `src/` master
  and the foundation every higher-level FFV1 encoder stage
  (Configuration Record write, Slice Header write, range-coded Slice
  Content write) will build on. The decoder-side `Decoder` /
  `Encoder` trait stitch is still pending the §4.2.14 docs gap
  resolution (#904) before [`decode_frame`] can be registered into
  [`RuntimeContext`], but the §3.8.1 / §3.8.1.2 primitive layer is
  now closed end-to-end.

- **End-to-end Golomb-Rice full-frame slice-assembly tests**
  (`tests/frame_assembly_golomb.rs`, round 136) — close the coverage gap
  on the `coder_type == 0` branch of [`decode_frame`]. Every shipped v3
  fixture uses the range coder, and the only `coder_type == 0` corpus
  fixture is FFV1 *version 0* (which the v3-targeted driver rejects), so
  the §4.7 / §4.8 Golomb-Rice driver path — driving
  [`PlaneReconstructor`] across every row of every slice and stitching
  each slice's plane into the frame buffer at its §4.8.3 / §4.7.4 pixel
  origin — previously had no validation *through* `decode_frame` itself.
  The new tests build a **synthetic, self-consistent** v3 Golomb-Rice
  frame with a tiny clean-room encoder (a §3.8.1 binary range encoder +
  §3.8.1.2 `put_ur` + §3.8.2.4 Golomb-Rice scalar encoder + §4.9.3 CRC
  parity solver, each the exact algebraic inverse of the in-tree
  decoder), then assert `decode_frame` reconstructs the known planar
  frame bit-exactly:
    - single-slice full-plane reconstruction (8-bit and 10-bit),
    - a **2×2 slice grid** over an 8×4 frame assembled bit-exact (the
      core assembly assertion — each of the four slices lands in its
      correct pixel quadrant),
    - a 1×3 vertical slice stack (catches `slice_pixel_y` / row-stride
      faults a square grid would mask),
    - determinism (no hidden cross-call state).
  The encoder lives only in the test; it is not part of the public
  surface — its sole job is to manufacture a known-good wire image so
  the **decoder's** assembly path is checked against a frame whose
  ground truth we chose. 6 new integration tests.

- **Frame-level decode driver** ([`decode_frame`]) — the round-11
  deliverable (round 129 of OxideAV-wide implementer rounds):
  - `decode_frame(frame_bytes: &[u8], cr: &Ffv1ConfigurationRecord,
    quant_table_sets: &[QuantizationTableSet], frame_dims:
    FramePixelDimensions, ec: bool) -> Result<DecodedFrame, Error>`
    wires every per-stage parser the prior rounds landed into ONE
    coherent end-to-end driver:
      1. §4.9.1 trailer-pointer chain walk
         ([`walk_trailer_chain`]) → forward-ordered `Vec<SliceExtent>`.
      2. Per Slice: §4.9 footer validate ([`parse_slice_footer`]
         — cross-checks the §4.9.1 size field + (`ec == 1`) the §4.9.3
         whole-Slice CRC).
      3. Build a [`RangeDecoder`] over the slice's body bytes.
      4. §4.6 Slice Header on that decoder
         ([`parse_slice_header_from_decoder`] — the round-11 sibling
         of [`parse_slice_header`] that takes a caller-owned decoder so
         the SliceContent range coder continues from the post-header
         cursor).
      5. §4.7 layout ([`compute_slice_content`]) → per-plane pixel
         dimensions.
      6. Route on `coder_type`:
           - 0 → byte-align after the range coder, build a
             [`BitReader`] from the post-header tail, drive
             [`PlaneReconstructor::reconstruct_plane`] (§3.8.2
             Golomb-Rice).
           - 1 or 2 → drive
             [`RangePlaneReconstructor::reconstruct_plane`] on the same
             `RangeDecoder` (§3.8.1.2 range coder), with the §3.3.1
             alt-16-bit-median predicate computed from
             `colorspace_type == 0 && bits_per_raw_sample == 16 &&
             (coder_type == 1 || 2)`.
      7. Copy each reconstructed Plane into the frame-level
         [`DecodedFrame`] at the slice's pixel-space origin
         (chroma-shifted for planes 1/2 when `chroma_planes == true`).
  - [`DecodedFrame { planes: Vec<DecodedFramePlane>, width, height,
    bits_per_raw_sample, colorspace }`] is the driver's output type —
    one [`DecodedFramePlane { plane_index, width, height, samples:
    Vec<i32> }`] per `primary_color_count` plane, each sample in
    `0 .. 2^bits_per_raw_sample`, assembled at frame resolution
    (chroma-subsampled where appropriate).
  - [`parse_slice_header_from_decoder`] (new public entry) is the
    refactor needed for §4.6 to compose with §4.8 — the existing
    [`parse_slice_header`] is now a thin wrapper that constructs a
    fresh decoder; callers that need the residual range-coder cursor
    (every multi-stage decoder above the header) pass their own
    [`RangeDecoder`] in.
  - Scope: YCbCr / `colorspace_type == 0` (plane-major §4.7
    traversal) is wired end-to-end for `coder_type ∈ {0, 1, 2}`. RGB /
    `colorspace_type == 1` (line-major / row-interleaved between
    Planes) surfaces [`Error::ColorspaceLayoutNotImplemented`] —
    follow-up round will add a row-by-row driver that keeps per-Plane
    entropy state external to the reconstructors.
  - `ec` is taken as an explicit `bool` parameter because
    [`parse_configuration_record`] does not yet decode the §4.2.x
    `ec` / `intra` / `initial_state_delta` fields (deferred from
    earlier rounds); callers obtain `ec` from a black-box source
    (container metadata, or a separate ec-only parser).
  - Frame-level CRC (§4.5 `frame_crc_parity` when `ec == 1 && !slicecrc`)
    is NOT yet wired — the v3-default / v3-grayscale / v3-rgb-bgr0
    fixtures all use per-Slice CRC mode so the round-7 §4.9.3
    validator is enough for the existing test corpus.
  - 19 new tests (197 total, was 178):
      - 12 unit tests in `frame::tests` (frame-plane dimensions for
        YUV420 / grayscale / extra plane / odd widths; plane origin
        chroma shift; `blit_into` full-rect / right-overshoot /
        bottom-overshoot clipping; driver error gates for v0/v1
        config / RGB layout / truncated frame; pre-allocated plane
        shape).
      - 7 integration tests in `tests/frame_driver.rs` driving the
        whole `decode_frame` pipeline against the v3-default
        (4-slice YUV420 128×96) and v3-grayscale (1-slice gray 32×24)
        fixtures + the RGB negative + a corrupt-chain negative + a
        determinism check.
  - This is the wiring layer the public `Decoder` trait
    implementation will sit on once the Configuration Record
    parser's deferred fields land. The crate still does NOT register
    a codec into the runtime context (`register()` stays a no-op);
    that final stitch is the next round.

- §4.9.1 **trailer-pointer chain walk** (`walk_trailer_chain`) — the
  round-10 deliverable:
  - `walk_trailer_chain(frame: &[u8], ec: bool) -> Result<Vec<SliceExtent>, Error>`
    walks the §4.9 Slice Footer `slice_size` (`u(24)`) field backwards
    from the end of a raw FFV1 frame payload and returns one
    [`SliceExtent`] per Slice in **forward** slice-index order
    (`extents[0]` = slice 0, `extents.last()` = highest-indexed slice).
  - `SliceExtent { start: usize, total_len: usize }` carries the byte
    offset and total whole-Slice length (body + footer); the byte
    range `frame[start .. start + total_len]` is exactly the buffer
    [`parse_slice_footer`] consumes. `SliceExtent::end()` returns
    `start + total_len` for callers that want the one-past-the-last
    offset.
  - `slice_footer_len(ec)` is exposed alongside (8 bytes for `ec == 1`,
    3 bytes for `ec == 0`) so callers that pre-compute byte ranges
    don't have to re-derive the constant.
  - The walker reads ONLY the §4.9.1 `u(24)` size field; per-Slice
    validation (CRC, header parsing, pixel reconstruction) stays in
    the existing modules — `parse_slice_footer`, `parse_slice_header`,
    `PlaneReconstructor`, `RangePlaneReconstructor`. This narrow
    contract lets a fuzz harness exercise the chain walk in isolation.
  - Malformed chains (frame shorter than one footer, declared size
    field overruns the cursor / `ec` mismatch) abort with
    `Error::TruncatedSliceFooter` rather than emitting partial
    extents; the §4.9.1 chain is tightly coupled (one mis-read
    `slice_size` shifts every preceding boundary) so partial answers
    would be actively misleading.
  - 19 new tests (178 total, was 159): 14 unit tests in
    `trailer_chain::tests` (single-slice ec=0 / ec=1 round trips,
    four-slice forward-ordering for both ec values, chain coverage of
    the whole frame, many-small-slice walking without stack issues,
    empty-frame + truncated-footer + ec-mismatch + declared-size-
    overrun rejection, `SliceExtent::end()` arithmetic) + 5 fixture
    integration tests in `tests/trailer_chain_fixtures.rs` exercising
    the walker against the same `*_FULL_SLICE*` byte constants the
    round-7 footer-parser fixtures use: the four v3-default slices
    (lens 237 / 316 / 560 / 580, summing to the 1693-byte frame) are
    concatenated into a synthetic frame, walked, and each recovered
    range parses back through `parse_slice_footer` to the same
    `slice_size` / `slice_crc_parity` values the round-7 tests
    already validated against `trace.txt`; the three single-slice
    fixtures (v3-grayscale / v3-rgb-bgr0 / v3-yuv444p16) round-trip
    too.
  - This is the byte-range plumbing the frame-level decode driver
    needs to feed the existing per-Slice parsers + reconstructors:
    a raw FFV1 frame in, an array of validated Slice byte ranges
    out, ready for `parse_slice_header` / `parse_slice_footer` /
    `PlaneReconstructor` / `RangePlaneReconstructor` to consume.

- Per-plane pixel reconstruction for the **range-coder slice path**
  (RFC 9043 §3.1 / §3.3 / §3.3.1 / §3.5 / §3.7 / §3.8 / §3.8.1.2 /
  §3.8.1.3 / §4.7 / §4.8) — the round-9 deliverable:
  - `RangePlaneReconstructor::reconstruct_plane(rc, qtable,
    context_count, width, height, bits, use_16bit_median)` mirrors the
    round-8 Golomb-Rice `PlaneReconstructor::reconstruct_plane` but
    decodes `sample_difference` through a `RangeDecoder` + one signed
    `get_symbol` call per Sample (Figure 21), rather than through a
    `BitReader` + adaptive `get_vlc_symbol`. Returns the reconstructed
    Plane as a row-major `Vec<i32>` of length `width * height` (each
    entry in `0 .. 2^bits`).
  - The range-coder slice path differs from the Golomb-Rice path in
    exactly three ways: (a) **no run mode** — §3.8.2.2 ("Run Mode") is
    explicitly part of §3.8.2 "Golomb-Rice Mode" and does not apply
    here; (b) **per-context 32-slot state windows**, all initialised to
    128 (§3.8.1.3) and laid out flat in `context_count * 32` bytes,
    each window indexed by the §3.5 absolute context; (c) the §3.3.1
    alternate 16-bit median predictor is opt-in via a
    `use_16bit_median: bool` flag (the caller computes the
    `colorspace_type == 0 && bits_per_raw_sample == 16 &&
    (coder_type == 1 || coder_type == 2)` predicate). The §3.1 border
    handling, §3.3 median, §3.5 sign-flip, and §3.8 modular add-back
    (`reconstruct_sample`) are byte-for-byte identical to the
    Golomb-Rice path.
  - The decoder is borrowed mutably (`&mut RangeDecoder<'_>`) so a
    caller can thread the same range coder across multiple Plane
    decodes — the §4.7 YCbCr `Plane then Line` interleave needs Y/U/V
    decoded from a single decoder cursor without re-seeding state.
  - Public `RangeDecoder` / `PARAMETERS_INITIAL_STATE` /
    `DEFAULT_ONE_STATE` re-exports surface the range coder so external
    callers can construct it from a slice's range-coded byte region.
  - 20 new tests (159 total, was 139): 12 unit tests in
    `range_reconstruct::tests` (per-context state-window init to 128
    across `context_count * 32` bytes, per-context window isolation
    via `window_mut`, zero-context fallback, the §3.3.1 alt-median
    formula at both halves of the 16-bit reinterpretation + parity
    with the default median for small values, empty-dimension guards,
    single-Sample reconstruction whose `is-zero` bit reproduces under
    a fresh state-128 window, 8-bit + 16-bit whole-Plane range
    invariants, per-context state isolation across multi-context
    qtables) + 8 integration tests in
    `tests/range_reconstruct_plane.rs` (8/10/16-bit whole-Plane range
    invariants, empty-dimension guards, determinism across two
    decoders on the same byte stream, distinct qtables yielding
    distinct Planes, decoder-cursor-advances-between-Plane-calls — the
    §4.7 YCbCr `Plane then Line` interleave contract — and both
    median branches producing valid 16-bit Planes).
  - The v3 fixtures all use `coder_type == 1`, so this engine is what
    they need to reach end-to-end Plane reconstruction; the
    frame-level driver that splits a Slice's byte regions across
    range-coded header + slice content + footer and assembles
    multi-slice output is queued for a later round.

- Per-plane pixel reconstruction for the Golomb-Rice path
  (RFC 9043 §3.1 / §3.3 / §3.5 / §3.8 / §4.8) — the round-8 deliverable:
  - `PlaneReconstructor::reconstruct_plane(br, qtable, context_count,
    width, height, bits)` decodes a full Plane's `sample_difference`
    stream and reconstructs Samples into a row-major `Vec<i32>` of
    length `width * height` (each entry in `0 .. 2^bits`). Unlike
    `decode_line` (which returns the raw `sample_difference` row), the
    reconstructor folds the §3.3 median predictor + §3.5 sign-flip into
    the per-pixel loop, because every Sample's §3.5 context and §3.3
    prediction depend on the **reconstructed** neighbours of the
    surrounding Samples, not on the raw differences.
  - `reconstruct_sample(pred, diff, bits)` implements the §3.8 modular
    add-back `(pred + diff) mod 2^bits` standalone — only the `n`
    (= `bits_per_raw_sample`) least-significant bits are coded, so the
    sum is reduced modulo `2^bits` and lands in `0 .. 2^bits`.
  - The §3.1 Slice border is maintained across the per-row decode:
    the column one left of the Slice is seeded from the previous row's
    first Sample (`sample[y][-1] = sample[y-1][0]`, `sample[0][-1] = 0`),
    the additional left column and the two rows above are zero, and the
    right border mirrors the rightmost Sample. `BORDER_LEFT` /
    `BORDER_RIGHT` expose the working-buffer pad widths.
  - Run mode (§3.8.2.2.1) and the per-context adaptive VLC state
    (§3.8.2.4 / §3.8.2.5) persist for the whole Plane; run mode resets
    per Plane, and the VLC contexts are keyframe-initialised.
  - The §3.3.1 16-bit median exception is deliberately NOT applied: its
    predicate requires the **range** coder (`coder_type == 1 || 2`),
    which the Golomb-Rice path excludes.
  - 16 new tests (12 module unit + 4 `reconstruct_plane.rs`
    integration), including a hand-traced byte-exact 2x2 scalar plane
    (`0x69 0x90` → `[3, 4, 5, 5]`) that exercises cross-row prediction
    chaining and the §3.1 left-of-slice border seed, plus a run-mode
    flat-zero plane and modular-wrap boundary cases at 8/10/16-bit.

- Slice Footer parser (RFC 9043 §4.9) — the round-7 deliverable:
  - `parse_slice_footer(full_slice_bytes, ec)` reads the §4.9
    `SliceFooter()`: `slice_size` (§4.9.1, `u(24)`), and — when `ec` is
    set — `error_status` (§4.9.2, `u(8)`) + `slice_crc_parity`
    (§4.9.3, `u(32)`). The footer is the trailing 8 bytes for `ec=1`
    or 3 bytes for `ec=0` (it is always byte-aligned per §4.9). The
    caller passes the *whole* Slice byte range (SliceHeader +
    SliceContent + Golomb-Rice padding + footer), typically obtained
    by walking the §4.9.1 trailer-pointer chain backwards from the end
    of the FFV1 frame.
  - `Ffv1SliceFooter { slice_size, total_size, error_status,
    error_status_raw, slice_crc_parity }` plus a `footer_len()` helper.
    `slice_size` is the footer-excluded body length; `total_size` is
    the on-wire length the caller supplied.
  - `SliceErrorStatus::{NoError, Correctable, Uncorrectable, Reserved}`
    typed mirror of §4.9.2 Table 16, with `from_wire` + an
    `is_uncorrectable()` helper. The raw wire byte is preserved on
    `error_status_raw` for `Reserved` diagnostics.
  - For `ec=1` the parser validates the §4.9.3 whole-Slice CRC: the
    same IEEE generator (poly `0x104C11DB7`, init 0, no inversion,
    MSB-first) as the §4.3.2 Configuration Record CRC — so it reuses
    the internal `crc::ffv1_crc32` — must leave a residue of zero over
    the entire Slice (footer included). RFC 9043 §4.9.3: "the Slice as
    a whole has a CRC remainder of 0."
  - Structural cross-check: the on-wire `slice_size` must equal
    `buffer_len - footer_len`; a mismatch (`SliceSizeOutOfRange`) is
    surfaced *before* the CRC check, so a mis-walked trailer chain or a
    wrong `ec` flag is diagnosed structurally rather than as a
    downstream CRC failure.
  - New `Error` variants: `TruncatedSliceFooter` (buffer shorter than
    the footer), `SliceSizeOutOfRange { field, expected }`, and
    `SliceCrcMismatch { residue, stored_parity }`.
  - `SLICE_FOOTER_LEN_EC0` (3) / `SLICE_FOOTER_LEN_EC1` (8) constants.
  - 21 new tests (123 total, was 102): 9 unit tests in
    `slice_footer::tests` (ec=0 size-zero / one-byte body, ec=0 + ec=1
    size-mismatch rejection, ec=0 + ec=1 truncated rejection, a
    solved-parity ec=1 round trip with residue 0, a corrupted-body
    rejection surfacing residue + stored parity, the §4.9.2 Table 16
    mapping) + 12 fixture tests in `tests/fixture_slice_footer.rs`
    reproducing the `trace.txt` `SLICE` `header_crc` parity bit-exactly
    for all four `v3-default` slices and slice 0 of `v3-grayscale` /
    `v3-rgb-bgr0` / `v3-yuv444p16` (residue 0 over each whole Slice),
    plus corrupted-body / corrupted-parity / truncated / wrong-`ec`
    rejection. The whole-Slice byte ranges were extracted black-box via
    `ffmpeg -c copy -f rawvideo` + the §4.9.1 trailer-pointer chain
    walk (`tests/data/slice_footer_fixtures.rs`).

- Configuration Record CRC validation (RFC 9043 §4.3.2) — the round-6
  deliverable:
  - `validate_configuration_record_crc(extradata)` runs the §4.9.3
    generator (IEEE polynomial `0x104C11DB7`, initial value 0, no
    pre-inversion, no post-inversion, MSB-first) over the *entire*
    extradata blob and asserts the residue is `0`. RFC 9043 §4.3.2:
    "configuration_record_crc_parity is 32 bits chosen so that the
    Configuration Record as a whole has a CRC remainder of zero." No
    table is used — the byte-at-a-time bit loop is exact and the record
    is tiny.
  - New internal `crc::ffv1_crc32` (crate-private, reused by the future
    §4.9.3 Slice Footer CRC) and the public
    `validate_configuration_record_crc` entry point.
  - New `Error` variant `ConfigurationRecordCrcMismatch(u32)` carrying
    the non-zero residue on failure; a too-short blob (< 4 parity
    bytes) returns `TruncatedRangeCoder`.
  - 12 new tests (102 total, was 90): 6 unit tests in `crc::tests`
    (all-zero input → 0, valid-record residue 0, corrupted-payload +
    corrupted-parity detection, too-short rejection, single-byte
    known-answers `0x80 → 0x690CE0EE` / `0xFF → 0xB1F740B4`) + 6
    fixture tests in `tests/fixture_config_crc.rs` confirming the four
    v3 fixtures (`v3-default` / `v3-grayscale` / `v3-rgb-bgr0` /
    `v3-yuv444p16`) all CRC to `0`, matching their `trace.txt`
    `GLOBAL_HEADER` `crcref=0x00000000`, plus flipped-byte and
    truncated-parity rejection.

- Quantization Table Set cascade decode (RFC 9043 §4.1) — the round-5
  deliverable, decoded from the same §4.2 Parameters range-coder
  stream:
  - `parse_quantization_table_sets(extradata)` walks the §4.2
    Parameters block (via the new internal `config::parse_parameters`
    helper, factored out of `parse_configuration_record`) and then
    continues into `for (i = 0; i < quant_table_set_count; i++)
    QuantizationTableSet( i )` (Figure 28). Returns a
    `ParametersWithQuantTables { record, quant_table_sets }`.
  - `QuantizationTableSet { tables: QuantTableSet, context_count }` —
    one parsed set: the five 256-entry signed tables (directly the
    `QuantTableSet` the §3.5 context computation indexes) plus
    `context_count = ceil(scale / 2)` (§4.1.2, bounded `<= 32768`).
  - `QuantizationTable(i, j, scale)` (§4.1): run-length `len - 1`
    (`ur`, §3.8.1.2 method) fills the first half with `scale * v`; the
    second half is the sign-flipped reflection
    (`table[256 - k] = -table[k]`, plus the dedicated
    `table[128] = -table[127]`). `scale *= 2 * len_count - 1` after
    each of the five sub-tables. `MAX_CONTEXT_INPUTS = 5`.
  - Reset granularity (the #904 context-buffer-width ambiguity): the
    per-context state window resets to 128 at the start of **each**
    `QuantizationTable`; the arithmetic coder (low / range / byte
    position) continues in the Parameters bitstream. This is the only
    interpretation reproducing the fixture `QUANT_TABLE`
    `context_count` trace values.
  - New `Error` variants: `InvalidQuantTableSetCount(u32)` (§4.2.13,
    `1..=8`), `QuantContextCountOutOfRange(u32)` (§4.1.2), and
    `MalformedQuantTable` (a run overruns the buffer or > 128 levels).
  - 10 new tests (90 total, was 80): 3 unit tests (sign-reflection
    mirror invariant, `ceil(scale/2)`, `MAX_CONTEXT_INPUTS == 5`) +
    7 fixture tests in `tests/fixture_quant_table.rs` reproducing the
    `QUANT_TABLE` `context_count` ground truth bit-exactly —
    `v3-default` / `v3-grayscale` / `v3-rgb-bgr0` → 666 / 7563 (8-bit;
    `len_count` cascades `{6,6,6,1,1}` / `{6,6,3,3,3}`) and
    `v3-yuv444p16` → 365 / 5063 (16-bit; `{5,5,5,1,1}` /
    `{5,5,3,3,3}`), plus the §4.1 second-half sign-flip invariant,
    the entry-0-is-zero invariant, and truncated-input rejection. The
    `v3-yuv444p16` extradata was extracted black-box via
    `ffprobe -show_data` on `input.mkv`.

- Per-row `sample_difference` decode (RFC 9043 §4.8 + §3.8.2) — the
  round-4 deliverable wires the §4.8 `Line( p, y )` body to the
  Golomb-Rice path:
  - `BitReader` — MSB-first `get_bits(n)` over a byte slice
    (RFC 9043 §2.2.9.4). Reads past the buffer end inject zero bits
    per §3.8.2's "padded with zeroes" rule.
  - `get_ur_golomb_esc(k, bits)` + `get_sr_golomb_esc(k, bits)` —
    unsigned/signed Golomb-Rice VLC per Figures 26 / 27 with ESC mode
    per §3.8.2.1.1 (12 zero prefix → flat `bits`-wide suffix + 11).
    The `get_ur_golomb(k)` form (no ESC width) is kept for stand-alone
    prefix-table tests.
  - `sign_extend(input_number, input_bits)` per §3.8.2.3.
  - `VlcState { drift, error_sum, bias, count }` + `VLC_STATE_INITIAL`
    (`{0, 4, 0, 1}`) per §3.8.2.5; `get_vlc_symbol` implements the
    full §3.8.2.4 scalar-mode pseudocode including adaptive `k`
    estimation, error/drift accumulation, count rescale at 128, and
    bias nudging.
  - `get_vlc_symbol_level` — the §3.8.2.4.1 level-coding variant that
    skips the zero value (used for the first post-run-break sample).
  - `LOG2_RUN` — the 41-entry run-mode log2-of-suffix-width table
    (§3.8.2.2.1).
  - `median_predict(l, t, tl)` per §3.3.
  - `NeighborSamples { tt, ll, t, tl, tr, l }` + `QuantTableSet` (5
    sub-tables of 256 i32 entries each per §3.4) + `raw_context` /
    `absolute_context` per §3.5. `AbsoluteContext.sign_flip` flags
    the §3.5 negative-context case so the caller knows to flip the
    decoded `sample_difference`'s sign.
  - `LineDecoderState { vlc: Vec<VlcState>, run_index, run_mode,
    run_count }` carries the per-Plane decoder state; `new(N)`
    initialises N VLC slots from `VLC_STATE_INITIAL`.
  - `LineNeighborBuffers { prev_row, prev_prev_row, current_row,
    plane_pixel_width }` is the view onto the §3.1 bordered sample
    grid the per-row decoder reads from / writes back into.
  - `decode_line(br, state, qtable, neighbours, bits) -> Vec<i32>`
    decodes one Line of sample_difference values: per pixel, computes
    §3.5 absolute context, dispatches to run-mode or scalar VLC per
    §3.8.2.2 / §3.8.2.4, applies the §3.5 sign-flip, writes the
    decoded value back into `current_row` so subsequent neighbour
    lookups see it. **Pixel reconstruction is NOT performed** — the
    returned values are decoded sample_difference symbols, not
    Sample values. The median-predict + modular-wrap reconstruction
    is queued for a later round.
  - 42 new tests (80 total, was 38):
    - 6 `bit_reader::tests` (MSB-first read, cross-byte, past-end
      zero, `bits_left`, 32-bit wide read).
    - 16 `golomb_rice::tests` (`sign_extend` at 0/8-bit, `ur_golomb`
      at k=0/k=2 against RFC §3.8.2.1.3 Table 3 entries, `sr_golomb`
      interleave mapping, ESC mode reading value 139 from the
      `0000_0000_0000 1000_0000` table-3 bit pattern, VLC state
      initialisation, `get_vlc_symbol` zero-decode, `get_vlc_symbol_level`
      zero-skip, `LOG2_RUN` length / monotonicity / last-entry).
    - 9 `predictor::tests` (median predictor on flat / gradient /
      sharp-edge configurations, `median_of_three` precondition,
      zero-quant-table context, identity-quant-table context, signed
      context with §3.5 sign-flip).
    - 6 `sample_diff::tests` (state init, run-state reset preserves
      VLC, decode emits correct row count, scalar path on nonzero
      context, row writeback persists).
    - 5 `tests/golomb_rice_decode.rs` integration tests driving
      `decode_line` end-to-end through hand-crafted bit streams
      against synthetic quant tables.

- Slice Content scaffold per RFC 9043 §4.7 / §4.8:
  - `FramePixelDimensions` value type for caller-supplied frame
    width / height (FFV1's Configuration Record carries neither
    field; the container reports them).
  - `SliceContent { planes: Vec<Plane>, slice_pixel_x/y/width/height,
    traversal }` materializing §4.7.1 `primary_color_count` planes,
    each pre-sized with `Plane.lines: Vec<Line>` identity-only
    placeholders so a future pixel-decode round can fill them.
  - `Plane` carries `plane_index` + `width` / `height` per §4.7.2 /
    §4.8.1 (chroma planes subsampled via
    `log2_h_chroma_subsample` / `log2_v_chroma_subsample`).
  - `PlaneTraversal::{PlaneMajor, LineMajor}` typed mirror of the
    §4.7 `colorspace_type == 0` vs `colorspace_type == 1` interleave;
    `SliceContent::line_visits()` enumerates `(plane_index, y)`
    pairs in the order the §4.7 pseudocode mandates.
  - `compute_slice_content(header, cr, frame)` does raster-bounds
    checks (`slice_x + slice_width <= num_h_slices` etc.), rejects
    zero frame dimensions, and rejects v0/v1 records (which have no
    Configuration-Record-level slice grid).
  - New `Error` variants: `SliceRequiresVersion3`,
    `InvalidFramePixelDimensions { width, height }`,
    `SliceRasterOutOfRange { slice_x, slice_y, slice_width,
    slice_height, num_h_slices, num_v_slices }`.
- 14 new tests (40 total, was 22):
  - 10 unit tests in `slice_content::tests` covering
    `primary_color_count` for {Y, YA, YUV, YUVA, RGB, RGBA},
    `plane_pixel_width` / `plane_pixel_height` at no-subsample / 4:2:0
    / odd-width-rounds-up, the v3-default 2×2 raster pixel rectangle,
    `for_colorspace` traversal mapping, zero-dimension rejection,
    off-raster rejection, v0/v1 rejection, YUV 4:2:0 plane shape +
    `line_visits()` output, RGB line-major interleave, grayscale
    single-plane shape, YUVA extra-plane append, and per-line identity
    invariant.
  - 4 fixture-based integration tests in
    `tests/fixture_slice_content.rs` against extracted slice bytes
    + the trace ground truth from `docs/video/ffv1/fixtures/v3-default/`,
    `/v3-grayscale/`, and `/v3-rgb-bgr0/` — per-plane pixel dimensions
    match the reference decoder's PLANE events bit-exactly, and the
    4-slice v3-default tiling test confirms the §5 "every position
    filled by exactly one Slice" restriction is honoured (12288 luma
    pixels + 6144 chroma pixels = the full 128×96 frame).
- Spec gap noted: RFC 9043 §4.7.3 reads
  `slice_pixel_height = floor((slice_y + slice_height) * slice_pixel_height / num_v_slices) - slice_pixel_y`,
  with `slice_pixel_height` on its own RHS. This is a documentation
  typo (the §4.8.2 sibling formula uses `frame_pixel_width`); we use
  the unambiguous `frame_pixel_height` reading and three fixture
  trace files confirm the per-plane pixel dimensions match bit-exactly.

- Slice Header parser per RFC 9043 §4.6:
  - `Ffv1SliceHeader` struct exposing `slice_x`, `slice_y`,
    `slice_width` (raster), `slice_height` (raster),
    `quant_table_set_index_count` + `quant_table_set_index[..]`,
    `picture_structure` (typed) + `picture_structure_raw`,
    `sar_num`, `sar_den`, plus `sar_is_known()` / `quant_table_indices()`
    helpers and a `MAX_QUANT_TABLE_SET_INDEXES` constant.
  - `parse_slice_header(slice_bytes, &Ffv1ConfigurationRecord)`
    composes with the round-1 range coder + `ur` symbol decoder.
  - Confirms the §4.6 "Slice Header has its own initial states"
    ambiguity resolves the same way as §4.2 Parameters: a single
    shared 32-slot context window. All 6 slice-header fixtures
    decode bit-correctly under that hypothesis.
- 7 fixture-based integration tests against extracted slice bytes
  from `docs/video/ffv1/fixtures/v3-default/` (all 4 slices),
  `/v3-grayscale/` (slice 0, validates `chroma_planes=0` with
  `quant_table_set_index_count=2` via §4.6.5 `version<=3`), and
  `/v3-rgb-bgr0/` (slice 0, RGB/RCT path). Slice bytes were extracted
  via a black-box `ffmpeg -c copy -f rawvideo` invocation + the
  trailer-pointer chain walk per §4.9.1.
- 5 unit tests covering `quant_table_set_index_count` arithmetic and
  truncated-slice rejection.

- Configuration Record parser per RFC 9043 §4.2 / §4.3:
  - `Ffv1ConfigurationRecord` struct exposing `version`,
    `micro_version`, `coder_type`, `state_transition_delta`,
    `colorspace_type`, `bits_per_raw_sample`, `chroma_planes`,
    `log2_h_chroma_subsample`, `log2_v_chroma_subsample`,
    `extra_plane`, and (for v3) `num_h_slices` / `num_v_slices` /
    `quant_table_set_count`.
  - Binary range decoder (RFC 9043 §3.8.1.1, Closed mode) with the
    default state-transition table (§3.8.1.5).
  - Scalar symbol decoder (`get_ur` / `get_sr` / `get_br`) per
    Figure 21.
  - `Ffv1Version`, `ColorspaceType`, and `PictureStructure` enums
    (the last is exported for future slice-header parsers; not
    decoded in this round).
- Unit tests (10 total): 5 range-coder primitives + 5 fixture-based
  Configuration Record tests using the v3-default / v3-rgb-bgr0 /
  v3-grayscale extradata blobs from `docs/video/ffv1/fixtures/`.

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`. The crates.io version
  (`0.0.7`) is preserved on the new master to avoid breaking
  downstream version pins; the published versions on crates.io will
  be yanked by the maintainer.
- The `oxideav-avi` / `oxideav-mkv` dev-dependencies (used by the
  prior crate's roundtrip integration test) are dropped from the
  scaffold and will be re-introduced in a future round if needed.

### Next

- Frame-level driver that splits a Slice's byte regions across
  range-coded header + slice content + footer, threads the per-slice
  `RangeDecoder` through `RangePlaneReconstructor` for each Plane in
  the §4.7 colorspace-defined order, and assembles the multi-slice
  output into a container-ready image — turning the round-9 bit
  engine into an end-to-end fixture decode against the v3 fixtures.
- `initial_state_delta` / `ec` / `intra` — the v3 tail of Parameters
  (§4.2.14 / §4.2.15), still blocked on the #904 DOCS-GAP.
- RCT colorspace post-transform (§3.7.2) for the `colorspace_type == 1`
  fixtures.
