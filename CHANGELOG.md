# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
