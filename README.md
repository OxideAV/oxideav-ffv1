# oxideav-ffv1

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

Clean-room rebuild, round 149 (2026-05-26). The prior implementation was
retired on 2026-05-18 under the workspace clean-room policy.

Round 1 landed the **Configuration Record parser** plus its
range-coder dependencies; round 2 added the **Slice Header parser**
(RFC 9043 §4.6); round 3 added the **Slice Content scaffold**
(§4.7 / §4.8); round 4 wired the §4.8 `Line( p, y )` body to the
**§3.8.2 Golomb-Rice decode path** (MSB-first bit reader,
unsigned/signed Golomb-Rice VLC + ESC mode, per-context adaptive
VLC state, `log2_run` run-mode primitives, the §3.3 median predictor,
and the §3.5 context computation including the negative-context
sign-flip); round 5 adds the **§4.1 Quantization Table Set cascade
decode** — the run-length `len - 1` decode (§3.8.1.2 method), the
`scale *= 2*len_count - 1` accumulation, the
`context_count = ceil(scale/2)` derivation, and the symmetric
sign-flipped second-half reflection. The new
`parse_quantization_table_sets` API decodes the whole §4.2 Parameters
block *and* its embedded §4.1 cascade from one extradata blob,
producing `quant_table_set_count` ready-to-use `QuantTableSet`s — so
the round-4 `decode_line` no longer needs a caller-supplied table.
Round 6 adds the **§4.3.2 Configuration Record CRC** check
(`validate_configuration_record_crc`): the §4.9.3 generator (poly
`0x104C11DB7`, init 0, no inversion) run over the whole extradata blob
must leave a remainder of zero, matching every fixture's
`trace.txt` `crcref=0x00000000`. Round 7 adds the **§4.9 Slice Footer
parser** (`parse_slice_footer`): it reads `slice_size` (§4.9.1),
`error_status` (§4.9.2), and `slice_crc_parity` (§4.9.3) from the
trailing 8 bytes (`ec=1`) or 3 bytes (`ec=0`) of a Slice, cross-checks
the size field against the buffer length, and validates the §4.9.3
whole-Slice CRC residue is zero — reproducing every fixture's
`trace.txt` `SLICE` `header_crc` parity bit-exactly. Round 8 adds
**per-plane pixel reconstruction for the Golomb-Rice path**
(`PlaneReconstructor::reconstruct_plane`): it folds the §3.3 median
predictor + §3.5 sign-flip into the per-pixel decode loop (so each
Sample's context and prediction read the *reconstructed* neighbours),
maintains the §3.1 Slice border, and applies the §3.8 modular add-back
(`reconstruct_sample`, `Sample = (pred + diff) mod 2^bits`) to recover
a full Plane as a row-major `Vec<i32>`. Round 9 mirrors that for the
**range-coder slice path** (`RangePlaneReconstructor::reconstruct_plane`,
RFC 9043 §3.7 / §3.8.1.2 / §4.8): same §3.1 border + §3.3 median +
§3.8 modular add-back, but each Sample's `sample_difference` is one
signed `get_symbol` call per Sample (Figure 21) against a
`RangeDecoder` rather than a Golomb-Rice VLC against a `BitReader`.
The range-coder path has **no run mode** (§3.8.2.2 is Golomb-Rice-only),
allocates one 32-slot state window per §3.5 absolute context (all
initialised to 128 per §3.8.1.3 — `context_count * 32` bytes flat),
and exposes the §3.3.1 alternate 16-bit median predictor through a
`use_16bit_median` flag. This is the bit engine the four v3 fixtures
(all `coder_type == 1`) need to reach end-to-end Plane reconstruction.
Round 10 adds the **§4.9.1 trailer-pointer chain walk**
(`walk_trailer_chain`): given a raw FFV1 frame payload + the
Configuration Record's `ec` flag, it walks the §4.9 Slice Footer
`slice_size` (`u(24)`) field backwards from the end of the frame and
returns one `SliceExtent` per Slice in forward slice-index order
(slice 0 first). Each extent's `frame[start .. start + total_len]`
range is the buffer `parse_slice_footer` (and downstream parsers)
consumes — round 10's deliverable is the byte-range plumbing the
frame-level decode driver needs to feed the existing per-Slice
parsers (`parse_slice_header`, `parse_slice_footer`) and per-plane
reconstructors (`PlaneReconstructor`, `RangePlaneReconstructor`). The
walker reads only the §4.9.1 `u(24)` size field; CRC validation,
header parsing, and pixel reconstruction stay where they already
live.

Round 11 adds the **frame-level decode driver** (`decode_frame`):
given the raw FFV1 v3 frame bytes, the parsed Configuration Record,
the §4.1 Quantization Table Sets, the surrounding container's pixel
dimensions, and the Configuration Record's `ec` flag, the driver
walks every per-stage parser in turn (§4.9.1 trailer chain → §4.9
footer validate → §4.6 header parse → §4.7 plane layout → §3.8.2
Golomb-Rice or §3.8.1.2 range-coder per-plane reconstruction) and
stitches each Slice's per-plane output into a frame-level
`DecodedFrame`. A small `parse_slice_header_from_decoder` refactor
lets §4.6 compose with §4.8 on the same range coder (required for
`coder_type == 1 || 2` where SliceHeader and SliceContent share one
decoder cursor). YCbCr / plane-major (`colorspace_type == 0`) is
wired end-to-end; RGB / line-major surfaces
`Error::ColorspaceLayoutNotImplemented` (the §4.7 row-interleaved
traversal needs a separate row-by-row driver — follow-up round). The
v3-default (4-slice YUV420 128×96) and v3-grayscale (1-slice gray
32×24) integration tests drive the whole pipeline end-to-end with no
panics and every sample in the §3.8 modular range.

Round 12 lands two things. First, the **§4.4 `keyframe` field**: a
Frame opens with a single range-coded boolean (`br`, initial state 128)
at the very start of the first Slice's range-coded region, *before* that
Slice's §4.6 header; later Slices carry no keyframe bit. The prior
driver skipped it, leaving the first Slice's range coder one bit out of
sync. With it consumed, the per-Slice range-coder content start now
matches the reference trace's `RAC_STATE` (`low` / `range`) **exactly
for every Slice**, and the YCbCr / plane-major path decodes the
v3-grayscale fixture **bit-exactly against `expected.raw`** (the
round-12 regression guard `decode_v3_grayscale_is_bit_exact_against_expected_raw`).
Second, the **RGB / JPEG 2000 RCT line-major driver** (`decode_frame_rgb`,
RFC 9043 §3.7.2 / §3.7.2.1 / §4.7 `colorspace_type == 1`): it keeps one
per-Plane reconstruction state alive across the §4.7 row-interleaved
traversal (`for y { for p { Line(p, y) } }`), decodes each Plane's Lines
with the `bits_per_raw_sample + 1` RCT coded width, then applies the
§3.7.1 inverse RCT — Figure 7 general (`g = Y - ((Cb+Cr)>>2); r = Cr+g;
b = Cb+g`) or the §3.7.2.1 Figure 9 exception (9..15-bit, no extra
plane) — de-offsetting Cb/Cr by `1 << bits_per_raw_sample` to recover
the R/G/B (and optional alpha) Planes. The inverse-RCT arithmetic is
covered bit-exactly by forward→inverse round-trip unit tests (8/16-bit
general + 12-bit exception). Against the v3-rgb-bgr0 fixture the Y and
Cb coded Planes reconstruct bit-exactly through the line-major
interleave; a whole-frame bit-exact RGB comparison is still gated on a
localised range-coder content-decode divergence on the third (Cr) Plane
(tracked as a follow-up).

Round 149 lands the **§3.8.2 Golomb-Rice content encoder primitives**
— the bit-coded symmetric inverses of the decode-side
`get_ur_golomb_esc` / `get_sr_golomb_esc` / `get_vlc_symbol` /
`get_vlc_symbol_level` family that the existing per-Line decoder uses.
A new MSB-first `BitWriter` (the inverse of the existing `BitReader`)
accumulates bits into a `u64` accumulator, commits a byte whenever
eight have buffered, and zero-pads the final partial byte on
`finish()` per the RFC 9043 §3.8.2 "padded with zeroes" rule (so a
written §4.8 content section ends on a byte boundary the way the §4.9
Slice Footer parser expects). `put_ur_golomb_esc(k, bits, value)`
emits the Figure 26 non-ESC unary-prefix-plus-`k`-bit-suffix encoding
when `value >> k < 12`, or the ESC twelve-zero-prefix plus flat
`bits`-wide field `value - 11` otherwise. `put_sr_golomb_esc(k, bits,
value)` folds signed values onto unsigned via the Figure 27 interleave
(`0, -1, 1, -2, 2, …` → `0, 1, 2, 3, 4, …`) and delegates; the
`i32::MIN` magnitude is handled through `unsigned_abs` so the
magnitude doubling never overflows. `put_vlc_symbol(state, bits,
target)` is the §3.8.2.4 adaptive scalar encoder: it picks the same
`k` the decoder will pick (via a shared `vlc_pick_k` helper extracted
from the existing decoder), inverts the sign-flip-and-bias
transformation (`v = target - bias; v_raw = flip ? -1 - v : v`),
emits the signed Golomb-Rice code, and updates the per-context
`VlcState` via the shared `vlc_update` helper so the encoder and
decoder state windows drift in lockstep across every symbol.
`put_vlc_symbol_level` is the §3.8.2.4.1 level-coded variant for the
first non-zero sample after a run-mode run breaks (inverts the
decoder's `if diff >= 0 { diff += 1 }` shift). All four primitives are
quant-table-independent and therefore unit-testable in isolation; they
are the per-Sample bit engine the higher-level §4.8 Golomb-Rice Slice
Content encoder (with run-mode + scalar-mode + level-mode dispatch)
will build on. 22 new tests (281 total, was 259): 6 `BitWriter` tests
in `bit_reader::tests` (byte-aligned MSB-first emission, bit-at-a-time
emission, cross-byte-boundary writes that re-decode through a fresh
`BitReader`, partial-byte zero padding, 32-bit-wide values, a
100+-bit deterministic bit-run round-trip, partial-state reporting via
`bits_buffered`), plus 16 `golomb_rice::tests` covering:
`put_ur_golomb_esc` non-ESC values at every `k ∈ 0..=4`, the non-ESC ↔
ESC `prefix == 12` boundary at every `k ∈ 0..=5` (both sides),
`value == 0` at every `k ∈ 0..=8`, plus a byte-image check against
RFC 9043 §3.8.2.1.3 Table 3 last row for the ESC value 139;
`put_sr_golomb_esc` paired-sign round trips at `k = 0` and `k = 2`,
large magnitudes through ESC at `k ∈ {0, 1, 3, 5}`, the `i32::MIN`
magnitude guard; `put_vlc_symbol` zero-only, alternating signs, a
500-zero constant run, count-rescale crossings at the `count == 128`
boundary, a 500-symbol xorshift Sample-Difference stream, the wider
`bits = 16` path, plus a strict step-by-step state-lockstep test that
snapshots the encoder state after each symbol and asserts the decoder
reproduces it exactly; and `put_vlc_symbol_level` paired-sign round
trips at `bits = 8`. The pre-existing test-local Golomb-Rice encoder
helpers in `tests/frame_assembly_golomb.rs` were left untouched
intentionally — this round only lifts the scalar / level path into
`src/`; folding the run-mode encoder dispatch in is a follow-up.

Round 146 lands the **§4.6 Slice Header encoder**
(`encode_slice_header` + `encode_slice_header_to_encoder`), the symmetric
inverse of `parse_slice_header` + `parse_slice_header_from_decoder`. It
walks the Figure-in-§4.6 fields in the same order — `slice_x`,
`slice_y`, `slice_width - 1`, `slice_height - 1`, the §4.6.5-derived
`quant_table_set_index[i]` loop, `picture_structure`, `sar_num`,
`sar_den` — each one a `put_ur` against the shared 32-slot context
window §4.6 places at the start of the Slice's range-coded region
(same `PARAMETERS_INITIAL_STATE = 128` seed the decode side uses). The
`_to_encoder` variant chains directly into a caller-owned
`RangeEncoder` for the `coder_type >= 1` Slices where SliceHeader and
SliceContent share one range coder cursor (the encode-side mirror of
the existing `parse_slice_header_from_decoder`); the freestanding
`encode_slice_header` returns the standalone byte region for
`coder_type == 0` Slices and standalone testing. `slice_width == 0` /
`slice_height == 0` is rejected (`SliceSizeOutOfRange`) — the wire
field is `slice_width - 1`, so 0 would underflow the round-trip and a
0-pixel Slice has no §4.7 layout to match anyway. A header whose
`quant_table_set_index_count` field disagrees with what the
Configuration Record's §4.6.5 derivation produces is rejected too —
emitting a different number of `ur` symbols than the decoder's matching
loop reads would desync every subsequent field. 17 new tests: 14 unit
tests in `slice_header::tests` cover per-field round trips
(`chroma_planes` true/false + extra_plane true/false for both count=2
and count=3, slice raster positions over a 16x16 grid, raster
dimensions from 1x1 through 255x191, all 4 quant-table-index
permutations, all 4 typed `PictureStructure` values + 4 reserved wire
bytes, 7 SAR pairs including (0,0), one-zero, and conformant pairs,
the full-field combo with every field non-default, encoder
determinism, the `slice_width == 0` and `slice_height == 0` rejection
paths, the `quant_table_set_index_count` mismatch rejection, and the
chained `_to_encoder` API agreeing with the freestanding entry point),
plus 3 integration tests in `tests/fixture_slice_header.rs` that parse
the real corpus fixtures (`v3-default` slices 0–3, `v3-grayscale`
slice 0, `v3-rgb-bgr0` slice 0), re-encode the parsed
`Ffv1SliceHeader`, and assert the re-encoded bytes re-parse to the
same struct — the encoder reproduces every field of the corpus's
SliceHeaders symbol-for-symbol on the shared context window.

Round 142 landed the first **frame-level encoder primitive** on top of
round 137's scalar building blocks: the **§4.9 Slice Footer writer**
(`encode_slice_footer` + `encode_slice_footer_with_raw_status`), the
symmetric inverse of `parse_slice_footer`. Given a Slice body
(SliceHeader + SliceContent + any Golomb-Rice padding) and an `ec`
flag, the writer emits the §4.9 trailer: 3 bytes (`slice_size` u(24))
for `ec == 0`, or 8 bytes (`slice_size` u(24) + `error_status` u(8) +
`slice_crc_parity` u(32)) for `ec == 1`. For the `ec == 1` path the
§4.9.3 parity word is solved by running the §4.9.3 generator (poly
`0x104C11DB7`, init 0, no inversion, MSB-first) over the prefix
`body || size(3) || error_status(1)` and appending its 32-bit CRC as
the trailing parity word — the generator's
`CRC(M || CRC(M)) == 0` property drives the whole-Slice CRC residue to
zero by construction, which is exactly the condition the `ec == 1`
branch of `parse_slice_footer` checks for. The §4.9.1 `u(24)` overflow
(`body.len() >= 1 << 24`) is surfaced via `SliceSizeOutOfRange`, and
the typed `SliceErrorStatus` (NoError / Correctable / Uncorrectable /
Reserved) round-trips against the §4.9.2 Table 16 wire byte; callers
needing a specific reserved value (3..=255) reach for the
`_with_raw_status` variant. 11 new tests cover the round-trips:
`ec == 0` (4 body shapes); `ec == 1` zero-residue (4 body shapes); the
4 typed `error_status` values; 5 reserved raw bytes; the `ec == 0`
"error_status argument ignored" guard; the `body.len() == 1 << 24`
overflow boundary and the `(1<<24) - 1` upper-fit case; encoder→parser
corrupted-body residue-mismatch sensitivity; encoder determinism;
parity-mixing under body and `error_status` bit-flips; and the solver
shape `parity == CRC(body || size || error_status)`.

Round 137 lands the first **encoder primitive** in `src/`: the
**§3.8.1 binary range encoder** (`RangeEncoder`) plus the §3.8.1.2
scalar `put_ur` / `put_sr` / `put_br` symbol encoders — the symmetric
inverses of `RangeDecoder` + `get_ur` / `get_sr` / `get_br`.
`RangeEncoder` mirrors the decoder's Figure-18/19/20 state machine
(16-bit `range` / `low`, `(range * state) / 256` split, one byte
emitted per renormalisation) with the classic delayed-byte +
pending-0xFF carry technique for byte emission; the encoded byte
stream re-decodes through a fresh `RangeDecoder` to the same bit
sequence the encoder consumed. The scalar `put_*` family walks the
same 32-slot context-window layout Figure 21 reads in the same order
(`is_zero` bit → unary exponent → MSB-first mantissa → optional sign
bit), with the `i32::MIN` magnitude-overflow corner handled the same
way the decoder's `-(a as i64) as i32` cast handles its inverse. This
is the foundation every higher-level FFV1 encoder stage (Configuration
Record write, Slice Header write, range-coded Slice Content write)
will build on. 21 new tests cover the round trips: 6 binary
(constant-zero / constant-one streams exercising both arithmetic
branches plus the high-side carry path, alternating, deterministic
pseudo-random, per-bit independent contexts, an empty-stream
seedable-flush guard) + 10 scalar (`ur` zero / small / power-of-two
boundaries / saturated exponent / mixed pseudo-random / per-symbol
independent contexts; `sr` zero-and-signed-pairs / int-boundary /
mixed signed pseudo-random; `br` alternating). Each test asserts
*both* the recovered value sequence and the post-trip context-window
state so any asymmetry between the decoder's state mutation and the
encoder's would surface immediately.

Round 136 added **end-to-end coverage for the Golomb-Rice
(`coder_type == 0`) full-frame slice-assembly path** of `decode_frame`.
Every shipped v3 fixture uses the range coder, and the only
`coder_type == 0` corpus fixture is FFV1 version 0 (which the
v3-targeted driver rejects), so the §4.7 / §4.8 Golomb-Rice branch —
driving `PlaneReconstructor` across every row of every slice and
stitching each slice's plane into the frame buffer at its §4.8.3 /
§4.7.4 pixel origin — had no validation *through* `decode_frame`
itself. `tests/frame_assembly_golomb.rs` builds a synthetic,
self-consistent v3 Golomb-Rice frame with a clean-room test-only
encoder (a §3.8.1 binary range encoder + §3.8.1.2 `put_ur` + §3.8.2.4
Golomb-Rice scalar encoder + §4.9.3 CRC parity solver, each the exact
inverse of the in-tree decoder), then asserts `decode_frame`
reconstructs the chosen planar frame bit-exactly: single-slice
full-plane (8- and 10-bit), a **2×2 slice grid** assembled bit-exact
(each of four slices landing in its correct pixel quadrant), a 1×3
vertical slice stack (catching `slice_pixel_y` / row-stride faults),
and determinism. The encoder is test-only — its job is to manufacture a
known-good wire image so the **decoder's** assembly path can be checked
against a frame whose ground truth we chose. 6 new integration tests.

Implemented (RFC 9043 §3.1 / §3.3 / §3.3.1 / §3.5 / §3.7 / §3.8 /
§3.8.1.1 / §3.8.1.2 / §3.8.1.3 / §3.8.2 / §4.1 / §4.2 / §4.3 / §4.3.2 /
§4.6 / §4.7 / §4.8 / §4.9 / §4.9.1 / §4.9.3):

- Binary range **decoder + encoder** (Closed mode), default
  state-transition table (`RangeDecoder` / `RangeEncoder`,
  RFC 9043 §3.8.1.1 / Figures 18–20). The encoder mirrors the
  decoder's renormalisation cadence with the classic delayed-byte +
  pending-0xFF carry technique and round-trips bit-exactly through a
  fresh decoder.
- Scalar symbol decoder + **encoder** (`ur` / `sr` / `br` →
  `get_ur` / `get_sr` / `get_br` + `put_ur` / `put_sr` / `put_br`)
  per Figure 21. The encoder side walks the same 32-slot context
  layout (`is_zero` bit, unary exponent, MSB-first mantissa,
  optional sign bit) in the same offset order, with the `i32::MIN`
  magnitude corner promoted through `i64` to mirror the decoder.
- MSB-first bit reader for the Golomb-Rice path (§2.2.9.4 /
  §3.8.2).
- Golomb-Rice VLC: `get_ur_golomb_esc(k, bits)` and
  `get_sr_golomb_esc(k, bits)` per Figures 26 / 27 + §3.8.2.1.1 ESC
  mode; `sign_extend` per §3.8.2.3.
- Adaptive VLC state (`VlcState { drift, error_sum, bias, count }`)
  + `get_vlc_symbol` / `get_vlc_symbol_level` per §3.8.2.4 / §3.8.2.4.1.
- `LOG2_RUN` run-mode table per §3.8.2.2.1.
- Median predictor `median(l, t, l+t-tl)` per §3.3.
- §3.5 context computation: 5-subtable quant-table lookup, sum, and
  absolute-value mapping with sign-flip flag.
- Per-Line `sample_difference` decode (`decode_line`) — drives the
  §4.8 `Line( p, y )` pseudocode over a caller-supplied
  `QuantTableSet`, emitting a `Vec<i32>` of decoded sample
  differences per row.
- Per-plane pixel reconstruction for the Golomb-Rice path
  (`PlaneReconstructor::reconstruct_plane`, §3.1 / §3.3 / §3.5 / §3.8 /
  §4.8): decodes a Plane's `sample_difference` stream and reconstructs
  Samples into a row-major `Vec<i32>` (each in `0 .. 2^bits`). The §3.3
  median predictor + §3.5 sign-flip are folded into the per-pixel loop
  (a Sample's context + prediction read the reconstructed neighbours,
  not the raw differences); the §3.1 border is maintained per row
  (left-of-slice column seeded from the previous row's first Sample,
  additional left column + two rows above zero, right border mirrored);
  the §3.8 add-back `(pred + diff) mod 2^bits` is exposed standalone as
  `reconstruct_sample`. Run mode + the per-context adaptive VLC state
  persist across the Plane's rows. The §3.3.1 16-bit median exception
  is N/A (it requires the range coder).
- Per-plane pixel reconstruction for the **range-coder slice path**
  (`RangePlaneReconstructor::reconstruct_plane`, §3.1 / §3.3 / §3.3.1
  / §3.5 / §3.7 / §3.8 / §3.8.1.2 / §3.8.1.3 / §4.8): decodes a
  Plane's `sample_difference` stream from a `RangeDecoder` (one signed
  `get_symbol` call per Sample, Figure 21) and reconstructs Samples
  into a row-major `Vec<i32>` (each in `0 .. 2^bits`). Differs from
  the Golomb-Rice path in exactly three ways: (a) **no run mode** —
  §3.8.2.2 is Golomb-Rice-only; (b) per-context **32-slot state
  windows** flat in `context_count * 32` bytes, all initialised to 128
  per §3.8.1.3; (c) the §3.3.1 alternate median predictor is opt-in via
  a `use_16bit_median` flag (caller computes the
  `colorspace_type == 0 && bits_per_raw_sample == 16 && coder_type
  in {1,2}` predicate). The §3.1 border / §3.3 median / §3.5 sign-flip
  / §3.8 modular add-back are byte-for-byte identical to the
  Golomb-Rice path. The decoder is passed by `&mut` so a caller can
  thread the same range coder across multiple Planes for the §4.7
  YCbCr "Plane then Line" interleave.
- Configuration Record fields: `version`, `micro_version`,
  `coder_type`, `state_transition_delta`, `colorspace_type`,
  `bits_per_raw_sample`, `chroma_planes`, `log2_h_chroma_subsample`,
  `log2_v_chroma_subsample`, `extra_plane`, `num_h_slices`,
  `num_v_slices`, `quant_table_set_count`.
- Slice Header fields: `slice_x`, `slice_y`, `slice_width` (raster),
  `slice_height` (raster), `quant_table_set_index[..]`,
  `picture_structure`, `sar_num`, `sar_den`.
- Slice Content scaffold (§4.7): `primary_color_count` (§4.7.1),
  `slice_pixel_x` / `slice_pixel_y` / `slice_pixel_width` /
  `slice_pixel_height`, per-plane `plane_pixel_width` /
  `plane_pixel_height` with 4:2:0 / 4:2:2 / 4:4:4 chroma subsampling,
  and the §4.7 plane-then-line (YCbCr) vs. line-then-plane (RGB)
  traversal order as a typed `PlaneTraversal` + `line_visits()`
  enumerator.
- Quantization Table Set cascade (§4.1): `parse_quantization_table_sets`
  decodes the §4.2 Parameters block plus its embedded
  `QuantizationTableSet( i )` cascade from one extradata blob,
  emitting `quant_table_set_count` `QuantizationTableSet`s. Each is a
  ready-to-use `QuantTableSet` (5 × 256 signed entries) plus its
  `context_count` (`ceil(scale/2)`, §4.1.2). Per-context state resets
  to 128 at the start of each of the five `QuantizationTable`s; the
  arithmetic coder continues in the Parameters bitstream.
- Configuration Record CRC validation (§4.3.2):
  `validate_configuration_record_crc(extradata)` runs the §4.9.3 CRC
  (poly `0x104C11DB7`, init 0, MSB-first, no pre/post-inversion) over
  the whole extradata blob and requires a zero residue. Reuses an
  internal `ffv1_crc32` that the §4.9.3 Slice Footer CRC shares.
  Returns `ConfigurationRecordCrcMismatch(residue)` on a non-zero
  residue.
- Trailer-pointer chain walk (§4.9.1):
  `walk_trailer_chain(frame, ec)` walks the §4.9 Slice Footer's
  `slice_size` (`u(24)`) field backwards from the end of a raw FFV1
  frame payload and returns one `SliceExtent { start, total_len }` per
  Slice in **forward** slice-index order (slice 0 first). Each
  `frame[start .. start + total_len]` range is the whole-Slice buffer
  `parse_slice_footer` consumes. Footer length is 8 bytes when
  `ec == 1` (the §4.9 `if (ec)` branch) and 3 bytes when `ec == 0`;
  `slice_footer_len(ec)` is exposed alongside for callers that
  pre-compute byte ranges. The walker only reads the `u(24)` size
  field — §4.9.3 CRC validation, header parsing, and pixel
  reconstruction stay in their existing modules. Malformed chains
  (size field overruns the cursor; frame shorter than one footer)
  abort with `Error::TruncatedSliceFooter` rather than emitting
  partial extents.
- Slice Footer parsing (§4.9): `parse_slice_footer(full_slice, ec)`
  reads `slice_size` (§4.9.1), `error_status` (§4.9.2, typed
  `SliceErrorStatus` per Table 16), and `slice_crc_parity` (§4.9.3)
  from the trailing 8 bytes (`ec=1`) / 3 bytes (`ec=0`) of a Slice.
  Cross-checks `slice_size == buffer_len - footer_len`
  (`SliceSizeOutOfRange`) and, for `ec=1`, validates the §4.9.3
  whole-Slice CRC residue is zero via the shared `ffv1_crc32`
  (`SliceCrcMismatch { residue, stored_parity }`). The whole-Slice
  byte range is what the §4.9.1 trailer-pointer chain walk yields.
- **Slice Header encoder** (§4.6): `encode_slice_header(header, cr)`
  (+ `encode_slice_header_to_encoder` for chaining into a caller-owned
  `RangeEncoder`) is the symmetric inverse of `parse_slice_header` /
  `parse_slice_header_from_decoder`. Walks the Figure-in-§4.6 fields
  in the same order — `slice_x`, `slice_y`, `slice_width - 1`,
  `slice_height - 1`, the §4.6.5-derived `quant_table_set_index[i]`
  loop, `picture_structure_raw`, `sar_num`, `sar_den` — each one a
  `put_ur` against the shared 32-slot context window §4.6 places at
  the start of the Slice's range-coded region (same
  `PARAMETERS_INITIAL_STATE = 128` seed the decoder uses).
  `slice_width == 0` / `slice_height == 0` and a
  `quant_table_set_index_count` mismatch are rejected as
  `SliceSizeOutOfRange`. Composes with the §4.9 Slice Footer encoder
  on the produced body (`encode_slice_header` → caller drives §4.8
  SliceContent encode → `encode_slice_footer` wraps).
- **Slice Footer encoder** (§4.9): `encode_slice_footer(body, ec,
  status)` (+ `encode_slice_footer_with_raw_status` for explicit
  reserved bytes) is the symmetric inverse of `parse_slice_footer`.
  Given a Slice body it appends 3 bytes (`ec == 0`) or 8 bytes
  (`ec == 1`); for `ec == 1` the §4.9.3 `slice_crc_parity` is solved
  by running `ffv1_crc32` over `body || size(3) || error_status(1)`
  and appending the resulting CRC, leveraging the polynomial-division
  property `CRC(M || CRC(M)) == 0` to drive the whole-Slice residue
  to zero by construction. The §4.9.1 `u(24)` `slice_size` overflow
  (`body.len() >= 1 << 24`) is surfaced as
  `SliceSizeOutOfRange { field, expected: 1 << 24 }`. The first
  frame-level FFV1 encoder primitive shipping in `src/`.
- **Frame-level decode driver** (`decode_frame`): wires §4.9.1 chain
  walk → §4.9 footer validate → §4.6 header parse (via
  `parse_slice_header_from_decoder`, the round-11 sibling that takes
  a caller-owned `RangeDecoder` so SliceHeader and SliceContent share
  the decoder cursor for `coder_type >= 1`) → §4.7 layout → §3.8.2
  Golomb-Rice (`PlaneReconstructor`) or §3.8.1.2 range-coder
  (`RangePlaneReconstructor`) per-plane reconstruction → per-slice
  blit into a frame-level `DecodedFrame`. YCbCr / plane-major path is
  end-to-end; RGB / line-major returns
  `Error::ColorspaceLayoutNotImplemented`. `ec` flows in as an
  explicit parameter pending the §4.2.14 `ec` parse.

Not yet implemented:

- `states_coded` / `initial_state_delta` / `ec` / `intra` (the v3 tail
  of Parameters) — **blocked** on a §4.2.14 loop-count discrepancy; see
  Notes for future rounds (#904 DOCS-GAP). Until those land,
  `decode_frame` takes `ec` as an explicit `bool` parameter.
- RGB / `colorspace_type == 1` line-major (§4.7 row-interleaved
  between Planes) frame driver — needs a row-by-row reconstructor
  variant that keeps per-Plane entropy state external. Follow-up
  round.
- Frame-level CRC (§4.5 `frame_crc_parity` when `ec == 1 && !slicecrc`)
   — every existing fixture uses per-Slice CRC mode so the round-7
  §4.9.3 per-Slice validator is enough for the test corpus.
- `Decoder` trait registration into `RuntimeContext` — small wiring
  step once the Configuration Record's deferred fields are parsed.
- RCT colorspace post-transform.
- Remaining higher-level encoder stages (Configuration Record write,
  range-coded Slice Content write). The §3.8.1 binary range encoder +
  §3.8.1.2 scalar `put_ur` / `put_sr` / `put_br` primitives the
  higher-level stages will compose on top of landed in round 137; the
  §4.9 Slice Footer writer (`encode_slice_footer`) — the first
  frame-level encoder primitive — landed in round 142; the §4.6 Slice
  Header writer (`encode_slice_header`) — symmetric inverse of
  `parse_slice_header` — landed in round 146.

Until the trait stitch lands, the public `Decoder` / `Encoder` traits
return `Error::NotImplemented` and no codec is registered into the
runtime — the `decode_frame` API is exposed directly for downstream
crates that want to drive the codec without the registry.

## Verification

The parsers are validated against the workspace's black-box FFV1
fixture corpus under `docs/video/ffv1/fixtures/`:

- Configuration Records are the Matroska CodecPrivate bytes of each
  fixture's `input.mkv`, checked against the `trace.txt`
  `GLOBAL_HEADER` event.
- Slice Headers are the leading bytes of each slice's range-coded
  region — located by walking the trailer-pointer chain backwards
  from the end of the raw FFV1 frame (extracted via a black-box
  `ffmpeg -c copy -f rawvideo` invocation). Expected `slice_x` /
  `slice_y` come straight from the `trace.txt` `SLICE` events.
- Quantization Table Set cascades are checked against the `trace.txt`
  `QUANT_TABLE` events' `context_count` field. `context_count` is a
  function of every `len_count` across all five sub-tables, so a
  single off-by-one anywhere in the run-length stream desynchronises
  it — making it a tight bit-exactness check on the whole cascade.
- Configuration Record CRCs are checked against the `trace.txt`
  `GLOBAL_HEADER` event's `crcref` field (`0x00000000` for every
  fixture): the §4.9.3 generator run over the whole extradata blob
  reproduces the reference decoder's zero residue. A clean-room CRC
  that hits the same `0` over the same bytes has the polynomial
  orientation and the no-inversion convention exactly right.
- Slice Footers are checked against each slice's `trace.txt` `SLICE`
  event: the parsed `slice_size` matches the trailer-chain-walked
  body length (and the trace's `len` minus the 8-byte footer), and the
  §4.9.3 whole-Slice CRC residue is zero — which is equivalent to the
  parsed `slice_crc_parity` reproducing the trace's `header_crc`
  bit-exactly (the encoder solved the parity word for the zero
  residue). The whole-Slice byte ranges are extracted black-box via
  `ffmpeg -c copy -f rawvideo` + the §4.9.1 trailer-pointer chain
  walk.

| Fixture | Round 1 (cfg record) | Round 2 (slice header) | Round 3 (slice content) |
| --- | --- | --- | --- |
| `v3-default` | v3 / 8-bit YUV 4:2:0 / range coder default / 2x2 slices | all 4 slices: raster cells (0,0)(1,0)(0,1)(1,1) | slice 0: 64x48 Y + 32x24 U + 32x24 V (matches trace PLANE), 4-slice tiling exhausts 128x96 |
| `v3-rgb-bgr0` | v3 / RGB (RCT) / chroma_planes=1 / no subsample | slice 0 (chroma_planes=1 / RCT path) | slice 0: 3 planes × 32x24, line-major traversal per §4.7 |
| `v3-grayscale` | v3 / single-plane / chroma_planes=0 | slice 0 (chroma_planes=0, count=2 via version<=3) | slice 0: single 32x24 plane |

Round 4's Golomb-Rice decode primitives are validated by 42 in-tree
tests including the §3.8.2.1.3 Table 3 examples (k=0/2 unary / suffix
decode, ESC mode reading the `0000_0000_0000 1000_0000` byte pattern
as value 139), the §3.3 / §3.5 predictor and context calculations,
and 5 integration tests driving `decode_line` end-to-end with
synthetic quant tables and hand-crafted bit streams.

Round 5's §4.1 quant-table cascade adds 10 tests (90 total, was 80):
3 unit tests (sign-reflection mirror, `ceil(scale/2)`,
`MAX_CONTEXT_INPUTS == 5`) + 7 fixture tests reproducing the
`QUANT_TABLE` `context_count` ground truth bit-exactly —
`v3-default` / `v3-grayscale` / `v3-rgb-bgr0` decode to 666 / 7563
(8-bit, sub-table `len_count` cascades `{6,6,6,1,1}` / `{6,6,3,3,3}`)
and `v3-yuv444p16` to 365 / 5063 (16-bit, `{5,5,5,1,1}` /
`{5,5,3,3,3}`), plus the §4.1 second-half sign-flip invariant and
truncated-input rejection.

Round 6's §4.3.2 Configuration Record CRC adds 12 tests (102 total,
was 90): 6 unit tests in `crc::tests` (all-zero input → 0, valid-record
residue 0, corrupted-payload + corrupted-parity rejection, too-short
rejection, single-byte known-answers `0x80 → 0x690CE0EE` /
`0xFF → 0xB1F740B4`) and 6 fixture tests reproducing `crcref=0x00000000`
for `v3-default` / `v3-grayscale` / `v3-rgb-bgr0` / `v3-yuv444p16` plus
flipped-byte and truncated-parity rejection.

Round 7's §4.9 Slice Footer parser adds 21 tests (123 total, was 102):
9 unit tests in `slice_footer::tests` (ec=0 size-zero / one-byte body
round trips, ec=0 + ec=1 size-mismatch rejection, ec=0 + ec=1
truncated rejection, a solved-parity ec=1 round trip with whole-Slice
residue 0, a corrupted-body rejection surfacing residue + stored
parity, and the §4.9.2 Table 16 `error_status` mapping) + 12 fixture
tests in `tests/fixture_slice_footer.rs` reproducing the `trace.txt`
`SLICE` `header_crc` parity bit-exactly for all four `v3-default`
slices (`0xCB530827` / `0xC93079C7` / `0xB8923B4F` / `0x42C8841D`) and
slice 0 of `v3-grayscale` (`0x44C7D58E`) / `v3-rgb-bgr0` (`0x3BBFE098`)
/ `v3-yuv444p16` (`0x0AD980DC`) — each whole-Slice CRC residue is 0 —
plus corrupted-body, corrupted-parity, truncated-footer, and
wrong-`ec`-flag rejection.

Round 8's §3.1 / §3.3 / §3.8 per-plane reconstruction adds 16 tests
(139 total, was 123): 12 unit tests in `reconstruct::tests`
(`reconstruct_sample` modular add-back at 8/10/16-bit across the sign
boundary, the median-predictor base, a run-mode flat plane, a 1x1 and a
2x1 scalar plane with cross-pixel prediction, empty-dimension guards,
and an 8x8 range invariant) + 4 integration tests in
`tests/reconstruct_plane.rs`: a hand-traced byte-exact 2x2 scalar plane
(`0x69 0x90` → `[3, 4, 5, 5]`) that exercises cross-row prediction
chaining and the §3.1 left-of-slice border seed, a run-mode flat-zero
5x4 plane, and modular-wrap boundary cases. The v3 fixtures use the
range coder (`coder=1`) so they cannot yet be reconstructed end-to-end;
the reconstruction logic is verified byte-exact via these synthetic
Golomb-Rice traces (see "Not yet implemented").

Round 9's §3.7 / §3.8.1.2 range-coder plane reconstruction adds 20
tests (159 total, was 139): 12 unit tests in `range_reconstruct::tests`
(per-context state-window init to 128 across `context_count * 32`
bytes, per-context window isolation, zero-context fallback, the §3.3.1
alt-median formula at both halves of the 16-bit reinterpretation +
parity with the default median for small values, empty-dimension
guards, a single-Sample reconstruction whose `is-zero` bit reproduces
under a fresh state-128 window, and 8-bit + 16-bit whole-Plane range
invariants) + 8 integration tests in `tests/range_reconstruct_plane.rs`
(8/10/16-bit whole-Plane range invariants, empty-dimension guards,
determinism across two decoders on the same byte stream, distinct
qtables yielding distinct Planes, decoder cursor advancing between
back-to-back Plane calls — the §4.7 YCbCr `Plane then Line` interleave
contract — and both median branches producing valid 16-bit Planes).
The range coder itself is exercised through the public
`RangeDecoder` API, not constructed inside the test, so the per-Sample
state-window plumbing is verified end-to-end.

Round 137's §3.8.1 binary range *encoder* + §3.8.1.2 scalar `put_*`
symbol encoders add 21 tests (199 total, was 178): 6 unit tests in
`range_coder::tests` (round-trip constant zeros, constant ones,
alternating, deterministic pseudo-random, per-bit independent contexts,
and the empty-stream seedable-flush guard) + 10 in `symbol::tests`
(`put_ur` zero / small values / power-of-two boundaries / saturated
exponent / mixed pseudo-random / per-symbol-independent-context;
`put_sr` zero-and-signed-pairs / int-boundary including `i32::MIN` /
mixed signed pseudo-random; `put_br` alternating). Each scalar test
asserts both the recovered value sequence and the post-trip
context-window state matches the encoder's, so any asymmetry between
decoder + encoder state mutation would surface immediately rather than
hiding behind a value-only comparison.

Round 142's §4.9 Slice Footer encoder adds 11 unit tests in
`slice_footer::tests` (210 total, was 199): `ec == 0` round-trips
(4 body shapes from empty to 256 bytes); `ec == 1` zero-residue
round-trips (4 body shapes); the 4 typed `SliceErrorStatus` values
each round-tripping through encode→parse; 5 reserved-range raw bytes
(3 / 7 / 99 / 254 / 255) folding back to `Reserved`; the `ec == 0`
"`error_status` argument is unused" invariant; the `body.len() == 1 << 24`
overflow boundary + the `(1 << 24) - 1` upper-fit case; corrupted-body
sensitivity (`parse_slice_footer` surfaces non-zero residue + unchanged
stored parity after a one-bit body flip); encoder determinism; parity
mixing under body bit-flips and `error_status` changes (both still
satisfy §4.9.3 residue zero — the parity adapts); and the solver-shape
pin `parity == ffv1_crc32(body || size(3) || error_status(1))` for
`ec == 1`. The encoder→parser symmetry is the primary correctness
test — every test ends with `parse_slice_footer` (or `ffv1_crc32`)
asserting the §4.9.3 residue-zero invariant after the encoder's parity
solver finished.

Round 146's §4.6 Slice Header encoder adds 17 tests (259 total, was
242): 14 unit tests in `slice_header::tests` covering per-field
round-trips (minimal `chroma_planes` true/false + extra_plane true
for count=3, raster positions over a 16x16 grid, raster dimensions
from 1x1 through 255x191, all 4 quant-table-index permutations, all
4 typed `PictureStructure` values + 4 reserved wire bytes, 7 SAR
pairs including (0,0) and one-zero degenerate, the full-field combo
asserting no cross-talk on the shared 32-slot state window between
fields, encoder determinism, the `slice_width == 0` /
`slice_height == 0` rejection paths, the
`quant_table_set_index_count` mismatch rejection, and the chained
`_to_encoder` API agreeing byte-for-byte with the freestanding entry
point) + 3 integration tests in `tests/fixture_slice_header.rs`
(`v3-default` all 4 slices, `v3-grayscale` slice 0, `v3-rgb-bgr0`
slice 0) that parse the corpus fixture's SliceHeader bytes,
re-encode the parsed [`Ffv1SliceHeader`], and assert the re-encoded
bytes re-parse to the same struct — the encoder reproduces every
field of the corpus's SliceHeaders symbol-for-symbol on the shared
context window even though the fixture bytes also carry downstream
SliceContent (so a byte-for-byte fixture comparison isn't applicable;
the round-trip-through-parser is the §4.6-isolated correctness
check).

Round 10's §4.9.1 trailer-pointer chain walk adds 19 tests (178 total,
was 159): 14 unit tests in `trailer_chain::tests` (single-slice ec=0 /
ec=1 round trips, four-slice forward-ordering for both ec values,
chain coverage of the whole frame, many-small-slice walking without
stack issues, empty-frame + truncated-footer + ec-mismatch + declared-
size-overrun rejection, `SliceExtent::end()` arithmetic) + 5 fixture
integration tests in `tests/trailer_chain_fixtures.rs` exercising the
walker against the same `*_FULL_SLICE*` byte constants the round-7
footer-parser fixtures use: the four v3-default slices (lens 237 / 316
/ 560 / 580, summing to the 1693-byte frame) are concatenated into a
synthetic frame, walked, and each recovered range parses back through
`parse_slice_footer` to the same `slice_size` / `slice_crc_parity`
values the round-7 tests already validated against `trace.txt`; the
three single-slice fixtures (v3-grayscale / v3-rgb-bgr0 / v3-yuv444p16)
round-trip too. This is the round-10 ↔ round-7 integration check: the
walker delivers exactly the bytes the footer parser consumes.

## Notes for future rounds

- **DOCS-GAP (#904, §4.2.14 `states_coded` loop count).** Round 6
  attempted the §4.2.14 / §4.2.15 / §4.2.16 / §4.2.17 Parameters tail
  (`states_coded` / `initial_state_delta` / `ec` / `intra`) but found a
  contradiction between Figure 28 and the corpus. Figure 28 reads
  `states_coded` once **per Quantization Table Set** — two `br` symbols
  for these `quant_table_set_count == 2` fixtures. Yet the trailing
  `ec` / `intra` (`ec=1 intra=0` in every `GLOBAL_HEADER`) are
  reproducible bit-exactly across **all four** v3 fixtures (8-bit and
  16-bit, three colorspaces) only when **exactly one** `states_coded`
  `br` is consumed before `ec` (after resetting the Parameters state
  window to 128 post-cascade). Every two-`br` model — residual state,
  whole-buffer reset, per-`br` reset, distinct per-set slots — desyncs
  the range coder and corrupts `ec` / `intra` for at least one fixture.
  This is the §4.2/§4.3 Parameters context-buffer-width ambiguity (the
  per-set loop semantics + reset granularity). Figures 29 / 30 are
  otherwise clear (`pred = j ? initial_states[i][j-1][k] : 128`;
  `initial_state[i][j][k] = (pred + initial_state_delta[i][j][k]) & 255`;
  `CONTEXT_SIZE = 32`; "k as context index"), but no fixture exercises
  `states_coded == 1` to validate the per-`k` window placement. Recommend
  a §4.2.14 clarification of how many `states_coded` symbols are coded
  for `quant_table_set_count > 1` and the exact state-window/reset model
  for the §4.2 tail, plus (ideally) one fixture with `states_coded == 1`.
- RFC 9043 §4.2 says "Parameters has its own initial states, all set
  to 128" without specifying the state-buffer width. Empirically, **all
  Parameters symbols share a single 32-slot context window**: the test
  fixtures decode correctly only with that interpretation. §4.6 has
  the same wording for the Slice Header — round 2 confirmed the
  shared-window hypothesis holds there too (all 6 slice-header
  fixtures decode bit-correctly with a single 32-slot window).
- §4.1's QuantizationTableSet carries the SAME "has its own initial
  states, all set to 128" wording (the #904 context-buffer-width
  ambiguity). Round 5 resolved its reset granularity empirically: the
  per-context state window resets to 128 at the start of **each of the
  five `QuantizationTable`s** (not once per Set, not shared with the
  Parameters prefix), while the arithmetic coder (low / range / byte
  position) continues uninterrupted. This is the *only* interpretation
  under which the `v3-default` (666 / 7563) and `v3-yuv444p16`
  (365 / 5063) `QUANT_TABLE` `context_count` trace values reproduce
  bit-exactly.
- The Configuration Record's last 4 bytes are
  `configuration_record_crc_parity`; the range decoder is in Closed
  mode and reads past-end as zero, so passing the full extradata blob
  (including those 4 bytes) is safe — the early Parameters symbols
  never reach them. Round 6's `validate_configuration_record_crc`
  consumes those bytes explicitly: §4.3.2 makes the whole-blob CRC
  residue (including the parity word) the canonical fixity check, so the
  validator never re-derives the stored parity — it just asserts the
  §4.9.3 generator leaves a remainder of zero.
- `parse_slice_header` takes the slice's range-coded byte region
  excluding its 8-byte SliceFooter (`ec=1`) or 3-byte footer
  (`ec=0`). It returns a typed [`Ffv1SliceHeader`] with the raster
  position and size (post-`+1`), the per-plane
  `quant_table_set_index` array, the picture structure, and SAR.
  The range coder's residual state is *not* exposed — slice content
  decode is a later round.
- `compute_slice_content` consumes a parsed `Ffv1SliceHeader` + the
  Configuration Record + a caller-supplied `FramePixelDimensions`
  (the container's reported frame width / height — FFV1's
  Configuration Record carries no width / height fields per §4.2)
  and returns a `SliceContent` with one `Plane` per
  `primary_color_count` entry. RFC 9043 §4.7.3 has a documentation
  typo: its right-hand side reads `slice_pixel_height` where it
  should read `frame_pixel_height` (the §4.8.2 sibling formula uses
  `frame_pixel_width`). We use the unambiguous reading and the three
  fixture trace files confirm the per-plane pixel dimensions match
  the reference decoder bit-exactly.

## License

MIT — see [LICENSE](./LICENSE).

[RFC 9043]: https://www.rfc-editor.org/rfc/rfc9043.html
