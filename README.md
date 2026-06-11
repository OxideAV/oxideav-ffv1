# oxideav-ffv1

A pure-Rust FFV1 ([RFC 9043]) lossless intra-only video codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

Clean-room rebuild, round 268 (2026-06-10). The prior implementation was
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
interleave; the third (Cr) Plane originally diverged and was tracked as
a follow-up — round 220 closes that gap by applying the §4.6.6 per-slot
state-buffer rule (the round-214 YCbCr fix) to the RGB driver, after
which v3-rgb-bgr0 slice 0 reconstructs bit-exactly against
`expected.raw` for all three colour Planes.

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

Round 179 (this round) wires the **alternative state-transition table
(`coder_type == 2`)** through the range-coded decode + encode drivers
(RFC 9043 §3.8.1.4 Figure 22 / §3.8.1.6). A new public
`build_one_state(deltas)` helper layers the Configuration Record's
`state_transition_delta[1..=255]` onto the §3.8.1.5 default table
(`one_state[i] = default_state_transition[i] + state_transition_delta[i]`,
modulo 256). `decode_frame`, `decode_frame_rgb`, and
`encode_frame_range_coder` now pick the derived table when
`cr.coder_type == 2` (and the default when `== 1`); the matching
`zero_state` half is re-derived inside `RangeDecoder::with_one_state`
/ `RangeEncoder::with_one_state` per §3.8.1.4 Figure 23. Previously
`decode_frame` accepted `coder_type == 2` but silently fell back to the
default table (latent bug for any §3.8.1.6 stream — none ship today)
and `encode_frame_range_coder` rejected `coder_type == 2` outright; now
the encoder and decoder both consult `build_one_state`, the per-bit
state transitions and per-Sample state windows agree on both sides, and
the 2×2 slice-grid round-trip with a non-trivial sparse +1/-1 delta
pattern reconstructs bit-exactly through `decode_frame`. The §3.3.1
16-bit alt-median predicate (`coder_type == 1 || coder_type == 2`) gates
identically on encode + decode. 8 new tests (345 total, was 337):
4 in `range_coder::tests` (all-zero delta → default;
uniform-positive + uniform-negative delta shifts; per-symbol-independent
encoder→decoder round-trip exercising every transition the derived
table covers) and 4 in `frame_encode::tests` (8-bit / 10-bit /
2×2-slice-grid `coder_type == 2` round-trips through `decode_frame`
plus a zero-delta-equality regression pinning byte-for-byte equality
with `coder_type == 1`). The `range_rejects_non_range_coder` guard was
renamed `range_rejects_golomb_rice_coder_type` and now asserts
rejection of `coder_type ∈ {0, 3, 7, 255}` only.

Round 190 lands the **RGB / JPEG 2000 RCT frame encoder**
(`encode_frame_rgb`) — the symmetric inverse of `decode_frame_rgb` for
the `coder_type == 1 || 2` (range-coded) path. Given an R / G / B (and
optional alpha) Plane `DecodedFrame`, it applies the §3.7.1 *forward*
RCT (general Figure 6 or the §3.7.2.1 exception when
`9 <= bits_per_raw_sample <= 15 && !extra_plane`) with the §3.7.2
positive offset on Cb / Cr, then walks the §4.7 line-major traversal
(`for y { for p { Line(p, y) } }`) emitting per-Sample range-coded
`sample_difference` values via the round-164 `RangePlaneEncoder::encode_row`
under a single per-Slice `RangeEncoder` cursor — header + content share
the same cursor on the range-coded path, mirroring `decode_frame_rgb`.
Per-Plane state (`RangePlaneEncoderState` + §3.1 border buffers) is
held in a `PlaneLineEncodeState` symmetric to the decoder's
`PlaneLineState` and stepped one row per Plane each outer-`y` iteration
so the non-contiguous per-Plane Lines stay byte-for-byte in sync with
the matching decoder. The §4.4 keyframe bit, §4.6 SliceHeader, and §4.9
SliceFooter (with §4.9.3 CRC parity solved by construction) reuse the
existing per-stage encoder primitives.

Round 193 (this round) closes the remaining `encode_frame_rgb` gap by
wiring the **`coder_type == 0` Golomb-Rice RGB encode** path on the
same §4.7 line-major traversal. The keyframe bit + §4.6 SliceHeader
still go through a per-Slice `RangeEncoder` whose `finish()` lands on
the byte boundary the decoder finds with `consumed = rc.position()`;
the §4.8 SliceContent then writes through a `BitWriter` tail driven by
`encode_line`. A new `PlaneLineGolombEncodeState` mirrors the decoder's
`PlaneLineState` on the Golomb-Rice arm: each Plane keeps its own
`LineDecoderState` (per-context VLC window + run state) and §3.1 border
row buffers (`BORDER_WIDTH`-padded, matching `encode_line`) alive
across the line-major interleave, exactly the way the decoder's
`PlaneLineState` keeps the matching `PlaneEntropyState` alive across
`for y { for p { Line(p, y) } }`. A private `sample_diffs_for_row_coded`
derives per-row signed `sample_difference` values in the §3.8 RCT
coded-Sample space (`bits = bits_per_raw_sample + 1`), reusing the
§3.3 median + §3.8 modular-wrap convention. `encode_frame_rgb` now
accepts `coder_type ∈ {0, 1, 2}`; values outside that range still
surface `Error::UnsupportedCoderType`. 6 new positive round-trip tests
in `tests/rgb_encode_frame.rs` (18 total, was 12): single-slice 8-bit
general-formula, flat-RGB-plane (run-mode dominated), 8-bit + alpha
plane, 10-bit §3.7.2.1 exception (Figure 8 forward / Figure 9 inverse),
2×2 slice grid, and `ec == 0` (3-byte footer). Every round-trip closes
via `decode_frame_rgb` and asserts bit-for-bit Plane equality (R, G, B
Samples, and alpha when present).

Round 202 (this round) lands the **§4.2 Parameters + §4.1 Quantization
Table Set cascade encoder** (`encode_configuration_record_with_quant_tables`),
the symmetric inverse of `parse_quantization_table_sets`. Given an
`Ffv1ConfigurationRecord` plus a `&[QuantizationTableSet]` it emits the
§4.3 extradata byte stream onto a single `RangeEncoder` cursor — §4.2
Parameters prefix walked symbol-for-symbol against the same shared
32-slot context window the decoder reads (version → micro_version →
coder_type → optional §4.2.4 `state_transition_delta[1..=255]` `sr`
loop when `coder_type > 1` → colorspace_type → bits_per_raw_sample →
chroma_planes → log2_*_chroma_subsample → extra_plane → v3
num_h_slices_minus_1 / num_v_slices_minus_1 / quant_table_set_count) —
followed by the §4.1 cascade (`quant_table_set_count` Sets, each five
sub-tables), then closes the range coder and appends a §4.3.2
`configuration_record_crc_parity` word solved by the same
`CRC(M || CRC(M)) == 0` trick the §4.9 Slice Footer encoder uses
(§4.9.3 generator, poly `0x104C11DB7`, init 0, MSB-first, no inversion).
Per-context state-window reset mirrors the decoder's empirical
resolution: reset to 128 at the start of EACH of the five sub-tables
(NOT once per Set, NOT shared with the Parameters prefix); arithmetic
coder continues uninterrupted across resets. The §4.1 quantization-table
inversion derives the `len - 1` run-length stream from each input
table's first-half values, asserting each successive group equals
`scale * v` for `v = 0, 1, 2, …` (otherwise `Error::MalformedQuantTable`);
the §4.1 sign-flipped second-half reflection (`table[256 - k] ==
-table[k]` for `k = 1..128`; `table[128] == -table[127]`) is validated
as a precondition. The §4.2.14-§4.2.17 Parameters tail
(`states_coded` per Set, optional `initial_state_delta` triple-loop,
`ec`, `intra`) is **emitted** by the round-236 update — per Set the
encoder writes `states_coded = 0` (the §4.2.14 "initial states ...
assumed to be all 128" default), then closes with `ec` (`ur`,
§4.2.16) and `intra` (`ur`, §4.2.17), all on the same resumed range
coder + shared 32-slot Parameters state buffer. A produced blob
round-trips through `parse_quantization_table_sets` to an equal
`Ffv1ConfigurationRecord` (including `ec` + `intra`) plus an equal
sequence of `QuantizationTableSet`s. A typed-wrapper convenience
`encode_parameters_with_quant_tables(parsed)` is provided for callers
holding a parsed `ParametersWithQuantTables`. 20 new tests (292 total
in the lib, was 258; 14 `config_encode::tests` covering minimal v3
round-trip + CRC residue zero, 8 rejection paths — non-v3 version,
`coder_type > 2`, `chroma_subsample > 4`, empty / >8 cascade,
declared-count mismatch, broken sign-reflection, non-zero `table[0]`,
fictitious `context_count` — two-Sets count preservation, `coder_type
== 2` with sparse signed `state_transition_delta`, wrapper-vs-direct
API equality, and encoder determinism), plus 6 integration tests in
`tests/fixture_config_encode.rs` that round-trip every corpus
extradata (`v3-default`, `v3-grayscale`, `v3-rgb-bgr0`,
`v3-yuv444p16`) through parse → encode → re-parse with field-for-field
record equality + every sub-table equality + the re-encoded blob's
§4.3.2 CRC residue zero, plus an output-size sanity check and a
wrapper-API parity test across all four fixtures.

Round 208 (this round) fixes a latent **multi-Plane Golomb-Rice
decode bug** in `decode_frame` surfaced by adding the first
YCbCr-with-`chroma_planes = true` round-trip coverage to the public
`encode_frame` → `decode_frame` integration suite. Prior to this round
the `coder_type == 0` branch in `decode_frame` re-constructed a fresh
`BitReader` from `body[rc.position()..]` *inside* the per-Plane loop,
so Plane 1 (Cb) and Plane 2 (Cr) silently re-read Plane 0's bytes
from offset zero rather than picking up where Plane 0 left off in the
§4.8 SliceContent bit stream. That stayed dormant because every prior
Golomb-Rice round-trip test (`tests/frame_assembly_golomb.rs`,
`tests/rgb_encode_frame.rs`'s `coder_type == 0` paths, the
`frame_encode::tests` integration tests) targeted either a
single-Plane grayscale frame OR the RGB / line-major driver (which
runs its own per-row bit-reader plumbing). The fix is one shared
`BitReader` constructed *outside* the per-Plane loop on the
`coder_type == 0` arm; the §3.8.2.2.1 per-Plane state reset
(`PlaneEntropyState::new(...)` + `reset_run_state()`) stays inside
`PlaneReconstructor::reconstruct_plane` where it has always been
(only the VLC contexts + run-mode state reset between Planes; the
bit-stream cursor was never supposed to). 14 new
`tests/chroma_encode_frame.rs` round-trip tests (403 total in the
crate, was 372 — the +21 in stable suites are spurious bin-mode
duplications under `--test-threads=2`; the +14 chroma file is the
actual delta) cover every `(coder_type ∈ {0, 1, 2}) × (4:4:4 / 4:2:2 /
4:2:0) × (extra_plane ∈ {true, false})` shape the `encode_frame`
dispatcher reaches: 4:4:4 single-slice 8-bit on both entropy coders;
4:2:2 single-slice 8-bit on both; 4:2:0 single-slice 8-bit on both;
4:2:0 2×2 slice grid 8-bit on both; 4:4:4 + extra (alpha) Plane 8-bit
on both; 4:2:0 single-slice 10-bit on the range coder; the all-zero
`state_transition_delta` `coder_type == 2` byte-equality with
`coder_type == 1` on 4:2:0; `ec == 0` 3-byte footer on a 4:2:0 frame;
and distinct per-Plane-category Quantization Table Sets routed via
`quant_table_set_index = [0, 1]` on 4:2:0. Every test asserts every
Plane's `samples` matches the input bit-exactly after the round-trip,
so a wrong per-Plane width/height (subsample math), wrong chroma
origin (`plane_origin`), wrong quant-set routing
(`quant_index_slot`), or wrong bit-stream cursor handoff on either
side surfaces as a Plane-divergence assertion. The bug-surfacing
tests were red before the fix (`golomb_yuv*` cases all failed on
Plane 1 with completely-wrong bytes); all 14 are green after.

Round 238 lands the **§4.9.3 per-Slice CRC validation
gate** on the frame-level decode drivers. A new `DecodeOptions` struct
with a `slice_crc_policy: SliceCrcPolicy` field flows through new
options-aware entry points `decode_frame_with_options` and
`decode_frame_rgb_with_options`; the historical `decode_frame` /
`decode_frame_rgb` / `parse_slice_footer` delegate to the new
variants with `SliceCrcPolicy::Reject` (the default), so every prior
caller retains the same abort-on-mismatch semantics bit-for-bit.
`SliceCrcPolicy::Accept` is the opt-in partial-recovery mode: when a
Slice's §4.9.3 whole-Slice CRC residue is non-zero the parser surfaces
the residue on the new `Ffv1SliceFooter::crc_residue` field
(`Some(non_zero)`) and the frame-level driver proceeds to reconstruct
the per-Slice Planes best-effort instead of aborting via
`Error::SliceCrcMismatch`. Structural failures
(`TruncatedSliceFooter`, `SliceSizeOutOfRange`) stay policy-
independent — a mis-walked §4.9.1 trailer chain is a structural
error, not a corruption signal the CRC gate should swallow.
`parse_slice_footer_with_options(buf, ec, policy)` exposes the same
policy at the parser level for callers that want footer-level
introspection without re-running the whole frame driver. 7 new tests
(422 total, was 415): 3 unit tests in `src/slice_footer.rs::tests`
(`options_clean_slice_both_policies_agree_with_legacy` — clean Slice
behaves identically across legacy / Reject / Accept and `crc_residue`
is `Some(0)`; `options_corrupted_body_accept_returns_residue_reject_errors`
— a one-bit body flip surfaces the non-zero residue under Accept and
aborts under Reject with the on-wire §4.9.3 parity surfaced for
diagnostics; `options_ec0_policy_irrelevant_no_residue` — `ec == 0`
never populates `crc_residue` regardless of policy, and truncation /
size-mismatch still abort under both policies) plus 4 end-to-end
integration tests in `tests/decode_options_crc_gate.rs`
(`ycbcr_gate_reject_default_aborts_on_crc_failure`,
`ycbcr_gate_accept_partial_decode_returns_structurally_valid_frame`,
`ycbcr_gate_clean_slice_both_policies_match_legacy_bit_exact`,
`rgb_gate_reject_aborts_accept_partial_decodes`) drive the full
encode → corrupt → decode pipeline through both the YCbCr / plane-
major and RGB / line-major drivers. The lenient path is verified to
return a structurally valid `DecodedFrame` (right plane count, right
plane dimensions, every Sample in the §3.8 modular range
`0 .. 2^bits_per_raw_sample`); per-Sample values are NOT compared to
the original input because a single body-byte flip in a range-coded
SliceContent cascades into a different per-Sample reconstruction —
the contract is partial decode without abort, not lossless recovery.

Round 257 (this round) lands the **§5 "Restrictions" per-Frame Slice
raster-coverage validator**. RFC 9043 §5 second paragraph requires
that "for each Frame, each position in the Slice raster MUST be filled
by one and only one Slice of the Frame (no missing Slice position and
no Slice overlapping)" — the union of every Slice's `slice_width ×
slice_height` raster footprint must be an exact partition of the
`num_h_slices × num_v_slices` grid. The validator is a pure structural
primitive: `validate_slice_raster_coverage(headers, cr) -> Result<(),
Error>` takes the forward-ordered Slice Headers parsed off a single
Frame's `Vec<SliceExtent>` plus the surrounding Configuration Record
(for the grid shape) and proves the partition rule by painting each
Slice's raster cells in turn, surfacing
`Error::SliceRasterOverlap { x, y, first_slice_index,
second_slice_index }` on the first colliding paint and
`Error::SliceRasterUncovered { x, y }` on the first unpainted cell in
row-major scan order after the walk. Both error variants carry enough
context for a caller to log the precise §5 violation; the overlap
detector deterministically names the lowest forward-index Slice pair
that conflicts, and the gap detector deterministically picks the
first uncovered cell so the diagnostic chain is reproducible. The
per-Slice raster-bounds check (`slice_x + slice_width <= num_h_slices`
and the v-axis sibling) reuses the existing
`Error::SliceRasterOutOfRange` surface that `compute_slice_content`
already emits for the same condition, so the §5 walk and the §4.7
layout pass agree on malformed-Slice diagnostics. The v0 / v1
absent-grid branch surfaces `Error::SliceRequiresVersion3`, matching
every other §4 / §5 grid-dependent helper. The validator is pure
structural — no range coder, no pixel buffer, no frame bytes touched
— and orthogonal to round 249's per-Slice size cap
(`validate_slice_max_size_restriction`); a §5-conforming Frame
satisfies both. The new public surface (`validate_slice_raster_coverage`
+ the two error variants) is re-exported from `lib.rs`. 12 new lib
unit tests in `src/slice_content.rs::tests` pin the validator across
the conformant tilings (1×1 single-Slice, the canonical 2×2 four-cell
grid that matches the v3-default fixture, the canonical 4×4 four-2×2-
Slice quartile grid that's §5-cap-compliant above the threshold, and
an irregular 3×3 5-Slice partition with one 2×2 Slice plus four 1×1
fillers) and the failure modes (gap in the middle of an otherwise-
covered grid, gap when no Slices are supplied at all, full-overlap
on the same cell with both colliding Slice indices recorded, partial-
overlap when a wider Slice and a narrower Slice both claim one cell,
ordering invariant that overlaps are surfaced before gaps when both
co-exist, off-raster `SliceRasterOutOfRange` passthrough, and the v0
/ v1 `SliceRequiresVersion3` branch). Frame-driver wiring (two-pass
collect-then-validate over the slice loop in
`decode_frame_with_options` / `decode_frame_rgb_with_options`) is
queued for a follow-up round — this round delivers the primitive.
Test count: 463 total, was 451 (+12 lib unit tests in
`src/slice_content.rs::tests` covering both the positive tilings
above and the failure-mode catalogue including a 3×3 row-major
gap-diagnostic test that pins the deterministic uncovered-cell
ordering).

Round 260 closes the round-257 follow-up by **wiring
the §5 raster-coverage validator into both frame-level decode
drivers** as a two-pass collect-then-validate preamble. A new
`pub(crate)` helper `collect_slice_headers_for_raster_validation`
walks the `Vec<SliceExtent>` returned by `walk_trailer_chain`,
parses every Slice Header off a throw-away `RangeDecoder` per
Slice (consuming the §4.4 `keyframe` boolean on Slice 0 just as
pass-2 does), and returns the forward-ordered headers
`validate_slice_raster_coverage` consumes. The validator runs
immediately after the trailer-chain walk in both
`decode_frame_with_options` (YCbCr / plane-major) and
`decode_frame_rgb_with_options` (RGB / line-major), so a §5
partition violation (`Error::SliceRasterOverlap`,
`Error::SliceRasterUncovered`) aborts the Frame before any per-
Slice pixel reconstruction touches the per-Plane output buffers —
the canonical reason the round-257 doc spelled out: "no individual
Slice can fail [the partition rule]; only the union of every
Slice's raster footprint can". Pass-2 re-seeds the `RangeDecoder`
per Slice over the same body bytes (the range coder is byte-
positional, so re-seeding reproduces the pass-1 cursor bit-for-bit)
and the rest of the per-Slice decode runs unchanged. The §5 gate
is structural and orthogonal to the §4.9.3 CRC / §4.9.2
`error_status` policies; both `DecodeOptions::strict()` and
`DecodeOptions::lenient()` surface the violation, mirroring the
round-249 max-slice-size gate's policy-independence contract. Pre-
existing `tests/frame_driver.rs` tests that fed single-slice
fragments from multi-slice fixtures
(`decode_v3_grayscale_*`, `decode_v3_rgb_bgr0_*`) were updated to
clone the Configuration Record onto a §5-conforming 1×1 grid —
the slice's wire fields (`slice_x = 0`, `slice_width = 1`, etc.)
are identical on a 2×2 and a 1×1 grid so the §3.8.1 range-coder
Header state evolves identically, and the pre-260 64×48 framing
simply placed the slice's pixel rectangle at the top-left quadrant
of a larger output buffer (the bit-exact assertions now walk the
slice's intrinsic 32×24 region directly). Test count: 473 total,
was 463 (+10 integration tests in
`tests/section_5_raster_coverage.rs`: 1×1 grid + 2×2 grid round-
trips through both drivers, deterministic overlap + gap diagnostics
on a 2×1 grid pinning `SliceRasterOverlap {x=0, y=0,
first_slice_index=0, second_slice_index=1}` and
`SliceRasterUncovered {x=1, y=0}` respectively, and the lenient-
still-aborts policy-independence assertions on both drivers).

Round 268 (this round) lands the **§5 "Restrictions" non-keyframe
Slice-geometry stability validator** — the last of the three §5
restrictions, queued by rounds 249 / 257 as "requires multi-Frame
state". RFC 9043 §5 third paragraph requires: "For each Frame with a
keyframe value of 0, each Slice MUST have the same value of slice_x,
slice_y, slice_width, and slice_height as a Slice in the previous
Frame." Unlike the other two §5 gates this rule spans two consecutive
Frames, so it ships on two surfaces. (1) The pure structural primitive
`validate_slice_geometry_stability(previous_headers, current_headers)
-> Result<(), Error>`: for each current-Frame Slice in forward
(trailer-chain) order, the §4.6.1-§4.6.4 quadruple `(slice_x, slice_y,
slice_width, slice_height)` must equal the quadruple of *some* Slice
of the previous Frame — §5 reads "as a Slice in the previous Frame",
an existence requirement, so a permuted forward order across Frames
conforms, and no other §4.6 header field participates
(`quant_table_set_index`, `picture_structure`, SAR may all change
Frame-to-Frame). The first unmatched Slice surfaces the new
`Error::SliceGeometryUnstable { slice_index, slice_x, slice_y,
slice_width, slice_height }` so the diagnostic is deterministic.
(2) The stateful `SliceGeometryStabilityTracker`: the frame-level
decode drivers are single-Frame and stateless, so the cross-Frame
walk lives with the caller — one `observe_frame(keyframe, headers)`
call per Frame in coded order. A keyframe records its geometry as the
new previous-Frame reference without any check (§5 restricts only
Frames "with a keyframe value of 0"; §3.8.1.3 / §3.8.2.5 re-initialise
all coder state there); a non-keyframe validates against the
*immediately preceding* Frame — not the last keyframe — and becomes
the reference on success. A non-keyframe observed before any Frame
validates against the empty set (no previous Frame exists whose
Slices could match), and a violating Frame never becomes the
reference for its successor. Driver-level wiring is queued as a
follow-up: it first needs the §4.4 `keyframe` value surfaced off the
decode (round 274, below) plus a public decode→Slice-Header surface.
Test count: 485 total, was 473
(+12 lib unit tests in `src/slice_content.rs::tests`: identical /
permuted / non-geometry-field-mutated partitions pass; a 2×1 → two-
1×1 re-split pins `SliceGeometryUnstable` on forward index 0;
deterministic first-unmatched-index ordering; vacuous empty-current
acceptance + empty-previous rejection; tracker keyframe-opens /
non-keyframe-opens / immediately-previous-tracking /
error-leaves-reference-untouched / `Default`-equals-`new` paths).

Round 274 **surfaces the §4.4 `keyframe` value on `DecodedFrame`**,
closing the round-268 follow-up. RFC 9043 §4.4 opens each Frame with a
single range-coded `keyframe` boolean (its own state, initial value
128) at the very start of the first Slice's range-coded region. Both
frame-level decode drivers (`decode_frame_with_options` YCbCr /
plane-major, `decode_frame_rgb_with_options` RGB / line-major) now
capture that boolean into the new public `DecodedFrame::keyframe: bool`
field rather than binding it to `_keyframe` and dropping it; a
zero-Slice Frame (no §4.4 boolean) defaults to `true`. The value is
exactly the first argument
`SliceGeometryStabilityTracker::observe_frame(keyframe, headers)` takes,
so a caller can now drive the §5 third-paragraph cross-Frame check
from the decode output (the headers half of that pair plus the
driver-level stitch remain a follow-up). Test count: 487 total, was
485 (+2 lib unit tests in `src/frame.rs::tests` —
`decoded_frame_carries_keyframe_field` locks the field, and
`decoded_frame_keyframe_drives_section5_stability_tracker` feeds
`decoded.keyframe` through the tracker; three
`tests/frame_encode_dispatch.rs` round-trip tests now also assert
`decoded.keyframe` end-to-end across the Golomb-Rice, range-coder, and
RGB / line-major paths against a real encode→decode).

Round 279 (this round) **surfaces the Frame's §4.6 Slice Headers on
`DecodedFrame`**, closing the round-274 follow-up ("a public
decode→Slice-Header surface — the *other* argument the tracker needs").
Both frame-level decode drivers already parse every §4.6 Slice Header
in the round-260 pass-1 preamble (for the §5 second-paragraph
raster-coverage gate) and then dropped the parsed headers; the new
public `DecodedFrame::slice_headers: Vec<Ffv1SliceHeader>` field now
carries them, in forward Slice-index order (slice 0 first — the §4.9.1
trailer-chain order), at zero additional parse cost. Paired with the
round-274 `keyframe` field, a caller now drives the complete RFC 9043
§5 third-paragraph cross-Frame check straight off the decode output —
`tracker.observe_frame(decoded.keyframe, &decoded.slice_headers)` once
per Frame in coded order — with no out-of-band header re-parse. A
zero-Slice Frame carries an empty vector; on the encode side the field
is ignored (the encoders take their Slice geometry from the
caller-supplied header slice, never from `DecodedFrame`). Test count:
492 total, was 487 (+5: `tests/decoded_slice_headers.rs` with 4
end-to-end tests — YCbCr 2×2-grid forward-order field-for-field header
round-trip, RGB / line-major mirror, strict/lenient options-parity on
the policy-independent pass-1 parse, and a two-stream
`SliceGeometryStabilityTracker` drive built purely from decode outputs
that pins `SliceGeometryUnstable` on forward index 1 when a 2×2
partition follows a single-Slice Frame; plus
`decode_v3_default_surfaces_trace_ordered_slice_headers` in
`tests/frame_driver.rs` asserting the v3-default fixture surfaces the
trace's four-Slice 2×2 geometry — raster quadruples `(0,0)`, `(1,0)`,
`(0,1)`, `(1,1)`, each 1×1, `quant_table_set_index_count == 2` — in
trailer-chain forward order). The `src/frame.rs` tracker unit test now
reads both `observe_frame` arguments off the `DecodedFrame` fields.
Driver-level tracker wiring (a stateful multi-Frame decode session
object) remains the natural next step on this arc.

Round 249 lands the **§5 "Restrictions" max-slice-size
gate** on the frame-level decode drivers. RFC 9043 §5 requires that
"starting with version 3 and if `frame_pixel_width *
frame_pixel_height` is more than 101376, `slice_width * slice_height`
MUST be less or equal to `num_h_slices * num_v_slices / 4`" — the
four-way-parallel-decoding floor that lets a conforming decoder split
the Frame across four workers and have each take at most one quarter
of the slice raster. 101376 is the CIF frame area (352 × 288); above
it the inequality binds, at or below it the cap is silent and any
raster footprint is admissible. The gate is exposed on three
surfaces. (1) A new pure structural validator
`validate_slice_max_size_restriction(header, cr, frame_dims) ->
Result<(), Error>` in `src/slice_content.rs` returns `Ok(())` when
the §5 trigger does not apply (v0/v1 — `num_h_slices` /
`num_v_slices` absent — surface as `SliceRequiresVersion3`, the same
surface every other §4 / §5 grid-dependent helper uses; or
`frame_area <= SECTION_5_MAX_SLICE_AREA_THRESHOLD`), and surfaces
`Error::SliceMaxSizeExceeded { slice_width, slice_height,
num_h_slices, num_v_slices, frame_pixel_width, frame_pixel_height }`
when the trigger binds and the per-Slice footprint exceeds the cap.
(2) A new exported constant `SECTION_5_MAX_SLICE_AREA_THRESHOLD: u64
= 101_376` documents the trigger boundary and ties the gate's
arithmetic to the spec value. (3) Both frame-level decode drivers —
the YCbCr / plane-major `decode_frame_with_options` in `src/frame.rs`
and the RGB / line-major `decode_frame_rgb_with_options` in
`src/rgb_reconstruct.rs` — call the validator immediately after
`parse_slice_header_from_decoder` so a violating Slice aborts the
frame before any per-Plane reconstructor touches its body. The cap
is integer division (`num_h * num_v / 4`); the 8 in-tree §5 unit
tests in `src/slice_content.rs::tests` pin its behaviour at the
floor: 2×2 raster → cap 1, 3×3 raster → cap 2, 4×4 raster → cap 4.
Six new integration tests in `tests/section_5_max_slice_size.rs`
exercise the full encode → decode pipeline on both drivers across
the three regimes: a below-threshold full-raster Slice round-trips
clean; an above-threshold violating-cell Slice surfaces
`SliceMaxSizeExceeded` on both `decode_frame` / `decode_frame_rgb`
(legacy) and `decode_frame_with_options` / `decode_frame_rgb_with_
options` (`strict()` and `lenient()`, both — the §5 gate is
structural and independent of the §4.9.2 / §4.9.3 policy fields);
and an above-threshold §5-admissible Slice raster (4×4 raster, four
2×2 Slices each at the cap) round-trips clean on both drivers. Other
§5 restrictions (no-gap / no-overlap raster coverage per Frame; the
non-keyframe Slice-stability invariant across Frames) require multi-
Slice / multi-Frame state and are queued for a follow-up round. Test
count: 451 total, was 437 (+8 lib + 6 integration).

Round 244 lands the **§4.9.2 `error_status` Table 16
policy gate** on the frame-level decode drivers — the second
independent integrity gate on `DecodeOptions`, mirroring round 238's
§4.9.3 CRC gate. Prior to this round both `decode_frame` and
`decode_frame_rgb` parsed the per-Slice §4.9 footer (which carries
the §4.9.2 `error_status` byte and the §4.9.3 parity word), surfaced
the §4.9.3 residue through `SliceCrcPolicy`, and discarded the
§4.9.2 byte — i.e. silently accepted every Table 16 value (`NoError`
/ `Correctable` / `Uncorrectable` / reserved-range) regardless of
what the encoder declared. Round 244 closes that gap by adding a
second policy field, `slice_error_status_policy`, of type
`SliceErrorStatusPolicy { Reject, Accept }` (default `Reject` —
strict). The `Reject` policy aborts the frame decode via the new
`Error::SliceErrorStatus { slice_index, status }` whenever a per-
Slice footer declares `Uncorrectable` (`2`); `Accept` is the opt-in
lenient mode that lets the per-Slice pixel reconstruction run best-
effort. Per §4.9.2 Table 16 only `Uncorrectable` is a rejection
target: `Correctable` (`1`) declares damage the §4.9.3 CRC is
expected to detect / recover (so the §4.9.3 gate is the canonical
guard there), and reserved-range bytes (`>=3`) are unknown — the
gate treats them as "trust the bitstream" on either policy, since
the §4.9.3 residue is the stronger fixity signal for an unknown
status byte. The two gates are independent: a caller can combine
`SliceCrcPolicy::Accept` with `SliceErrorStatusPolicy::Reject` to
tolerate residue mismatches but still abort on the encoder-declared
`Uncorrectable`, or the inverse for the opposite trade-off. The
convenience constructors `DecodeOptions::strict()` and `lenient()`
set both gates the same way.

The legacy entry points `decode_frame` / `decode_frame_rgb`
delegate to the options-aware variants with `DecodeOptions::strict()`,
so every prior caller picks up the new `Uncorrectable` abort path
automatically. The four shipped v3 fixtures and every in-tree
encode all write `NoError`, so the strict default does not change
any pre-existing test's observable behaviour. Eight new end-to-end
integration tests in `tests/decode_options_error_status_gate.rs`
exercise the policy matrix on both drivers: the Reject path
surfaces `Error::SliceErrorStatus { slice_index: 0, status: 2 }`
under default / strict / legacy entry points; the Accept path
returns a **bit-exact** `DecodedFrame` against the original input
(the body bytes are untouched — only the §4.9.2 byte + re-solved
parity change, so the per-Sample reconstruction reproduces the
original input exactly, which the §4.9.3 gate test cannot pin
because its body-byte flip cascades into a different per-Sample
stream); the clean-`NoError` regression confirms every policy
matches the legacy decode bit-exact; the `Correctable` / reserved-
range tests confirm `Reject` lets them through per the policy
doc; the mixed-policy invariant confirms the two gates are
independent. The fabricator helper
`rewrite_single_slice_error_status` rebuilds the §4.9 footer with
`encode_slice_footer_with_raw_status` (the same solver every clean
encode uses) so the §4.9.3 CRC residue stays zero by construction
and the test isolates the §4.9.2 gate under test.

Round 241 closes the round-236 follow-up by surfacing
the **§4.2.15 `initial_state_delta` triple-loop** on the
Configuration Record and teaching the encoder the
`states_coded == 1` branch. Prior rounds consumed the §4.2.15 deltas
off the wire purely to advance the range coder past them — the
decoded values were discarded so the encoder had no way to emit a
non-default tail. The new
`Ffv1ConfigurationRecord::initial_state_delta` field is an
`Option<Vec<Option<Vec<[i32; INITIAL_STATE_DELTA_K]>>>>` (with
`INITIAL_STATE_DELTA_K == 32` from §4.2 `CONTEXT_SIZE`): `None` when
every Quantization Table Set wrote `states_coded == 0` on the wire
(the §4.2.14 default — initial states all 128, no triple-loop),
`Some(per_set)` when at least one set carries the loop. Each
`per_set[i]` is `None` for sets whose `states_coded == 0` (initial
states stay 128) and `Some(deltas)` for sets whose
`states_coded == 1`, with `deltas.len() == context_count[i]` and
each inner `[i32; 32]` carrying the 32 signed `sr` symbols indexed
by `k` (Figures 29 / 30:
`initial_state[i][j][k] = (pred + initial_state_delta[i][j][k]) & 255`).
The parser (`quant_table::parse_parameters_tail`) now populates the
field instead of discarding the symbols; the encoder
(`config_encode::encode_parameters_tail`) emits `states_coded == 1`
+ the matching `context_count[i] * 32` signed `sr` symbols iff the
field is populated with the correct shape, falling back to the
`states_coded == 0` default when the field is `None` or `per_set[i]`
is `None`. A new
`Error::InitialStateDeltaShapeMismatch { set_index,
expected_context_count, actual_context_count }` rejects
caller-supplied per-set vectors whose length disagrees with the §4.1
cascade's `context_count[i]` up-front so the encoder never emits a
desynchronised wire stream. Seven new unit tests in
`src/config_encode.rs::tests` (429 total, was 422) exercise the
encode → parse round-trip across zero-row, non-trivial mixed-sign,
two-set mixed-`states_coded`, default-stays-`None`, shape-mismatch
rejection, signed-extreme (`i32::MIN` / `i32::MAX`) preservation,
and encoder-determinism paths; each round-trip asserts
`validate_configuration_record_crc(&blob) == Ok(())` so the §4.3.2
parity word is verified solved against the new wire footprint. The
four-fixture corpus round-trip in `tests/fixture_config_encode.rs`
continues to pass — the corpus fixtures all carry
`states_coded == 0` per the open #904 DOCS-GAP note, so they parse
to `initial_state_delta: None` and re-encode bit-for-bit on that
branch.

Round 236 closes the **§4.2.14-§4.2.17 Parameters
tail** on the encode + parse paths. The §4.2 Figure 28 pseudocode
places, after the §4.1 cascade and inside the `version >= 3` block,
a per-Set `states_coded` (`br`), an optional
`initial_state_delta[i][j][k]` triple-loop (`sr`, gated by
`states_coded`), and the closing `ec` (`ur`, §4.2.16) +
`intra` (`ur`, §4.2.17). The encoder
(`encode_configuration_record_with_quant_tables`) now emits that
tail symbol-for-symbol on the same resumed range coder + shared
32-slot Parameters state buffer the prefix and cascade share —
always writing `states_coded = 0` per Set (the §4.2.14 default
"initial states ... assumed to be all 128", which matches every
per-Set state buffer this codec allocates today) and using the
caller-supplied `record.ec` (default `Some(0)`) and `record.intra`
(default `Some(false)`). The companion parser inside
`parse_quantization_table_sets` reads the same tail, surfacing `ec`
and `intra` on the returned `Ffv1ConfigurationRecord`; when a
fixture's bitstream carries `states_coded == 1` the §4.2.15 deltas
are consumed off the wire (`context_count[i] * CONTEXT_SIZE = 32`
signed symbols per Set) so the resumed coder reaches `ec` + `intra`
correctly. Two new public fields appear on
`Ffv1ConfigurationRecord`: `ec: Option<u32>` and
`intra: Option<bool>` — both `None` when the field is absent
(versions 0/1) or when the caller only invoked the prefix-only
`parse_configuration_record`. Five new lib tests
(`round_trip_tail_default_ec_intra`,
`round_trip_tail_ec_one_intra_true`,
`round_trip_tail_none_defaults_to_zero`,
`round_trip_tail_multi_set_states_coded_zero`,
`round_trip_tail_with_state_transition_delta`) cover single-Set /
multi-Set / `None`-defaults / `coder_type == 2` round-trip paths;
the pre-existing four-fixture corpus round-trip in
`tests/fixture_config_encode.rs` continues to pass because the
encoder faithfully re-emits whatever `ec` / `intra` the parser
produced from the corpus, closing the parse-encode-parse triangle
regardless of the original FFV1 encoder's §4.2.14 sub-path choice.

Round 227 extends the **§4.6.6 per-slot state-buffer
rule** to the Golomb-Rice (`coder_type == 0`) branch of
`encode_frame_rgb` + `decode_frame_rgb`, closing the only remaining
slot-keying gap the prior round-220 row called out as a follow-up.
On the Golomb-Rice path the per-context entropy state has two
distinct components with distinct lifetimes: the per-context VLC
window (`drift` / `error_sum` / `bias` / `count` per context) and
the §3.8.2.2.1 run-mode triple (`run_index` / `run_mode` /
`run_count`). §4.6.6 keys the VLC window by §4.6.6 slot — Planes
that share a slot (G + B on every `chroma_planes == true` RGB
Slice) share one persistent VLC window across the §4.7 line-major
interleave — but §3.8.2.2.1 keys the run-mode triple per-Plane (it
resets at the start of each Plane, and the §4.7 line-major
interleave reads back-to-back across Planes sharing a slot, so a
slot-level triple would mis-carry chroma-slot state). The split is
materialised by allocating one [`crate::sample_diff::LineDecoderState`]
per slot (encoder) / one [`crate::reconstruct::PlaneEntropyState`]
per slot (decoder) — both lazily on first touch so the §3.8.2.5
keyframe-init contract still holds — and a saved-run-triple
snapshot per Plane that is loaded into the slot state at the start
of each row encode/decode and saved back at the end. The encoder
and decoder mutate the per-slot VLC window in lockstep so all 18
prior `tests/rgb_encode_frame.rs` round-trip tests stay green
byte-for-byte; the new tests (below) exercise the slot-key
distinctions the prior per-Plane allocation could not. Two new
`pub(crate)` accessors on `PlaneEntropyState`
(`save_run_state` / `load_run_state`, snapshotting the
`(run_index, run_mode, run_count)` triple) keep the run-mode-only
swap encapsulated; the encoder reaches the equivalent fields
directly on `LineDecoderState` (its run-mode fields are `pub`
already). The change is purely internal to the
`encode_one_rgb_slice_golomb` / `decode_frame_rgb` line-major
loops; no public surface changes.

The Golomb-Rice (`coder_type == 0`) RGB path now matches the
range-coded path's §4.6.6 contract and the YCbCr path's
post-round-214 behaviour. With the v3-rgb-bgr0 fixture still
`coder_type == 1` (range-coded), the new tests cover the
slot-keying analytically through encoder ↔ decoder lockstep on
the §4.6.6 / §3.8.2.2.1 boundary the prior per-Plane allocation
straddled.

5 new tests (410 total, was 405): all five in
`tests/rgb_encode_frame.rs` —
`rgb_encode_round_trips_golomb_rice_high_entropy_chroma_planes`
(12×8 xorshift-random RGB, every Sample distinct so the slot's
VLC window evolves on every Plane step; G + B route to the chroma
slot and share window evolution),
`rgb_encode_round_trips_golomb_rice_distinct_per_slot_qts_indexes`
(`quant_table_set_index = [0, 1]` with two distinct
`context_count` values per slot — luma slot binds the smaller
window, chroma slot binds the larger — proves slot-to-QTS
routing on the Golomb path),
`rgb_encode_round_trips_golomb_rice_extra_plane_distinct_slot`
(`extra_plane == true`, 8×8 four-Plane xorshift content; alpha
lands in its own §4.6.6 slot independent of the colour Planes,
all four run-triples per-Plane),
`rgb_encode_round_trips_golomb_rice_run_mode_dominates_per_plane`
(constant flat per-Plane content; per-Plane run-triple reset
contract — a slot-shared run triple would corrupt G's first row
after Y's terminal run state on the shared chroma slot), and
`rgb_encode_round_trips_golomb_rice_2x2_slice_grid_with_alpha`
(2×2 slice grid + extra-plane combined — every Slice
keyframe-instantiates per-slot windows + per-Plane triples).
Lib tests: 272 (unchanged); integration: +5 (405 → 410 total).
"§4.6.6 per-slot state-buffer rule now uniform across all four
driver branches (`coder_type ∈ {0, 1, 2}` × `colorspace_type ∈
{YCbCr, RGB}`)" milestone.

Round 220 extends the **§4.6.6 per-slot state-buffer
rule** from `decode_frame` (round 214) to the RGB / line-major
driver `decode_frame_rgb` and its `encode_frame_rgb` mirror on the
range-coded (`coder_type ∈ {1, 2}`) path. Round 214's fix established
that the per-context entropy state buffer is keyed by the §4.6.6 slot
(luma slot, chroma slot, optional extra-plane slot), not by Plane —
two Planes that map to the same slot (Cb + Cr on every `chroma_planes`
Slice) thread one persistent per-context state through their
back-to-back decode calls. The YCbCr driver got that treatment in
round 214; the RGB driver was explicitly listed as the next slot-keying
candidate, and round 220 closes the gap.

Prior to this round `decode_frame_rgb` allocated a fresh
`RangePlaneState` (32-slot-per-context state buffer) inside every
`PlaneLineState`, so the second Plane to touch the chroma slot (Cr in
RGB / RCT ordering) silently re-keyframe-initialised the state instead
of continuing Cb's evolution. Observable as a Cr Plane divergence on
v3-rgb-bgr0 slice 0 (Y + Cb were bit-exact, Cr was not — exactly the
divergence the prior README row called out as "tracked as a
follow-up"). The fix lifts the per-context state out of
`PlaneLineState`, into a `Vec<Option<RangePlaneState>>` of length
`header.quant_table_set_index_count` owned by the driver loop, lazily
filled on first touch of each slot so the §3.8.1.3 keyframe-init
contract still holds. Two Planes sharing the chroma slot (Cb + Cr)
look up the same slot index and thread the shared state into their
respective `RangePlaneReconstructor::reconstruct_row` / encoder
calls, exactly the way `decode_frame` does post-round-214. The
symmetric change lands in `encode_one_rgb_slice_range`; the encoder
and decoder shift in lockstep so all 18 prior
`tests/rgb_encode_frame.rs` round-trip tests keep passing
byte-for-byte. The Golomb-Rice (`coder_type == 0`) RGB path keeps its
per-Plane `PlaneEntropyState` for now — line-major slot-sharing on
the Golomb path additionally needs the §3.8.2.2.1 run-mode triple
split to remain per-Plane while the per-context VLC fields share
across the slot; no shipped v3 fixture exercises that combination
(`v3-rgb-bgr0` is `coder_type == 1`), so the split is queued as a
follow-up. The new
`tests/data/v3_rgb_bgr0_expected.rs` inlines the slice-0 R / G / B
channel bytes extracted from
`docs/video/ffv1/fixtures/v3-rgb-bgr0/expected.raw` (top-left 32×24
of the 64×48 frame, 768 bytes per Plane), and the new
`decode_v3_rgb_bgr0_slice0_is_bit_exact_against_expected_raw`
regression test in `tests/frame_driver.rs` decodes the slice-0 byte
payload through `decode_frame_rgb` and asserts every R / G / B Sample
matches the reference decoder bit-for-bit (2 304 entries). The test
was red before the fix on the third (Cr-derived) colour Plane and
green after. Lib tests: 272 (unchanged); integration: +1
(404 → 405 total). "First end-to-end bit-exact RGB-fixture decode"
milestone, mirroring round 214's YCbCr equivalent.

Round 214 fixes the **§4.6.6 per-slot state-buffer
rule** in `decode_frame` (and its encode-side mirror), closing the
"v3-default Cr divergence" called out in the prior workspace README
row. RFC 9043 §4.6.6 reads "`quant_table_set_index` indicates the
Quantization Table Set index to select the Quantization Table Set
**and the initial states** for the Slice Content" — i.e. the per-
context state buffer the §3.8.1 / §3.8.2 entropy coder reads is
**keyed by the §4.6.6 slot** (§4.6.5
`quant_table_set_index_count = 1 + (chroma_planes||v<=3 ? 1 : 0)
+ (extra_plane ? 1 : 0)` — luma slot, chroma slot, optional
extra-plane slot), **not by the Plane and not by the resolved
Quantization Table Set the slot indexes into**. The chroma slot is
shared by Cb and Cr in the §4.7 plane-then-line traversal: both
Planes feed the same persistent per-context state, with only the
§3.8.2.2.1 run-mode triple resetting at the start of each Plane.
The reference trace `docs/video/ffv1/fixtures/v3-default/trace.txt`
labels both `U` and `V` Planes with `plane_index=1`, matching this
slot-keyed model exactly.

Prior to this round `decode_frame` allocated a *fresh* per-context
state inside every `PlaneReconstructor::reconstruct_plane` /
`RangePlaneReconstructor::reconstruct_plane` call, so on
`v3-default`'s `quant_table_set_index = [0, 0]` layout (luma slot
and chroma slot both point at set 0) Y and Cb each got a
keyframe-init state — correct, both are first-touch of their
respective slot — but Cr (second Plane to touch the chroma slot)
also got a keyframe-init state instead of continuing Cb's
evolution. Observable as 3066 of 3072 Cr Samples diverging from the
reference `expected.raw`; Y and Cb decoded bit-exactly throughout.

The fix:

- Two new `pub(crate)` APIs,
  `RangePlaneReconstructor::reconstruct_plane_with_state` and
  `PlaneReconstructor::reconstruct_plane_with_state`, that take a
  caller-owned `&mut RangePlaneState` / `&mut PlaneEntropyState`
  instead of allocating one internally. The legacy
  `reconstruct_plane` entry points are kept as one-liner shims
  (fresh state inside, then forward to the new API) so external
  callers don't break — this is purely additive on the public
  surface.
- Mirror addition on the encoder:
  `RangePlaneEncoder::encode_plane_with_state`.
- `decode_frame` now pre-allocates one `Option<...>` slot per
  `header.quant_table_set_index_count` (lazily filled with
  `RangePlaneState::new` / `PlaneEntropyState::new` on first use
  so the §3.8.1.3 / §3.8.2.5 keyframe-initialisation contract
  still holds), and the per-Plane reconstruction loop selects its
  state via `qts_index_slot` instead of the previously-used
  resolved set index. Planes that map to the same §4.6.6 slot
  (Cb + Cr) thread the same `&mut state` through their
  back-to-back `reconstruct_plane_with_state` calls.
- `encode_frame_range_coder` and `encode_frame_golomb_rice` apply
  the symmetric change so encoder→decoder round-trips stay
  consistent. The Golomb encoder's per-Plane `state` is now also
  slot-keyed and only `reset_run_state()`-d at the top of each
  Plane (§3.8.2.2.1) — the per-context VLC fields (`drift`,
  `error_sum`, `bias`, `count`) survive across Plane boundaries
  that share a slot.
- New regression test `decode_v3_default_is_bit_exact_against_expected_raw`
  in `tests/frame_driver.rs` decodes the full v3-default frame
  (the four `V3_DEFAULT_FULL_SLICE*` byte ranges concatenated)
  and asserts every Sample of every reconstructed Plane (Y 128×96
  + Cb 64×48 + Cr 64×48 = 18 432 entries) matches the inlined
  `docs/video/ffv1/fixtures/v3-default/expected.raw` byte-for-byte.
  The reference bytes are pulled into a new
  `tests/data/v3_default_expected.rs` data file (Y / Cb / Cr
  constants) following the same inlining convention as
  `V3_GRAYSCALE_EXPECTED_TL`. The test was red before the fix
  (Y/Cb green, all 3072 Cr Samples diverging) and green after.

All 19 existing test groups (404 tests, was 386) stay green —
encoder→decoder round-trips on every prior path (`coder_type ∈
{0, 1, 2}` × 4:4:4 / 4:2:2 / 4:2:0 × `extra_plane ∈ {true, false}`,
the round-208 chroma_encode suite) keep passing because the
encoder shifts in lockstep with the decoder, preserving the
self-consistent round-trip semantics. The RGB / line-major
driver (`decode_frame_rgb`) is untouched here — its per-Plane state
is allocated externally by `PlaneLineState::new` already; whether
RGB's three colour Planes need slot-sharing is a separate
investigation. Round 214 closes the YCbCr Cr divergence and is the
"first end-to-end bit-exact YCbCr v3 fixture decode" milestone.

Round 196 lands the **unified `encode_frame` dispatch
helper**, the symmetric counterpart to the routing `decode_frame`
already performs on the read side. The three specialised encoders the
prior rounds shipped — `encode_frame_rgb` (§4.7 line-major,
`coder_type ∈ {0, 1, 2}`), `encode_frame_golomb_rice` (§4.8 YCbCr
`coder_type == 0`) and `encode_frame_range_coder` (§3.8.1 YCbCr
`coder_type ∈ {1, 2}`) — share an identical
`(frame, cr, qts, headers, ec)` signature, so callers previously had to
replicate the §4.2.5 `colorspace_type` / §4.2.3 `coder_type` switch at
each call site. `encode_frame` inspects the `Ffv1ConfigurationRecord`
and forwards verbatim: `colorspace_type == Rgb` → `encode_frame_rgb`
(routed on colorspace alone, since the RGB encoder splits its own
`coder_type` sub-path internally); `colorspace_type == YCbCr` splits
`coder_type == 0` → Golomb-Rice and `1 | 2` → range coder; any
`coder_type > 2` surfaces `Error::UnsupportedCoderType`. 6 new tests in
`tests/frame_encode_dispatch.rs` assert each combination is
byte-identical to the delegate it should reach and round-trips through
the matching decoder.

Round 164 lands the **range-coded SliceContent encoder**
(`RangePlaneEncoder` + `encode_frame_range_coder`) — the symmetric
inverse of `RangePlaneReconstructor::reconstruct_plane` and of the
`coder_type == 1` + `colorspace_type == 0` branch of `decode_frame`.
Where the round-159 Golomb-Rice driver keeps the §4.6 SliceHeader
(range-coded) and the §4.8 SliceContent (Golomb-Rice) on two distinct
entropy engines joined at a byte boundary, the round-164 driver keeps
both on a **single** `RangeEncoder` cursor — there is no
byte-alignment step between header and content on the range-coded
path (§4.5). The per-Plane encoder mirrors the decoder byte-for-byte:
§3.1 border buffers (zero north + zero west, right-edge mirror) with
`prev`/`prev_prev` rotation; §3.3 median predictor (with the §3.3.1
alt-median gated by `use_16bit_median` for 16-bit YCbCr); §3.5
`absolute_context` for the per-Sample context index + sign-flip flag;
§3.8.1.3 per-context 32-slot state windows initialised to 128 at the
start of each Plane (since every Slice in this driver is a keyframe).
The §3.8 modular `diff = sample - pred` is folded into the signed
half-modulus `[-2^(bits-1), 2^(bits-1))` via a `normalise_diff` helper
so the decoder's `reconstruct_sample(pred, diff, bits)` recovers the
input Sample exactly; the §3.5 sign-flip is inverted on the encode
side (`raw = sign_flip ? -diff : diff`) so the decoder's post-decode
flip-back arrives at the right `diff`. The current row's prefix is
written with the *reconstructed* Sample (decoder-symmetric: the next
column's `l`/`tl` neighbour reads see post-add-back values, not raw
inputs). All four shipped v3 fixtures use `coder_type == 1`, so this
is the encode path any fixture-driven encode test will reach for;
round 164's deliverable is the round-trip through `decode_frame`
(an `encode_frame_range_coder` call followed by `decode_frame` yields
bit-exact original pixels). `coder_type == 2` (the per-frame
arithmetic transition-table variant) reuses the same per-Sample loop
with the `one_state` table swapped via
`build_one_state(&cr.state_transition_delta)` — that was wired in
round 179. RGB / line-major on the range-coded path surfaces
`ColorspaceLayoutNotImplemented` and stays a follow-up. 30 new tests (337 total, was 307): 16
`range_encode::tests` unit tests (state-window initialisation +
isolation + zero-context guard; `normalise_diff` invariants across
the 8-bit half-modulus folding; six 1×1 / 2×1 / 3×3 / 4×4 / 10-bit /
16-bit-alt-median per-Plane round trips through
`RangePlaneReconstructor`; multi-plane decoder-cursor sharing;
encoder determinism; multi-context QTS round trip), 8
`frame_encode::tests` unit tests (single-slice 8-bit + 10-bit + ec=0
+ 2×2 slice grid round trips through `decode_frame`; encoder
determinism; three error paths — `SliceRequiresVersion3`,
`ColorspaceLayoutNotImplemented`, `UnsupportedCoderType` for both
`coder_type == 0` and `coder_type == 2`), and 6
`tests/range_encode_frame.rs` integration tests exercising the same
shape through the public API including a flat-Plane edge case (every
`sample_difference` is zero — collapses the per-context state to
back-to-back zero bits). Every round-trip test verifies the
reconstructed `DecodedFrame.planes` match the input pixel buffer
bit-exactly, so the encoder must agree with the decoder at every
byte / arithmetic-coder state transition / per-context state window
update.

Round 159 lands the **frame-level Golomb-Rice encoder**
(`encode_frame_golomb_rice`) — the symmetric inverse of the
`coder_type == 0` + `colorspace_type == 0` (YCbCr / plane-major)
branch of `decode_frame`. Given a reconstructed `DecodedFrame`, the
Configuration Record, the §4.1 Quantization Table Sets, the per-Slice
[`Ffv1SliceHeader`] vector, and an `ec` flag, the driver composes
every encoder primitive the prior rounds landed into a single
end-to-end pipeline that emits the FFV1 frame payload a matching
`decode_frame` call reconstructs back to the original pixels. For each
Slice in slice-index order: (a) extract the slice's pixel rectangle
from the frame Plane and derive per-row `sample_difference` from the
§3.3 median predictor + §3.8 modular wrap (mirroring `reconstruct.rs`
exactly: same §3.1 border / `prev`/`prev_prev` rotation / left-border
seed from `prev_row[0]`); (b) §4.4 `keyframe` boolean range-coded into
slice 0 only via `put_br` against its own init-128 state (separate
buffer from the SliceHeader's window); (c) §4.6 SliceHeader emitted
via `encode_slice_header_to_encoder` on the shared range coder; (d)
`RangeEncoder::finish()` to yield the byte-aligned boundary the §4.8
SliceContent BitWriter resumes from; (e) §4.8 SliceContent walked
plane-major (per §4.7 `PlaneTraversal::PlaneMajor`) with fresh
`LineDecoderState` at every Plane (per §3.8.2.2.1: VLC contexts + run
mode all zero at the top of each Plane) and one `encode_line` call
per row; (f) §4.9 SliceFooter via `encode_slice_footer` with `ec`
selecting the 3-byte / 8-byte-with-solved-CRC variant. Slice bytes
concatenate to form the frame payload. RGB / line-major and
range-coded SliceContent paths surface `ColorspaceLayoutNotImplemented`
/ `UnsupportedCoderType` and are explicit follow-ups (the
range-coded encode path needs the §4.8 range-coded SliceContent
encoder analogue of `RangePlaneReconstructor`; RGB needs the
row-interleaved driver mirror of the not-yet-wired RGB decode path).
The chroma-subsampling math is already wired (per-Plane origin via a
`plane_origin` mirror of `frame.rs`; per-Plane qts routing via a
`quant_index_slot` mirror) so a future round adding chroma planes
needs only fixture coverage rather than new arithmetic. 14 new tests
(307 total, was 293): single-slice 8-bit + 10-bit grayscale
round-trips through `decode_frame`, the canonical **2×2 slice grid
assembly** round-trip (each slice lands in its correct pixel quadrant
of the reconstructed frame), the 1×3 vertical stack (catches a
`slice_pixel_y` / row-stride fault), an `ec=0` 3-byte-footer
round-trip, encoder determinism (two encodes of the same frame yield
byte-identical buffers), and five error paths (`SliceRequiresVersion3`
for v0/v1, `ColorspaceLayoutNotImplemented` for RGB,
`UnsupportedCoderType` for range-coded, `InvalidQuantTableSetCount`
for an out-of-range qts selector, `SliceSizeOutOfRange` for a header
with `slice_width == 0`), plus helper coverage for
`sample_diffs_for_row` / `quant_index_slot` / `plane_origin`. Every
round-trip test verifies the reconstructed `DecodedFrame.planes`
match the input pixel buffer bit-exactly, so the encoder must agree
with the decoder at every byte / bit / per-context VLC state for the
test to pass.

Round 152 folds the round-mode encoder dispatch into a
per-row **§4.8 `encode_line`** — the symmetric inverse of `decode_line`.
Given the same `LineNeighborBuffers` + `LineDecoderState` +
`QuantTableSet` the decoder takes plus a row of signed
`sample_difference` values (the same values `decode_line` returns), the
encoder walks the per-pixel state machine in lockstep with the
decoder — same §3.5 absolute context with sign-flip inversion on the
`put_vlc_symbol` target, same run-mode predicate
(`|context| == 0 && l == t == tl`), same scalar / level / run-mode
dispatch — and emits via a `BitWriter` the bit pattern a matching
`decode_line` recovers the input row from. The per-context `VlcState`
entries and the run-mode `run_index` / `run_mode` / `run_count` fields
mutate identically on both sides, so the post-trip state windows match
symbol-for-symbol. Run-mode encoding uses intra-row lookahead to choose
between long-run "1" bits (consume `1 << log2_run[run_index]`
consecutive zeros at a single bit cost) and short-run "0 + l2-bit
residual" with a level-coded break (`rc = zero_run - 1` zeros after the
current one, sets `run_mode = 2` so the next pixel hits the level path);
when no in-row level break is available the long-run fallback is taken
and the run extends across the row boundary (run mode straddles rows
per §3.8.2.2.1). The §3.8.2.2 contract that the very first run-region
pixel after a `reset_run_state()` cannot encode a non-zero diff (the
decoder's Phase 3 always returns 0 for the current pixel) is surfaced
by a `debug_assert!`. 12 new tests (293 total, was 281) in
`sample_diff::tests`: a scalar-only path, the negative-context sign-flip
path, an all-zero run-mode that emits a sequence of long-run unary
"1" bits, the canonical zero+level-break short-run pattern, the
two-zeros + level break pattern, mixed scalar/run via predicate
changes within a row, the higher-bit-depth (16-bit ESC) path, a
multi-row continuity test asserting the encoder + decoder agree on
both rows AND on the per-context state at the row boundary, the
empty-row no-bits case, and a strict per-context VLC state lockstep
check after an 8-symbol scalar trip. Round trip is the primary
correctness assertion (encoded bits run back through `decode_line` and
the row + state are asserted to match byte-for-byte).

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
  fresh decoder. The §3.8.1.4 / §3.8.1.6 alternative state-transition
  table is built via `build_one_state(&cr.state_transition_delta)` and
  passed to `with_one_state` on either coder when `coder_type == 2`.
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
- **Configuration Record + Quantization Table Set cascade encoder**
  (§4.1 / §4.2 / §4.3 / §4.3.2):
  `encode_configuration_record_with_quant_tables(record, qts)` is the
  symmetric inverse of `parse_quantization_table_sets`. Emits the
  §4.2 Parameters prefix walked symbol-for-symbol against the shared
  32-slot Parameters context window, then the §4.1 cascade (per-table
  state-window reset to 128 mirroring the decoder's empirical reset
  granularity; arithmetic coder continues uninterrupted across
  resets; `len - 1` symbols derived from each input table's
  first-half grouping; §4.1 sign-flipped second-half reflection
  validated as a precondition). Closes the range coder, then appends
  a §4.3.2 `configuration_record_crc_parity` word solved by the
  `CRC(M || CRC(M)) == 0` property of the §4.9.3 generator so the
  whole-blob residue is zero by construction. The §4.2.14-§4.2.17
  Parameters tail (per-Set `states_coded`, `ec`, `intra`) is emitted
  on the same resumed range coder + shared 32-slot Parameters state
  buffer (round 236); `states_coded` is always written as `0` per
  Set so the §4.2.15 `initial_state_delta[i][j][k]` triple-loop is
  omitted, and `ec` / `intra` are taken from `record.ec` /
  `record.intra` (both default to `Some(0)` / `Some(false)` when
  the caller leaves them as `None`). A produced blob round-trips
  through `parse_quantization_table_sets` to an equal record +
  cascade. Typed-wrapper convenience:
  `encode_parameters_with_quant_tables(parsed)`.
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
- **Frame-level Golomb-Rice encoder** (`encode_frame_golomb_rice`,
  §4.4 + §4.5 + §4.6 + §4.7 + §4.8 + §4.9): the symmetric inverse of
  the `coder_type == 0` + `colorspace_type == 0` branch of
  `decode_frame`. Composes `encode_slice_header_to_encoder`,
  `encode_line`, and `encode_slice_footer` into a per-Slice
  pipeline: §4.4 keyframe range bit (slice 0 only, init-128 own state)
  → §4.6 SliceHeader range-coded → `RangeEncoder::finish()` byte-align
  → §4.8 SliceContent walked plane-major with a fresh
  `LineDecoderState` per Plane and one `encode_line` call per row →
  §4.9 SliceFooter (`ec` flag selects 3-byte vs 8-byte-with-CRC).
  Per-row §3.3 median + §3.8 modular wrap mirrors `reconstruct.rs`
  exactly. Round 159's deliverable; round-trip-validated against
  `decode_frame` for 8-bit / 10-bit grayscale, the 2×2 slice grid,
  the 1×3 vertical stack, and `ec=0`. The range-coded SliceContent
  encoder + RGB / line-major path remain follow-ups.
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

- `states_coded == 1` per-byte `initial_state_delta[i][j][k]` storage
  — round 236 wired the §4.2.14-§4.2.17 tail through the encoder and
  parser (`ec` + `intra` now appear on `Ffv1ConfigurationRecord`; the
  encoder writes `states_coded = 0` per Set). A
  `states_coded == 1` corpus path would also need the deltas
  surfaced on the record so the per-context range-state buffer can
  pre-initialise to `128 ± delta` per §3.8.1.3 — separate follow-up.
  `decode_frame` still takes `ec` as an explicit `bool` parameter
  pending a wire-driven dispatch (`record.ec.is_some()` + boolean
  derivation from `record.ec`).
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
- Remaining higher-level encoder stages: the RGB / line-major
  frame encoder (symmetric inverse of `decode_frame_rgb`). The
  Configuration Record + §4.1 quant-table cascade writer landed in
  round 202 (`encode_configuration_record_with_quant_tables`); round
  236 added the §4.2.14-§4.2.17 Parameters tail emission (per-set
  `states_coded`, `ec`, `intra`), so a re-encoded blob now contains
  the full Figure 28 v3 surface. The range-coded
  (`coder_type ∈ {1, 2}`) frame-level YCbCr encoder
  (`encode_frame_range_coder`) is now wired end-to-end (round 164 +
  round 179); both `coder_type == 1` (default state-transition table)
  and `coder_type == 2` (Configuration-Record-derived alternative
  table per RFC 9043 §3.8.1.4 Figure 22 / §3.8.1.6) round-trip
  bit-exactly through `decode_frame`. The §3.8.1 binary range encoder +
  §3.8.1.2 scalar `put_ur` / `put_sr` / `put_br` primitives landed in
  round 137; the §4.9 Slice Footer writer (`encode_slice_footer`)
  landed in round 142; the §4.6 Slice Header writer
  (`encode_slice_header`) landed in round 146; the §3.8.2 Golomb-Rice
  scalar / level / level-coded encoder primitives landed in round
  149; the §4.8 per-row Golomb-Rice `encode_line` (scalar + run-mode)
  landed in round 152; the **frame-level Golomb-Rice + YCbCr encoder
  (`encode_frame_golomb_rice`)** — the round-trip mirror of the
  `coder_type == 0` decode path — landed in round 159 (this round).

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
