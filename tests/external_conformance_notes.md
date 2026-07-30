# External-conformance corpus — generation + validation notes (r411, updated r416/r420/r434)

`tests/external_conformance.rs` pins a 40-stream self-encoded corpus
(SHA-256 per packet + per §4.3.3 extradata blob) spanning the encoder's
RFC 9043 matrix:

* **versions** 0 / 1 / 3;
* **coders** (§4.2.3) Golomb-Rice (0), range default (1), range custom
  table (2, ±1 deltas on every live `DEFAULT_ONE_STATE` transition);
* **colour** gray, YUV 4:2:0 / 4:2:2 / 4:4:4, YUVA, RGB, RGBA (RCT);
* **depths** 8 / 9 / 10 / 12 / 14 / 16 bit (alpha included at 9 and
  14 bits as of r434);
* **structure** single-slice, 2×2, and non-uniform §4.8 floor-division
  grids on odd dimensions — 3×2, plus (r434) 3×3 non-uniform on both
  axes and a Golomb-Rice non-uniform 2×2 — `ec == 0` and `ec == 1`
  (§4.9.3 slice CRCs);
* **temporal** every stream is a §4.4 keyframe **plus carried
  non-keyframes** (§3.8.1.3 / §3.8.2.5 inter-Frame coder-state carry —
  the v3 per-Slice carry and the r411 v0/v1 single-Slice carry),
  including a 4-frame v3 2×2 chain and a 3-frame v0 Golomb chain
  (state evolving across multiple inter Frames), plus a §4.2.17
  `intra == 1` stream whose every Frame is a keyframe and (r416) a
  **mid-stream-keyframe** stream (`v3-yuv420p8-range-kf2`, keyframes at
  Frames 0 and 2 — the §3.8.1.3 state re-initialises mid-stream; the
  validator reports the §4.4 pattern `1 0 1 0` and decodes bit-exactly).

Sources are synthesised in-test from a deterministic xorshift32 +
band pattern (ramp / noise / flat / texture per plane), so the whole
corpus regenerates from the test alone; no binary fixtures are
committed.

## Black-box validation procedure

Validator: **ffmpeg 8.1** (Homebrew arm64 build, macOS 15), invoked
strictly as an opaque process — no library or source access. Because
this ffmpeg has no raw-FFV1 demuxer, each exported stream was wrapped
in a minimal RIFF/AVI (container layer only, independent of the FFV1
bitstream): FourCC `FFV1`, the §4.2 Configuration Record appended to
`strf` past the 40-byte header for version-3 streams (RFC 9043
§4.3.3.1), packets as `00dc` chunks with an `idx1` index.

Per stream:

```
FFV1_CONFORMANCE_EXPORT_DIR=<dir> cargo test --test external_conformance
# wrap <dir>/<name>/pkt*.bin (+ extradata.bin) into wrapped.avi, then
ffmpeg -threads 1 -i wrapped.avi -f rawvideo -pix_fmt <manifest pix_fmt> out.raw
cmp out.raw <(cat src0.raw src1.raw)
```

`src*.raw` is the test's export of the source planes in the decoder's
raw layout (planar, little-endian 16-bit above 8 bits; RGB exported in
G, B, R (, A) plane order to match `gbrp*` — plane-order conversions
like `bgr0 → gbrp` are lossless byte reorders).

## Results (2026-07-17, r416 — supersedes the 2026-07-11 run)

* **28 / 28 streams decode bit-exactly** (every frame, zero decoder
  warnings — in particular no "bytestream end" mismatch, the r411
  §3.8.1.1.1 termination finding). The validator's info-level
  "ignoring invalid SAR: 0/0" message is cosmetic: RFC 9043 §4.6.8 /
  §4.6.9 say encoders **MUST** write 0/0 when the Sample aspect ratio
  is unknown, which is exactly what these headers carry.

### r416: the `v0-yuv420p8-custom` cell, root-caused and closed

r411 recorded this stream (version 0, `coder_type == 2`) as rejected
by the validator at the first packet and attributed the rejection to a
validator limitation. r416 re-ran the failure and got a *semantic*
diagnostic — `invalid state transition 0` — which pointed at the table
content, not at a missing capability. Three v0 `coder_type == 2` probe
streams discriminated the cause (all validated black-box the same
way):

| probe deltas | transmitted `one_state` | validator result |
| --- | --- | --- |
| r411 scheme (±1 on live entries, 0 on zero-default entries) | zero at `1..=8`, `249..=255` (as in the §3.8.1.5 default) | rejected, `invalid state transition 0` |
| all-zero deltas (custom table == §3.8.1.5 default) | zero at `1..=8`, `249..=255` | rejected, `invalid state transition 0` |
| r411 scheme + self-loop lift (`one_state[i] = i`) on every zero-default entry | fully nonzero | **decodes bit-exactly** |

Conclusion: the validator's version-0/1 inline-Parameters path rejects
a transmitted custom table containing **any** zero transition — even
the table that exactly equals the §3.8.1.5 default — while its
version-3 Configuration Record path accepts the very same delta block
(`v3-yuv420p8-custom` passed in r411 with zero entries). Both table
shapes are RFC-conforming (Figure 28 places no version condition on
the §4.2.4 block, and the zeroed states are unreachable from the
§3.8.1.3 initial state 128), so which one to transmit is encoder
freedom — and the fully-live table is the interoperable choice.
`custom_transition_deltas()` now lifts every zero-default entry to the
self-loop `i`, the corpus was regenerated, and **all 27 streams now
validate bit-exactly**. Byte impact: only the two custom streams
changed (the v3 stream only in its extradata blob — the lifted entries
are never visited while coding, so every coded packet outside the v0
inline-Parameters keyframe is byte-identical).

Cross-check in the opposite direction (decode axis): eleven
reference-encoded keyframe+inter probe streams (gray/yuv/yuva/rgb/rgba
× range/Golomb × v0/v1/v3, produced black-box from `testsrc2` with
`-g 2`) decode bit-exactly through this crate's carry drivers.

## r420: six deep / off-grid-depth cells added (2026-07-21)

The corpus grew from 28 to 34 streams with the round-420 deep-format
registry work:

* `v3-yuva422p10-range`, `v3-yuva444p12-range`, `v3-yuva444p16-range`
  — the deep alpha-carrying Yuva family (§4.2.10 `extra_plane` at
  10 / 12 / 16 bits, 4:2:2 and 4:4:4);
* `v3-yuv444p14-range`, `v3-yuv420p9-range` — the off-grid 14-bit and
  9-bit depths (the depths the framework mapping rides on deeper
  surfaces with a significant-bits record);
* `v3-yuv444p16-range-3x2-odd` — 16-bit 4:4:4 on a non-uniform §4.8
  3×2 grid over odd 63×47 dimensions (deep + floor-division geometry
  in one stream).

Validation run 2026-07-21, same wrap procedure and validator build
class as r416 (ffmpeg 8.1 black-box; minimal RIFF/AVI wrap, `-threads
1`, rawvideo out, `cmp` against the exported sources): **all 34 / 34
streams decode bit-exactly with zero warnings** — the 28 pre-existing
pins revalidated unchanged alongside the six new cells.

## r434: six corpus-C writer-mirror cells added (2026-07-31)

The docs-staged corpus C (`docs/video/ffv1/fixtures/`, 13
reference-encoded non-uniform / deep-colour streams — consumed on the
decode axis by `tests/staged_corpus_c.rs`) exposed writer shapes the
matrix did not yet emit. The corpus grew from 34 to 40 streams:

* `v3-gray14-range-2x2`, `v3-rgba14-range-2x2` — 14-bit single-plane
  gray and 14-bit RGB + alpha (RCT), each on a 2×2 grid like the
  staged `deep-gray14` / `deep-gbrap14` streams;
* `v3-yuva420p9-range-2x2`, `v3-yuva444p9-range-2x2` — 9-bit YUVA at
  4:2:0 and 4:4:4 (the alpha-carrying off-grid depth; previously only
  9-bit YCbCr without alpha was emitted);
* `v3-yuv444p10-range-3x3-odd` — 10-bit 4:4:4 on a 3×3 grid over odd
  97×65, non-uniform on BOTH axes (97 = 3·32+1, 65 = 3·21+2);
* `v3-gray8-golomb-2x2-odd` — §3.8.2 Golomb-Rice on a non-uniform 2×2
  grid over odd 61×47 (every earlier Golomb cell was single-slice on
  even dimensions).

Validation run 2026-07-31, same wrap procedure (minimal RIFF/AVI,
`-threads 1`, rawvideo out, `cmp` against the exported sources;
validator: ffmpeg 8.1 Homebrew arm64, black-box): **all 40 / 40
streams decode bit-exactly with zero warnings** — the 34 pre-existing
pins revalidated unchanged alongside the six new cells.

## Re-pinning discipline

Any encoder change that alters emitted bytes fails the pin gate. Do
NOT re-pin from the new hashes until the full black-box procedure
above has been re-run and every stream decodes bit-exactly again
(since r416 there is no exception entry: all 40 validate).
