# External-conformance corpus — generation + validation notes (r411)

`tests/external_conformance.rs` pins a 24-stream self-encoded corpus
(SHA-256 per packet + per §4.3.3 extradata blob) spanning the encoder's
RFC 9043 matrix:

* **versions** 0 / 1 / 3;
* **coders** (§4.2.3) Golomb-Rice (0), range default (1), range custom
  table (2, ±1 deltas on every live `DEFAULT_ONE_STATE` transition);
* **colour** gray, YUV 4:2:0 / 4:2:2 / 4:4:4, YUVA, RGB, RGBA (RCT);
* **depths** 8 / 10 / 12 / 14 / 16 bit;
* **structure** single-slice, 2×2, and non-uniform 3×2 grids on odd
  dimensions (§4.8 floor division), `ec == 0` and `ec == 1` (§4.9.3
  slice CRCs);
* **temporal** every stream is a §4.4 keyframe **plus one carried
  non-keyframe** (§3.8.1.3 / §3.8.2.5 inter-Frame coder-state carry —
  the v3 per-Slice carry and the r411 v0/v1 single-Slice carry).

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

## Results (2026-07-11)

* **23 / 24 streams decode bit-exactly** (both frames, zero decoder
  warnings — in particular no "bytestream end" mismatch, the r411
  §3.8.1.1.1 termination finding).
* **`v0-yuv420p8-custom` (version 0, `coder_type == 2`)** is rejected
  by the validator at the first packet. The stream is RFC-conforming —
  Figure 28 places the §4.2.4 `state_transition_delta` block behind
  `coder_type > 1` with **no version condition**, and §4.2.3 Table 7
  imposes no version restriction — but the validator does not
  implement v0/v1 custom-table streams: its own encoder silently
  coerces a custom-table request at versions 0/1 (the produced file is
  byte-identical to the default-table encode). The stream therefore
  stays in the corpus pinned on bit-exact self round-trip only,
  flagged here as a validator limitation rather than an encoder
  defect.

Cross-check in the opposite direction (decode axis): eleven
reference-encoded keyframe+inter probe streams (gray/yuv/yuva/rgb/rgba
× range/Golomb × v0/v1/v3, produced black-box from `testsrc2` with
`-g 2`) decode bit-exactly through this crate's carry drivers.

## Re-pinning discipline

Any encoder change that alters emitted bytes fails the pin gate. Do
NOT re-pin from the new hashes until the full black-box procedure
above has been re-run and every stream (except the documented
validator-limitation entry) decodes bit-exactly again.
