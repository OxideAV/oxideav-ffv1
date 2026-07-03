# oxideav-ffv1 benchmarks

Criterion micro-benchmarks for the RFC 9043 frame decode + encode
paths across the full coder × depth × colorspace × slice-count matrix.
All inputs are synthesised in-bench from deterministic xorshift32
seeds — no committed fixture files, no `docs/` access at run time, no
third-party samples. Every scenario benchmarked here is also pinned
byte-exact by `tests/optimization_pins.rs` (encode side) and the
reference-fixture corpus in `tests/reference_fixture_decode.rs`
(decode side), so a hot-path change that flips one output byte fails
CI before it can ship.

```
cargo bench -p oxideav-ffv1 --bench decode
cargo bench -p oxideav-ffv1 --bench encode
```

## Coverage

One 320×240 Frame per scenario (`benches/common/mod.rs`), on a
realistic §4.1-shaped Quantization Table Set (11 symmetric levels on
the three §3.5 Figure 5 primary differences → `context_count == 666`,
the same shape the registry's default set uses), with mixed content —
a flat half that drives the §3.8.2.2 run mode / zero contexts and a
gradient+noise half that drives the scalar paths. `ec == 1` (per-Slice
§4.9.3 CRC) everywhere, so the CRC gate and parity solver are inside
the measured loop, as they are in real streams.

| axis        | values                                                        |
| ----------- | ------------------------------------------------------------- |
| direction   | `decode_frame` / `decode_frame_rgb`, `encode_frame`           |
| coder §4.2.3| 0 Golomb-Rice, 1 range default-table, 2 range custom-table    |
| depth §4.2.7| 8, 10, 16 bits (16-bit YCbCr range = §3.3.1 alternate median) |
| colorspace  | YCbCr 4:2:0 plane-major, RGB/RCT line-major                   |
| slice grid  | 1×1, 2×2, 4×4 (§4.9.1 trailer chain + per-Slice CRC)          |

Throughput is normalised to **raw-sample bytes** (1 byte/Sample at
≤ 8 bits, 2 otherwise) so MiB/s figures compare across depths and
layouts.

## Round-386 results

Measured on aarch64 (Apple, macOS), Criterion 0.5, `sample_size 20`,
`measurement_time 3 s`. Run-to-run noise on this machine is ±2–3%;
absolute wall times are hardware-relative — the cross-scenario ranking
and the pre→post deltas are the stable signal. "pre" is the round-386
baseline (recorded immediately after the round's §3.8.2.2 run-mode
encoder correctness fix); "post" is after the round's four
optimization commits.

### Decode

| scenario                         | pre       | post      | Δ time  | post thrpt |
| -------------------------------- | --------- | --------- | ------- | ---------- |
| ycbcr420/golomb/8bit/1slice      | 862.9 µs  | 702.3 µs  | −18.6%  | 156 MiB/s  |
| ycbcr420/golomb/16bit/1slice     | 1.082 ms  | 888.6 µs  | −17.8%  | 247 MiB/s  |
| ycbcr420/range/8bit/1slice       | 1.668 ms  | 1.342 ms  | −19.5%  | 82 MiB/s   |
| ycbcr420/range/10bit/1slice      | 2.014 ms  | 1.684 ms  | −16.4%  | 130 MiB/s  |
| ycbcr420/range/16bit/1slice      | 3.397 ms  | 3.053 ms  | −10.1%  | 72 MiB/s   |
| ycbcr420/range-custom/8bit/1slice| 1.671 ms  | 1.341 ms  | −19.7%  | 82 MiB/s   |
| rgb/golomb/8bit/1slice           | 4.424 ms  | 3.696 ms  | −16.4%  | 59 MiB/s   |
| rgb/range/8bit/1slice            | 4.311 ms  | 4.170 ms  | −3.3%   | 53 MiB/s   |
| rgb/range/16bit/1slice           | 5.453 ms  | 5.140 ms  | −5.8%   | 86 MiB/s   |
| ycbcr420/range/8bit/4slices      | 1.540 ms  | 1.361 ms  | −11.6%  | 81 MiB/s   |
| ycbcr420/range/8bit/16slices     | 1.614 ms  | 1.412 ms  | −12.5%  | 78 MiB/s   |
| ycbcr420/golomb/8bit/4slices     | 836.7 µs  | 707.9 µs  | −15.4%  | 155 MiB/s  |

### Encode

| scenario                         | pre       | post      | Δ time  | post thrpt |
| -------------------------------- | --------- | --------- | ------- | ---------- |
| ycbcr420/golomb/8bit/1slice      | 1.235 ms  | 983.4 µs  | −20.3%  | 112 MiB/s  |
| ycbcr420/golomb/16bit/1slice     | 1.443 ms  | 1.092 ms  | −24.3%  | 201 MiB/s  |
| ycbcr420/range/8bit/1slice       | 1.688 ms  | 1.541 ms  | −8.7%   | 71 MiB/s   |
| ycbcr420/range/10bit/1slice      | 1.947 ms  | 1.772 ms  | −8.9%   | 124 MiB/s  |
| ycbcr420/range/16bit/1slice      | 3.065 ms  | 2.944 ms  | −3.9%   | 75 MiB/s   |
| ycbcr420/range-custom/8bit/1slice| 1.691 ms  | 1.544 ms  | −8.7%   | 71 MiB/s   |
| rgb/golomb/8bit/1slice           | 4.901 ms  | 4.580 ms  | −6.5%   | 48 MiB/s   |
| rgb/range/8bit/1slice            | 4.372 ms  | 4.167 ms  | −4.7%   | 53 MiB/s   |
| rgb/range/16bit/1slice           | 5.161 ms  | 4.958 ms  | −3.9%   | 89 MiB/s   |
| ycbcr420/range/8bit/4slices      | 1.680 ms  | 1.535 ms  | −8.6%   | 72 MiB/s   |
| ycbcr420/range/8bit/16slices     | 1.725 ms  | 1.623 ms  | −5.9%   | 68 MiB/s   |
| ycbcr420/golomb/8bit/4slices     | 1.271 ms  | 1.021 ms  | −19.7%  | 108 MiB/s  |

Matrix shape notes:

* **Golomb-Rice decodes ~2× faster than the range coder per raw byte**
  (156 vs 82 MiB/s at 8-bit YCbCr): the §3.8.2.2 run mode covers flat
  regions with a handful of bits, while the §3.8.1 range coder pays
  one multi-`get_rac` symbol per Sample regardless.
* **16-bit YCbCr range decode is the per-Sample-slowest cell** — two
  bytes per Sample halves the MiB/s denominator advantage and the
  §3.3.1 alternate median adds three reinterpret selects per Sample.
* **RGB is per-Sample slower than YCbCr** mostly because it carries 3
  full-resolution Planes (2× the Samples of 4:2:0) plus the §3.7.1
  RCT pass and the `bits + 1` coded width.
* **Slice-count scaling is flat** (1 → 16 Slices costs ≤ 5%): per-Slice
  overhead (header, footer CRC, state re-init) is well amortised at
  320×240.

## Profile (macOS `sample`, post-optimization)

* `decode range 8-bit`: `symbol::decode_symbol` ~78%,
  `reconstruct_row` ~19%, slice-footer CRC ~1.4%.
* `encode rgb range 8-bit`: `symbol::encode_symbol` ~80%,
  `encode_row` ~13%, forward RCT ~2.4%, `RangeEncoder::shift` ~2.2%.
* `decode golomb 8-bit`: `get_ur_golomb_esc` + `get_vlc_symbol` ~58%,
  `reconstruct_row` ~35%, CRC ~1.6%.

The remaining dominance of `decode_symbol` / `encode_symbol` is the
§3.8.1 binary range coder's inherently serial
`low`/`range`-dependency chain — each `get_rac` depends on the
previous one's renormalisation. Further gains there would need
speculative multi-symbol decoding, which is out of proportion for
this crate today.

## Round-386 optimization log

Every step kept outputs byte-identical (encoder pins + reference
fixture corpus green throughout).

1. **Slicing-by-8 §4.9.3 CRC** (`src/crc.rs`): the bit-at-a-time
   register ran over every Slice byte on both the decode gate and the
   encode parity solver (~4% range / ~8% Golomb decode). A plain
   single-table form measured **no better** than the branchless bit
   loop on this core (serial load-to-use latency ≈ eight ALU steps);
   slicing-by-8 — eight compile-time tables, eight independent loads
   per 8-byte block — cut the CRC to ~1.5%. Golomb 16-bit decode
   −12.9% on its own.
2. **Fixed 32-slot context windows** (`src/symbol.rs` + both range
   plane coders): the §3.8.1.2 Figure 21 slot accesses went through
   unsized `&mut [u8]` with per-slot bounds checks; `&mut [u8; 32]`
   windows (one range check at `window_mut`, none inside) plus
   `#[inline]` on `RangeDecoder::get_rac`/`refill` took range decode
   −7…−9%. The mirror `#[inline]` hints on the *encoder*'s
   `put_rac`/`shift`/`renorm` were tried and **reverted**: they
   regressed 16-bit range encode by +39% (renorm-loop code bloat).
3. **§3.8.2 bit-engine fast paths** (`src/bit_reader.rs`,
   `src/golomb_rice.rs`): 32-bit-word accumulator refill; the
   §3.8.2.1 unary prefix decoded by peeking the 12-bit window once
   and counting leading zeros (instead of twelve `get_bit` loop
   rounds); `BitWriter::put_bits` accumulating whole fields; unary
   prefixes emitted as single fields. Golomb decode −13%, Golomb
   encode −22…−25% on top of (1)+(2).
4. **Neighbour-carry stencil loops** (both reconstructors + the range
   encoder row): `l`/`ll` are the Samples the loop just wrote and
   `tl`/`t` slide along the row above, so each Sample loads only
   `tr` + `tt` instead of re-reading all six §3.2 Figure 3 cells
   through bounds-checked indexing. Broad −4…−10% additional across
   the matrix.

Found by the same harness (not a speed item): the §3.8.2.2 run-mode
encoder desync on multi-context quantization tables — see
`tests/golomb_run_mode_multicontext.rs` and the CHANGELOG `Fixed`
entry; the fix landed *before* the baseline was recorded.
