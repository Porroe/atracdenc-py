# Low-risk encoder optimizations — November 19, 2026

These changes preserve the DSP, quantization arithmetic, rounding, codebook
selection/tie-breaking, packet-size search order, and RIFF metadata:

- Cache best Huffman encodings by the complete quantized int32 block bytes and
  word length. Byte length captures coefficient count. The LRU is limited to
  512 blocks and stores immutable tuples; an evicted block is recomputed normally.
- Cache word-length header fields by the exact word-length tuple and channel
  count, limited to 64 entries.
- Replace the packet-sized growing integer writer with a bytearray and a small
  pending-bit accumulator. Whole 32-bit words are flushed to amortize buffer
  operations; at most 31 bits remain pending between writes. Bit order, original
  bit count, non-consuming finish(), and zero padding are preserved.
- Count the exact bits for each packet-size candidate before serializing it.
  Oversized candidates are skipped; only the first candidate that fits is written.
  The original 32 -> 28 -> 27 ... quant-unit search is unchanged. The serialized
  count is also checked against the budget calculation.

Caches are shared within the process and bounded; the codec's internal block
sizes are at most 128 int32 coefficients. DSP buffers/state are not cached. No
multithreading, multiprocessing, different FFT, or lower-precision math is added.

## Measured full-encode times

One Linux x86-64 host, Python 3.14.8, NumPy 2.4.6. Each fixture is one
second of stereo audio. Values below are medians of three complete encode calls,
with caches cleared before every call. Cache reuse within a file counts. Imports,
file I/O, optional normalization, RIFF packaging and decoding are excluded.
Baseline: commit `5ff4d6e`. This is a small synthetic benchmark, not an AT3Tool
comparison or a guarantee for every workload.

| Input | kbps | Before (seconds) | After (seconds) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| silence | 64 | 0.4763 | 0.0657 | 7.25x |
| silence | 128 | 0.3821 | 0.0680 | 5.62x |
| silence | 320 | 0.3762 | 0.0671 | 5.60x |
| music | 64 | 0.6524 | 0.3848 | 1.70x |
| music | 128 | 0.5997 | 0.3894 | 1.54x |
| music | 320 | 0.4352 | 0.4044 | 1.08x |
| noise | 64 | 0.8073 | 0.4212 | 1.92x |
| noise | 128 | 0.7650 | 0.4243 | 1.80x |
| noise | 320 | 0.5117 | 0.4431 | 1.15x |

All nine benchmark packet payloads have identical SHA-256 hashes and lengths.
Raw timings, hashes, Python/NumPy versions and cache policy are recorded in
[benchmarks/low-risk-optimizations.json](benchmarks/low-risk-optimizations.json).

## Exact-output checks

The regression suite retains a frozen copy of the pre-optimization bitstream
algorithm in `tests/legacy_bitstream.py`. It runs both implementations under the
same NumPy environment so FFT rounding differences between environments do not
invalidate the comparison. This module is tests-only and excluded from the wheel.

Checks compare bytes directly for 84 encodes across six fixture types, mono and
stereo, and all seven CLI bitrates, plus five explicit packet sizes. Fixtures cover
one-sample silence, frame-aligned silence, boundary/head/tail impulses, music,
seeded noise, and alternating signed-16 extrema. Stereo packet comparisons also
check 126 complete RIFF files with absent, whole-track and custom loops.

Additional tests cover arbitrary field boundaries, wide fields, padding, repeated
finish/write operations, invalid values, cache hits/key distinctions, immutable
results, and eviction beyond the 512-entry limit. All 17 local tests pass,
including optional upstream C++ differential checks and FFmpeg decode tests.

## Reproduce the benchmark

Use two checkouts with the same Python and NumPy environment and no competing
CPU-heavy work:

```sh
python tools/benchmark_optimizations.py --source /path/to/baseline --output /tmp/before.json
python tools/benchmark_optimizations.py --source . --output /tmp/after.json
```

The helper prints per-case medians and records timings and payload hashes. Runtime
tests assert exact outputs; timing thresholds are deliberately not CI assertions.
