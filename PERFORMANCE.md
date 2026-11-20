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

## Loop-allocation audit and second pass

Baseline for this pass: `19c0e52` (the already optimized encoder above).
The review found and removed the following repeated allocations:

| Location | Repeated creation | Change |
| --- | --- | --- |
| `encode_spectra` | Temporary sign list and sign-field tuples for each Huffman symbol | Reuse precomputed immutable sign sequences for up to four coefficients |
| `encode_spectra` | Range objects and repeated bit-mask computation inside symbol loops | Construct them once per codebook call |
| `_best_quantized` | NumPy scalar wrappers on each coefficient access in eight codebook searches | Convert the quantized block to Python integers once per cache miss |
| `_best_quantized` | Tuples for intermediate winning candidates | Keep local references and freeze only the final winner |
| `encode_pcm` | A fresh zero-filled input array for each frame | Reuse one input buffer, explicitly clearing it before every frame/tail block |
| `MDCT.forward/inverse` | Transform-index arrays and derived index expressions for each call | Precompute integer gather/scatter indices when the transform is constructed |

No floating-point expression, evaluation order, precision, FFT, or packet decision
was changed. The full input buffer is cleared on every frame, so a partial final
frame and tail flushing cannot retain samples from an earlier frame.

The following allocations remain intentionally: per-frame FFT outputs and
arithmetic intermediates, scaled-block objects/arrays, packet output bytes and
cache-key bytes. Reusing them would require more care around buffer ownership,
aliasing, or NumPy output semantics; the existence of an allocation alone does not
mean it is safe or useful to remove it.

Same benchmark protocol as above: three medians per one-second stereo fixture,
caches cold before each encode, same host/Python/NumPy. This table measures the
additional improvement from the allocation pass, not speed versus AT3Tool.

| Input | kbps | Before (seconds) | After (seconds) | Additional speedup |
| --- | ---: | ---: | ---: | ---: |
| silence | 64 | 0.0634 | 0.0605 | 1.05x |
| silence | 128 | 0.0693 | 0.0651 | 1.07x |
| silence | 320 | 0.0719 | 0.0656 | 1.10x |
| music | 64 | 0.3902 | 0.2608 | 1.50x |
| music | 128 | 0.3831 | 0.2526 | 1.52x |
| music | 320 | 0.3946 | 0.2545 | 1.55x |
| noise | 64 | 0.4109 | 0.2718 | 1.51x |
| noise | 128 | 0.4136 | 0.2792 | 1.48x |
| noise | 320 | 0.4328 | 0.2926 | 1.48x |

All nine payload hashes and lengths match. Raw results are in
[benchmarks/allocation-reduction.json](benchmarks/allocation-reduction.json).
All 19 local tests pass, including the previous exact-output corpus, direct
comparisons of all 112 Huffman codebooks with both array and Python-list inputs,
and fresh-versus-reused PCM buffer checks around frame boundaries. Existing
MDCT-definition and upstream C++ DSP/packet comparisons also pass.
