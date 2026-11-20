# Port boundaries and reproduction

The production encoder is Python/NumPy. It does not call atracdenc, bind a native
ATRAC codec, invoke a compiler, or use FFmpeg for encoding. Python dispatches all
codec decisions and packet generation; NumPy supplies general numerical kernels.

## Implemented

1. A 384-tap FIR polyphase analysis filter with 368-sample history and 16 subbands.
2. Float32 MDCT pre/post rotations and FFT, sine-window overlap, odd-band reversal.
3. Upstream 64-entry scale selection, mantissa quantization (ties to even), and
   canonical Huffman tables with the original spectral codebook search.
4. Upstream initial word lengths, word-length delta coding, and decreasing-quant-
   unit packet budgeting (32 -> 28 -> 27 ...), without dedicated tonal data.
5. Python-native packet-size patch: `ceil(kbps*1000*2048/(44100*64))*8` bytes,
   propagated consistently into budgeting, padding, fmt block alignment, byte rate,
   and codec parameters. An explicit packet-size override is also available.
6. Channel history/lookahead, zero tail flush, original sample count, and PSP loops.

## Intentional first-port limits

The full upstream default GHA analysis and dedicated tonal bitstream are not yet
translated. All input is passed through the transform spectral path, subject to
its normal quantization and packet-driven bandwidth reduction. There is no
upstream-default quality or bit-exactness guarantee. The initial port uses the
long sine-window path; upstream gain compensation and steep windows are absent.
The requested first codec is ATRAC3plus; ATRAC1 and plain ATRAC3 are not implemented.

The included reference harness explicitly bypasses GHA to compare that same
spectral path. Matching it does not establish parity with upstream's default GHA
encoder or AT3Tool. NumPy FFT rounding may also differ across versions/platforms.

## Reproduce tables

```sh
git clone https://github.com/dcherednik/atracdenc.git /tmp/atracdenc-reference
git -C /tmp/atracdenc-reference checkout --detach cb170709f836cb43dc0312455188c791159d81dc
python tools/import_tables.py /tmp/atracdenc-reference --output /tmp/tables.json
cmp atracdenc_py/tables.json /tmp/tables.json
```

The importer parses numeric initializers with `ast.literal_eval`; it does not
execute C++ or Python expressions from the upstream source. It handles hexadecimal
constants without losing trailing `f` digits and records source-file hashes.

## Optional native reference tests

A C++ reference is only a development check. It is not a runtime dependency and
is not included in the Python wheel. On Linux with CMake/C++/libsndfile development
headers installed:

```sh
python tools/build_reference.py --source /tmp/atracdenc-reference
ATRACDENC_PY_REFERENCE="$PWD/.reference/atracdenc-reference" \
  python -m unittest discover -s tests -v
```

The helper requires an existing checkout of the exact revision, initializes its
pinned libgha submodule, builds upstream libraries, and compiles tools/reference.cpp.
The native harness compares both DSP stages and full packets with tonal data off.
It never changes upstream source. The normal CLI never discovers or invokes it.
