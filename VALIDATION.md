# Validation — November 19, 2026

Local environment: Linux x86-64, Python 3.14.8, NumPy 2.4.6, FFmpeg.

`ATRACDENC_PY_REFERENCE=/tmp/at3-python-reference python3 -m unittest discover -s tests -v`:
**14 tests passed, no skips** (6.1 seconds in the initial full run).

- FFmpeg decoded generated stereo files at all seven offered CLI bitrates:
  64, 96, 128, 160, 192, 256, and 320 kbps, without decoder errors.
- C++ differential tests compare against the pinned upstream spectral-only path,
  with GHA bypassed in the included tools/reference.cpp harness.
- Impulse-fixture packets were byte-identical to that reference at 64, 128, 192,
  and 320 kbps. This is fixture parity, not universal/default-upstream bit identity.
- PQF and MDCT results match upstream within the test's explicit float tolerances.
  An independent direct cosine-definition test also validates the MDCT.
- Raw decoded timing shows the same 4,464-sample delay. Leading and final source
  impulses survive flushing, and metadata retains the original source length.
- One-sample, silent, and frame-boundary tracks pass. Whole-track, custom and WAV
  imported loops, batch conversion, optional normalization, overwrite protection,
  malformed RIFF, and padded chunk parsing pass.
- A runtime test makes subprocess creation fail and still encodes valid packets.
  WAV encoding requires no external ATRAC codec or compiler.
- Two-second synthetic stereo music/noise samples encoded and decoded at 64, 128,
  192 and 320 kbps. Observed encode times were 1.40, 1.31, 1.22 and 0.90 seconds.
  Aligned synthetic-source SNRs were 14.74, 15.39, 16.14 and 19.77 dB respectively.
  These are illustrative measurements, not a perceptual quality claim or benchmark
  guarantee. GHA omission can affect tonal quality and usable bandwidth.

Not verified: full upstream GHA behavior or quality, AT3Tool parity, actual PSP/
PC Engine game playback, audible loop seams, and other platforms/NumPy versions.
The GitHub workflow checks Python 3.10, 3.12 and 3.14 with FFmpeg; optional C++
reference checks are local developer checks and explicitly skip in the default CI.

Generic FFmpeg WAV decoding exposes raw priming/tail samples in these tests. The
PSP metadata skip is accounted for explicitly when measuring source alignment.
The test suite does not treat generic-player duration as PSP playback validation.

The table importer reproduced the committed JSON byte-for-byte. Python wheel and
source archive builds passed. The wheel bundles tables/licenses and no native
codec; the source archive includes test/reference and table reproduction tools.

Low-risk optimization update: all 17 tests passed, including frozen-baseline byte
comparisons, cache eviction and bit-buffer boundary tests. Nine benchmark outputs
retain identical hashes. See PERFORMANCE.md for measured times and coverage.
