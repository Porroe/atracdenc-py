# atracdenc-py

A **Python/NumPy ATRAC3plus spectral encoder port** derived from
[dcherednik/atracdenc](https://github.com/dcherednik/atracdenc), with the selectable
packet-size patch implemented directly in Python and PSP RIFF loop metadata.

WAV-to-AT3 encoding runs in Python and NumPy: **no atracdenc executable, Sony tool,
codec binding, C/C++ compiler, or FFmpeg encoder is used at runtime**. The codec
tables are included as JSON and do not need downloading.

This first port implements the polyphase analysis filter bank, long-window MDCT,
scaling, quantization, Huffman coding, packet budgeting, encoder delay/tail flush,
and PSP `fmt`/`fact`/`smpl` metadata. It encodes the full input through the spectral
path. **Dedicated GHA tonal extraction is not ported yet**, so this is not a full
feature/quality-equivalent replacement for upstream atracdenc. ATRAC1 and plain
ATRAC3 are outside this first release.

Files decode in local FFmpeg tests. PC Engine game playback, PSP hardware, audible
loop transitions, and AT3Tool parity remain unverified.

## Install

Requires Python 3.10+ and NumPy. From a checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

No compiler or native ATRAC library is required. NumPy is a general numerical
library, usually installed as a prebuilt wheel; this is not a standard-library-only
implementation. FFmpeg is optional for non-WAV normalization and decode tests.

## Convert

```sh
# Input: 44.1 kHz, 16-bit stereo PCM WAV. Default target: 128 kbps.
python -m atracdenc_py convert track.wav track.at3

# Infinite whole-track loop.
atracdenc-py convert track.wav track.at3 --wholeloop

# Loop samples [44100, 882000); sample positions are at the output rate.
atracdenc-py convert track.wav track.at3 --loop-start 44100 --loop-end 882000

# Import exactly one integer, infinite, forward WAV smpl loop.
atracdenc-py convert track.wav track.at3 --import-loop

# Select packet size through bitrate, or set it explicitly.
atracdenc-py convert track.wav track.at3 --bitrate 192
atracdenc-py convert track.wav track.at3 --packet-bytes 1120

# Batch WAV files without changing track stems/numbers.
atracdenc-py batch extracted-wavs converted-at3 --wholeloop

# Optional FFmpeg preprocessing only; encoding remains Python/NumPy.
atracdenc-py convert song.flac song.at3 --normalize

atracdenc-py inspect track.at3
atracdenc-py compare reference.at3 track.at3
```

`--loop-end` is **exclusive**. Stored RIFF endpoints are inclusive, with the PSP
sample offset applied. Source-loop import cannot be combined with normalization;
provide explicit rescaled positions instead. Whole-track loops are never implicit.

Existing outputs require `--overwrite`. Outputs are written atomically after
successful conversion. Batch conversion stops at the first error and keeps earlier
successes. Exit codes: 0 success/equal, 1 metadata warnings/comparison differences,
2 input/conversion errors.

## Encoder API

```python
import numpy as np
from atracdenc_py.encoder import EncoderSettings, encode_pcm
from atracdenc_py.riff import make_at3

pcm = np.zeros((2, 44100), dtype=np.int16)  # (channels, samples), signed-16 units
settings = EncoderSettings(bitrate=128)
packets = encode_pcm(pcm, settings)
with open("silence.at3", "wb") as f:
    f.write(make_at3(packets, pcm.shape[1], settings, loop=(0, pcm.shape[1])))
```

The lower-level `Encoder.encode_block()` API accepts `(channels, 2048)` blocks and
preserves filter history. It consumes one lookahead block before emitting packets;
`encode_pcm()` handles block padding and tail flushing automatically. The core can
emit mono packets; the current PSP RIFF profile and CLI require stereo.

The command-line bitrate choices are 64/96/128/160/192/256/320 kbps. Packet size is
rounded up to a multiple of eight bytes, e.g. 128 kbps gives 744-byte packets
(about 128.166 kbps). `--packet-bytes` overrides bitrate sizing and is experimental
outside the tested sizes. Full-input and output buffers are held in memory.

## Test

```sh
python -m unittest discover -s tests -v
```

FFmpeg decode tests explicitly skip when FFmpeg is absent. Optional upstream C++
differential tests also skip unless a reference harness is supplied; they are not
needed to use the encoder. See [PORTING.md](PORTING.md) for reproduction and
[VALIDATION.md](VALIDATION.md) for measured results and limitations.

## License and provenance

The translated codec and its tables are **LGPL-2.1-or-later**, preserving atracdenc
and FFmpeg attribution. The reused CLI, RIFF helpers, and converter tests retain
their MIT notices. See [NOTICE.md](NOTICE.md), [LICENSE](LICENSE), and
[LICENSE-MIT](LICENSE-MIT) for exact scope.

This is an independent derived project, not an official upstream release. Sony
binaries, game assets, and runtime codec binaries are not included.
