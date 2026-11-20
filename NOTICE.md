# Copyright, licensing, and modification notices

Python translation/adaptation by atracdenc-py contributors, November 19, 2026.
Based on [dcherednik/atracdenc](https://github.com/dcherednik/atracdenc), revision
`cb170709f836cb43dc0312455188c791159d81dc`, by Daniil Cherednik and contributors.
The translated upstream code and data remain LGPL-2.1-or-later. The complete
license text is included in LICENSE; original notices must be retained.

The upstream source headers state that AtracDEnc is free software under the GNU
Lesser General Public License, version 2.1 or (at your option) any later version,
without any warranty, including merchantability or fitness for a particular purpose.

## Source mapping

| Python file | Upstream source/derivation | License |
| --- | --- | --- |
| `dsp.py` | `src/atrac/atrac3plus_pqf/atrac3plus_pqf.c`, `src/lib/mdct/mdct.h`, `src/lib/mdct/mdct.cpp`, `src/atrac/at3p/at3p_mdct.cpp` | LGPL-2.1-or-later |
| `bitstream.py` | `src/atrac/atrac_scale.cpp`, `src/atrac/at3p/at3p_bitstream.cpp`, `src/atrac/at3p/at3p_bitstream_impl.h` | LGPL-2.1-or-later |
| `tables.py` / `tables.json` | `src/atrac/at3p/at3p_tables.cpp`, `at3p_tables.h`, `ff/atrac3plus_data.h`, `src/atrac/atrac3plus_pqf/atrac3plus_pqf_data.h` | LGPL-2.1-or-later |
| `encoder.py` | `src/atrac/at3p/at3p.cpp`, plus the pce-at3 configurable-packet-size adaptation | LGPL-2.1-or-later |
| `tools/import_tables.py`, `tools/reference.cpp`, `tools/build_reference.py`, encoder tests | Port/reproduction and validation support | LGPL-2.1-or-later |
| `cli.py`, `riff.py`, `tests/test_converter.py` | Adapted from Porroe/pce-at3; copyright (c) 2026 pce-at3 contributors | MIT; see LICENSE-MIT |
| Other original project files | atracdenc-py contributors | LGPL-2.1-or-later unless explicitly marked otherwise |

The table and FIR coefficient sources include FFmpeg-derived material:

**Copyright (c) 2010–2013 Maxim Poliakovski.** Those files identify themselves as
part of FFmpeg and are licensed under LGPL-2.1-or-later, without warranty. This
attribution covers the imported Huffman, mantissa, spectral-code, and PQF data.
The JSON records the exact source revision and SHA-256 hashes of its input files.
No codec tables were extracted from Sony executables.

## Changes from upstream, dated November 19, 2026

- Translated the PQF, FFT-based MDCT rotations, spectral scaling/quantization,
  Huffman table construction, and packet writer to Python/NumPy.
- Replaced C++ object ownership/buffers and Kiss FFT calls with NumPy arrays/FFT.
  No Kiss FFT source or libgha source is bundled in the runtime implementation.
- Implemented the configurable packet-size patch in EncoderSettings and passed
  the same packet size through bit allocation, packet padding, and RIFF metadata.
- Ported the spectral-only path. Dedicated GHA extraction and tone payloads,
  gain compensation, steep windows, ATRAC1, and plain ATRAC3 are not ported.
- Added source-length-aware flushing and PSP metadata from the pce-at3 CLI.

This is a standalone derived project. A GitHub fork label is not required by the
license. Publishing this source must retain these notices and license texts.
When distributing compiled/dependency binaries, separately comply with their
licenses and any LGPL source/relinking requirements. NumPy is an external BSD
licensed dependency; optional FFmpeg licensing depends on the installed build.
No compiled encoder, FFmpeg, NumPy, upstream source tree, or game audio is bundled.

ATRAC and related names belong to their respective trademark owners. Software
licensing does not determine rights to input audio or patent clearance.
