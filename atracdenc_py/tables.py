# SPDX-License-Identifier: LGPL-2.1-or-later
# Translation/adaptation of atracdenc/FFmpeg tables, 2026-11-19; see NOTICE.md.
"""Pinned codec tables and canonical Huffman encoder construction."""
import json
from importlib.resources import files
import numpy as np

TABLES = json.loads(files(__package__).joinpath('tables.json').read_text())
SIZES = TABLES['SpecsPerBlock']
STARTS = TABLES['BlockSizeTab']
SCALE = np.array(TABLES['scale_factors'], dtype=np.float32)
SCALE = np.float32(SCALE / SCALE[-1])
MANT = np.array(TABLES['atrac3p_mant_tab'], dtype=np.float32)
WORD_LENGTHS = (7,)*17 + (6,)*9 + (5, 5, 4, 3, 2, 1)


def canonical(counts, symbols, size):
    result = [(0, 0)] * size
    index, code = 0, 0
    for bits, count in enumerate(counts, 1):
        for _ in range(count):
            result[symbols[index]] = (code, bits)
            index += 1
            code += 1
        code <<= 1
    return result, index


def build_huffman():
    wl = []
    offset = 0
    for word, code in zip(TABLES['atrac3p_wl_cbs'], TABLES['atrac3p_ct_cbs']):
        table, used = canonical(word, TABLES['atrac3p_wl_ct_xlats'][offset:], 8)
        wl.append(table)
        offset += used
        _, used = canonical(code, TABLES['atrac3p_wl_ct_xlats'][offset:], 8)
        offset += used
    spectra = []
    offset = 0
    for counts in TABLES['atrac3p_spectra_cbs']:
        if counts[0] < 0:
            spectra.append(spectra[-counts[0]])
        else:
            table, used = canonical(counts, TABLES['atrac3p_spectra_xlats'][offset:], 256)
            spectra.append(table)
            offset += used
    return wl, spectra


WORD_HUFFMAN, SPECTRAL_HUFFMAN = build_huffman()
