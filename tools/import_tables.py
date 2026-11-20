#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Reproduce numeric tables from the pinned atracdenc source checkout.

The resulting tables derive from atracdenc and FFmpeg (LGPL-2.1-or-later).
No Sony binaries or separately sourced proprietary codec tables are used.
"""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

REVISION = 'cb170709f836cb43dc0312455188c791159d81dc'


def extract(text, name):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    match = re.search(r'\b' + re.escape(name) + r'\s*(?:\[[^\]]*\])*\s*=\s*\{', text)
    if not match:
        raise ValueError(f'array not found: {name}')
    start = match.end() - 1
    depth = 0
    for end in range(start, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            body = text[start:end + 1]
            body = re.sub(r'static_cast<int8_t>\((\d+)\)', r'\1', body)
            body = re.sub(r'(?<![a-zA-Z0-9_])([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)[fF]\b', r'\1', body)
            body = re.sub(r',\s*}', '}', body)
            return ast.literal_eval(body.replace('{', '[').replace('}', ']'))
    raise ValueError(f'unterminated array: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'atracdenc_py/tables.json')
    args = p.parse_args()
    groups = {
        'src/atrac/at3p/ff/atrac3plus_data.h': [
            'atrac3p_mant_tab', 'atrac3p_wl_cbs', 'atrac3p_ct_cbs', 'atrac3p_wl_ct_xlats',
            'atrac3p_spectra_cbs', 'atrac3p_spectra_xlats', 'atrac3p_spectra_tabs',
            'atrac3p_qu_to_subband', 'atrac3p_subband_to_num_powgrps',
            'atrac3p_tone_cbs', 'atrac3p_tone_xlats'],
        'src/atrac/atrac3plus_pqf/atrac3plus_pqf_data.h': ['ff_ipqf_coeffs1', 'ff_ipqf_coeffs2'],
        'src/atrac/at3p/at3p_tables.cpp': ['src'],
        'src/atrac/at3p/at3p_tables.h': ['SpecsPerBlock', 'BlockSizeTab'],
    }
    result = {'upstream_revision': REVISION, 'license': 'LGPL-2.1-or-later', 'source_sha256': {}}
    for path, names in groups.items():
        data = (args.source / path).read_bytes()
        result['source_sha256'][path] = hashlib.sha256(data).hexdigest()
        for name in names:
            result['scale_factors' if name == 'src' else name] = extract(data.decode(), name)
    args.output.write_text(json.dumps(result, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
