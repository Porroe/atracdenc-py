# SPDX-License-Identifier: LGPL-2.1-or-later
# Python translation/adaptation of atracdenc at3p_bitstream.cpp, 2026-11-19.
# See NOTICE.md for original authors, source mapping, and modification notices.
"""Frozen 5ff4d6e compatibility oracle; tests only, never used at runtime.

Keep the original algorithm here so exact-output regressions are checked in the
same numerical environment, independent of platform FFT rounding differences.
"""
from dataclasses import dataclass
import numpy as np
from atracdenc_py.tables import (TABLES, SIZES, STARTS, SCALE, MANT, WORD_LENGTHS,
                     WORD_HUFFMAN, SPECTRAL_HUFFMAN)


class Bits:
    def __init__(self):
        self.value = 0
        self.size = 0

    def put(self, value, count):
        value, count = int(value), int(count)
        if count < 0 or value < 0 or value >= (1 << count):
            raise ValueError(f'code {value} does not fit in {count} bits')
        self.value = (self.value << count) | value
        self.size += count

    def extend(self, parts):
        for value, size in parts:
            self.put(value, size)

    def finish(self, packet_bytes):
        if self.size > packet_bytes * 8:
            raise ValueError('packet exceeded bit budget')
        return (self.value << (packet_bytes * 8 - self.size)).to_bytes(packet_bytes, 'big')


@dataclass
class Scaled:
    index: int
    values: np.ndarray


def scale_spectra(spectra):
    result = []
    for start, size in zip(STARTS, SIZES):
        src = spectra[start:start + size]
        maximum = min(float(np.max(np.abs(src))), 1.)
        idx = min(int(np.searchsorted(SCALE, maximum)), 63)
        values = np.float32(src / SCALE[idx])
        # Match upstream: only replace samples at/over unity, not all > .99999.
        values = np.where(np.abs(values) >= 1., np.where(values > 0, .99999, -.99999), values).astype(np.float32)
        result.append(Scaled(idx, values))
    return result


def quantize(block, word_length):
    multiplier = np.float32(1.) / MANT[word_length]
    # lrint under FE_TONEAREST: round ties to even.
    return np.rint(np.float32(block.values * multiplier)).astype(np.int32)


def encode_spectra(values, table_index):
    group, coeffs, bits, signed = TABLES['atrac3p_spectra_tabs'][table_index]
    table = SPECTRAL_HUFFMAN[table_index]
    out = []
    pos = 0
    while pos < len(values):
        if group != 1:
            out.append((1, 1))
        for _ in range(group):
            code = 0
            signs = []
            for i in range(coeffs):
                value = int(values[pos])
                pos += 1
                if not signed and value:
                    signs.append((int(value < 0), 1))
                    value = abs(value)
                else:
                    value &= (1 << bits) - 1
                code |= value << (bits * i)
            vlc = table[code]
            if not vlc[1]:
                raise ValueError('quantized value has no Huffman code')
            out.append(vlc)
            out.extend(signs)
    return out


def best_spectra(block, word_length):
    values = quantize(block, word_length)
    best = None
    for i in range(8):
        parts = encode_spectra(values, word_length - 1 + 7*i)
        size = sum(bits for _, bits in parts)
        if best is None or size < best[0]:
            best = (size, i, parts)
    return best[1], best[2]


def word_lengths(writer, lengths, channels):
    delta = [(lengths[i] - lengths[i-1]) & 7 for i in range(1, len(lengths))]
    max_delta = 0
    for a, b in zip(lengths[1:], lengths):
        max_delta |= abs(a-b)
    choices = (2, 3) if max_delta >= 3 else ((1,) if max_delta == 2 else (0,))
    best = min(choices, key=lambda t: sum(WORD_HUFFMAN[t][d][1] for d in delta))
    writer.extend([(3, 2), (0, 2), (0, 2), (best, 2), (lengths[0], 3)])
    writer.extend(WORD_HUFFMAN[best][d] for d in delta)
    if channels == 2:
        # Both channels use the same upstream initial word-length allocation.
        writer.extend([(1, 2), (0, 2), (0, 2)])
        writer.extend(WORD_HUFFMAN[0][0] for _ in lengths)


def frame_packet(spectra, packet_bytes):
    """Port the upstream spectral path and decreasing-quant-unit budget loop.

    Dedicated GHA tone separation is not used in this initial port: all signal
    energy is encoded in the transform residual. This is a valid ATRAC3plus mode.
    """
    channels = len(spectra)
    scaled = [scale_spectra(ch) for ch in spectra]
    cache = {}
    units = 32
    while units:
        writer = Bits()
        writer.extend([(0, 1), (channels - 1, 2), (units - 1, 5), (0, 1)])
        lengths = WORD_LENGTHS[:units]
        word_lengths(writer, lengths, channels)
        for ch in range(channels):
            writer.put(0, 2)
            for qu in range(units):
                writer.put(scaled[ch][qu].index, 6)
        selections = []
        for ch in range(channels):
            row = []
            for qu in range(units):
                key = ch, qu, lengths[qu]
                if key not in cache:
                    cache[key] = best_spectra(scaled[ch][qu], lengths[qu])
                row.append(cache[key])
            selections.append(row)
        writer.put(1, 1)  # full code table set
        for row in selections:
            writer.extend([(0, 1), (0, 2), (0, 1)])
            for idx, _ in row:
                writer.put(idx, 3)
        for row in selections:
            for _, parts in row:
                writer.extend(parts)
            groups = TABLES['atrac3p_subband_to_num_powgrps'][TABLES['atrac3p_qu_to_subband'][units - 1]]
            for _ in range(groups):
                writer.put(15, 4)  # power compensation off
        if channels == 2:
            writer.put(0, 2)  # no swaps or sign inversion
        for _ in range(channels):
            writer.put(0, 1)  # long sine window
        for _ in range(channels):
            writer.put(0, 1)  # no gain compensation
        writer.extend([(0, 1), (0, 1), (3, 2)])  # no tones, no noise, terminator
        if writer.size <= packet_bytes * 8:
            return writer.finish(packet_bytes)
        units = 28 if units == 32 else units - 1
    raise ValueError('packet too small for channel metadata')
