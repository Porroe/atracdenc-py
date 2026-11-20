# SPDX-License-Identifier: LGPL-2.1-or-later
# Python translation/adaptation of atracdenc at3p_bitstream.cpp, 2026-11-19.
# See NOTICE.md for original authors, source mapping, and modification notices.
"""ATRAC3plus scaling, quantization, Huffman coding, and packet budgeting."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from .tables import (TABLES, SIZES, STARTS, SCALE, MANT, WORD_LENGTHS,
                     WORD_HUFFMAN, SPECTRAL_HUFFMAN)


class Bits:
    """MSB-first writer with a byte buffer and at most 31 pending bits."""
    def __init__(self):
        self._buffer = bytearray()
        self._pending = 0
        self._pending_bits = 0
        self.size = 0

    @property
    def value(self):
        # Preserve read access for callers of the original writer. Encoding does
        # not reconstruct this growing integer on the hot path.
        return (int.from_bytes(self._buffer, 'big') << self._pending_bits) | self._pending

    def put(self, value, count):
        value, count = int(value), int(count)
        if count < 0 or value < 0 or value >= (1 << count):
            raise ValueError(f'code {value} does not fit in {count} bits')
        combined = (self._pending << count) | value
        total = self._pending_bits + count
        if total >= 32:
            # Flush whole 32-bit words, amortizing bytearray/bytes operations
            # without returning to a packet-sized growing integer.
            remainder = total & 31
            self._buffer.extend((combined >> remainder).to_bytes((total >> 5) * 4, 'big'))
            combined &= (1 << remainder) - 1
            total = remainder
        self._pending = combined
        self._pending_bits = total
        self.size += count

    def extend(self, parts):
        for value, size in parts:
            self.put(value, size)

    def finish(self, packet_bytes):
        if self.size > packet_bytes * 8:
            raise ValueError('packet exceeded bit budget')
        result = bytes(self._buffer)
        if self._pending_bits:
            result += (self._pending << (-self._pending_bits & 7)).to_bytes(
                (self._pending_bits + 7) // 8, 'big')
        return result + bytes(packet_bytes - len(result))


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


@lru_cache(maxsize=512)
def _best_quantized(word_length, quantized_bytes):
    # The key includes every encoded integer, its count (via byte length), and
    # the word length. Channel, scale index, and source floats no longer affect
    # Huffman selection once quantization has finished. Cache only immutable data.
    values = np.frombuffer(quantized_bytes, dtype=np.int32)
    best = None
    for i in range(8):
        parts = encode_spectra(values, word_length - 1 + 7*i)
        size = sum(bits for _, bits in parts)
        if best is None or size < best[0]:
            best = (size, i, tuple(parts))
    return best[1], best[2]


def best_spectra(block, word_length):
    values = quantize(block, word_length)
    return _best_quantized(word_length, values.tobytes())


@lru_cache(maxsize=64)
def _word_length_parts(lengths, channels):
    delta = [(lengths[i] - lengths[i-1]) & 7 for i in range(1, len(lengths))]
    max_delta = 0
    for a, b in zip(lengths[1:], lengths):
        max_delta |= abs(a-b)
    choices = (2, 3) if max_delta >= 3 else ((1,) if max_delta == 2 else (0,))
    best = min(choices, key=lambda t: sum(WORD_HUFFMAN[t][d][1] for d in delta))
    parts = [(3, 2), (0, 2), (0, 2), (best, 2), (lengths[0], 3)]
    parts.extend(WORD_HUFFMAN[best][d] for d in delta)
    if channels == 2:
        # Both channels use the same upstream initial word-length allocation.
        parts.extend([(1, 2), (0, 2), (0, 2)])
        parts.extend(WORD_HUFFMAN[0][0] for _ in lengths)

    return tuple(parts)


def word_lengths(writer, lengths, channels):
    writer.extend(_word_length_parts(tuple(lengths), channels))


def frame_packet(spectra, packet_bytes):
    """Port the upstream spectral path and decreasing-quant-unit budget loop.

    Dedicated GHA tone separation is not used in this initial port: all signal
    energy is encoded in the transform residual. This is a valid ATRAC3plus mode.
    """
    channels = len(spectra)
    scaled = [scale_spectra(ch) for ch in spectra]
    # Quantization and codebook choice are independent of the packet's final
    # number of quant units. Compute each block once, then budget its prefix.
    selections = [[best_spectra(block, WORD_LENGTHS[qu])
                   for qu, block in enumerate(row)] for row in scaled]
    payload_bits = [0]
    for qu in range(32):
        payload_bits.append(payload_bits[-1] + sum(
            size for row in selections for _, size in row[qu][1]))

    units = 32
    while units:
        lengths = WORD_LENGTHS[:units]
        word_parts = _word_length_parts(lengths, channels)
        groups = TABLES['atrac3p_subband_to_num_powgrps'][TABLES['atrac3p_qu_to_subband'][units - 1]]
        size = (9 + sum(count for _, count in word_parts)
                + channels * (2 + 6 * units)  # scale indices
                + 1 + channels * (4 + 3 * units)  # spectral table indices
                + payload_bits[units] + channels * 4 * groups
                + (2 if channels == 2 else 0) + 2 * channels + 4)  # footer
        if size <= packet_bytes * 8:
            break
        units = 28 if units == 32 else units - 1
    if not units:
        raise ValueError('packet too small for channel metadata')

    writer = Bits()
    writer.extend([(0, 1), (channels - 1, 2), (units - 1, 5), (0, 1)])
    writer.extend(word_parts)
    for ch in range(channels):
        writer.put(0, 2)
        for qu in range(units):
            writer.put(scaled[ch][qu].index, 6)
    writer.put(1, 1)  # full code table set
    for row in selections:
        writer.extend([(0, 1), (0, 2), (0, 1)])
        for idx, _ in row[:units]:
            writer.put(idx, 3)
    for row in selections:
        for _, parts in row[:units]:
            writer.extend(parts)
        for _ in range(groups):
            writer.put(15, 4)  # power compensation off
    if channels == 2:
        writer.put(0, 2)  # no swaps or sign inversion
    for _ in range(channels):
        writer.put(0, 1)  # long sine window
    for _ in range(channels):
        writer.put(0, 1)  # no gain compensation
    writer.extend([(0, 1), (0, 1), (3, 2)])  # no tones, no noise, terminator
    assert writer.size == size, 'packet budget differs from serialized bit count'
    return writer.finish(packet_bytes)
