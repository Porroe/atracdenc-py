# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pce-at3 contributors
"""Strict RIFF parsing and PSP ATRAC3plus metadata (no codec implementation)."""
import hashlib
import struct
from pathlib import Path
from .encoder import EncoderSettings, RAW_DELAY

GUID = bytes.fromhex('bfaa23e958cb7144a119fffa01e4ce62')
RATE = 44100
SPF = 2048
# PSP adds 368 samples to the fact offset (PPSSPP Core/Util/AtracTrack.h).
EXTRA = 368


def chunks(blob):
    if len(blob) < 12 or blob[:4] != b'RIFF' or blob[8:12] != b'WAVE':
        raise ValueError('expected RIFF/WAVE')
    end = struct.unpack_from('<I', blob, 4)[0] + 8
    if end != len(blob):
        raise ValueError('RIFF size does not match file length')
    pos = 12
    result = []
    while pos < end:
        if pos + 8 > end:
            raise ValueError('truncated chunk header')
        tag, size = struct.unpack_from('<4sI', blob, pos)
        stop = pos + 8 + size
        if stop + (size & 1) > end:
            raise ValueError('truncated chunk payload or padding')
        result.append((tag, blob[pos + 8:stop]))
        pos = stop + (size & 1)
    return result


def pack(items):
    body = b'WAVE' + b''.join(tag + struct.pack('<I', len(data)) + data +
                            (b'\0' if len(data) & 1 else b'') for tag, data in items)
    if len(body) > 0xffffffff:
        raise ValueError('file exceeds RIFF size limit')
    return b'RIFF' + struct.pack('<I', len(body)) + body


def unique(items, tag):
    matches = [data for key, data in items if key == tag]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one {tag!r} chunk')
    return matches[0]


def read_loops(data):
    if len(data) < 36:
        raise ValueError('smpl chunk is too short')
    count = struct.unpack_from('<I', data, 28)[0]
    if count > (len(data) - 36) // 24:
        raise ValueError('truncated smpl loop records')
    result = []
    for i in range(count):
        cue, kind, start, end, fraction, plays = struct.unpack_from('<6I', data, 36 + i * 24)
        result.append(dict(start=start, end_inclusive=end, type=kind,
                           fraction=fraction, play_count=plays))
    return result


def describe(blob):
    items = chunks(blob)
    fmt = unique(items, b'fmt ')
    if len(fmt) < 16:
        raise ValueError('fmt chunk is too short')
    tag, channels, rate, byte_rate, align, bits = struct.unpack_from('<HHIIHH', fmt)
    codec = 'atrac3plus' if tag == 0xfffe and len(fmt) >= 40 and fmt[24:40] == GUID else ('atrac3' if tag == 0x270 else 'other')
    data = unique(items, b'data')
    if align == 0 or len(data) % align:
        raise ValueError('data size is not a whole number of frames')
    facts = [value for key, value in items if key == b'fact']
    if len(facts) > 1 or (facts and (len(facts[0]) < 4 or len(facts[0]) % 4)):
        raise ValueError('invalid fact chunk')
    fact = list(struct.unpack('<' + 'I' * (len(facts[0]) // 4), facts[0])) if facts else []
    loops = []
    for key, value in items:
        if key == b'smpl':
            loops.extend(read_loops(value))
    offset = fact[2] if len(fact) >= 3 else (fact[1] if len(fact) >= 2 else 0)
    for loop in loops:
        loop['pcm_start'] = loop['start'] - offset
        loop['pcm_end_exclusive'] = loop['end_inclusive'] - offset + 1
    spf = 2048 if codec == 'atrac3plus' else 1024
    warnings = []
    if fact and codec in ('atrac3', 'atrac3plus'):
        skip = (fact[1] if len(fact) > 1 else 0) + (368 if codec == 'atrac3plus' else 69)
        if fact[0] + skip > len(data) // align * spf:
            warnings.append('fact duration plus decoder skip exceeds encoded capacity')
        for loop in loops:
            if not 0 <= loop['pcm_start'] < loop['pcm_end_exclusive'] <= fact[0]:
                warnings.append('loop outside source sample range')
    return dict(codec=codec, channels=channels, sample_rate=rate, block_align=align,
                bitrate=round(align * 8 * rate / spf), bits_per_sample=bits,
                fmt_hex=fmt.hex(), fact=fact, frames=len(data) // align,
                duration_seconds=fact[0] / rate if fact and rate else None,
                loops=loops, chunks=[dict(id=k.decode('ascii', 'replace'), size=len(v)) for k, v in items],
                data_sha256=hashlib.sha256(data).hexdigest(), warnings=warnings)


def mux(encoded, samples, delay, loop=None):
    info = describe(encoded)
    if info['codec'] != 'atrac3plus' or info['channels'] != 2 or info['sample_rate'] != RATE:
        raise ValueError('encoder must produce 44.1 kHz stereo ATRAC3plus RIFF')
    align = info['block_align']
    if align % 8 or not 8 <= align <= 8192:
        raise ValueError('invalid PSP ATRAC3plus frame size')
    if not 0 < samples <= 0xffffffff - delay - EXTRA:
        raise ValueError('sample count exceeds supported range')
    if samples + delay + EXTRA > info['frames'] * SPF:
        raise ValueError('encoder did not flush enough frames for the original audio')
    # Sony-style 52-byte extensible fmt, version 1 and big-endian codec params.
    params = 0x2800 | (align // 8 - 1)
    fmt = struct.pack('<HHIIHHHHI16s', 0xfffe, 2, RATE, align * RATE // SPF,
                      align, 0, 34, 0x800, 3, GUID)
    fmt += struct.pack('<H', 1) + struct.pack('>H', params) + bytes(8)
    items = [(b'fmt ', fmt), (b'fact', struct.pack('<III', samples, delay, delay))]
    if loop:
        start, end = loop
        if not 0 <= start < end <= samples:
            raise ValueError('loop must satisfy 0 <= start < end <= source samples')
        smpl = struct.pack('<9I', 0, 0, 1_000_000_000 // RATE, 60, 0, 0, 0, 1, 0)
        smpl += struct.pack('<6I', 0, 0, start + delay, end - 1 + delay, 0, 0)
        items.append((b'smpl', smpl))
    items.append((b'data', unique(chunks(encoded), b'data')))
    return pack(items)


def make_at3(data, samples, settings=None, loop=None):
    """Package packets produced by the Python encoder with PSP metadata."""
    settings = settings or EncoderSettings()
    channels, align = settings.channels, settings.frame_bytes
    if channels != 2:
        raise ValueError('the pce-psp RIFF profile requires stereo')
    fmt = struct.pack('<HHIIHHHHI16s', 0xfffe, channels, RATE, align * RATE // SPF,
                      align, 0, 22, 0x800, 3, GUID)
    raw = pack([(b'fmt ', fmt), (b'fact', struct.pack('<I', samples)), (b'data', data)])
    return mux(raw, samples, RAW_DELAY - EXTRA, loop)
