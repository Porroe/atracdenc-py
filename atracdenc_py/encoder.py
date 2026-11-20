# SPDX-License-Identifier: LGPL-2.1-or-later
# Python translation/adaptation of atracdenc ATRAC3plus, 2026-11-19.
# See NOTICE.md for original authors, source mapping, and modification notices.
"""ATRAC3plus encoder: no native ATRAC codec, subprocess, or runtime compilation."""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .dsp import PolyphaseAnalysis, SubbandMDCT
from .bitstream import frame_packet

RATE = 44100
FRAME_SAMPLES = 2048
# Analysis PQF + MDCT / delayed-frame latency. Independently verified in tests.
RAW_DELAY = 4096 + 368


@dataclass(frozen=True)
class EncoderSettings:
    bitrate: int = 128
    channels: int = 2
    packet_bytes: int | None = None

    @property
    def frame_bytes(self):
        # Port of the pce-at3 patch: decimal kbps, round packet size up to 8 bytes.
        value = self.packet_bytes
        if value is None:
            value = ((self.bitrate * 1000 * FRAME_SAMPLES + RATE*64 - 1) // (RATE*64)) * 8
        if self.channels not in (1, 2):
            raise ValueError('ATRAC3plus supports one or two channels in this port')
        if not 32 <= self.bitrate <= 384:
            raise ValueError('bitrate must be between 32 and 384 kbps')
        if isinstance(value, bool) or not isinstance(value, int) or value % 8 or not 96 <= value <= 8192:
            raise ValueError('packet_bytes must be a multiple of eight between 96 and 8192')
        return value


class Encoder:
    """Stateful port of the transform/spectral encoder path.

    Input blocks have shape (channels, 2048), in signed-16 PCM units. One block
    of lookahead is consumed before the first output packet, as in atracdenc.
    Callers must feed zero blocks to flush lookahead, transform history and FIR.
    """
    def __init__(self, settings=None):
        self.settings = settings or EncoderSettings()
        self.frame_bytes = self.settings.frame_bytes
        self.pqf = PolyphaseAnalysis(self.settings.channels)
        self.mdct = SubbandMDCT(self.settings.channels)
        self.previous = np.zeros((self.settings.channels, 16, 128), dtype=np.float32)
        self.started = False

    def encode_block(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32)
        if (pcm.shape != (self.settings.channels, FRAME_SAMPLES) or not np.isfinite(pcm).all()
                or np.any(pcm < -32768) or np.any(pcm > 32767)):
            raise ValueError('PCM block must be finite signed-16 values, shape (channels, 2048)')
        current = self.pqf.analyse(pcm / np.float32(32768.))
        if not self.started:
            self.started = True
            self.previous = current
            # Upstream first emitted frame transforms a zero previous buffer.
            self.pending = np.zeros_like(current)
            return None
        bands = self.pending / np.float32(32768. / 1.122018)
        spectra = self.mdct.analyse(bands)
        self.pending = self.previous
        self.previous = current
        return frame_packet(spectra, self.frame_bytes)


def encode_pcm(pcm, settings=None, progress: Callable[[int, int], None] | None = None):
    """Encode an entire per-channel PCM array, including all tail flush packets."""
    settings = settings or EncoderSettings()
    pcm = np.asarray(pcm)
    if pcm.ndim != 2 or pcm.shape[0] != settings.channels or pcm.shape[1] == 0:
        raise ValueError('PCM must have shape (channels, nonzero samples)')
    if not np.isfinite(pcm).all() or np.any(pcm < -32768) or np.any(pcm > 32767):
        raise ValueError('PCM must contain finite signed-16 sample values')
    samples = pcm.shape[1]
    frames = (samples + RAW_DELAY + FRAME_SAMPLES - 1) // FRAME_SAMPLES
    encoder = Encoder(settings)
    result = bytearray()
    total = frames + 1
    if progress is not None:
        progress(0, total)
    # encode_block consumes/copies the input; it never retains this PCM buffer.
    block = np.zeros((settings.channels, FRAME_SAMPLES), dtype=np.float32)
    for index in range(total):
        block.fill(0)
        start = index * FRAME_SAMPLES
        if start < samples:
            segment = pcm[:, start:start+FRAME_SAMPLES]
            block[:, :segment.shape[1]] = segment
        packet = encoder.encode_block(block)
        if packet is not None:
            result.extend(packet)
        if progress is not None:
            progress(index + 1, total)
    return bytes(result)
