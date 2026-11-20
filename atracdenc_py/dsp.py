# SPDX-License-Identifier: LGPL-2.1-or-later
# Python translation/adaptation of atracdenc, 2026-11-19.
# Upstream: dcherednik/atracdenc, revision cb170709f836cb43dc0312455188c791159d81dc.
# See NOTICE.md for original authors, source mapping, and modification notices.
"""ATRAC3plus polyphase analysis and MDCT, implemented with Python/NumPy."""
import numpy as np
from .tables import TABLES


class MDCT:
    """Port of lib/mdct/mdct.h, preserving float32 pre/post rotations."""
    def __init__(self, n, scale=1.0):
        self.n = n
        # These integer gather/scatter indices depend only on transform length,
        # not on the frame. Precompute them without changing the DSP arithmetic.
        self._k = np.arange(0, n // 2, 2)
        self._reverse_k = n // 2 - 1 - self._k
        a, b = self._k[:n // 8], self._k[n // 8:]
        self._indices = ((3*n//4 - 1 - a, 3*n//4 + a),
                         (n//4 + a, n//4 - 1 - a),
                         (3*n//4 - 1 - b, b - n//4),
                         (n//4 + b, 5*n//4 - 1 - b))
        # Upstream rounds the angular constants to float before evaluating trig.
        alpha = np.float32(2 * np.pi / (8 * n))
        omega = np.float32(2 * np.pi / n)
        angle = np.float32(omega * np.arange(n // 4, dtype=np.float32) + alpha)
        factor = np.float32(np.sqrt(np.float32(scale / n)))
        self.c = np.float32(factor * np.cos(angle.astype(np.float64)))
        self.s = np.float32(factor * np.sin(angle.astype(np.float64)))

    def forward(self, data):
        n = self.n
        k = self._k
        indices = self._indices
        split = n // 8
        r = np.empty(data.shape[:-1] + (n // 4,), dtype=np.float32)
        i = np.empty_like(r)
        r[..., :split] = data[..., indices[0][0]] + data[..., indices[0][1]]
        i[..., :split] = data[..., indices[1][0]] - data[..., indices[1][1]]
        r[..., split:] = data[..., indices[2][0]] - data[..., indices[2][1]]
        i[..., split:] = data[..., indices[3][0]] + data[..., indices[3][1]]
        transformed = np.fft.fft(np.complex64((r*self.c + i*self.s) + 1j*(i*self.c - r*self.s)), axis=-1).astype(np.complex64)
        out = np.empty(data.shape[:-1] + (n//2,), dtype=np.float32)
        out[..., k] = -transformed.real*self.c - transformed.imag*self.s
        out[..., self._reverse_k] = -transformed.real*self.s + transformed.imag*self.c
        return out

    def inverse(self, data):
        n = self.n
        k = self._k
        indices = self._indices
        r, i = data[..., k], data[..., self._reverse_k]
        xr = np.float32(-2 * (i*self.s + r*self.c))
        xi = np.float32(-2 * (i*self.c - r*self.s))
        transformed = np.fft.fft(np.complex64(xr + 1j*xi), axis=-1).astype(np.complex64)
        r1 = transformed.real*self.c + transformed.imag*self.s
        i1 = transformed.real*self.s - transformed.imag*self.c
        out = np.empty(data.shape[:-1] + (n,), dtype=np.float32)
        half = n // 8
        out[..., indices[0][0]] = r1[..., :half]
        out[..., indices[0][1]] = r1[..., :half]
        out[..., indices[1][0]] = i1[..., :half]
        out[..., indices[1][1]] = -i1[..., :half]
        out[..., indices[2][0]] = r1[..., half:]
        out[..., indices[2][1]] = -r1[..., half:]
        out[..., indices[3][0]] = i1[..., half:]
        out[..., indices[3][1]] = i1[..., half:]
        return out


class PolyphaseAnalysis:
    """Port of atrac3plus_pqf.c: 384-tap FIR, 16 bands, 368 history samples."""
    def __init__(self, channels):
        self.history = np.zeros((channels, 368), dtype=np.float32)
        fir = np.zeros(384, dtype=np.float32)
        c1 = np.array(TABLES['ff_ipqf_coeffs1'], dtype=np.float32)
        c2 = np.array(TABLES['ff_ipqf_coeffs2'], dtype=np.float32)
        for i in range(16):
            if i >= 8:
                fir[96+(i-8)*12:108+(i-8)*12] = c1[:, i]
                fir[288+(i-8)*12:300+(i-8)*12] = c2[:, i]
            else:
                fir[i*12:i*12+12] = c1[:, i]
                fir[192+i*12:204+i*12] = c2[:, i]
        self.fir = fir.reshape(32, 12).T
        self.dct = MDCT(32, 16 * 128 * 512.)

    def analyse(self, pcm):
        data = np.concatenate((self.history, np.asarray(pcm, dtype=np.float32)), axis=1)
        self.history[:] = data[:, 2048:]
        windows = np.lib.stride_tricks.sliding_window_view(data, 384, axis=1)[:, ::16, :]
        # The original multiplies in float32, then accumulates into double.
        y = np.sum(windows.reshape(len(pcm), 128, 12, 32) * self.fir[None, None, :, :], axis=2, dtype=np.float64)
        yy = np.empty((len(pcm), 128, 16), dtype=np.float32)
        yy[..., :8] = y[..., 8:16] + y[..., 7::-1]
        yy[..., 8:] = y[..., 16:24] + y[..., 31:23:-1]
        res = -self.dct.inverse(yy)[..., 8:24]
        return np.ascontiguousarray(res[..., ::-1].transpose(0, 2, 1))


class SubbandMDCT:
    """Port of at3p_mdct.cpp's long sine-window encoding path."""
    def __init__(self, channels):
        self.history = np.zeros((channels, 16, 128), dtype=np.float32)
        angles = np.float32((np.arange(128) + .5) * (np.pi / 256))
        self.window = np.float32(2 * np.sin(angles))
        self.mdct = MDCT(256)

    def analyse(self, bands):
        data = np.concatenate((self.history, bands * self.window[::-1]), axis=-1)
        result = self.mdct.forward(data)
        result[:, 1::2] = result[:, 1::2, ::-1]
        self.history[:] = bands * self.window
        return result.reshape(len(bands), 2048)
