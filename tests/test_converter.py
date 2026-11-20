# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pce-at3 contributors
import contextlib
import io
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from atracdenc_py.cli import BITRATES, DELAY, main
from atracdenc_py.riff import EXTRA, SPF, chunks, describe, mux, pack


def wav(path, samples, pulses=None, channels=2, rate=44100):
    pulses = pulses or {}
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
        stream.writeframes(b''.join(struct.pack('<' + 'h' * channels,
            *pulses.get(i, (0,) * channels)) for i in range(samples)))


class MetadataTests(unittest.TestCase):
    def test_rejects_truncation_bad_sizes_duplicate_fmt(self):
        for blob in (b'', b'RIFF\x04\0\0\0WAVEjunk',
                     b'RIFF\x0d\0\0\0WAVEJUNK\x01\0\0\0x'):
            with self.assertRaises(ValueError):
                chunks(blob)
        with self.assertRaises(ValueError):
            describe(pack([(b'fmt ', bytes(16)), (b'fmt ', bytes(16)), (b'data', b'')]))

    def test_odd_chunks_roundtrip(self):
        items = [(b'JUNK', b'x'), (b'data', b'abc')]
        self.assertEqual(chunks(pack(items)), items)


@unittest.skipUnless(shutil.which('ffmpeg'), 'install ffmpeg for decode integration tests')
class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def cli(self, *args, status=0):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            actual = main([str(x) for x in args])
        self.assertEqual(actual, status, err.getvalue())
        return out.getvalue()

    def decode(self, path):
        result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
                                 '-af', 'asetpts=N/SR/TB', '-f', 's16le', '-'], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stderr, b'')
        return array('h', result.stdout)

    def test_all_bitrates_decode_and_keep_impulse_alignment(self):
        source = self.root / 'impulses.wav'
        positions = {0: (28000, 0), 4410: (28000, 0), 8820: (0, 18000),
                     22050: (28000, 0), 49999: (28000, 0)}
        wav(source, 50000, positions)
        for bitrate in BITRATES:
            with self.subTest(bitrate=bitrate):
                output = self.root / f'{bitrate}.at3'
                self.cli('convert', source, output, '--bitrate', bitrate, '--wholeloop')
                info = describe(output.read_bytes())
                self.assertEqual(info['fact'], [50000, DELAY, DELAY])
                self.assertEqual(info['warnings'], [])
                self.assertEqual([c['id'] for c in info['chunks']], ['fmt ', 'fact', 'smpl', 'data'])
                # Independently implement PPSSPP loop adjustment: stored + 368,
                # then subtract fact[1] + 368 (fact[1] == fact[2]).
                self.assertEqual(info['loops'][0]['start'] + 368 - (4096 + 368), 0)
                self.assertEqual(info['loops'][0]['end_inclusive'] - 4096, 49999)
                pcm = self.decode(output)
                skip = DELAY + EXTRA
                self.assertGreaterEqual(len(pcm) // 2, 50000 + skip)
                for pos in positions:
                    channel = 1 if pos == 8820 else 0
                    expected = pos + skip
                    peak = max(range(expected - 30, expected + 31), key=lambda i: abs(pcm[2*i+channel]))
                    # Lossy quantization can move the largest ringing peak by one sample
                    # at 64 kbps; the stream/framing offset remains fixed.
                    self.assertLessEqual(abs(peak - expected), 1)
                    self.assertGreater(abs(pcm[2*peak+channel]), 500)

    def test_short_silent_and_frame_boundary_tracks(self):
        for count in (1, 2047, 2048, 2049, 4096):
            with self.subTest(samples=count):
                source, output = self.root / 'short.wav', self.root / 'short.at3'
                wav(source, count)
                self.cli('convert', source, output, '--overwrite')
                info = describe(output.read_bytes())
                self.assertEqual(info['fact'][0], count)
                self.assertFalse(info['warnings'])
                self.assertFalse(info['loops'])
                pcm = self.decode(output)
                self.assertTrue(all(x == 0 for x in pcm))
                self.assertGreaterEqual(len(pcm) // 2, count + DELAY + EXTRA)

    def test_loop_import_explicit_and_errors_preserve_output(self):
        source, output = self.root / 'source.wav', self.root / 'output.at3'
        wav(source, 8193)
        raw = source.read_bytes()
        smpl = struct.pack('<9I6I', 0, 0, 22675, 60, 0, 0, 0, 1, 0, 0, 0, 512, 8000, 0, 0)
        source.write_bytes(pack(chunks(raw) + [(b'smpl', smpl)]))
        self.cli('convert', source, output, '--import-loop')
        info = describe(output.read_bytes())
        self.assertEqual(info['loops'][0]['pcm_start'], 512)
        self.assertEqual(info['loops'][0]['pcm_end_exclusive'], 8001)
        before = output.read_bytes()
        self.cli('convert', source, output, status=2)
        self.cli('convert', source, output, '--overwrite', '--loop-start', 10, '--loop-end', 10000, status=2)
        self.cli('convert', source, output, '--overwrite', '--wholeloop', '--loop-end', 3, status=2)
        self.assertEqual(output.read_bytes(), before)
        self.cli('convert', source, output, '--overwrite', '--loop-start', 10, '--loop-end', 8000)
        self.assertEqual(describe(output.read_bytes())['loops'][0]['pcm_start'], 10)
        self.cli('compare', output, output)
        self.cli('inspect', output)
        self.cli('compare', output, self.root / 'missing', status=2)
        with self.assertRaises(ValueError):
            mux(before, 9999999, DELAY)

    def test_normalization_and_batch(self):
        source = self.root / 'mono.wav'
        wav(source, 22050, channels=1, rate=22050)
        self.cli('convert', source, self.root / 'normalized.at3', status=2)
        self.cli('convert', source, self.root / 'normalized.at3', '--normalize')
        info = describe((self.root / 'normalized.at3').read_bytes())
        self.assertEqual(info['fact'][0], 44100)
        inputs, outputs = self.root / 'inputs', self.root / 'outputs'
        inputs.mkdir()
        wav(inputs / 'track 02.WAV', 500)
        wav(inputs / 'track 03.wav', 501)
        output = self.cli('batch', inputs, outputs, '--wholeloop')
        self.assertIn('[1/2] Encoding ', output)
        self.assertIn('[2/2] Encoding ', output)
        self.assertEqual(len(list(outputs.glob('*.at3'))), 2)
        self.cli('convert', inputs / 'track 03.wav', inputs / 'track 03.wav', '--overwrite', status=2)


if __name__ == '__main__':
    unittest.main()
