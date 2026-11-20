# SPDX-License-Identifier: LGPL-2.1-or-later
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from atracdenc_py.encoder import Encoder, EncoderSettings, encode_pcm, RAW_DELAY
from atracdenc_py.dsp import MDCT, PolyphaseAnalysis, SubbandMDCT
from atracdenc_py.bitstream import Bits, frame_packet, scale_spectra, quantize
from atracdenc_py.tables import TABLES, SCALE, WORD_HUFFMAN, SPECTRAL_HUFFMAN
from tools.import_tables import extract


class EncoderTests(unittest.TestCase):
    def test_patch_packet_sizing(self):
        for kbps, size in [(64,376),(96,560),(128,744),(192,1120),(256,1488),(320,1864)]:
            self.assertEqual(EncoderSettings(bitrate=kbps).frame_bytes, size)
        self.assertEqual(EncoderSettings(packet_bytes=2048).frame_bytes,2048)
        for settings in [EncoderSettings(bitrate=0), EncoderSettings(channels=3),
                         EncoderSettings(packet_bytes=743),EncoderSettings(packet_bytes=9000)]:
            with self.assertRaises(ValueError):
                _ = settings.frame_bytes

    def test_encoder_never_calls_native_codec_or_subprocess(self):
        pcm=np.zeros((2,2049),np.int16)
        pcm[0,0]=28000
        with patch('subprocess.run',side_effect=AssertionError('must not call native encoder')):
            with patch('subprocess.Popen',side_effect=AssertionError('must not spawn a native encoder')):
                out=encode_pcm(pcm)
        frames=(2049+RAW_DELAY+2047)//2048
        self.assertEqual(len(out),frames*744)
        self.assertNotEqual(out,bytes(len(out)))

    def test_state_and_input_validation(self):
        enc=Encoder()
        self.assertIsNone(enc.encode_block(np.zeros((2,2048))))
        self.assertEqual(len(enc.encode_block(np.zeros((2,2048)))),744)
        for pcm in [np.zeros(2048),np.zeros((2,1)),np.full((2,2048),np.nan),np.full((2,2048),40000)]:
            with self.assertRaises(ValueError):
                enc.encode_block(pcm)
        for pcm in [np.zeros((2,0)),np.zeros((3,4)),np.full((2,2),np.inf),np.full((2,2),40000)]:
            with self.assertRaises(ValueError):
                encode_pcm(pcm)

    def test_quantization_scaling_and_bit_order(self):
        writer=Bits();writer.extend([(0,1),(1,2),(3,2)])
        self.assertEqual(writer.finish(1),b'\x38')
        with self.assertRaises(ValueError):writer.put(4,2)
        self.assertEqual(extract('int a[] = {0x1f, 0xff, 1.25f, -1};','a'),[31,255,1.25,-1])
        self.assertEqual(len(TABLES['atrac3p_spectra_tabs']),112)
        self.assertEqual(TABLES['atrac3p_spectra_cbs'][33][8],172)
        for tab in SPECTRAL_HUFFMAN:
            codes=[(code,bits) for code,bits in tab if bits]
            self.assertEqual(len(codes),len(set(codes)))
        scaled=scale_spectra(np.zeros(2048,np.float32))
        self.assertEqual(len(scaled),32)
        self.assertTrue(all(block.index==0 for block in scaled))
        self.assertTrue(all(np.all(quantize(block,7)==0) for block in scaled))

    def test_numpy_mdct_matches_definition(self):
        # Independent direct cosine definition, not the FFT algorithm under test.
        rng=np.random.default_rng(42)
        x=rng.normal(size=256).astype(np.float32)
        k=np.arange(128)[:,None]
        n=np.arange(256)[None,:]
        basis=np.cos(np.pi/128*(n+.5+64)*(k+.5))/256
        expected=basis@x
        np.testing.assert_allclose(MDCT(256).forward(x),expected,rtol=2e-5,atol=2e-7)

    def test_mono_packets(self):
        packets=encode_pcm(np.zeros((1,2048),np.int16),EncoderSettings(channels=1))
        self.assertEqual(len(packets)%744,0)
        # First three bits: start=0, channel unit=mono (00).
        self.assertEqual(packets[0]>>5,0)


REFERENCE=os.environ.get('ATRACDENC_PY_REFERENCE')


@unittest.skipUnless(REFERENCE,'set ATRACDENC_PY_REFERENCE for optional C++ differential tests')
class ReferenceTests(unittest.TestCase):
    def test_pqf_and_mdct_against_upstream(self):
        rng=np.random.default_rng(27)
        pcm=rng.uniform(-.95,.95,(8,2048)).astype(np.float32)
        reference=np.frombuffer(subprocess.check_output([REFERENCE,'dsp'],input=pcm.tobytes()),np.float32).reshape(8,2,2048)
        pqf,mdct=PolyphaseAnalysis(1),SubbandMDCT(1)
        for index,block in enumerate(pcm):
            bands=pqf.analyse(block[None,:])
            spectra=mdct.analyse(bands/np.float32(32768/1.122018))
            np.testing.assert_allclose(bands.ravel(),reference[index,0],atol=.003,rtol=2e-6)
            np.testing.assert_allclose(spectra.ravel(),reference[index,1],atol=1e-8,rtol=3e-5)

    def test_packets_match_cpp_spectral_path(self):
        pcm=np.zeros((2,16000),np.float32)
        pcm[0,[0,4410,15999]]=28000;pcm[1,8820]=18000
        for bitrate in [64,128,192,320]:
            settings=EncoderSettings(bitrate=bitrate)
            frames=(pcm.shape[1]+RAW_DELAY+2047)//2048
            padded=np.zeros((frames+1,2,2048),np.float32)
            for i in range(frames+1):
                segment=pcm[:,i*2048:(i+1)*2048]
                padded[i,:,:segment.shape[1]]=segment/32768
            ref=subprocess.check_output([REFERENCE,'packets',str(settings.frame_bytes)],input=padded.tobytes())
            self.assertEqual(encode_pcm(pcm,settings),ref)


if __name__=='__main__':unittest.main()
