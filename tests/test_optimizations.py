# SPDX-License-Identifier: LGPL-2.1-or-later
"""Exact-output regression checks against the frozen pre-optimization encoder."""
import unittest
from unittest.mock import patch
import numpy as np
from atracdenc_py import bitstream
from atracdenc_py.bitstream import Bits, Scaled, best_spectra
from atracdenc_py.encoder import EncoderSettings, encode_pcm
from atracdenc_py.riff import make_at3
from legacy_bitstream import Bits as LegacyBits, best_spectra as legacy_best, frame_packet as legacy_packet


class OptimizationTests(unittest.TestCase):
    def tearDown(self):
        bitstream._best_quantized.cache_clear()
        bitstream._word_length_parts.cache_clear()

    def test_writer_unaligned_fields_padding_and_repeated_finish(self):
        rng=np.random.default_rng(129)
        for fields in [0,1,17,300,1000]:
            writer,reference=Bits(),LegacyBits()
            for _ in range(fields):
                count=int(rng.integers(0,64))
                value=int(rng.integers(0,2**min(count,62)))
                writer.put(value,count);reference.put(value,count)
                self.assertEqual(writer.size,reference.size)
                self.assertEqual(writer.value,reference.value)
                self.assertLess(writer._pending_bits,32)
            length=(reference.size+7)//8
            for padded in [length,length+1,length+17]:
                self.assertEqual(writer.finish(padded),reference.finish(padded))
                self.assertEqual(writer.finish(padded),reference.finish(padded))
            # finish() must not consume or pad the writer's state.
            writer.put(5,3);reference.put(5,3)
            self.assertEqual(writer.finish(length+1),reference.finish(length+1))
        for value,count in [(-1,3),(8,3),(0,-1),(1,0)]:
            with self.assertRaises(ValueError):Bits().put(value,count)
        writer=Bits();writer.put(1,1)
        with self.assertRaises(ValueError):writer.finish(0)

    def test_cache_key_bounds_hits_and_immutable_results(self):
        bitstream._best_quantized.cache_clear()
        block=Scaled(0,np.zeros(16,dtype=np.float32))
        first=best_spectra(block,7)
        second=best_spectra(Scaled(63,np.zeros(16,dtype=np.float32)),7)
        self.assertEqual(first,second)
        self.assertEqual(bitstream._best_quantized.cache_info().hits,1)
        self.assertIsInstance(first[1],tuple)
        for size,wl in [(16,6),(32,7),(128,1)]:
            block=Scaled(0,np.zeros(size,dtype=np.float32))
            actual=best_spectra(block,wl);expected=legacy_best(block,wl)
            self.assertEqual(actual[0],expected[0])
            self.assertEqual(list(actual[1]),expected[1])
        # More distinct blocks than capacity must evict rather than grow forever.
        rng=np.random.default_rng(135)
        for _ in range(520):
            block=Scaled(0,rng.uniform(-.9,.9,16).astype(np.float32))
            best_spectra(block,7)
        info=bitstream._best_quantized.cache_info()
        self.assertEqual(info.maxsize,512)
        self.assertEqual(info.currsize,512)

    def test_packets_and_riff_equal_previous_encoder(self):
        rng=np.random.default_rng(26)
        count=5001
        t=np.arange(count)/44100
        impulses=np.zeros((2,count),np.int16)
        impulses[0,[0,2047,2048,count-1]]=28000
        impulses[1,[1,4095]]=18000
        music=np.stack([12000*np.sin(2*np.pi*440*t),
                        9000*np.sin(2*np.pi*659.25*t)]).astype(np.int16)
        extrema=np.tile(np.array([-32768,32767],np.int16),(2,(count+1)//2))[:,:count]
        corpus=[np.zeros((2,1),np.int16),np.zeros((2,2048),np.int16),impulses,music,
                rng.integers(-18000,18001,(2,count),dtype=np.int16),extrema]
        for pcm in corpus:
            for channels in (1,2):
                for bitrate in (64,96,128,160,192,256,320):
                    settings=EncoderSettings(bitrate=bitrate,channels=channels)
                    with self.subTest(samples=pcm.shape[1],channels=channels,bitrate=bitrate):
                        with patch('atracdenc_py.encoder.frame_packet',legacy_packet):
                            expected=encode_pcm(pcm[:channels],settings)
                        actual=encode_pcm(pcm[:channels],settings)
                        self.assertEqual(actual,expected)
                        if channels==2:
                            count=pcm.shape[1]
                            for loop in (None,(0,count),((1,count) if count>1 else (0,1))):
                                self.assertEqual(make_at3(actual,count,settings,loop),
                                                 make_at3(expected,count,settings,loop))
        # Explicit packet sizes use the same budgeting boundaries, too.
        for size in (96,200,744,2048,8192):
            settings=EncoderSettings(packet_bytes=size)
            with patch('atracdenc_py.encoder.frame_packet',legacy_packet):
                expected=encode_pcm(impulses,settings)
            self.assertEqual(encode_pcm(impulses,settings),expected)
