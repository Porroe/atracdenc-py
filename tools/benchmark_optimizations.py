#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Measure full encoding and exact hashes using a chosen checkout.

Example: python tools/benchmark_optimizations.py --source /tmp/baseline --output /tmp/before.json
Run again with --source .; compare hashes and per-case medians. All bounded caches
are cleared before each timed encode; hits within a file still count. No native
encoder, disk output, normalization, or decoder is included in these timings.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seconds',type=float,default=1.)
    args=p.parse_args()
    if args.repeats<1 or args.seconds<=0:p.error('repeats and seconds must be positive')
    sys.path.insert(0,str(args.source.resolve()))
    import numpy as np
    import atracdenc_py.bitstream as bitstream
    from atracdenc_py.encoder import encode_pcm,EncoderSettings
    samples=round(args.seconds*44100)
    t=np.arange(samples,dtype=np.float64)/44100
    music=np.stack((8000*np.sin(2*np.pi*440*t)+4000*np.sin(2*np.pi*659.25*t),
                    8000*np.sin(2*np.pi*329.63*t)+4000*np.sin(2*np.pi*523.25*t))).astype(np.int16)
    rng=np.random.default_rng(112)
    cases={'silence':np.zeros((2,samples),np.int16),'music':music,
           'noise':rng.integers(-16000,16001,(2,samples),dtype=np.int16)}
    results=[]
    for name,pcm in cases.items():
        for bitrate in (64,128,320):
            elapsed=[]
            hashes=[]
            for _ in range(args.repeats):
                for cache in ('_best_quantized','_word_length_parts'):
                    fn=getattr(bitstream,cache,None)
                    if fn is not None:fn.cache_clear()
                start=time.perf_counter()
                result=encode_pcm(pcm,EncoderSettings(bitrate=bitrate))
                elapsed.append(time.perf_counter()-start)
                hashes.append(hashlib.sha256(result).hexdigest())
            assert len(set(hashes))==1
            row=dict(case=name,bitrate=bitrate,seconds=elapsed,median=statistics.median(elapsed),
                     sha256=hashes[0],bytes=len(result))
            results.append(row)
            print(f'{name} {bitrate}: {row["median"]:.4f}s',flush=True)
    args.output.write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,
        audio_seconds=args.seconds,cache_policy='cold before each encode',results=results),indent=2)+'\n')


if __name__=='__main__':main()
