#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Build an optional C++ differential-test harness, never a runtime codec."""
import argparse
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REVISION='cb170709f836cb43dc0312455188c791159d81dc'


def run(*args):
    subprocess.run([str(x) for x in args],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--cxx',default='c++')
    p.add_argument('--cmake-arg',action='append',default=[])
    args=p.parse_args()
    source=args.source.resolve()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=REVISION:
        raise SystemExit(f'Expected pinned upstream {REVISION}; found {revision}')
    run('git','-C',source,'submodule','update','--init','--recursive')
    root=ROOT/'.reference'
    root.mkdir(exist_ok=True)
    build=root/'build'
    run('cmake','-S',source,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*args.cmake_arg)
    run('cmake','--build',build,'--parallel','4','--target','atracdenc_impl')
    run(args.cxx,'-std=c++17','-I'+str(source/'src'),'-I'+str(source/'src/lib'),
        ROOT/'tools/reference.cpp',build/'src/libatracdenc_impl.a',
        build/'src/libbitstream.a',build/'src/lib/libgha/libgha.a',
        build/'src/libfft_impl.a','-o',root/'atracdenc-reference')
    print(f'ATRACDENC_PY_REFERENCE={root/"atracdenc-reference"}')


if __name__=='__main__':main()
