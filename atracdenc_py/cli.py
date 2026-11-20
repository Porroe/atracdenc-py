# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pce-at3 contributors
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from .riff import EXTRA, RATE, SPF, chunks, describe, make_at3, read_loops
from .encoder import EncoderSettings, encode_pcm
import numpy as np

# Measured on pinned atracdenc using impulses in both channels: raw delay 4464.
# PSP adds 368 to the fact skip, leaving an explicit skip of two 2048 frames.
DELAY = 4096
BITRATES = (64, 96, 128, 160, 192, 256, 320)


def run(args):
    result = subprocess.run([str(x) for x in args], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'{args[0]} failed ({result.returncode}):\n{result.stderr[-3000:]}\n{result.stdout[-1000:]}')


def wav_info(path):
    with wave.open(str(path), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (2, 2, RATE, 'NONE'):
            raise ValueError('input must be 44.1 kHz, 16-bit stereo PCM WAV; use --normalize for other audio')
        count = wav.getnframes()
        if not count:
            raise ValueError('input audio is empty')
        # wave can report a declared length beyond the physical end of the file.
        remaining = count
        while remaining:
            block = wav.readframes(min(65536, remaining))
            if not block or len(block) % 4:
                raise ValueError('truncated PCM input')
            remaining -= len(block) // 4
        return count


def select_loop(args, source, samples):
    if args.wholeloop:
        return (0, samples)
    if args.loop_start is not None or args.loop_end is not None:
        if args.loop_start is None or args.loop_end is None:
            raise ValueError('--loop-start and --loop-end must be supplied together')
        loop = (args.loop_start, args.loop_end)
    elif args.import_loop:
        if args.normalize:
            raise ValueError('--import-loop cannot be combined with --normalize; use explicit output-rate sample points')
        records = [read_loops(data) for tag, data in chunks(source.read_bytes()) if tag == b'smpl']
        loops = [loop for record in records for loop in record]
        if len(loops) != 1 or loops[0]['type'] != 0 or loops[0]['fraction'] != 0 or loops[0]['play_count'] != 0:
            raise ValueError('WAV must contain exactly one infinite, integer-sample, forward smpl loop')
        loop = (loops[0]['start'], loops[0]['end_inclusive'] + 1)
    else:
        return None
    if not 0 <= loop[0] < loop[1] <= samples:
        raise ValueError('loop must satisfy 0 <= start < end <= source samples (end is exclusive)')
    return loop


def write_output(path, blob, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.at3-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(blob)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication, including a concurrent creator.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def convert(args, source=None, target=None):
    source, target = Path(source or args.input), Path(target or args.output)
    if source.resolve() == target.resolve():
        raise ValueError('input and output must be different files')
    if target.exists() and not args.overwrite:
        raise ValueError(f'output exists: {target}; use --overwrite')
    with tempfile.TemporaryDirectory(prefix='pce-at3-') as tmp:
        tmp = Path(tmp)
        pcm = source
        if args.normalize:
            pcm = tmp / 'normalized.wav'
            run([args.ffmpeg, '-v', 'error', '-nostdin', '-i', source, '-map', '0:a:0', '-vn',
                 '-ar', str(RATE), '-ac', '2', '-c:a', 'pcm_s16le', '-bitexact', pcm])
        samples = wav_info(pcm)
        if samples > 0xffffffff - DELAY - EXTRA:
            raise ValueError('input exceeds supported RIFF sample count')
        loop = select_loop(args, source, samples)
        with wave.open(str(pcm), 'rb') as wav:
            pcm_data = np.frombuffer(wav.readframes(samples), dtype='<i2').reshape(samples, 2).T
        settings = EncoderSettings(bitrate=args.bitrate, packet_bytes=args.packet_bytes)
        progress = None
        if not getattr(args, 'no_progress', False):
            started = time.monotonic()
            last = [-1.0]
            def show(done, total):
                now = time.monotonic()
                if done != total and done and now - last[0] < 0.2:
                    return
                last[0] = now
                fraction = done / total if total else 1.0
                elapsed = now - started
                eta = (elapsed / fraction - elapsed) if fraction else 0.0
                print(f'\rencoding {fraction * 100:6.2f}%  elapsed {elapsed:6.1f}s  '
                      f'ETA {eta:6.1f}s', end='', file=sys.stderr, flush=True)
                if done == total:
                    print(file=sys.stderr)
            progress = show
        packets = encode_pcm(pcm_data, settings, progress=progress)
        blob = make_at3(packets, samples, settings, loop)
        report = describe(blob)
        if report['warnings']:
            raise ValueError('; '.join(report['warnings']))
        write_output(target, blob, args.overwrite)
        print(f'{target}: ATRAC3plus {report["bitrate"] / 1000:.1f} kbps, {samples} samples, '
              f'{"loop " + str(loop) if loop else "no loop"}')


def options(parser):
    parser.add_argument('--profile', choices=['pce-psp'], default='pce-psp')
    parser.add_argument('--bitrate', type=int, choices=BITRATES, default=128, help='target kbps; default 128')
    parser.add_argument('--packet-bytes', type=int, help='explicit packet size, multiple of eight; overrides bitrate sizing')
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--normalize', action='store_true', help='use FFmpeg to convert input to CD-quality stereo PCM')
    loop = parser.add_mutually_exclusive_group()
    loop.add_argument('--wholeloop', action='store_true')
    loop.add_argument('--import-loop', action='store_true', help='import one forward infinite WAV smpl loop')
    loop.add_argument('--loop-start', type=int, help='zero-based PCM sample at 44100 Hz')
    parser.add_argument('--loop-end', type=int, help='exclusive PCM sample at 44100 Hz')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no-progress', action='store_true', help='disable conversion progress on stderr')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Python ATRAC3plus encoder with PSP RIFF metadata (PCE hardware validation pending)')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('convert', help='convert a track')
    p.add_argument('input'); p.add_argument('output'); options(p)
    p = commands.add_parser('batch', help='convert WAV files from a directory; fail on first error')
    p.add_argument('input', type=Path); p.add_argument('output', type=Path); options(p)
    p = commands.add_parser('inspect', help='inspect RIFF metadata as JSON')
    p.add_argument('input', type=Path)
    p = commands.add_parser('compare', help='compare headers, duration, loops and encoded data; exit 1 on differences')
    p.add_argument('reference', type=Path); p.add_argument('input', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ('convert', 'batch'):
            if args.loop_end is not None and (args.wholeloop or args.import_loop):
                raise ValueError('--loop-end cannot be combined with --wholeloop or --import-loop')
            if args.command == 'convert':
                convert(args)
            else:
                files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() == '.wav')
                if not files:
                    raise ValueError('no WAV files found')
                names = [p.stem.casefold() for p in files]
                if len(set(names)) != len(names):
                    raise ValueError('input filenames would collide in the output directory')
                for source in files:
                    convert(args, source, args.output / (source.stem + '.at3'))
        elif args.command == 'inspect':
            info = describe(args.input.read_bytes())
            print(json.dumps(info, indent=2))
            return 1 if info['warnings'] else 0
        else:
            reference, result = describe(args.reference.read_bytes()), describe(args.input.read_bytes())
            differences = {key: {'reference': reference[key], 'input': result[key]} for key in reference if reference[key] != result[key]}
            print(json.dumps({'equal': not differences, 'differences': differences}, indent=2))
            return 1 if differences else 0
    except (ValueError, OSError, wave.Error, EOFError) as exc:
        print(f'atracdenc-py: {exc}', file=sys.stderr)
        return 2
    return 0
