"""Paired CLI benchmark for the fixed recon-all mri_convert --conform stage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def payload(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--native-bin', required=True, type=Path)
    parser.add_argument('--python-module', required=True, type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--license', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    python_module = args.python_module.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['FREESURFER_HOME'] = str(args.native_bin.parent.parent)
    env['FS_LICENSE'] = str(args.license)
    runs = []
    for repeat in range(args.repeats):
        outputs = {'native': args.work / f'native_{repeat}.mgz',
                   'python': args.work / f'python_{repeat}.mgz'}
        order = ['native', 'python'] if repeat % 2 == 0 else ['python', 'native']
        times = {}
        for name in order:
            if name == 'native':
                command = [str(args.native_bin), str(args.input), str(outputs[name]), '--conform']
            else:
                command = [sys.executable, str(python_module), str(args.input),
                           str(outputs[name]), '--device', args.device]
            start = time.perf_counter()
            completed = subprocess.run(command, env=env, cwd=args.work, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            times[name] = time.perf_counter() - start
            if completed.returncode:
                raise RuntimeError(f'{name} returned {completed.returncode}: {completed.stdout[-1500:]}')
        native, candidate = (payload(outputs[key]) for key in ('native', 'python'))
        prefix_len = 284 + 256**3
        if native[:prefix_len] != candidate[:prefix_len]:
            raise AssertionError('MGH header or voxel payload differs')
        runs.append({'order': order, 'wall_seconds': times,
                     'header_and_voxel_sha256': hashlib.sha256(native[:prefix_len]).hexdigest(),
                     'native_footer_bytes': len(native) - prefix_len,
                     'python_footer_bytes': len(candidate) - prefix_len})
        print(f'pair {repeat + 1}: {times}', flush=True)
    report = {'host': platform.node(), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'device': args.device, 'timing_scope': 'fresh CLI process, input read and MGZ write; '
              'sequential A/B pairs; OS cache not cleared', 'runs': runs,
              'median_seconds': {key: statistics.median(run['wall_seconds'][key] for run in runs)
                                 for key in ('native', 'python')}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['median_seconds']), flush=True)


if __name__ == '__main__':
    main()
