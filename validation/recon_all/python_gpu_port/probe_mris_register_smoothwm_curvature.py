"""Compare independent smoothwm mean curvature with the native sno2 array."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_smoothwm import smoothwm_mean_curvature


def describe(candidate, reference):
    delta = np.abs(candidate - reference)
    return {'exact_vertices': int(np.count_nonzero(candidate == reference)),
            'max_abs_error': float(delta.max()),
            'median_abs_error': float(np.median(delta)),
            'p95_abs_error': float(np.percentile(delta, 95)),
            'correlation': float(np.corrcoef(candidate, reference)[0, 1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('smoothwm', type=Path)
    parser.add_argument('native_raw_curvature', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--predicted-raw', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = perf_counter()
    vertices, faces = fsio.read_geometry(str(args.smoothwm))
    raw = smoothwm_mean_curvature(torch.from_numpy(vertices.astype(np.float32)),
                                  torch.from_numpy(faces.astype(np.int64)))
    if args.predicted_raw is not None:
        raw.numpy().tofile(args.predicted_raw)
    reference = np.fromfile(args.native_raw_curvature, dtype='<f4')
    assert raw.shape == reference.shape
    report = {'smoothwm_sha256': hashlib.sha256(args.smoothwm.read_bytes()).hexdigest(),
              'native_raw_curvature_sha256': hashlib.sha256(
                  args.native_raw_curvature.read_bytes()).hexdigest(),
              'predicted_raw_sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
              'vertices': len(raw), 'raw_mean': float(raw.double().mean()),
              'raw_std': float(raw.double().std(unbiased=False)),
              **describe(raw.numpy(), reference),
              'seconds_excluding_io': perf_counter() - start}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
