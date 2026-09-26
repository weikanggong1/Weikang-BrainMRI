"""Compare Python overlap repair with a completed native sphere registration."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.mris_register_nonlinear import face_area_normals
from fnit.recon_all.mris_register_overlap import remove_overlap_sphere


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('input_sphere', 'native_final', 'native_log', 'report'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--start-iteration', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    torch.set_num_threads(4)
    original, faces = fsio.read_geometry(str(args.input_sphere))
    reference, reference_faces = fsio.read_geometry(str(args.native_final))
    if not np.array_equal(faces, reference_faces):
        raise ValueError('input and native final faces differ')
    source = torch.from_numpy(original.astype(np.float32)).to(args.device)
    triangles = torch.from_numpy(faces.astype(np.int64)).to(args.device)
    matches = re.findall(r'(?m)^(\d+): dt=([0-9.]+),\s+(\d+) negative triangles',
                         args.native_log.read_text())
    native = [(int(epoch), float(dt), int(count)) for epoch, dt, count in matches
              if int(epoch) >= args.start_iteration]
    start = perf_counter()
    result, history = remove_overlap_sphere(source, triangles,
                                            start_iteration=args.start_iteration)
    seconds = perf_counter() - start
    result = result.cpu().numpy()
    error = np.abs(result - reference)
    area, _ = face_area_normals(torch.from_numpy(result),
                                torch.from_numpy(faces.astype(np.int64)), signed_sphere=True)
    report = {
        'input_sha256': hashlib.sha256(args.input_sphere.read_bytes()).hexdigest(),
        'native_final_sha256': hashlib.sha256(args.native_final.read_bytes()).hexdigest(),
        'native_log_sha256': hashlib.sha256(args.native_log.read_bytes()).hexdigest(),
        'device': str(args.device), 'start_iteration': args.start_iteration,
        'native_count_entries': len(native), 'python_count_entries': len(history),
        'first_count_mismatch': next((entry[0] for entry, observed in zip(native, history)
                                      if entry[2] != observed), None),
        'negative_after_last_step': int((area < 0).sum()),
        'exact_vertices': int(np.all(result == reference, axis=1).sum()),
        'total_vertices': len(result), 'max_abs_error': float(error.max()),
        'p95_abs_error': float(np.percentile(error, 95)),
        'seconds_excluding_io': seconds,
    }
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if (len(history) != len(native) or report['first_count_mismatch'] is not None
            or report['exact_vertices'] != len(result)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
