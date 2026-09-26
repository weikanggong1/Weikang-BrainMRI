"""Test bitwise source-order Numba registration averaging against PyTorch."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np
import torch
from fnit.recon_all.mris_register_average_numba import average_gradients_exact_cpu
from fnit.recon_all.mris_register_nonlinear import average_gradients, ordered_neighbors_from_faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sphere", type=Path)
    parser.add_argument("gradient", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--iterations", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    vertices, faces = fsio.read_geometry(str(args.sphere))
    force = np.fromfile(args.gradient, dtype='<f4').copy().reshape(len(vertices), 3)
    neighbors, degrees = ordered_neighbors_from_faces(
        torch.from_numpy(faces.astype(np.int64)), len(vertices))
    average_gradients_exact_cpu(torch.from_numpy(force), neighbors, degrees, 1)
    started = perf_counter()
    reference = average_gradients(torch.from_numpy(force), neighbors, degrees,
                                  args.iterations).numpy()
    torch_seconds = perf_counter() - started
    started = perf_counter()
    actual = average_gradients_exact_cpu(
        torch.from_numpy(force), neighbors, degrees, args.iterations).numpy()
    numba_seconds = perf_counter() - started
    error = np.abs(reference - actual)
    report = {"vertices": len(vertices), "iterations": args.iterations,
              "exact_vertices": int(np.all(reference == actual, axis=1).sum()),
              "max_abs_error": float(error.max()),
              "p95_abs_error": float(np.percentile(error, 95)),
              "pytorch_seconds": torch_seconds, "numba_seconds": numba_seconds}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
