"""Compare Python's seeded standard-sphere distance sample with native log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.sphere_standard_python import (
    FreeSurferSphereRandom,
    sample_standard_metric_neighbors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothwm")
    parser.add_argument("native_log", type=Path)
    parser.add_argument("--vertex", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    xyz, faces = fsio.read_geometry(args.smoothwm)
    native = [(int(line.split(": ")[1].split(", ")[0]),
               float(line.split(", ")[1]))
              for line in args.native_log.read_text().splitlines()]
    started = perf_counter()
    neighbors = ordered_neighbors(faces, len(xyz))
    rng = FreeSurferSphereRandom(args.seed)
    for vertex in range(args.vertex + 1):
        selected, distances = sample_standard_metric_neighbors(
            xyz, faces, vertex, rng, neighbors)
    seconds = perf_counter() - started
    native_ids = [vertex for vertex, _ in native]
    report = {
        "vertex": args.vertex,
        "seed": args.seed,
        "python_seconds_including_prior_vertices": seconds,
        "native_neighbor_count": len(native),
        "python_neighbor_count": len(selected),
        "ordered_neighbor_ids_exact": selected == native_ids,
        "exact_neighbor_ids": sum(a == b for a, b in zip(selected, native_ids)),
        "pre_symmetry_distance_error_max_mm": float(np.max(np.abs(
            distances - [distance for _, distance in native]))),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
