"""Compare source-order pre-sampling smoothwm distances with native vertex-0 log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_standard_python import original_metric_distance_rings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothwm")
    parser.add_argument("native_v0_log", type=Path)
    parser.add_argument("--vertex", type=int, default=0)
    args = parser.parse_args()
    vertices, faces = fsio.read_geometry(args.smoothwm)
    start = perf_counter()
    rings, values = original_metric_distance_rings(vertices, faces, args.vertex)
    seconds = perf_counter() - start
    native = [(int(line.split(": ")[1].split(", ")[0]),
               float(line.split(", ")[1]))
              for line in args.native_v0_log.read_text().splitlines()]
    metric = {vertex: (depth, float(distance))
              for depth, (ring, distances) in enumerate(zip(rings, values), 1)
              for vertex, distance in zip(ring, distances)}
    immediate = sum((ring for ring in rings[:3]), [])
    sampled = native[len(immediate):]
    immediate_two = native[:len(rings[0]) + len(rings[1])]
    original = np.asarray(vertices, dtype=np.float32)
    euclidean = []
    for other, _ in immediate_two:
        delta = original[args.vertex] - original[other]
        squared = np.float32(delta[0] * delta[0] + delta[1] * delta[1])
        euclidean.append(np.float32(np.sqrt(np.float32(
            squared + delta[2] * delta[2]))))
    report = {"vertex": args.vertex, "python_seconds": seconds,
              "candidate_count_by_ring": [len(ring) for ring in rings],
              "native_sample_count": len(native),
              "native_ids_in_candidates": sum(vertex in metric for vertex, _ in native),
              "initial_three_ring_order_exact":
                  [vertex for vertex, _ in native[:len(immediate)]] == immediate,
              "initial_two_ring_distance_six_decimal_matches":
                  sum(f"{float(value):.6f}" == f"{distance:.6f}"
                      for value, (_, distance) in zip(euclidean, immediate_two)),
              "sampled_by_ring": {}}
    for depth in range(3, 8):
        selected = sampled[(depth - 3) * 8:(depth - 2) * 8]
        errors = np.asarray([abs(metric[vertex][1] - native_distance)
                             for vertex, native_distance in selected])
        report["sampled_by_ring"][str(depth)] = {
            "count": len(selected),
            "ids_in_candidate_ring": sum(metric[vertex][0] == depth
                                         for vertex, _ in selected),
            "distance_error_median_mm": float(np.median(errors)),
            "distance_error_max_mm": float(np.max(errors)),
            "distance_six_decimal_matches": int(np.count_nonzero(errors < .0000005)),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
