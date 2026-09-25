"""Check native standard-sphere vertex-0 initial distance neighborhood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.smooth_surface_python import ordered_neighbors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoothwm")
    parser.add_argument("native_v0_log", type=Path)
    args = parser.parse_args()
    vertices, faces = fsio.read_geometry(args.smoothwm)
    vertices = np.asarray(vertices, np.float32)
    neighbors = ordered_neighbors(faces, len(vertices))
    native = [(int(row.split(": ")[1].split(", ")[0]),
               float(row.split(", ")[1]))
              for row in args.native_v0_log.read_text().splitlines()]
    seen = {0}
    rings = [[0]]
    for _ in range(2):
        ring = []
        for vertex in rings[-1]:
            for other in neighbors[vertex]:
                if other not in seen:
                    seen.add(other)
                    ring.append(other)
        rings.append(ring)
    actual_ids = rings[1] + rings[2]
    native_ids = [vertex for vertex, _ in native[:len(actual_ids)]]
    offset = vertices[0] - vertices[actual_ids]
    squared = offset[:, 0] * offset[:, 0]
    squared += offset[:, 1] * offset[:, 1]
    squared += offset[:, 2] * offset[:, 2]
    distance = np.sqrt(squared)
    matched_distance = sum(f"{float(value):.6f}" == f"{native[i][1]:.6f}"
                           for i, value in enumerate(distance))
    print(json.dumps({
        "one_ring": len(rings[1]),
        "two_ring": len(rings[2]),
        "ordered_ids_exact": actual_ids == native_ids,
        "distance_six_decimal_matches": matched_distance,
        "distance_count": len(distance),
        "native_total_sampled_neighbors": len(native),
    }, indent=2))


if __name__ == "__main__":
    main()
