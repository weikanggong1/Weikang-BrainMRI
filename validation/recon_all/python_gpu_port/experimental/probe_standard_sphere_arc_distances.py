"""Compare first-epoch spherical current distances with native v0_dist log."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_python import project_radially
from fnit.recon_all.sphere_standard_unfold import (
    _sphere_radius_units, _spherical_distance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("native_v0_dist", type=Path)
    parser.add_argument("--project-start", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    xyz, _ = fsio.read_geometry(str(args.snapshot))
    xyz = np.asarray(xyz, np.float32)
    if args.project_start:
        xyz = project_radially(xyz, already_sphere=True)
    radius, unit = _sphere_radius_units(xyz)
    rows = [line.split(": ")[1].split(", ")
            for line in args.native_v0_dist.read_text().splitlines()]
    ids = np.asarray([int(row[0]) for row in rows], np.int32)
    native = np.asarray([float(row[1].split()[1]) for row in rows])
    current = np.asarray([_spherical_distance(xyz, radius, unit, 0, int(other))
                          for other in ids], np.float32)
    chord = np.linalg.norm(xyz[0] - xyz[ids], axis=1)

    def c_reference(other: int) -> float:
        x0, y0, z0 = xyz[0]
        x2, y2, z2 = xyz[other]
        r0 = np.float32(math.sqrt(float(x0) ** 2 + float(y0) ** 2 + float(z0) ** 2))
        r2 = np.float32(math.sqrt(float(x2) ** 2 + float(y2) ** 2 + float(z2) ** 2))
        inv = np.float32(np.float32(1) / r0)
        ux, uy, uz = (np.float32(x0 * inv), np.float32(y0 * inv), np.float32(z0 * inv))
        dot = np.float32(float(ux) * float(x2) + float(uy) * float(y2) + float(uz) * float(z2))
        norm = np.float32(max(float(r2), abs(float(dot))))
        cosine = float(np.float32(dot / norm))
        angle = np.float32(math.acos(cosine) if cosine < 0.99 else math.sqrt(2 * (1 - cosine)))
        return float(np.float32(angle * r0))

    c_dist = np.asarray([c_reference(int(other)) for other in ids], np.float32)
    result = {"rows": len(rows),
              "arc_six_decimal_exact": int(np.count_nonzero(
                  [f"{a:.6f}" == f"{b:.6f}" for a, b in zip(current, native)])),
              "c_reference_six_decimal_exact": int(np.count_nonzero(
                  [f"{a:.6f}" == f"{b:.6f}" for a, b in zip(c_dist, native)])),
              "chord_six_decimal_exact": int(np.count_nonzero(
                  [f"{a:.6f}" == f"{b:.6f}" for a, b in zip(chord, native)])),
              "arc_max_abs_mm": float(np.max(np.abs(current - native))),
              "chord_max_abs_mm": float(np.max(np.abs(chord - native))),
              "c_reference_max_abs_mm": float(np.max(np.abs(c_dist - native))),
              "numba_vs_c_reference_max_abs_mm": float(np.max(np.abs(c_dist - current))),
              "largest_arc_differences": [
                  {"neighbor": int(ids[i]), "native": float(native[i]),
                   "arc": float(current[i]), "chord": float(chord[i])}
                  for i in np.argsort(np.abs(current - native))[-5:][::-1]
              ]}
    output = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
