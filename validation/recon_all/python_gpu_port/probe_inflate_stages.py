"""Compare source-ordered Python inflation updates with native snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fnit.recon_all.inflate_python import inflate_updates
from probe_inflate import read_surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--native-prefix", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    xyz, faces, _ = read_surface(args.input)
    xyz = xyz.astype(np.float32)
    faces = faces.astype(np.int32)
    rows = []

    def compare(step, candidate):
        native, native_faces, _ = read_surface(Path(f"{args.native_prefix}{step:04d}"))
        if not np.array_equal(native_faces, faces):
            raise ValueError(f"faces differ at step {step}")
        distance = np.linalg.norm(native.astype(np.float64) - candidate.astype(np.float64), axis=1)
        rows.append({"step": step, "byte_identical_vertices": int(np.all(native == candidate, axis=1).sum()),
                     "vertices": len(native), "max_vertex_distance_mm": float(distance.max()),
                     "rms_vertex_distance_mm": float(np.sqrt(np.mean(distance ** 2)))})

    inflate_updates(xyz, faces, niterations=args.iterations, snapshot=compare)
    if not rows or rows[0]["byte_identical_vertices"] != len(xyz):
        raise ValueError("first update failed byte parity")
    if any(row["max_vertex_distance_mm"] > 0.001 for row in rows):
        raise ValueError("an update exceeded the 0.001 mm vertex tolerance")
    report = {"input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "native_prefix": str(args.native_prefix), "iterations_per_averaging_level": args.iterations,
              "averaging_levels": [16, 8, 4, 2, 1, 0], "rows": rows}
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
