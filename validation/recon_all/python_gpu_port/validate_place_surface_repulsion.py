"""Compare first pial surface repulsion with a pinned-source hash probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    mismatch = np.argwhere(actual != expected)
    return {
        "exact_elements": int(np.count_nonzero(actual == expected)),
        "total_elements": int(expected.size),
        "max_abs_error": float(difference.max(initial=0)),
        "first_mismatch": mismatch[0].tolist() if len(mismatch) else None,
    }


def source_buckets(path: Path, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    records = np.fromfile(path, dtype="<i4")
    offsets = np.zeros(vertex_count + 1, dtype=np.int32)
    vertices: list[int] = []
    cursor = 0
    for vertex in range(vertex_count):
        if cursor < len(records) and records[cursor] == vertex:
            count = int(records[cursor + 1])
            vertices.extend(records[cursor + 2:cursor + 2 + count])
            cursor += count + 2
        offsets[vertex + 1] = len(vertices)
    if cursor != len(records):
        raise ValueError("repulsion bucket dump has trailing records")
    return offsets, np.asarray(vertices, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_repulsion import (
        original_vertex_normals, surface_repulsion_gradient, vertex_buckets,
    )

    start = time.perf_counter()
    original, faces = nib.freesurfer.read_geometry(args.subject / "surf/lh.white")
    original = original.astype(np.float32)
    before = np.fromfile(args.probe / "lh.gradient.intensity", dtype=STATE)
    after = np.fromfile(args.probe / "lh.gradient.surface_repulsion", dtype=STATE)
    xyz = before["floats"][:, :3]
    normals = before["floats"][:, 3:6]
    ripped = before["flags"][:, 0]
    original_normals = original_vertex_normals(original, faces)
    offsets, candidates = vertex_buckets(xyz, original, ripped)
    expected_offsets, expected_candidates = source_buckets(args.probe / "lh.repulse_buckets", len(xyz))
    delta = surface_repulsion_gradient(xyz, normals, original, original_normals, ripped, offsets, candidates)
    candidate = np.float32(before["floats"][:, 6:9] + delta)
    checked = {
        "original_xyz": compare(original, xyz),
        "original_normals": compare(original_normals, np.fromfile(args.probe / "lh.gradient.orig_normals", dtype="<f4").reshape(-1, 3)),
        "bucket_offsets": compare(offsets, expected_offsets),
        "bucket_candidates": compare(candidates, expected_candidates) if candidates.shape == expected_candidates.shape else {
            "exact_elements": 0, "total_elements": len(expected_candidates), "max_abs_error": None,
            "candidate_count": len(candidates), "source_count": len(expected_candidates),
        },
        "repulsion_gradient": compare(candidate, after["floats"][:, 6:9]),
    }
    report = {
        "reference": "copied pinned FreeSurfer 8.2 first pial step source probe",
        "comparisons": checked,
        "seconds_including_io_and_jit": time.perf_counter() - start,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(item["exact_elements"] != item["total_elements"] for item in checked.values()):
        raise SystemExit("surface repulsion differs from pinned source")


if __name__ == "__main__":
    main()
