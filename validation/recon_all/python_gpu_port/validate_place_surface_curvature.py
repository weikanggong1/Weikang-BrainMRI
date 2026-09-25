"""Compare first pial curvature inputs with copied pinned-source checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def native_neighbors(path: Path, vertices: int) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    offsets = np.zeros(vertices + 1, dtype=np.int32)
    flat: list[int] = []
    cursor = 0
    for vertex in range(vertices):
        count = int(np.frombuffer(raw, dtype="<i2", count=1, offset=cursor)[0])
        cursor += 2
        flat.extend(np.frombuffer(raw, dtype="<i4", count=count, offset=cursor))
        cursor += 4 * count
        offsets[vertex + 1] = len(flat)
    if cursor != len(raw):
        raise ValueError("native neighbor dump has trailing bytes")
    return offsets, np.asarray(flat, dtype=np.int32)


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        return {"exact_elements": 0, "total_elements": int(expected.size), "actual_elements": int(actual.size)}
    mismatch = np.argwhere(actual != expected)
    return {
        "exact_elements": int(np.count_nonzero(actual == expected)),
        "total_elements": int(expected.size),
        "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
        "first_mismatch": mismatch[0].tolist() if len(mismatch) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_curvature import quadratic_curvature, tangent_basis, two_ring_neighbors
    from fnit.recon_all.place_surface_spring import spring_gradient

    _, faces = nib.freesurfer.read_geometry(args.surface)
    before = np.fromfile(args.probe / "lh.gradient.normal_spring", dtype=STATE)
    start = time.perf_counter()
    basis = tangent_basis(before["floats"][:, 3:6])
    offsets, candidates = two_ring_neighbors(faces, len(before))
    expected_basis = np.fromfile(args.probe / "lh.curvature.tangent_basis", dtype="<f4").reshape(-1, 6)
    expected_offsets, expected_candidates = native_neighbors(args.probe / "lh.gradient.neighbors_total", len(before))
    scalar = quadratic_curvature(
        before["floats"][:, :3], before["floats"][:, 3:6], basis,
        before["flags"][:, 0], offsets, candidates,
    )
    expected_scalar = np.fromfile(args.probe / "lh.curvature.scalar", dtype="<f4")
    expected_curvature = np.fromfile(args.probe / "lh.gradient.curvature", dtype=STATE)
    curvature_gradient = np.float32(before["floats"][:, 6:9] + np.float32(scalar[:, None] * before["floats"][:, 3:6]))
    tangent = spring_gradient(
        before["floats"][:, :3], before["floats"][:, 3:6], faces,
        before["flags"][:, 0], weight=0.3, direction="tangent",
        border=before["flags"][:, 1], negative=before["flags"][:, 2],
    )
    final_gradient = np.float32(curvature_gradient + tangent)
    expected_final = np.fromfile(args.probe / "lh.gradient.tangential_spring", dtype=STATE)
    checked = {
        "tangent_basis": compare(basis, expected_basis),
        "two_ring_offsets": compare(offsets, expected_offsets),
        "two_ring_candidates": compare(candidates, expected_candidates),
        "quadratic_scalar": compare(scalar, expected_scalar),
        "post_curvature_gradient": compare(curvature_gradient, expected_curvature["floats"][:, 6:9]),
        "final_gradient_from_python_curvature": compare(final_gradient, expected_final["floats"][:, 6:9]),
    }
    active = before["flags"][:, 0] == 0
    error = np.abs(scalar[active].astype(np.float64) - expected_scalar[active].astype(np.float64))
    checked["quadratic_scalar"]["p99_abs_error"] = float(np.percentile(error, 99))
    active_ids = np.flatnonzero(active)
    checked["quadratic_scalar"]["above_1e_5"] = int(np.count_nonzero(error > 1e-5))
    checked["quadratic_scalar"]["above_1e_6"] = int(np.count_nonzero(error > 1e-6))
    checked["quadratic_scalar"]["largest"] = [
        {
            "vertex": int(active_ids[index]),
            "candidate": float(scalar[active_ids[index]]),
            "source": float(expected_scalar[active_ids[index]]),
            "abs_error": float(error[index]),
        }
        for index in np.argsort(error)[-5:][::-1]
    ]
    report = {
        "reference": "copied pinned FreeSurfer 8.2 first pial step source probe",
        "comparisons": checked,
        "seconds_including_jit": time.perf_counter() - start,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(
        checked[name]["exact_elements"] != checked[name]["total_elements"]
        for name in checked
    ):
        raise SystemExit("curvature inputs differ from pinned source")


if __name__ == "__main__":
    main()
