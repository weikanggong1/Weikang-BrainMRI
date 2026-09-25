"""Recheck the first close-neighbor collision divergence from fixed source inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets

    surf = args.probe_root / "surf"
    before, faces = nib.freesurfer.read_geometry(surf / "lh.probe0006")
    expected, source_faces = nib.freesurfer.read_geometry(surf / "lh.probe0007")
    before = before.astype(np.float32)
    expected = expected.astype(np.float32)
    clear = np.fromfile(args.probe_root / "lh.gradient.step07.clear", dtype=STATE)
    gradient = np.fromfile(args.probe_root / "lh.gradient.step07.tangential_spring", dtype=STATE)
    ripped = clear["flags"][:, 0].astype(np.bool_)
    ordered = _ordered_neighbors(faces, len(before))[:2]
    proposed, offsets = unconstrained_step_with_offsets(
        before, gradient["floats"][:, 6:9], ripped,
    )
    actual, _ = asynchronous_first_step(
        before, faces, proposed, ripped, offsets=offsets, ordered_neighbors=ordered,
    )
    unequal = np.flatnonzero(np.any(actual != expected, axis=1))
    vertex = 9328
    report = {
        "reference": "fixed native step-seven gradient and step-six mesh; diagnostic only",
        "exact_components": int(np.count_nonzero(actual == expected)),
        "total_components": int(expected.size),
        "ordered_faces_exact": bool(np.array_equal(faces, source_faces)),
        "first_different_vertices": unequal[:10].tolist(),
        "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected))),
        "vertex_9328": {
            "before": before[vertex].tolist(),
            "unconstrained": proposed[vertex].tolist(),
            "python": actual[vertex].tolist(),
            "source": expected[vertex].tolist(),
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if unequal.size or not report["ordered_faces_exact"]:
        raise SystemExit("step-seven geometry differs")


if __name__ == "__main__":
    main()
