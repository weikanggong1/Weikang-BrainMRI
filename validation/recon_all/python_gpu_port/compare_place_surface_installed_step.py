"""Summarize the fixed installed-machine-code pial one-step diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def comparison(candidate: np.ndarray, reference: np.ndarray) -> dict:
    distance = np.linalg.norm(candidate.astype(np.float64) - reference.astype(np.float64), axis=1)
    return {
        "exact_vertices": int(np.count_nonzero(np.all(candidate == reference, axis=1))),
        "vertices": len(candidate),
        "median_mm": float(np.median(distance)),
        "p99_mm": float(np.percentile(distance, 99)),
        "max_mm": float(distance.max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--source-step", type=Path, required=True)
    parser.add_argument("--source-initial", type=Path, required=True)
    parser.add_argument("--installed-with-cleanup", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    log = args.log.read_text()
    if "INSTALLED_RAM_NITER=1" not in log or "SKIPPED_FINAL_INTERSECTIONS" not in log:
        raise ValueError("installed diagnostic hooks not recorded")
    meshes = [nib.freesurfer.read_geometry(path) for path in (
        args.installed, args.source_step, args.source_initial, args.installed_with_cleanup,
    )]
    coordinates, faces = meshes[0]
    if any(not np.array_equal(faces, other_faces) for _, other_faces in meshes[1:]):
        raise ValueError("ordered faces differ")
    report = {
        "reference": "installed FreeSurfer 8.2 binary; child RAM niterations=1; no pin; cleanup skipped",
        "vertices": len(coordinates),
        "ordered_faces": len(faces),
        "ordered_faces_exact": True,
        "installed_vs_recompiled_source_first_step": comparison(coordinates, meshes[1][0]),
        "installed_vs_initial": comparison(coordinates, meshes[2][0]),
        "installed_cleanup_skipped_vs_run": comparison(coordinates, meshes[3][0]),
        "native_internal_time_line": next((line for line in log.splitlines() if line.startswith("#ET# mris_place_surface")), ""),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
