"""Compare paired full native pial CLI outputs from identical frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def surface(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    vertices, faces = nib.freesurfer.read_geometry(path)
    triangle = vertices[faces].astype(np.float64)
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1).sum()
    volume = np.einsum("ij,ij->", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2])) / 6
    return vertices, faces, {"area_mm2": float(area), "signed_volume_mm3": float(volume)}


def comparison(a: np.ndarray, b: np.ndarray) -> dict:
    distance = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=1)
    exact = np.all(a == b, axis=1)
    first = int(np.flatnonzero(~exact)[0]) if not np.all(exact) else None
    return {
        "exact_vertices": int(exact.sum()), "vertex_count": len(a),
        "first_different_vertex": first,
        "first_difference_mm": float(distance[first]) if first is not None else 0.0,
        "median_mm": float(np.median(distance)),
        "p99_mm": float(np.percentile(distance, 99)),
        "max_mm": float(distance.max(initial=0)),
        "vertices_over_tolerance_mm": {
            f"{threshold:g}": int(np.count_nonzero(distance > threshold))
            for threshold in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--installed-log", type=Path, required=True)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    installed, installed_faces, installed_metrics = surface(args.installed)
    source, source_faces, source_metrics = surface(args.source)
    if installed.shape != source.shape:
        raise ValueError("vertex shape mismatch")
    logs = {name: path.read_text() for name, path in (
        ("installed", args.installed_log), ("source", args.source_log),
    )}
    report = {
        "reference_class": "paired normal CLI files from same frozen scientific inputs; no debugger or diagnostic source patch",
        "installed": {**digest(args.installed), **installed_metrics},
        "source": {**digest(args.source), **source_metrics},
        "ordered_faces_exact": bool(np.array_equal(installed_faces, source_faces)),
        "ordered_face_count": len(installed_faces),
        "coordinate_comparison": comparison(installed, source),
        "optimizer_accepted_steps": {
            name: len(re.findall(r"(?m)^\d{3}: dt: (?!0\.0000)", log))
            for name, log in logs.items()
        },
        "native_elapsed_lines": {
            name: re.findall(r"(?m)^#ET# mris_place_surface.*$", log)
            for name, log in logs.items()
        },
    }
    if args.historical is not None:
        old, old_faces, old_metrics = surface(args.historical)
        report["historical_original_lh_output"] = {
            **digest(args.historical), **old_metrics,
            "basis": "original recon-all LH pial output; its LH initial objective matches the current installed rerun and all coordinates are directly compared",
            "ordered_faces_equal_to_current_installed": bool(np.array_equal(installed_faces, old_faces)),
            "current_installed_coordinate_difference": comparison(installed, old),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
