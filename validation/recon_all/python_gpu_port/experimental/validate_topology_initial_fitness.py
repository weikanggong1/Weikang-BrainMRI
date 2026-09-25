"""Check first-defect native fitness trace against source-order Python composition/rank."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_fitness_search import (
    compose_patch_fitness, rank_patch_fitness,
)


COMPONENTS = re.compile(
    r"^fll=([-\d.]+) \([^)]+\), vll=([-\d.]+) \([^)]+\),"
    r"cll=([-\d.]+) \([^)]+\),qcll=([-\d.]+) \([^)]+\)"
    r" umll=([-\d.]+) \([^)]+\)$", re.M,
)
FITNESS = re.compile(r"^for the patch #(\d+), we have fitness = ([-\d.]+)\s*$", re.M)
BEST = re.compile(r"^best fitness at (\d+): ([-\d.]+) \(([-\d.]+) \+- ([-\d.]+)\)$", re.M)


def validate(log: Path, snapshots: Path, hemi: str) -> dict:
    source = log.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    scores = [(int(index), float(value)) for index, value in FITNESS.findall(source)]
    if [index for index, _ in scores] != list(range(10)):
        raise ValueError("expected source-order first 10 initial patch fitness scores")
    fitness = np.asarray([value for _, value in scores], np.float64)
    printed = COMPONENTS.findall(source)
    if len(printed) < 2:
        raise ValueError("missing first two printed fitness component rows")
    component_errors = []
    for index in (0, 1):
        face, vertex, normal, quadratic, volume = map(float, printed[index])
        composed = compose_patch_fitness(face, vertex, normal, quadratic, volume, True)
        component_errors.append(composed - fitness[index])
    ranks = rank_patch_fitness(fitness)
    selected = int(np.argmin(ranks))
    best_serial = 0
    running_best = fitness[0]
    for value in fitness[1:]:
        if value > running_best:
            running_best = value
            best_serial += 1
    summary = BEST.search(source)
    if summary is None:
        raise ValueError("missing native first population best/mean/sigma")
    native_best, native_value, native_mean, native_sigma = summary.groups()
    mean = float(fitness.mean())
    sigma = float(np.sqrt(np.mean(fitness * fitness) - mean * mean))
    selected_surface = snapshots / f"rh.defect_0_select{selected}"
    best_surface = snapshots / f"rh.defect_0_best_{best_serial}"
    xyz, faces = fsio.read_geometry(str(selected_surface))
    best_xyz, best_faces = fsio.read_geometry(str(best_surface))
    result = {
        "hemisphere": hemi,
        "native_verbose_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "native_selected_snapshot_sha256": hashlib.sha256(selected_surface.read_bytes()).hexdigest(),
        "native_best_snapshot_sha256": hashlib.sha256(best_surface.read_bytes()).hexdigest(),
        "scope": "first defect initial population; printed native likelihood components only; no independent component generation or GA evolution",
        "initial_fitness": fitness.tolist(),
        "source_ranks": ranks.tolist(),
        "python_best_initial_index": selected,
        "native_best_initial_index": int(native_best),
        "python_initial_best_snapshot_serial": best_serial,
        "python_best_initial_fitness": float(fitness[selected]),
        "native_best_initial_fitness_printed_4dp": float(native_value),
        "composition_errors_from_4dp_components": component_errors,
        "python_initial_mean": mean,
        "native_initial_mean_printed_4dp": float(native_mean),
        "python_initial_sigma": sigma,
        "native_initial_sigma_printed_4dp": float(native_sigma),
        "selected_vs_best_ordered_vertices_equal": bool(np.array_equal(xyz, best_xyz)),
        "selected_vs_best_ordered_faces_equal": bool(np.array_equal(faces, best_faces)),
        "selected_surface_vertices": len(xyz),
        "selected_surface_faces": len(faces),
    }
    if (selected != int(native_best) or abs(fitness[selected] - float(native_value)) > 5e-5
            or max(abs(x) for x in component_errors) > 7e-4
            or abs(mean - float(native_mean)) > 5e-5
            or abs(sigma - float(native_sigma)) > 5e-5
            or not result["selected_vs_best_ordered_vertices_equal"]
            or not result["selected_vs_best_ordered_faces_equal"]):
        raise AssertionError(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemi", choices=("lh", "rh"), required=True)
    ap.add_argument("--native-verbose-log", type=Path, required=True)
    ap.add_argument("--native-snapshots", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    result = validate(args.native_verbose_log, args.native_snapshots, args.hemi)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
