"""Compare a frozen LH pial first step with the installed FreeSurfer RAM mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])
SELECTED = (79330, 80852, 84480, 97803)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    if actual.dtype != np.float32 or reference.dtype != np.float32:
        raise TypeError("comparison requires stored float32 surface coordinates")
    exact = np.all(actual.view(np.uint32) == reference.view(np.uint32), axis=1)
    distances = np.linalg.norm(actual.astype(np.float64) - reference.astype(np.float64), axis=1)
    return {
        "exact_vertices": int(np.count_nonzero(exact)),
        "total_vertices": len(exact),
        "max_distance_mm": float(distances.max(initial=0)),
        "first_different_vertices": np.flatnonzero(~exact)[:12].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--installed-binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_collision import asynchronous_first_step
    from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets

    probe = args.root / "probe_official_pass0_run"
    clear_path = probe / "lh.gradient.step01.clear"
    final_path = probe / "lh.gradient.step01.tangential_spring"
    source_path = probe / "lh.gradient.step01.after_collision"
    installed_path = args.root / "probe_installed_ram_official_step1_exact/surf/lh.pial.ram"
    initial_path = probe / "surf/lh.probe0001"
    clear = np.fromfile(clear_path, STATE)
    final = np.fromfile(final_path, STATE)
    source = np.fromfile(source_path, STATE)
    installed, faces = nib.freesurfer.read_geometry(installed_path)
    installed = installed.astype(np.float32)
    source_mesh, source_faces = nib.freesurfer.read_geometry(initial_path)
    source_mesh = source_mesh.astype(np.float32)
    if not (len(clear) == len(final) == len(source) == len(installed)):
        raise ValueError("vertex counts differ")
    if not np.array_equal(faces, source_faces):
        raise ValueError("ordered faces differ")
    xyz = clear["floats"][:, :3].copy()
    gradient = final["floats"][:, 6:9].copy()
    ripped = clear["flags"][:, 0].astype(bool)
    proposal, offsets = unconstrained_step_with_offsets(xyz, gradient, ripped)
    neighbor_indices, neighbor_valid, _ = _ordered_neighbors(faces, len(xyz))
    accepted, order = asynchronous_first_step(
        xyz, faces, proposal, ripped, offsets=offsets,
        ordered_neighbors=(neighbor_indices, neighbor_valid), fast=True,
    )
    report = {
        "reference": "installed FreeSurfer 8.2.0-1 LH pial first-step RAM mesh; frozen copied-source gradient as input",
        "limitation": "Only v80852 gradient was confirmed bitwise equal to installed native; other vertices and later steps require independent installed gradient checks.",
        "sha256": {
            "installed_binary": sha256(args.installed_binary),
            "source_clear": sha256(clear_path),
            "source_final_gradient": sha256(final_path),
            "source_after_collision": sha256(source_path),
            "installed_ram_mesh": sha256(installed_path),
        },
        "ordered_faces_exact": True,
        "processed_vertices": len(order),
        "proposal_vs_installed": compare(proposal, installed),
        "accepted_vs_installed": compare(accepted, installed),
        "accepted_vs_copied_source": compare(accepted, source["floats"][:, :3]),
        "accepted_vs_copied_source_mesh": compare(accepted, source_mesh),
        "selected": {},
    }
    for vertex in SELECTED:
        report["selected"][str(vertex)] = {
            "gradient_bits": gradient[vertex].view(np.uint32).tolist(),
            "offset_bits": offsets[vertex].view(np.uint32).tolist(),
            "proposal_bits": proposal[vertex].view(np.uint32).tolist(),
            "accepted_bits": accepted[vertex].view(np.uint32).tolist(),
            "installed_bits": installed[vertex].view(np.uint32).tolist(),
            "source_accepted_bits": source["floats"][vertex, :3].view(np.uint32).tolist(),
            "accepted_installed_exact": bool(np.array_equal(
                accepted[vertex].view(np.uint32), installed[vertex].view(np.uint32),
            )),
        }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "proposal_vs_installed": report["proposal_vs_installed"],
        "accepted_vs_installed": report["accepted_vs_installed"],
        "selected_accepted_exact": {
            key: value["accepted_installed_exact"] for key, value in report["selected"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
