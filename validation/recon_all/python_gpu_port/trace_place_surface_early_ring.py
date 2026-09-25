"""Trace source/installed pial coordinate drift inside one vertex's 16-ring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vertex", type=int, default=81282)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    source1, faces = nib.freesurfer.read_geometry(root / "probe_official_pass0_run/surf/lh.probe0001")
    adjacency = [set() for _ in range(len(source1))]
    for x, y, z in faces:
        adjacency[x].update((int(y), int(z)))
        adjacency[y].update((int(x), int(z)))
        adjacency[z].update((int(x), int(y)))
    seen = {args.vertex}
    edge = seen.copy()
    shells = [sorted(edge)]
    for _ in range(args.radius):
        edge = set().union(*(adjacency[v] for v in edge)) - seen
        seen.update(edge)
        shells.append(sorted(edge))
    support = np.array(sorted(seen))
    report = {
        "reference": "installed FreeSurfer RAM first-outer checkpoints versus copied-source diagnostic geometry; selected fixed 16-ring, not full causal lineage through all earlier steps",
        "vertex": args.vertex,
        "radius": args.radius,
        "support_vertices": len(support),
        "ordered_neighbors": sorted(adjacency[args.vertex]),
        "shell_sizes": [len(row) for row in shells],
        "checkpoints": [],
    }
    installed_dirs = {
        1: "probe_installed_ram_official_step1_exact",
        5: "probe_installed_ram_official_step5",
        10: "probe_installed_ram_official_step10",
        12: "probe_installed_ram_official_step12",
        13: "probe_installed_ram_official_step13",
    }
    for step, dirname in installed_dirs.items():
        source, source_faces = nib.freesurfer.read_geometry(
            root / f"probe_official_pass0_run/surf/lh.probe{step:04d}",
        )
        installed, installed_faces = nib.freesurfer.read_geometry(
            root / dirname / "surf/lh.pial.ram",
        )
        if not np.array_equal(source_faces, faces) or not np.array_equal(installed_faces, faces):
            raise ValueError((step, "ordered faces differ"))
        distances = np.linalg.norm(source - installed, axis=1)
        local = distances[support]
        changed = support[local > 0]
        ranked = np.argsort(local)[::-1][:10]
        report["checkpoints"].append({
            "step": step,
            "center_distance_mm": float(distances[args.vertex]),
            "support_exact_vertices": int(np.count_nonzero(local == 0)),
            "support_max_mm": float(local.max()),
            "support_p99_mm": float(np.percentile(local, 99)),
            "support_above_1e-4_mm": int(np.count_nonzero(local > 1e-4)),
            "support_first_different_vertex": int(changed[0]) if len(changed) else None,
            "support_top_errors": [
                {"vertex": int(support[index]), "distance_mm": float(local[index])}
                for index in ranked if local[index] > 0
            ],
        })
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
