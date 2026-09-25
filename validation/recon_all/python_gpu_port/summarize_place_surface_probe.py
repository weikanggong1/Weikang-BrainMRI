"""Summarize frozen input, target, and first-step vertex checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np


def displacement(reference: Path, candidate: Path) -> dict[str, object]:
    a, a_faces = fs.read_geometry(reference)
    b, b_faces = fs.read_geometry(candidate)
    distance = np.linalg.norm(a - b, axis=1)
    return {
        "vertices": len(a),
        "faces_equal": bool(np.array_equal(a_faces, b_faces)),
        "changed_vertices": int(np.count_nonzero(distance)),
        "median_mm": float(np.median(distance)),
        "p99_mm": float(np.quantile(distance, 0.99)),
        "max_mm": float(distance.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkpoints
    results = {"installed_binary": {}, "source_probe": {}}
    for hemi in ("lh", "rh"):
        white = root / f"{hemi}.pre_iter_nsmooth5"
        pial = args.subject / "surf" / f"{hemi}.white"
        results["installed_binary"][f"{hemi}.white_target_vs_pre_iteration"] = displacement(
            white, root / f"{hemi}.target_diag"
        )
        results["installed_binary"][f"{hemi}.pial_target_vs_input"] = displacement(
            pial, root / f"{hemi}.pial.target_diag"
        )
        results["installed_binary"][f"{hemi}.pial_pre_iteration_vs_input"] = displacement(
            pial, root / f"{hemi}.pial.pre_iter"
        )
    white = root / "lh.pre_iter_nsmooth5"
    pial = args.subject / "surf/lh.white"
    for surface, baseline, probe_dir, target in (
        ("white", white, root / "probe_run/surf", root / "lh.target_diag"),
        ("pial", pial, root / "probe_pial/surf", root / "lh.pial.target_diag"),
    ):
        results["source_probe"][f"lh.{surface}.step0_vs_installed_pre_iteration"] = displacement(
            baseline, probe_dir / "lh.probe0000"
        )
        results["source_probe"][f"lh.{surface}.step1_vs_pre_iteration"] = displacement(
            baseline, probe_dir / "lh.probe0001"
        )
        results["source_probe"][f"lh.{surface}.target_vs_installed_target"] = displacement(
            target, probe_dir / ("lh.target_probe_no_opt" if surface == "white" else "lh.target")
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(args.out.read_text())


if __name__ == "__main__":
    main()
