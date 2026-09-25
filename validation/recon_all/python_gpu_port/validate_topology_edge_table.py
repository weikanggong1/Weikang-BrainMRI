"""Compare the pre-score topology EDGE table with a native qsort capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_preflight_python import (
    center_sphere,
    genetic_candidate_edge_table,
    project_and_smooth_sphere,
)


EDGE_DTYPE = np.dtype([("vno1", "<i4"), ("vno2", "<i4"), ("length", "<f4"),
                       ("used", "<i2"), ("padding", "V2")])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {
        "freesurfer_source": "d932c45b7941662ea380a05efef580568b98d41a",
        "scope": "first-defect EDGE identities and used flags before MRI scoring sort; no GA patch parity",
        "hemispheres": {},
    }
    for hemi in ("lh", "rh"):
        surf = args.diagnostics / "fs_sub01" / "surf"
        vertices, faces = fsio.read_geometry(str(surf / f"{hemi}.qsphere.nofix"))
        sphere, _ = center_sphere(project_and_smooth_sphere(vertices, faces))
        labels = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_labels"))).astype(np.int32)
        status = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_status"))).astype(np.int8)
        table, total = genetic_candidate_edge_table(sphere, faces, labels, status, 0)
        capture_dir = args.capture_root / f"capture_{hemi}"
        before_file = capture_dir / f"{hemi}.edge.before.bin"
        after_file = capture_dir / f"{hemi}.edge.after.bin"
        before = np.fromfile(before_file, dtype=EDGE_DTYPE)
        after = np.fromfile(after_file, dtype=EDGE_DTYPE)
        native_table = np.column_stack((before["vno1"], before["vno2"], before["used"]))
        n = min(len(table), len(native_table))
        identity_differences = int(np.count_nonzero(np.any(table[:n, :2] != native_table[:n, :2], axis=1)))
        status_differences = int(np.count_nonzero(table[:n, 2] != native_table[:n, 2]))
        before_keys = set(zip(before["vno1"].tolist(), before["vno2"].tolist()))
        after_keys = set(zip(after["vno1"].tolist(), after["vno2"].tolist()))
        log = (capture_dir / f"{hemi}.capture.stderr").read_text()
        wall_match = re.search(r"wall_seconds=([0-9.]+)", log)
        result = {
            "candidate_vertices": int((1 + np.sqrt(1 + 8 * total)) / 2),
            "all_pairs": total,
            "python_edges_after_base_prune": len(table),
            "native_edges_before_sort": len(before),
            "native_edges_after_sort": len(after),
            "ordered_endpoint_differences": identity_differences,
            "ordered_used_flag_differences": status_differences,
            "python_only_edges": len(set(map(tuple, table[:, :2])) - before_keys),
            "native_only_edges": len(before_keys - set(map(tuple, table[:, :2]))),
            "native_sort_retained_same_edge_set": before_keys == after_keys,
            "native_scores_finite": bool(np.all(np.isfinite(before["length"]))),
            "native_scores_non_decreasing_after_sort": bool(np.all(np.diff(after["length"]) >= 0)),
            "native_score_range": [float(before["length"].min()), float(before["length"].max())],
            "native_used_counts": {str(value): int(np.count_nonzero(before["used"] == value))
                                   for value in (0, 1, 2)},
            "native_capture_wall_seconds": float(wall_match.group(1)) if wall_match else None,
            "native_before_sha256": _sha(before_file),
            "native_after_sha256": _sha(after_file),
        }
        report["hemispheres"][hemi] = result
        print(hemi, result)
        if (len(table) != len(before) or len(after) != len(before)
                or identity_differences or status_differences
                or result["python_only_edges"] or result["native_only_edges"]
                or not result["native_sort_retained_same_edge_set"]):
            raise SystemExit(f"{hemi}: candidate EDGE table does not match native")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
