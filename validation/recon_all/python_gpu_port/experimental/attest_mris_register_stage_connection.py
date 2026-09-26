"""Check that a Python sulc seed equals the previously validated smoothwm input."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python_seed", "native_seed", "sulc_report",
                 "smoothwm_report", "smoothwm_final_compare", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    python_xyz, python_faces = fsio.read_geometry(str(args.python_seed))
    native_xyz, native_faces = fsio.read_geometry(str(args.native_seed))
    sulc = json.loads(args.sulc_report.read_text())
    smoothwm = json.loads(args.smoothwm_report.read_text())
    final = json.loads(args.smoothwm_final_compare.read_text())
    matching_inputs = all(digest(Path(sulc[key])) == digest(Path(smoothwm[key]))
                          for key in ("sphere", "smoothwm", "atlas"))
    matching_seed_iteration = sulc["last_iteration"] == smoothwm["seed_iteration"]
    previous_seed_file_equal = digest(Path(smoothwm["sulc_seed"])) == digest(args.native_seed)
    exact_vertices = int(np.all(python_xyz == native_xyz, axis=1).sum())
    report = {
        "python_seed_sha256": digest(args.python_seed),
        "previous_seed_sha256": digest(args.native_seed),
        "sulc_report_sha256": digest(args.sulc_report),
        "smoothwm_report_sha256": digest(args.smoothwm_report),
        "smoothwm_final_compare_sha256": digest(args.smoothwm_final_compare),
        "exact_seed_vertices": exact_vertices,
        "seed_vertices": len(native_xyz),
        "ordered_faces_equal": bool(np.array_equal(python_faces, native_faces)),
        "sphere_smoothwm_atlas_file_hashes_equal": matching_inputs,
        "seed_iteration_equal": matching_seed_iteration,
        "previous_seed_file_equal": previous_seed_file_equal,
        "previous_final_mesh_exact": (
            final["exact_vertices"] == len(native_xyz)
            and final["ordered_faces_equal"]
            and final["volume_geometry_equal"]),
        "previous_final_selected_dt_equal": final["all_selected_dt_equal"],
        "equivalent_smoothwm_inputs": (
            exact_vertices == len(native_xyz)
            and np.array_equal(python_faces, native_faces)
            and matching_inputs and matching_seed_iteration
            and previous_seed_file_equal),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not (report["equivalent_smoothwm_inputs"]
            and report["previous_final_mesh_exact"]
            and report["previous_final_selected_dt_equal"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
