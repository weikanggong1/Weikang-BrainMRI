"""Check a one-call native-free sphere registration against FreeSurfer and validated stages."""

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
    for name in ("candidate", "native_final", "combined_report", "sulc_report",
                 "smoothwm_report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    xyz, faces, info = fsio.read_geometry(str(args.candidate), read_metadata=True)
    reference, reference_faces, reference_info = fsio.read_geometry(
        str(args.native_final), read_metadata=True)
    combined = json.loads(args.combined_report.read_text())
    sulc = json.loads(args.sulc_report.read_text())
    smoothwm = json.loads(args.smoothwm_report.read_text())
    equal_vertices = np.all(xyz == reference, axis=1)
    input_hashes_equal = all(
        digest(Path(combined[name])) == digest(Path(sulc[name]))
        for name in ("sphere", "smoothwm", "sulc", "atlas"))
    report = {
        "candidate_sha256": digest(args.candidate),
        "native_final_sha256": digest(args.native_final),
        "combined_report_sha256": digest(args.combined_report),
        "reference_sulc_report_sha256": digest(args.sulc_report),
        "reference_smoothwm_report_sha256": digest(args.smoothwm_report),
        "vertices": len(xyz),
        "exact_vertices": int(equal_vertices.sum()),
        "max_abs_error_mm": float(np.max(np.abs(xyz - reference))),
        "ordered_faces_equal": bool(np.array_equal(faces, reference_faces)),
        "volume_geometry_equal": bool(info.keys() == reference_info.keys() and all(
            np.array_equal(info[key], reference_info[key]) for key in info)),
        "input_hashes_equal": input_hashes_equal,
        "sulc_seed_file_equal": (combined["sulc_seed_sha256"] == digest(Path(sulc["output"]))
                                 if "sulc_seed_sha256" in combined else None),
        "sulc_selected_dt_equal": [row["dt"] for row in combined["sulc_pass"]["updates"]]
        == [row["dt"] for row in sulc["updates"]],
        "smoothwm_selected_dt_equal": [row["dt"] for row in combined["smoothwm_pass"]["updates"]]
        == [row["dt"] for row in smoothwm["updates"]],
        "sulc_updates": len(combined["sulc_pass"]["updates"]),
        "smoothwm_updates": len(combined["smoothwm_pass"]["updates"]),
        "total_seconds_including_io": combined["total_seconds_including_io"],
        "sulc_seconds_including_io": combined["sulc_pass"]["total_seconds_including_io"],
        "smoothwm_seconds_including_io": combined["smoothwm_pass"]["total_seconds_including_io"],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if (report["exact_vertices"] != len(reference)
            or not all(report[key] for key in (
                "ordered_faces_equal", "volume_geometry_equal", "input_hashes_equal",
                "sulc_selected_dt_equal", "smoothwm_selected_dt_equal"))):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
