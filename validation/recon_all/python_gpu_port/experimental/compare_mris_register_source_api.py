"""Compare a source-scheduled Python registration stage with FreeSurfer 8.2."""

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
    for name in ("candidate", "native_final", "api_report", "probe_report", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    candidate, faces, info = fsio.read_geometry(str(args.candidate), read_metadata=True)
    reference, native_faces, native_info = fsio.read_geometry(
        str(args.native_final), read_metadata=True)
    api = json.loads(args.api_report.read_text())
    probe = json.loads(args.probe_report.read_text())
    api_dt = [row["dt"] for row in api["updates"]]
    probe_dt = [probe["dt"], *(row["dt"] for row in probe["continuation"])]
    mismatch = np.flatnonzero(~np.all(candidate == reference, axis=1))
    metadata_equal = (info.keys() == native_info.keys() and all(
        np.array_equal(info[key], native_info[key]) for key in info))
    report = {
        "candidate_sha256": digest(args.candidate),
        "native_final_sha256": digest(args.native_final),
        "api_report_sha256": digest(args.api_report),
        "probe_report_sha256": digest(args.probe_report),
        "input_sha256": {name: digest(Path(api[name])) for name in
                         ("sphere", "smoothwm", "sulc_seed", "atlas")},
        "vertices": len(candidate),
        "exact_vertices": int((len(candidate) - len(mismatch)) if len(candidate) == len(reference) else 0),
        "first_mismatch_vertex": None if not len(mismatch) else int(mismatch[0]),
        "max_abs_error_mm": float(np.max(np.abs(candidate - reference))),
        "ordered_faces_equal": bool(np.array_equal(faces, native_faces)),
        "volume_geometry_equal": bool(metadata_equal),
        "api_updates": len(api_dt),
        "probe_updates": len(probe_dt),
        "all_selected_dt_equal": api_dt == probe_dt,
        "first_dt_mismatch": next((index for index, (one, two) in
                                    enumerate(zip(api_dt, probe_dt)) if one != two), None),
        "negative_count_entries": len(api["negative_counts"]),
        "api_total_seconds_including_io": api["total_seconds_including_io"],
        "api_integration_seconds": api["integration_seconds"],
        "api_repair_seconds": api["repair_seconds"],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if (report["exact_vertices"] != len(reference)
            or not report["ordered_faces_equal"]
            or not report["volume_geometry_equal"]
            or not report["all_selected_dt_equal"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
