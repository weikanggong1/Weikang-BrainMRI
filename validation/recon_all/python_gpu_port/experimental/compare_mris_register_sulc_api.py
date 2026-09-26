"""Compare an independent sulc-pass registration with the installed surface."""

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
    for name in ("candidate", "native_seed", "api_report", "schedule_audit", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    candidate, faces, info = fsio.read_geometry(str(args.candidate), read_metadata=True)
    native, native_faces, native_info = fsio.read_geometry(
        str(args.native_seed), read_metadata=True)
    api = json.loads(args.api_report.read_text())
    audit = json.loads(args.schedule_audit.read_text())
    hemisphere = args.native_seed.name[:2]
    audited = audit[hemisphere]["rows"]
    actual_dt = {row["iteration"]: row["dt"] for row in api["updates"]}
    mismatch = np.flatnonzero(~np.all(candidate == native, axis=1))
    matching_dt = all(actual_dt.get(row["epoch"]) == row["selected_dt"]
                      for row in audited)
    matching_info = info.keys() == native_info.keys() and all(
        np.array_equal(info[key], native_info[key]) for key in info)
    report = {
        "candidate_sha256": digest(args.candidate),
        "native_seed_sha256": digest(args.native_seed),
        "api_report_sha256": digest(args.api_report),
        "schedule_audit_sha256": digest(args.schedule_audit),
        "input_sha256": {key: digest(Path(api[key])) for key in
                         ("sphere", "smoothwm", "sulc", "atlas")},
        "exact_vertices": int(len(candidate) - len(mismatch)),
        "total_vertices": len(candidate),
        "first_mismatch_vertex": None if not len(mismatch) else int(mismatch[0]),
        "max_abs_error_mm": float(np.abs(candidate - native).max()),
        "ordered_faces_equal": bool(np.array_equal(faces, native_faces)),
        "volume_geometry_equal": bool(matching_info),
        "api_updates": len(api["updates"]),
        "last_iteration": api["last_iteration"],
        "audited_dt_equal": bool(matching_dt),
        "audited_dt_count": len(audited),
        "api_total_seconds_including_io": api["total_seconds_including_io"],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if (report["exact_vertices"] != len(native)
            or not report["ordered_faces_equal"]
            or not report["volume_geometry_equal"]
            or not report["audited_dt_equal"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
