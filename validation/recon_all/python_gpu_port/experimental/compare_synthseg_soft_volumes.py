"""List every official-versus-PyTorch SynthSeg soft-volume difference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict[str, str]:
    with path.open(newline="") as stream:
        return next(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    a, b = read(args.reference), read(args.candidate)
    assert a.keys() == b.keys()
    columns = []
    for name in a.keys():
        if name == "subject":
            continue
        left, right = float(a[name]), float(b[name])
        columns.append({"name": name, "reference_mm3": left,
                        "candidate_mm3": right, "difference_mm3": right - left,
                        "relative_percent": 100 * (right - left) / left if left else None})
    report = {
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "reference_subject_field": a["subject"],
        "candidate_subject_field": b["subject"],
        "columns": columns,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(columns)} soft-volume columns compared")


if __name__ == "__main__":
    main()
