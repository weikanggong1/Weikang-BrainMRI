"""Audit official float32 CSV rendering without rerunning SynthSeg inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with args.official.open(newline="") as stream:
        official = list(csv.reader(stream))
    with args.candidate.open(newline="") as stream:
        candidate = list(csv.reader(stream))
    if len(official) != 2 or len(candidate) != 2 or official[0] != candidate[0]:
        raise ValueError("CSV header or row count differs")
    if official[1][0] != candidate[1][0]:
        raise ValueError("CSV subject differs")
    rows = []
    for name, reference_text, candidate_text in zip(
            official[0][1:], official[1][1:], candidate[1][1:]):
        reference = np.float32(reference_text)
        value = np.float32(candidate_text)
        formatted = str(value)
        rows.append({
            "structure": name,
            "official_text": reference_text,
            "candidate_text_before": candidate_text,
            "candidate_text_after": formatted,
            "float32_bit_equal": bool(reference.view(np.uint32) == value.view(np.uint32)),
            "float32_abs_difference_mm3": abs(float(reference) - float(value)),
            "formatted_abs_difference_mm3": abs(float(reference_text) - float(formatted)),
        })
    report = {
        "scope": "same-output CSV rendering audit; no inference rerun",
        "official_sha256": sha256(args.official),
        "candidate_before_sha256": sha256(args.candidate),
        "columns": len(rows),
        "float32_bit_equal_columns": sum(r["float32_bit_equal"] for r in rows),
        "formatted_text_equal_columns": sum(r["official_text"] == r["candidate_text_after"] for r in rows),
        "formatted_within_0.005_mm3_columns": sum(r["formatted_abs_difference_mm3"] <= 0.005 for r in rows),
        "max_float32_abs_difference_mm3": max(r["float32_abs_difference_mm3"] for r in rows),
        "max_formatted_abs_difference_mm3": max(r["formatted_abs_difference_mm3"] for r in rows),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
