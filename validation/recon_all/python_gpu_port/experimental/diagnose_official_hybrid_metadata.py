"""Classify metadata differences in an existing official/hybrid subject pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from nibabel import freesurfer as fs
import numpy as np


def value(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return item


def rows(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and (not line.startswith("#")
                                 or line.startswith("# Measure "))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    failed = {name: result for name, result in
              json.loads(args.comparison.read_text())["files"].items()
              if not result["pass"]}
    details = {}
    for name, result in failed.items():
        left, right = args.reference / name, args.candidate / name
        if name.startswith("surf/") and "coordinates_mm" in result:
            va = fs.read_geometry(str(left), read_metadata=True)[2]
            vb = fs.read_geometry(str(right), read_metadata=True)[2]
            details[name] = {key: [value(va.get(key)), value(vb.get(key))]
                             for key in set(va) | set(vb)
                             if not np.array_equal(va.get(key), vb.get(key))}
        elif name.startswith("stats/"):
            a, b = rows(left), rows(right)
            details[name] = [{"row": index, "reference": x, "candidate": y}
                             for index, (x, y) in enumerate(zip(a, b)) if x != y]
        elif name.startswith("mri/") and result.get("voxels", {}).get("outliers") == 0:
            ha, hb = nib.load(str(left)).header, nib.load(str(right)).header
            details[name] = {key: [value(ha[key]), value(hb[key])]
                             for key in ha.keys()
                             if not np.array_equal(ha[key], hb[key])}
    args.report.write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({"items": len(details), "names": list(details)}, indent=2))


if __name__ == "__main__":
    main()
