"""Compare a connected T1-to-Talairach Python run with two archived subjects."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np


FILES = ("mri/orig/001.mgz", "mri/rawavg.mgz", "mri/orig.mgz",
         "mri/synthstrip.mgz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xfm(path: Path) -> np.ndarray:
    body = path.read_text().partition("Linear_Transform =")[2].partition(";")[0]
    matrix = np.eye(4)
    matrix[:3] = np.fromstring(body, sep=" ").reshape(3, 4)
    return matrix


def compare(candidate: Path, reference: Path) -> dict:
    volumes = {}
    for name in FILES:
        a_path, b_path = reference / name, candidate / name
        a, b = nib.load(str(a_path)), nib.load(str(b_path))
        va, vb = np.asarray(a.dataobj), np.asarray(b.dataobj)
        raw_a, raw_b = gzip.decompress(a_path.read_bytes()), gzip.decompress(b_path.read_bytes())
        end = 284 + va.size * va.dtype.itemsize
        volumes[name] = {
            "reference_sha256": sha(a_path), "candidate_sha256": sha(b_path),
            "voxel_mismatches": int(np.count_nonzero(va != vb)),
            "dtype_equal": va.dtype == vb.dtype,
            "affine_max_abs_mm": float(np.max(np.abs(a.affine - b.affine))),
            "mgh_header_equal": raw_a[:284] == raw_b[:284],
            "mgh_header_and_payload_equal": raw_a[:end] == raw_b[:end],
        }
    a_path = reference / "mri/transforms/talairach.xfm"
    b_path = candidate / "mri/transforms/talairach.xfm"
    a, b = xfm(a_path), xfm(b_path)
    orig = nib.load(str(candidate / "mri/orig.mgz"))
    corners = np.asarray(list(itertools.product(*[(0, size - 1) for size in orig.shape])))
    ras = (orig.affine @ np.c_[corners, np.ones(len(corners))].T).T
    displacement = np.linalg.norm((b @ ras.T - a @ ras.T)[:3], axis=0)
    return {"volumes": volumes, "xfm": {
        "reference_sha256": sha(a_path), "candidate_sha256": sha(b_path),
        "max_matrix_abs": float(np.max(np.abs(a - b))),
        "corner_max_mm": float(displacement.max()),
        "corner_rms_mm": float(np.sqrt(np.mean(displacement ** 2))),
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("official", type=Path)
    parser.add_argument("hybrid", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = {"official": compare(args.candidate, args.official),
              "hybrid": compare(args.candidate, args.hybrid)}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {"xfm_corner_max_mm": value["xfm"]["corner_max_mm"],
                            "voxel_mismatches": {name: row["voxel_mismatches"]
                                                 for name, row in value["volumes"].items()}}
                      for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
