"""Compare a connected Python Talairach voxel LTA with archived official output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from itertools import product

import numpy as np
import surfa as sf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = (sf.load_affine(str(path)) for path in
                            (args.official, args.candidate))
    corners = np.array([(*xyz, 1) for xyz in product((0., 255.), repeat=3)])
    mapped_reference = corners @ reference.matrix.T
    mapped_candidate = corners @ candidate.matrix.T
    report = {
        "official_sha256": hashlib.sha256(args.official.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "official_space": str(reference.space),
        "candidate_space": str(candidate.space),
        "source_shapes_equal": bool(np.array_equal(reference.source.shape, candidate.source.shape)),
        "target_shapes_equal": bool(np.array_equal(reference.target.shape, candidate.target.shape)),
        "source_affine_max_abs_mm": float(np.max(np.abs(
            reference.source.vox2world.matrix - candidate.source.vox2world.matrix))),
        "target_affine_max_abs_mm": float(np.max(np.abs(
            reference.target.vox2world.matrix - candidate.target.vox2world.matrix))),
        "matrix_max_abs": float(np.max(np.abs(reference.matrix - candidate.matrix))),
        "corner_max_euclidean_mm": float(np.max(np.linalg.norm(
            mapped_reference[:, :3] - mapped_candidate[:, :3], axis=1))),
        "official_eTIV_mm3": float(1_948_106 / np.linalg.det(reference.matrix[:3, :3])),
        "candidate_eTIV_mm3": float(1_948_106 / np.linalg.det(candidate.matrix[:3, :3])),
    }
    report["eTIV_abs_difference_mm3"] = abs(
        report["official_eTIV_mm3"] - report["candidate_eTIV_mm3"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
