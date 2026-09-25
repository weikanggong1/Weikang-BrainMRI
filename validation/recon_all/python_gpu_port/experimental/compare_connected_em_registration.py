"""Compare the connected Python Talairach LTA to the archived official one."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_mri_em_stage import _parts, _sample_voxels
from fnit.recon_all.mri_em_register import find_all_samples, read_gca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    candidate, candidate_meta = _parts(args.candidate)
    reference, reference_meta = _parts(args.reference)
    points = find_all_samples(read_gca(args.atlas)).coordinates
    differing = np.any(_sample_voxels(candidate, points) !=
                       _sample_voxels(reference, points), axis=1)

    def basename_only(row: str) -> str:
        return "filename = " + Path(row.split("=", 1)[1].strip()).name \
            if row.startswith("filename =") else row

    report = {
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "atlas_sha256": hashlib.sha256(args.atlas.read_bytes()).hexdigest(),
        "matrix_max_abs_error": float(np.max(np.abs(candidate - reference))),
        "matrix_elements_above_2e-5": int(np.count_nonzero(np.abs(candidate - reference) > 2e-5)),
        "atlas_samples": len(points),
        "atlas_samples_with_different_source_voxel": int(np.count_nonzero(differing)),
        "metadata_equal_after_path_basename": list(map(basename_only, candidate_meta)) ==
                                              list(map(basename_only, reference_meta)),
        "metadata_differences": [
            {"candidate": left, "reference": right}
            for left, right in zip(candidate_meta, reference_meta) if left != right
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
