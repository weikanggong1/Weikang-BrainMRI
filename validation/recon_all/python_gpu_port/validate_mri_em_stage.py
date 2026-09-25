"""Compare an independently generated Python LTA with a native reference."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all.mri_em_register import (
    _vnl_affine_inverse, find_all_samples, read_gca,
)


def _parts(path: Path) -> tuple[np.ndarray, list[str]]:
    lines = path.read_text().splitlines()
    first = lines.index("1 4 4")
    matrix = np.array([[float(value) for value in row.split()]
                       for row in lines[first + 1:first + 5]], np.float32)
    nonmatrix = [" ".join(row.split()) for row in lines[2:first + 1] + lines[first + 5:]
                 if row.strip()]
    return matrix, nonmatrix


def _sample_voxels(matrix: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    transform = _vnl_affine_inverse(matrix) @ np.diag(np.array([2, 2, 2, 1], np.float32))
    points = coordinates.astype(np.float32)
    mapped = np.zeros_like(points)
    for axis in range(3):
        for input_axis in range(3):
            mapped[:, axis] = np.float32(
                mapped[:, axis] + np.float32(transform[axis, input_axis] * points[:, input_axis]))
        mapped[:, axis] = np.float32(mapped[:, axis] + transform[axis, 3])
    return np.where(mapped < 0, np.ceil(mapped - .5), np.floor(mapped + .5)).astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--matrix-atol", type=float, default=2e-5)
    args = parser.parse_args()
    candidate, candidate_metadata = _parts(args.candidate)
    reference, reference_metadata = _parts(args.reference)
    difference = np.abs(candidate - reference)
    sample_points = find_all_samples(read_gca(args.atlas)).coordinates
    candidate_voxels = _sample_voxels(candidate, sample_points)
    reference_voxels = _sample_voxels(reference, sample_points)
    voxel_mismatches = int(np.count_nonzero(np.any(candidate_voxels != reference_voxels,
                                                    axis=1)))
    report = {
        "matrix_max_abs_error": float(difference.max()),
        "matrix_elements_above_tolerance": int(np.count_nonzero(difference > args.matrix_atol)),
        "geometry_and_other_metadata_equal": candidate_metadata == reference_metadata,
        "metadata_line_count": len(candidate_metadata),
        "atlas_sample_count": len(sample_points),
        "atlas_samples_with_different_source_voxel": voxel_mismatches,
    }
    print(json.dumps(report, indent=2))
    return int(report["matrix_elements_above_tolerance"] != 0 or
               not report["geometry_and_other_metadata_equal"] or
               voxel_mismatches != 0)


if __name__ == "__main__":
    raise SystemExit(main())
