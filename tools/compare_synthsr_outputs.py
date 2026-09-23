"""Compare SynthSR NIfTI outputs from two arms on the same input cases.

NRMSE divides the voxel RMSE by the reference image's RMS intensity.
Metrics are computed only when shapes match; affine differences are reported
separately because unequal geometry makes direct voxel comparisons misleading.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def compare_case(reference_dir, candidate_dir, case):
    reference = nib.load(str(reference_dir / f"{case}_synthsr.nii.gz"))
    candidate = nib.load(str(candidate_dir / f"{case}_synthsr.nii.gz"))
    same_shape = reference.shape == candidate.shape
    report = {
        "case": case,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "shape_match": same_shape,
        "reference_dtype": str(reference.get_data_dtype()),
        "candidate_dtype": str(candidate.get_data_dtype()),
        "both_uint8": (reference.get_data_dtype() == np.dtype("uint8")
                       and candidate.get_data_dtype() == np.dtype("uint8")),
        "affine_max_abs": float(np.max(np.abs(reference.affine - candidate.affine))),
    }
    if not same_shape:
        report.update(mae=None, rmse=None, nrmse=None, exact_fraction=None,
                      max_abs_difference=None)
        return report
    first = np.asarray(reference.dataobj, dtype=np.float64)
    second = np.asarray(candidate.dataobj, dtype=np.float64)
    delta = second - first
    rmse = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(first * first)))
    report.update({
        "mae": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "nrmse": (rmse / reference_rms if reference_rms else (0.0 if rmse == 0 else None)),
        "exact_fraction": float(np.mean(first == second)),
        "max_abs_difference": float(np.max(np.abs(delta))),
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", help="case stems; default: all reference outputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_dir = args.reference_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    cases = args.cases or sorted(
        path.name.removesuffix("_synthsr.nii.gz")
        for path in reference_dir.glob("*_synthsr.nii.gz")
    )
    if not cases:
        parser.error("No reference SynthSR outputs found")
    reports = []
    for case in cases:
        try:
            reports.append(compare_case(reference_dir, candidate_dir, case))
        except Exception as exc:
            reports.append({"case": case, "error": f"{type(exc).__name__}: {exc}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    failures = sum("error" in row or not row["shape_match"] or not row["both_uint8"]
                   for row in reports)
    print(f"Compared {len(reports)} cases; {failures} incomplete/invalid: {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
