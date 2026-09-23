"""Compare saved WMH-SynthSeg outputs on identical image grids.

Expected names in each directory: CASE_seg.nii.gz,
CASE_seg.lesion_probs.nii.gz, and CASE_volumes.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def load_case(directory, case):
    seg = nib.load(directory / f"{case}_seg.nii.gz")
    prob = nib.load(directory / f"{case}_seg.lesion_probs.nii.gz")
    with (directory / f"{case}_volumes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one CSV row for {case} in {directory}")
    return seg, prob, rows[0]


def compare_case(reference_dir, candidate_dir, case):
    reference_seg, reference_prob, reference_volumes = load_case(reference_dir, case)
    candidate_seg, candidate_prob, candidate_volumes = load_case(candidate_dir, case)
    if reference_seg.shape != candidate_seg.shape or reference_prob.shape != candidate_prob.shape:
        raise ValueError(f"Output shape differs for {case}")
    if reference_seg.shape != reference_prob.shape:
        raise ValueError(f"Segmentation/probability shape differs for {case}")
    reference = np.asarray(reference_seg.dataobj)
    candidate = np.asarray(candidate_seg.dataobj)
    first = reference == 77
    second = candidate == 77
    lesion_denominator = int(first.sum() + second.sum())
    lesion_dice = (2 * np.count_nonzero(first & second) / lesion_denominator
                   if lesion_denominator else 1.0)
    probability_reference = reference_prob.get_fdata(dtype=np.float32)
    probability_candidate = candidate_prob.get_fdata(dtype=np.float32)
    delta = probability_candidate.astype(np.float64) - probability_reference.astype(np.float64)
    probability_rms = np.sqrt(np.mean(probability_reference.astype(np.float64) ** 2))
    volume_columns = [name for name in reference_volumes if name != "Input-file"]
    if set(reference_volumes) != set(candidate_volumes):
        raise ValueError(f"CSV columns differ for {case}")
    volume_errors = {name: abs(float(reference_volumes[name]) - float(candidate_volumes[name]))
                     for name in volume_columns}
    return {
        "case": case,
        "shape": list(reference_seg.shape),
        "segmentation_voxel_agreement": float(np.mean(reference == candidate)),
        "segmentation_disagreeing_voxels": int(np.count_nonzero(reference != candidate)),
        "wmh_hard_dice": float(lesion_dice),
        "wmh_voxels_reference": int(first.sum()),
        "wmh_voxels_candidate": int(second.sum()),
        "lesion_probability_mae": float(np.mean(np.abs(delta))),
        "lesion_probability_nrmse": float(np.sqrt(np.mean(delta ** 2)) / probability_rms)
        if probability_rms else 0.0,
        "lesion_probability_max_abs": float(np.max(np.abs(delta))),
        "segmentation_affine_max_abs": float(np.max(np.abs(reference_seg.affine - candidate_seg.affine))),
        "probability_affine_max_abs": float(np.max(np.abs(reference_prob.affine - candidate_prob.affine))),
        "wmh_soft_volume_mm3_reference": float(reference_volumes["WMH(77)"]),
        "wmh_soft_volume_mm3_candidate": float(candidate_volumes["WMH(77)"]),
        "max_csv_volume_abs_error_mm3": max(volume_errors.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = [compare_case(args.reference_dir, args.candidate_dir, case) for case in args.cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Compared {len(report)} cases: {args.output}")


if __name__ == "__main__":
    main()
