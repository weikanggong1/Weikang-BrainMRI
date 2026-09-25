"""Compare SynthSeg posterior reduction orders on one fixed T1 input."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthseg_parc import SynthSeg
from fnit.synthseg_parc.postprocess import postprocess_segmentation
from fnit.synthseg_parc.preprocess import preprocess_t1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_orig", type=Path)
    parser.add_argument("official_csv", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    model = SynthSeg(weights=args.weights, device=args.device, threads=4)
    with torch.inference_mode():
        prepared = preprocess_t1(args.input_orig, device=args.device)
        posterior = model.segmenter.posterior(prepared.image)
        _, posterior = postprocess_segmentation(
            posterior, model.segmenter.labels, model.topology, prepared.content_slices)
        torch_soft = posterior[1:].sum(dim=(1, 2, 3)).cpu().numpy()
        array = posterior.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    aligned_soft = np.sum(array[..., 1:], axis=(0, 1, 2))
    original_orientation = nib.orientations.io_orientation(prepared.input_affine)
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(prepared.aligned_affine), original_orientation)
    restored = nib.orientations.apply_orientation(array, transform)
    restored_soft = np.sum(restored[..., 1:], axis=(0, 1, 2))
    source = nib.load(str(args.input_orig))
    spacing = float(np.prod(source.header["delta"]))
    determinant = float(abs(np.linalg.det(prepared.aligned_affine[:3, :3])))
    with args.official_csv.open(newline="") as stream:
        rows = list(csv.reader(stream))
    reference = np.array([float(value) for value in rows[1][1:]])

    candidates = {}
    for name, soft, scale in (
        ("torch_aligned_determinant", torch_soft, determinant),
        ("numpy_aligned_determinant", aligned_soft, determinant),
        ("numpy_restored_determinant", restored_soft, determinant),
        ("numpy_restored_header_spacing", restored_soft, spacing),
    ):
        values = np.around(np.concatenate(([np.sum(soft)], soft)) * scale, 3)
        if len(values) != len(reference):
            raise ValueError("SynthSeg CSV label count differs")
        difference = values - reference
        candidates[name] = {
            "max_abs_difference_mm3": float(np.max(np.abs(difference))),
            "total_intracranial_difference_mm3": float(difference[0]),
            "exact_columns": int(np.count_nonzero(difference == 0)),
            "differing_columns": [
                {"name": label, "reference_mm3": float(ref),
                 "candidate_mm3": float(value), "difference_mm3": float(delta)}
                for label, ref, value, delta in zip(rows[0][1:], reference, values, difference)
                if delta != 0
            ],
        }
    report = {
        "input": str(args.input_orig),
        "reference": str(args.official_csv),
        "device": args.device,
        "seconds": time.perf_counter() - started,
        "posterior_dtype": str(array.dtype),
        "posterior_shape": list(array.shape),
        "orientation_transform": transform.tolist(),
        "determinant_mm3": determinant,
        "header_spacing_mm3": spacing,
        "candidate_reductions": candidates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in result.items()
                             if key != "differing_columns"}
                      for name, result in candidates.items()}, indent=2))


if __name__ == "__main__":
    main()
