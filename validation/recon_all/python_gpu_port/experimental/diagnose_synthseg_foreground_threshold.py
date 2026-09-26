"""Audit SynthSeg foreground-component threshold near 0.25 on one T1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fnit.synthseg_parc import SynthSeg
from fnit.synthseg_parc.preprocess import preprocess_t1
from fnit.synthseg_parc.synthseg import _restore_posterior_orientation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_orig", type=Path)
    parser.add_argument("official_post", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--band", type=float, default=1e-5)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    model = SynthSeg(weights=args.weights, device=args.device, threads=4)
    prepared = preprocess_t1(args.input_orig, device=args.device)
    with torch.inference_mode():
        raw = model.segmenter.posterior(prepared.image)
        sums = raw[1:].sum(0)
        index_tensor = torch.nonzero((sums - 0.25).abs() < args.band)
        indices = index_tensor.cpu().numpy()
        selected = raw[:, index_tensor[:, 0], index_tensor[:, 1], index_tensor[:, 2]].T.cpu().numpy()
        gpu_sum = sums[index_tensor[:, 0], index_tensor[:, 1], index_tensor[:, 2]].cpu().numpy()
    marker = np.zeros((*sums.shape, 1), dtype=np.int32)
    for index, coordinate in enumerate(indices):
        marker[tuple(coordinate) + (0,)] = index + 1
    restored = _restore_posterior_orientation(marker, prepared.volume_affine)
    official = np.asarray(nib.load(str(args.official_post)).dataobj)
    rows = []
    for index, coordinate in enumerate(indices):
        location = np.argwhere(restored[..., 0] == index + 1)
        if len(location) != 1:
            raise ValueError("orientation did not preserve marker")
        x, y, z = map(int, location[0])
        native = official[x, y, z]
        rows.append({
            "aligned_voxel": list(map(int, coordinate)),
            "original_voxel": [x, y, z],
            "torch_foreground_sum": float(gpu_sum[index]),
            "numpy_foreground_sum": float(np.sum(selected[index, 1:], dtype=np.float32)),
            "raw_background": float(selected[index, 0]),
            "raw_csf": float(selected[index, 16]),
            "official_post_foreground_sum": float(np.sum(native[1:], dtype=np.float32)),
            "official_post_background": float(native[0]),
            "official_post_csf": float(native[16]),
        })
    report = {"band": args.band, "near_threshold_voxels": len(rows), "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
