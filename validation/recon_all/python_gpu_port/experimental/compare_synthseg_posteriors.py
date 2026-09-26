"""Compare saved official SynthSeg posteriors to the PyTorch result on one T1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from fnit.synthseg_parc import SynthSeg
from fnit.synthseg_parc.postprocess import postprocess_segmentation
from fnit.synthseg_parc.preprocess import preprocess_t1
from fnit.synthseg_parc.synthseg import _restore_posterior_orientation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_orig", type=Path)
    parser.add_argument("official_post", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    model = SynthSeg(weights=args.weights, device=args.device, threads=4)
    with torch.inference_mode():
        prepared = preprocess_t1(args.input_orig, device=args.device)
        raw_posterior = model.segmenter.posterior(prepared.image)
        _, posterior = postprocess_segmentation(
            raw_posterior, model.segmenter.labels, model.topology, prepared.content_slices)
        spatial = torch.empty((*posterior.shape[1:], posterior.shape[0]),
                              dtype=posterior.dtype, device="cpu")
        spatial.copy_(posterior.permute(1, 2, 3, 0))
    candidate = _restore_posterior_orientation(spatial.numpy(), prepared.volume_affine)
    official = np.asarray(nib.load(str(args.official_post)).dataobj)
    if candidate.shape != official.shape:
        raise ValueError(f"posterior shapes differ: {candidate.shape} != {official.shape}")
    channels = candidate.shape[-1]
    maximum = np.zeros(channels, dtype=np.float64)
    absolute_total = np.zeros(channels, dtype=np.float64)
    count_1e6 = np.zeros(channels, dtype=np.int64)
    count_1e4 = np.zeros(channels, dtype=np.int64)
    exact = np.zeros(channels, dtype=np.int64)
    large_entries = []
    for start in range(0, candidate.shape[0], 8):
        difference = np.abs(candidate[start:start + 8] - official[start:start + 8])
        for row in np.argwhere(difference > 1e-4):
            x, y, z, channel = map(int, row)
            x += start
            large_entries.append({"voxel": [x, y, z], "channel": channel,
                                  "official": float(official[x, y, z, channel]),
                                  "candidate": float(candidate[x, y, z, channel])})
        maximum = np.maximum(maximum, difference.max(axis=(0, 1, 2)))
        absolute_total += difference.sum(axis=(0, 1, 2), dtype=np.float64)
        count_1e6 += np.count_nonzero(difference > 1e-6, axis=(0, 1, 2))
        count_1e4 += np.count_nonzero(difference > 1e-4, axis=(0, 1, 2))
        exact += np.count_nonzero(difference == 0, axis=(0, 1, 2))
    if large_entries:
        raw = raw_posterior[(slice(None), *prepared.content_slices)]
        raw_spatial = torch.empty((*raw.shape[1:], raw.shape[0]),
                                  dtype=raw.dtype, device="cpu")
        raw_spatial.copy_(raw.permute(1, 2, 3, 0))
        raw_restored = _restore_posterior_orientation(
            raw_spatial.numpy(), prepared.volume_affine)
        for entry in large_entries:
            x, y, z = entry["voxel"]
            channel = entry["channel"]
            entry["raw_candidate"] = float(raw_restored[x, y, z, channel])
            entry["topology_class"] = int(model.topology[channel])
    voxels = int(np.prod(candidate.shape[:3]))
    report = {
        "input_sha256": hashlib.sha256(args.input_orig.read_bytes()).hexdigest(),
        "official_post": str(args.official_post),
        "official_post_bytes": args.official_post.stat().st_size,
        "device": args.device,
        "shape": list(candidate.shape),
        "dtype_candidate": str(candidate.dtype),
        "dtype_official": str(official.dtype),
        "seconds": time.perf_counter() - started,
        "max_abs_probability": float(maximum.max()),
        "mean_abs_probability": float(absolute_total.sum() / (voxels * channels)),
        "count_abs_gt_1e6": int(count_1e6.sum()),
        "count_abs_gt_1e4": int(count_1e4.sum()),
        "exact_entries": int(exact.sum()),
        "total_entries": voxels * channels,
        "large_entries": large_entries,
        "channels": [
            {"index": index, "max_abs": float(maximum[index]),
             "mean_abs": float(absolute_total[index] / voxels),
             "count_abs_gt_1e6": int(count_1e6[index]),
             "count_abs_gt_1e4": int(count_1e4[index]),
             "exact_entries": int(exact[index])}
            for index in range(channels)
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "channels"}, indent=2))


if __name__ == "__main__":
    main()
