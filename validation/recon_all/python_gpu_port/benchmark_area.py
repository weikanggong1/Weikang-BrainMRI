"""Paired native/CUDA area-map timing on the same four FreeSurfer surfaces."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from fnit.recon_all.surface_area_gpu import area_map


def compare(reference, candidate):
    original = fsio.read_morph_data(str(reference))
    actual = fsio.read_morph_data(str(candidate))
    if original.shape != actual.shape:
        raise ValueError(f"Vertex count differs: {reference}, {candidate}")
    delta = np.abs(actual - original)
    return {"max_error_mm2": float(delta.max()),
            "mean_error_mm2": float(delta.mean()),
            "outliers": int((delta > 0.001 + 0.001 * np.abs(original)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surf-dir", type=Path, required=True)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or not args.device.startswith("cuda"):
        parser.error("Use at least one repeat and a CUDA device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.empty(1, device=args.device).sum().item()  # Shared-process CUDA warmup.
    env = {**os.environ, "FREESURFER_HOME": str(args.native_bundle),
           "FREESURFER": str(args.native_bundle),
           "FS_LICENSE": str(args.license),
           "LD_LIBRARY_PATH": str(args.native_bundle / "lib"),
           "PATH": f"{args.native_bundle / 'bin'}:/usr/bin:/bin"}
    rows = []
    for hemi in ("lh", "rh"):
        for kind in ("white", "pial"):
            name = f"{hemi}.area" if kind == "white" else f"{hemi}.area.pial"
            surface, reference = args.surf_dir / f"{hemi}.{kind}", args.surf_dir / name
            timings = {"native": [], "gpu": []}
            for trial in range(args.repeats):
                outputs = {side: args.output_dir / f"{name}.{side}.{trial}"
                           for side in ("native", "gpu")}
                for side in (("native", "gpu") if trial % 2 == 0 else ("gpu", "native")):
                    began = time.perf_counter()
                    if side == "gpu":
                        area_map(surface, outputs[side], device=args.device)
                    else:
                        subprocess.run([str(args.native_bundle / "bin/mris_place_surface"),
                                        "--area-map", str(surface), str(outputs[side])],
                                       env=env, capture_output=True, check=True)
                    timings[side].append(time.perf_counter() - began)
                for side in outputs:
                    comparison = compare(reference, outputs[side])
                    if comparison["outliers"]:
                        raise ValueError(f"{name} {side} failed on trial {trial}: {comparison}")
            rows.append({"map": name, "vertices": len(fsio.read_morph_data(str(reference))),
                         "gpu_vs_reference": compare(reference, outputs["gpu"]),
                         "native_vs_reference": compare(reference, outputs["native"]),
                         "wall_seconds": timings,
                         "median_seconds": {side: statistics.median(values)
                                            for side, values in timings.items()}})
    report = {"device": args.device, "repeats": args.repeats,
              "timing_scope": "Each command/function reads surface and writes map; Python/Torch/CUDA import and first context creation are excluded",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
