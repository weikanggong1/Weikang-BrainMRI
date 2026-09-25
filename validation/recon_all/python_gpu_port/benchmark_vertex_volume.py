"""Paired mris_convert --volume and Python/CUDA TH3 volume-map replay."""

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

from fnit.recon_all.surface_roi_gpu import vertex_volume_map


def compare(reference, candidate):
    original = np.asarray(fsio.read_morph_data(str(reference)), dtype=np.float32)
    actual = np.asarray(fsio.read_morph_data(str(candidate)), dtype=np.float32)
    if original.shape != actual.shape:
        raise ValueError("Vertex count differs")
    delta = np.abs(actual - original)
    return {"vertices": len(original), "max_error_mm3": float(delta.max()),
            "mean_error_mm3": float(delta.mean()),
            "outliers": int((delta > 0.005 + 0.001 * np.abs(original)).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use at least one repeat")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device).sum().item()
    env = {**os.environ, "FREESURFER_HOME": str(args.native_bundle),
           "FREESURFER": str(args.native_bundle),
           "FS_LICENSE": str(args.license),
           "SUBJECTS_DIR": str(args.subject.parent),
           "LD_LIBRARY_PATH": str(args.native_bundle / "lib"),
           "PATH": f"{args.native_bundle / 'bin'}:/usr/bin:/bin"}
    rows = []
    for hemi in ("lh", "rh"):
        surf = args.subject / "surf"
        reference = surf / f"{hemi}.volume"
        timings = {"native": [], "candidate": []}
        for trial in range(args.repeats):
            outputs = {side: args.output_dir / f"{hemi}.volume.{side}.{trial}"
                       for side in timings}
            for side in (("native", "candidate") if trial % 2 == 0 else
                         ("candidate", "native")):
                start = time.perf_counter()
                if side == "candidate":
                    vertex_volume_map(surf / f"{hemi}.white", surf / f"{hemi}.pial",
                                      args.subject / "label" / f"{hemi}.cortex.label",
                                      outputs[side], device=args.device)
                else:
                    subprocess.run([str(args.native_bundle / "bin/mris_convert"),
                                    "--volume", args.subject.name, hemi,
                                    str(outputs[side])], env=env,
                                   capture_output=True, check=True)
                timings[side].append(time.perf_counter() - start)
                comparison = compare(reference, outputs[side])
                if comparison["outliers"]:
                    raise ValueError(f"{hemi} {side} trial {trial}: {comparison}")
        rows.append({"hemi": hemi,
                     "median_seconds": {side: statistics.median(values)
                                        for side, values in timings.items()},
                     "wall_seconds": timings,
                     "candidate_vs_reference": compare(reference, outputs["candidate"]),
                     "native_vs_reference": compare(reference, outputs["native"])})
    report = {"device": args.device, "repeats": args.repeats,
              "timing_scope": "Full file I/O for each stage; Python/Torch startup and initial CUDA context excluded",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
