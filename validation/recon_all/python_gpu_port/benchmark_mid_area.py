"""Compare the two-command FreeSurfer mid-area step with one CUDA call."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from benchmark_area import compare
from fnit.recon_all.surface_area_gpu import mid_area_map


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
    torch.empty(1, device=args.device).sum().item()
    env = {**os.environ, "FREESURFER_HOME": str(args.native_bundle),
           "FREESURFER": str(args.native_bundle),
           "FS_LICENSE": str(args.license),
           "LD_LIBRARY_PATH": str(args.native_bundle / "lib"),
           "PATH": f"{args.native_bundle / 'bin'}:/usr/bin:/bin"}
    rows = []
    for hemi in ("lh", "rh"):
        white = args.surf_dir / f"{hemi}.area"
        pial = args.surf_dir / f"{hemi}.area.pial"
        reference = args.surf_dir / f"{hemi}.area.mid"
        timings = {"native": [], "gpu": []}
        for trial in range(args.repeats):
            outputs = {side: args.output_dir / f"{hemi}.area.mid.{side}.{trial}"
                       for side in ("native", "gpu")}
            for side in (("native", "gpu") if trial % 2 == 0 else ("gpu", "native")):
                began = time.perf_counter()
                if side == "gpu":
                    mid_area_map(white, pial, outputs[side], device=args.device)
                else:
                    binary = str(args.native_bundle / "bin/mris_calc")
                    for command in ([binary, "-o", str(outputs[side]), str(white), "add", str(pial)],
                                    [binary, "-o", str(outputs[side]), str(outputs[side]), "div", "2"]):
                        subprocess.run(command, env=env, capture_output=True, check=True)
                timings[side].append(time.perf_counter() - began)
                comparison = compare(reference, outputs[side])
                if comparison["outliers"]:
                    raise ValueError(f"{hemi} {side} failed on trial {trial}: {comparison}")
        rows.append({"hemi": hemi, "median_seconds":
                     {side: statistics.median(values) for side, values in timings.items()},
                     "wall_seconds": timings,
                     "gpu_vs_reference": compare(reference, outputs["gpu"]),
                     "native_vs_reference": compare(reference, outputs["native"])})
    report = {"device": args.device, "repeats": args.repeats,
              "timing_scope": "Read two area maps and write mid-area; shared Python/CUDA startup excluded",
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
