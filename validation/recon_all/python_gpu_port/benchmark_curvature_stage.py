"""Time the four recon-all curvature maps against FreeSurfer on one host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import nibabel.freesurfer as fs
import numpy as np
import torch

from fnit.recon_all.surface_curvature_gpu import curvature_map


def compare(reference: Path, candidate: Path) -> dict:
    expected = fs.read_morph_data(str(reference))
    actual = fs.read_morph_data(str(candidate))
    if expected.shape != actual.shape:
        raise ValueError(f"vertex count differs for {candidate}")
    delta = np.abs(expected - actual)
    return {"vertices": int(len(expected)), "maximum_abs": float(delta.max()),
            "outliers": int(np.count_nonzero(delta > .005 + .001 * np.abs(expected)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surf-dir", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda"):
        torch.empty(1, device=args.device).sum().item()
    env = {**os.environ, "FS_LICENSE": str(args.license)}
    for hemi in ("lh", "rh"):
        for kind in ("white", "pial"):
            name = f"{hemi}.curv" if kind == "white" else f"{hemi}.curv.pial"
            surface = args.surf_dir / f"{hemi}.{kind}"
            reference = args.surf_dir / name
            outputs = {side: args.output_dir / f"{name}.{side}"
                       for side in ("native", "candidate")}
            command = [str(args.native_binary), "--curv-map", str(surface),
                       "2", "10", str(outputs["native"])]
            start = time.perf_counter()
            subprocess.run(command, env=env, capture_output=True, check=True)
            native_seconds = time.perf_counter() - start
            start = time.perf_counter()
            curvature_map(surface, outputs["candidate"], device=args.device)
            candidate_seconds = time.perf_counter() - start
            native_check = compare(reference, outputs["native"])
            candidate_check = compare(reference, outputs["candidate"])
            row = {"map": name, "device": args.device, "native_command": command,
                   "native_seconds": native_seconds,
                   "candidate_seconds": candidate_seconds,
                   "native_vs_reference": native_check,
                   "candidate_vs_reference": candidate_check}
            print(json.dumps(row), flush=True)
            if native_check["outliers"] or candidate_check["outliers"]:
                raise ValueError(f"{name} failed vertexwise accuracy")


if __name__ == "__main__":
    main()
