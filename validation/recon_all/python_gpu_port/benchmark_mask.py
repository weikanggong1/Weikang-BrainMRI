"""Replay and time the four recon-all mri_mask calls on one frozen subject."""

import argparse
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


CASES = (
    ("brainmask", (), "T1", "synthstrip"),
    ("finalsurfs_threshold", ("-T", "5"), "brain", "brainmask"),
    ("finalsurfs_mcadura", ("-oval", "1", "-invert"),
     "finalsurfs_threshold", "mca-dura"),
    ("finalsurfs_vsinus", ("-oval", "1", "-invert"),
     "finalsurfs_mcadura", "vsinus"),
)


def compare(native, candidate):
    left = nib.load(str(native))
    right = nib.load(str(candidate))
    a = np.asarray(left.dataobj)
    b = np.asarray(right.dataobj)
    return {"shape": list(a.shape), "voxel_mismatches": int(np.count_nonzero(a != b)),
            "affine_max_error": float(np.max(np.abs(left.affine - right.affine))),
            "decompressed_mgh_identical": gzip.decompress(Path(native).read_bytes()) ==
                                          gzip.decompress(Path(candidate).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mri-dir", type=Path, required=True)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shared-python", action="store_true",
                        help="Keep one Python/CUDA context across all candidate calls")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Use at least one repeat")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shared_python:
        import torch
        from fnit.recon_all.mri_mask_gpu import mask_volume
        if args.device.startswith("cuda"):
            torch.empty(1, device=args.device).sum().item()
    env = {**os.environ, "FREESURFER_HOME": str(args.native_bundle),
           "FREESURFER": str(args.native_bundle),
           "FS_LICENSE": str(args.license),
           "LD_LIBRARY_PATH": str(args.native_bundle / "lib"),
           "PATH": f"{args.native_bundle / 'bin'}:/usr/bin:/bin"}
    report = {"device": args.device, "repeats": args.repeats,
              "timing_scope": (f"Native subprocess and resident Python call on {args.device}, "
                               "both with file I/O; import/device initialization excluded"
                               if args.shared_python else
                               "Full fresh command startup and file I/O for native and Python; shared host"),
              "rows": []}
    previous = set()
    for name, options, image_name, mask_name in CASES:
        timings = {"native": [], "candidate": []}
        for trial in range(args.repeats):
            for side in (("native", "candidate") if trial % 2 == 0 else
                         ("candidate", "native")):
                folder = args.output_dir / side
                folder.mkdir(exist_ok=True)
                image = (folder if image_name in previous else args.mri_dir) / f"{image_name}.mgz"
                mask = (folder if mask_name in previous else args.mri_dir) / f"{mask_name}.mgz"
                output = folder / f"{name}.mgz"
                if side == "native":
                    command = [str(args.native_bundle / "bin/mri_mask"),
                               *options, str(image), str(mask), str(output)]
                else:
                    command = [sys.executable, "-m", "fnit.recon_all.mri_mask_gpu",
                               *options, "--device", args.device,
                               str(image), str(mask), str(output)]
                start = time.perf_counter()
                if side == "candidate" and args.shared_python:
                    mask_volume(image, mask, output,
                                threshold=5 if "-T" in options else None,
                                invert="-invert" in options,
                                outside_value=1 if "-oval" in options else 0,
                                device=args.device)
                else:
                    completed = subprocess.run(command, env=env, capture_output=True, text=True)
                    if completed.returncode:
                        raise RuntimeError(f"{name} {side} trial {trial} failed:\n"
                                           f"{completed.stdout}\n{completed.stderr}")
                timings[side].append(time.perf_counter() - start)
            comparison = compare(args.output_dir / "native" / f"{name}.mgz",
                                 args.output_dir / "candidate" / f"{name}.mgz")
            if comparison["voxel_mismatches"] or not comparison["decompressed_mgh_identical"]:
                raise ValueError(f"{name} trial {trial}: {comparison}")
        report["rows"].append({"stage": name,
                               "median_seconds": {side: statistics.median(values)
                                                  for side, values in timings.items()},
                               "wall_seconds": timings,
                               "comparison": comparison})
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        previous.add(name)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
