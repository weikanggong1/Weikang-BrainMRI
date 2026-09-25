"""Pair the fixed SynthSeg cortical-mask command with FreeSurfer mri_binarize."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


def _run(argv: list[str], env: dict[str, str]) -> float:
    started = time.perf_counter()
    result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"{argv[0]} exited {result.returncode}: {result.stdout[-1000:]}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--native-binary", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FS_LICENSE=args.license)
    times: dict[str, list[float]] = {"native": [], "pytorch": []}
    paths = {}
    paired_bytes_equal = []
    for round_index in range(-1, args.repeats):
        for name in (("native", "pytorch") if round_index % 2 else ("pytorch", "native")):
            path = output / f"{name}.{round_index}.mgz"
            if name == "native":
                argv = [args.native_binary, "--i", args.input, "--match", "3", "42",
                        "--inv", "--o", str(path)]
            else:
                argv = [sys.executable, "-m", "fnit.recon_all.mri_binarize_gpu",
                        "--i", args.input, "--match", "3", "42", "--inv", "--o",
                        str(path), "--device", args.device]
            elapsed = _run(argv, env)
            if round_index >= 0:
                times[name].append(elapsed)
                paths[name] = path
            if name == "pytorch" and args.device.startswith("cuda"):
                time.sleep(3)
        if round_index >= 0:
            identical = (gzip.decompress(paths["native"].read_bytes()) ==
                         gzip.decompress(paths["pytorch"].read_bytes()))
            paired_bytes_equal.append(identical)
            if not identical:
                raise ValueError(f"decompressed MGH differs in trial {round_index}")

    native, candidate = (nib.load(str(paths[name])) for name in ("native", "pytorch"))
    native_voxels = np.asarray(native.dataobj)
    candidate_voxels = np.asarray(candidate.dataobj)
    native_bytes = gzip.decompress(paths["native"].read_bytes())
    candidate_bytes = gzip.decompress(paths["pytorch"].read_bytes())
    offset = int(native.header.get_data_offset())
    end = offset + native_voxels.size * native.get_data_dtype().itemsize
    report = {
        "command": "mri_binarize --i synthseg.rca.mgz --match 3 42 --inv --o ctxsegmask.mgz",
        "device": args.device,
        "repeats": args.repeats,
        "timing_scope": "Fresh command startup, device initialization, and file I/O on the same host; one warm-up pair excluded",
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "decompressed_output_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "shape": [int(length) for length in native.shape],
        "native_dtype": str(native.get_data_dtype()),
        "pytorch_dtype": str(candidate.get_data_dtype()),
        "voxel_mismatches": int(np.count_nonzero(native_voxels != candidate_voxels)),
        "native_positive_voxels": int(np.count_nonzero(native_voxels)),
        "pytorch_positive_voxels": int(np.count_nonzero(candidate_voxels)),
        "affine_equal": bool(np.array_equal(native.affine, candidate.affine)),
        "mgh_header_equal": native_bytes[:offset] == candidate_bytes[:offset],
        "mgh_footer_equal": native_bytes[end:] == candidate_bytes[end:],
        "mgh_bytes_equal": native_bytes == candidate_bytes,
        "paired_decompressed_mgh_equal": paired_bytes_equal,
        "mgh_footer_lengths": [len(native_bytes[end:]), len(candidate_bytes[end:])],
        "wall_seconds": times,
        "median_wall_seconds": {name: statistics.median(values) for name, values in times.items()},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["voxel_mismatches"] or not report["affine_equal"] or not report["mgh_bytes_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
