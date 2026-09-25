"""Paired same-host benchmark for recon-all's first NIfTI-to-MGZ conversion."""

import argparse
import gzip
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

import nibabel as nib
import numpy as np


def _timed(command):
    start = perf_counter()
    subprocess.run(command, check=True, capture_output=True, text=True)
    return perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = gzip.decompress(Path(args.reference).read_bytes())
    count = int(np.prod(nib.load(args.reference).shape)) * 4
    end = 284 + count
    trials = []
    for index in range(args.repeat):
        native = output_dir / f"native_{index}.mgz"
        candidate = output_dir / f"python_{index}.mgz"
        native_cmd = [args.native, args.input, str(native)]
        python_cmd = [sys.executable, args.script, args.input, str(candidate)]
        if index % 2:
            python_seconds = _timed(python_cmd)
            native_seconds = _timed(native_cmd)
        else:
            native_seconds = _timed(native_cmd)
            python_seconds = _timed(python_cmd)
        native_raw = gzip.decompress(native.read_bytes())
        candidate_raw = gzip.decompress(candidate.read_bytes())
        trials.append({
            "native_seconds": native_seconds,
            "python_seconds": python_seconds,
            "native_header_match": native_raw[:284] == reference[:284],
            "python_header_match": candidate_raw[:284] == reference[:284],
            "native_voxels_match": native_raw[284:end] == reference[284:end],
            "python_voxels_match": candidate_raw[284:end] == reference[284:end],
            "native_scan_parameters_match": native_raw[end:end + 20] == reference[end:end + 20],
            "python_scan_parameters_match": candidate_raw[end:end + 20] == reference[end:end + 20],
            "native_footer_bytes": len(native_raw) - end,
            "python_footer_bytes": len(candidate_raw) - end,
        })
    report = {
        "stage": "mri_convert NIfTI T1 to mri/orig/001.mgz",
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "voxel_count": count // 4,
        "timing_scope": "cold CLI, including startup, NIfTI read and MGZ write",
        "native_median_seconds": statistics.median(t["native_seconds"] for t in trials),
        "python_median_seconds": statistics.median(t["python_seconds"] for t in trials),
        "trials": trials,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
