"""Pair the isolated Python mri_cc stage with a frozen native 8.2 output."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.mri_cc_python import run_mri_cc


def _matrix(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = lines.index("1 4 4") + 1
    return np.asarray([[float(value) for value in line.split()]
                       for line in lines[start:start + 4]])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("aseg", "norm", "native-aseg", "native-lta", "native-stderr", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python_aseg = args.output_dir / "python_aseg.auto.mgz"
    python_lta = args.output_dir / "python_cc_up.lta"
    began = time.perf_counter()
    info = run_mri_cc(args.aseg, args.norm, python_aseg, python_lta)
    python_seconds = time.perf_counter() - began
    original = nib.load(str(args.aseg))
    native = nib.load(str(args.native_aseg))
    python = nib.load(str(python_aseg))
    before = np.asarray(original.dataobj)
    reference = np.asarray(native.dataobj)
    candidate = np.asarray(python.dataobj)
    raw_native = gzip.decompress(args.native_aseg.read_bytes())
    raw_python = gzip.decompress(python_aseg.read_bytes())
    offset = int(native.header.get_data_offset())
    end = offset + candidate.size * candidate.dtype.itemsize
    wall_match = re.search(r"wall_seconds=([0-9.]+)", args.native_stderr.read_text())
    native_seconds = float(wall_match.group(1)) if wall_match else None
    report = {
        "pinned_freesurfer_source": "d932c45b7941662ea380a05efef580568b98d41a",
        "native_version": "8.2.0",
        "stage": "mri_cc -aseg aseg.auto_noCCseg.mgz -o aseg.auto.mgz -lta transforms/cc_up.lta fs_sub01",
        "shape": list(candidate.shape),
        "native_output_dtype": str(reference.dtype),
        "python_output_dtype": str(candidate.dtype),
        "input_sha256": _sha(args.aseg),
        "native_output_sha256": _sha(args.native_aseg),
        "python_output_sha256": _sha(python_aseg),
        "native_changed_voxels": int(np.count_nonzero(reference != before)),
        "python_changed_voxels": int(np.count_nonzero(candidate != before)),
        "native_vs_python_voxel_differences": int(np.count_nonzero(reference != candidate)),
        "cc_label_counts_native": {str(label): int(np.count_nonzero(reference == label))
                                   for label in range(251, 256)},
        "cc_label_counts_python": {str(label): int(np.count_nonzero(candidate == label))
                                   for label in range(251, 256)},
        "lta_max_abs_matrix_difference": float(np.max(np.abs(_matrix(args.native_lta) - _matrix(python_lta)))),
        "mgh_header_byte_differences": int(np.count_nonzero(
            np.frombuffer(raw_native[:offset], dtype=np.uint8) !=
            np.frombuffer(raw_python[:offset], dtype=np.uint8))),
        "mgh_voxel_byte_differences": int(np.count_nonzero(
            np.frombuffer(raw_native[offset:end], dtype=np.uint8) !=
            np.frombuffer(raw_python[offset:end], dtype=np.uint8))),
        "mgh_trailer_lengths": [len(raw_native) - end, len(raw_python) - end],
        "native_wall_seconds": native_seconds,
        "python_wall_seconds": python_seconds,
        "native_to_python_speedup": native_seconds / python_seconds if native_seconds else None,
        "python_details": info,
    }
    path = args.output_dir / "mri_cc_python_pair_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["native_vs_python_voxel_differences"] or report["mgh_voxel_byte_differences"]:
        raise SystemExit("mri_cc voxel parity failed")


if __name__ == "__main__":
    main()
