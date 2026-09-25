"""Compare the fixed white.preaparc pre-iteration smoothing checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.place_surface_smoothing import average_vertex_positions


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--native-checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"device": args.device, "iterations": 5, "hemispheres": {}}
    for hemi in ("lh", "rh"):
        source = args.subject / "surf" / f"{hemi}.orig"
        reference = args.native_checkpoints / f"{hemi}.pre_iter_nsmooth5"
        source_xyz, source_faces = fs.read_geometry(source)
        reference_xyz, reference_faces = fs.read_geometry(reference)
        started = time.perf_counter()
        predicted_xyz = average_vertex_positions(source_xyz, source_faces, 5, device=args.device)
        seconds = time.perf_counter() - started
        predicted_path = args.out / f"{hemi}.python.pre_iter_nsmooth5"
        fs.write_geometry(predicted_path, predicted_xyz, source_faces)
        delta = np.abs(predicted_xyz - reference_xyz.astype(np.float32))
        report["hemispheres"][hemi] = {
            "input_sha256": file_hash(source),
            "native_checkpoint_sha256": file_hash(reference),
            "vertices": int(len(source_xyz)),
            "faces": int(len(source_faces)),
            "faces_equal": bool(np.array_equal(source_faces, reference_faces)),
            "coordinates_equal": bool(np.array_equal(predicted_xyz, reference_xyz)),
            "mismatched_vertices": int(np.count_nonzero(np.any(delta != 0, axis=1))),
            "max_abs_mm": float(delta.max()),
            "p99_abs_mm": float(np.quantile(delta, 0.99)),
            "python_seconds": seconds,
            "python_output": str(predicted_path),
        }
    output = args.out / "place_surface_smoothing_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output.read_text())


if __name__ == "__main__":
    main()
