"""Locate the first inflation update difference against native snapshots."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.inflate_center_python import center_vertices
from fnit.recon_all.sphere_python import initial_scale, inflate_before_quick_sphere


def summary(candidate: np.ndarray, native: np.ndarray) -> dict[str, float | int]:
    residual = np.linalg.norm(candidate.astype(np.float64) - native.astype(np.float64), axis=1)
    return {
        "exact_vectors": int(np.count_nonzero(np.all(candidate == native, axis=1))),
        "total_vectors": len(candidate),
        "median_mm": float(np.median(residual)),
        "max_mm": float(np.max(residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_surface", type=Path)
    parser.add_argument("native_step_0", type=Path)
    parser.add_argument("native_step_1", type=Path)
    parser.add_argument("--native-momentum", type=Path)
    args = parser.parse_args()
    original, faces = fs.read_geometry(args.input_surface)
    native0, faces0 = fs.read_geometry(args.native_step_0)
    native1, faces1 = fs.read_geometry(args.native_step_1)
    assert np.array_equal(faces, faces0) and np.array_equal(faces, faces1)
    python0 = center_vertices(center_vertices(initial_scale(original)))
    source, start = inspect.getsourcelines(inflate_before_quick_sphere)
    before_scale = next(start + offset for offset, line in enumerate(source)
                        if line.strip() == "xyz = scale_about_bbox(xyz, np.float32(0.5))")
    captured: dict[str, np.ndarray] = {}
    code = inflate_before_quick_sphere.__code__

    def local_trace(frame, event, _arg):
        if event == "line" and frame.f_lineno == before_scale:
            captured["xyz"] = frame.f_locals["xyz"].copy()
            captured["momentum"] = frame.f_locals["previous"].copy()
        return local_trace

    def global_trace(frame, event, _arg):
        return local_trace if event == "call" and frame.f_code is code else None

    sys.settrace(global_trace)
    try:
        inflate_before_quick_sphere(original, faces, iterations=1)
    finally:
        sys.settrace(None)
    report = {
        "start": summary(python0, native0),
        "after_one": summary(captured["xyz"], native1),
        "momentum_vs_snapshot_delta": summary(captured["momentum"], native1 - native0),
        "first_native_start": native0[0].tolist(),
        "first_python_start": python0[0].tolist(),
        "first_native_after": native1[0].tolist(),
        "first_python_after": captured["xyz"][0].tolist(),
        "first_python_momentum": captured["momentum"][0].tolist(),
    }
    if args.native_momentum is not None:
        native_momentum = np.fromfile(args.native_momentum, dtype="<f4").reshape(-1, 3)
        report["momentum_vs_native"] = summary(captured["momentum"], native_momentum)
        report["first_native_momentum"] = native_momentum[0].tolist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
