"""Compare Python sphere-inflation positions and inherited momentum with native state."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np

from fnit.recon_all.sphere_python import inflate_before_quick_sphere


def summary(candidate: np.ndarray, native: np.ndarray) -> dict[str, float | int]:
    residual = np.linalg.norm(candidate.astype(np.float64) - native.astype(np.float64), axis=1)
    return {
        "exact_vectors": int(np.count_nonzero(np.all(candidate == native, axis=1))),
        "total_vectors": len(candidate),
        "median_error": float(np.median(residual)),
        "p95_error": float(np.percentile(residual, 95)),
        "max_error": float(np.max(residual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_surface", type=Path)
    parser.add_argument("native_projected", type=Path)
    parser.add_argument("native_momentum", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-raw", type=Path)
    parser.add_argument("--native-before-projection", type=Path)
    args = parser.parse_args()
    original, faces = fs.read_geometry(args.input_surface)
    native_projected, native_faces = fs.read_geometry(args.native_projected)
    assert np.array_equal(faces, native_faces)
    native_momentum = np.fromfile(args.native_momentum, dtype="<f4").reshape(-1, 3)
    captured: dict[str, np.ndarray] = {}
    code = inflate_before_quick_sphere.__code__
    source, start = inspect.getsourcelines(inflate_before_quick_sphere)
    before_scale = next(start + offset for offset, line in enumerate(source)
                        if line.strip() == "xyz = scale_about_bbox(xyz, np.float32(0.5))")

    def local_trace(frame, event, _arg):
        if event == "line" and frame.f_lineno == before_scale:
            captured["raw"] = frame.f_locals["xyz"].copy()
        if event == "return":
            captured["momentum"] = frame.f_locals["previous"].copy()
            captured["before_projection"] = frame.f_locals["xyz"].copy()
        return local_trace

    def global_trace(frame, event, _arg):
        return local_trace if event == "call" and frame.f_code is code else None

    sys.settrace(global_trace)
    try:
        projected = inflate_before_quick_sphere(original, faces)
    finally:
        sys.settrace(None)
    report = {
        "positions_mm": summary(projected, native_projected),
        "momentum_mm": summary(captured["momentum"], native_momentum),
        "first_python_momentum": captured["momentum"][0].tolist(),
        "first_native_momentum": native_momentum[0].tolist(),
    }
    if args.native_raw is not None:
        native_raw, raw_faces = fs.read_geometry(args.native_raw)
        assert np.array_equal(faces, raw_faces)
        report["raw_positions_mm"] = summary(captured["raw"], native_raw)
    if args.native_before_projection is not None:
        native_before, before_faces = fs.read_geometry(args.native_before_projection)
        assert np.array_equal(faces, before_faces)
        report["before_projection_mm"] = summary(captured["before_projection"], native_before)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
