"""Compare first Python sphere-inflation gradient with native GDB state."""
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
    delta = candidate.astype(np.float64) - native.astype(np.float64)
    residual = np.linalg.norm(delta, axis=1)
    return {
        "exact_components": int(np.count_nonzero(candidate == native)),
        "total_components": int(candidate.size),
        "exact_vectors": int(np.count_nonzero(np.all(candidate == native, axis=1))),
        "median_norm": float(np.median(residual)),
        "max_norm": float(np.max(residual)),
        "first_python": candidate[0].tolist(),
        "first_native": native[0].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_surface", type=Path)
    parser.add_argument("native_state_dir", type=Path)
    args = parser.parse_args()
    original, faces = fs.read_geometry(args.input_surface)
    source, start = inspect.getsourcelines(inflate_before_quick_sphere)
    lines = {line.strip(): start + offset for offset, line in enumerate(source)}
    checkpoints = {
        lines["spring = np.zeros_like(xyz)"]: "after_sphere",
        lines["gradient = add_spring(gradient, xyz, normals, one, degree,"]: "after_convexity",
        lines["previous = (gradient.astype(np.float64) * 0.9"]: "after_normalized_spring",
    }
    captured: dict[str, np.ndarray] = {}
    code = inflate_before_quick_sphere.__code__

    def local_trace(frame, event, _arg):
        if event == "line" and frame.f_lineno in checkpoints:
            captured[checkpoints[frame.f_lineno]] = frame.f_locals["gradient"].copy()
            if "xyz" not in captured:
                captured["xyz"] = frame.f_locals["xyz"].copy()
                captured["normal"] = frame.f_locals["normals"].copy()
        return local_trace

    def global_trace(frame, event, _arg):
        return local_trace if event == "call" and frame.f_code is code else None

    sys.settrace(global_trace)
    try:
        inflate_before_quick_sphere(original, faces, iterations=1)
    finally:
        sys.settrace(None)
    report = {}
    for key in ("xyz", "normal", "after_sphere", "after_convexity",
                "after_normalized_spring"):
        name = (f"inflate_first_{key}.bin" if key in ("xyz", "normal") else
                f"inflate_first_gradient_{key}.bin")
        native = np.fromfile(args.native_state_dir / name, dtype="<f4").reshape(-1, 3)
        report[key] = summary(captured[key], native)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
