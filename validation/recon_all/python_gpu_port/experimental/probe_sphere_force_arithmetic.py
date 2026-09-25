"""Probe C++ promotion and operation order in the first sphere-inflation force."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(candidate: np.ndarray, native: np.ndarray) -> dict[str, float | int]:
    delta = candidate.astype(np.float64) - native.astype(np.float64)
    return {
        "exact_components": int(np.count_nonzero(candidate == native)),
        "total_components": int(candidate.size),
        "max_absolute_error": float(np.max(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native_state_dir", type=Path)
    args = parser.parse_args()
    xyz = np.fromfile(args.native_state_dir / "inflate_first_xyz.bin", dtype="<f4").reshape(-1, 3)
    native = np.fromfile(args.native_state_dir / "inflate_first_gradient_after_sphere.bin",
                         dtype="<f4").reshape(-1, 3)
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    centers = {
        "python_center": np.float32(.5) * (low.astype(np.float64) + high.astype(np.float64)).astype(np.float32),
        "cpp_center": (low + high) / np.float32(2),
    }
    results = {}
    for center_name, center in centers.items():
        centered = xyz - center
        squared = centered[:, 0] * centered[:, 0]
        squared += centered[:, 1] * centered[:, 1]
        squared += centered[:, 2] * centered[:, 2]
        length = np.sqrt(squared)
        unit = centered / length[:, None]
        ratio = (np.float32(200) - length) / np.float32(200)
        variants = {
            "python_float32": np.float32(.025) * ratio[:, None] * unit,
            "cpp_float64": (ratio[:, None].astype(np.float64) * float(np.float32(.025))
                            * unit.astype(np.float64)).astype(np.float32),
            "float32_ratio_after_weight": ratio[:, None] * np.float32(.025) * unit,
        }
        for name, candidate in variants.items():
            report = metrics(candidate, native)
            if name == "cpp_float64" and center_name == "cpp_center":
                mismatch = np.flatnonzero(np.any(candidate != native, axis=1))[:8]
                report["first_mismatch_vertices"] = mismatch.tolist()
                report["first_candidate_values"] = candidate[mismatch].tolist()
                report["first_native_values"] = native[mismatch].tolist()
                report["first_radius"] = length[mismatch].tolist()
                report["first_ratio"] = ratio[mismatch].tolist()
                report["first_unit"] = unit[mismatch].tolist()
            results[center_name + "/" + name] = report
    print(json.dumps({"centers": {key: value.tolist() for key, value in centers.items()},
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
