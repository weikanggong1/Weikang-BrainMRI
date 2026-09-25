"""Native-free ``mri_normalize -aseg -mask`` for matched T1 voxel grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from ..mgh_compat import save_same_dtype_mgh
from .normalize_aseg_ridge import medial_ridge
from .normalize_aseg_source import (apply_initial_aseg_bias, filter_aseg_ridge,
                                    prepare_aseg_source)
from .normalize_gentle_source import gentle_controls
from .normalize_3d_controls import controls_3d
from .normalize_gaussian_source import (apply_gentle_bias, apply_gentle_bias_float,
                                        smooth_bias, smooth_bias_torch)
from .normalize_voronoi_source import voronoi_fill, voronoi_fill_torch


def _complete_from_initial(initial: np.ndarray, device: str,
                           three_d_iterations: int) -> tuple[np.ndarray, dict]:
    """Run native gentle and 3D passes from the float32 aseg bias output."""
    source = torch.as_tensor(np.ascontiguousarray(initial, dtype=np.float32), device=device)
    steps = {}
    tick = time.perf_counter()
    control, details = gentle_controls(source)
    steps["gentle_controls_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    if source.is_cuda:
        voronoi, _ = voronoi_fill_torch(source, control)
        bias, _ = smooth_bias_torch(voronoi, source, control)
    else:
        control_cpu = control.numpy()
        voronoi, _ = voronoi_fill(source.numpy(), control_cpu)
        bias, _ = smooth_bias(voronoi, source.numpy(), control_cpu)
        bias = torch.as_tensor(bias, device=device)
    steps["gentle_bias_seconds"] = time.perf_counter() - tick
    if not three_d_iterations:
        return apply_gentle_bias(source, bias).cpu().numpy(), {"steps": steps, "gentle_controls": details}
    current = apply_gentle_bias_float(source, bias)
    for index in range(three_d_iterations):
        tick = time.perf_counter()
        control_3d, detail = controls_3d(current.cpu().numpy())
        steps[f"three_d_{index + 1}_controls_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        if source.is_cuda:
            control_tensor = torch.as_tensor(control_3d, device=device)
            voronoi, _ = voronoi_fill_torch(current, control_tensor)
            bias, _ = smooth_bias_torch(voronoi, current, control_tensor)
        else:
            current_cpu = current.numpy()
            voronoi, _ = voronoi_fill(current_cpu, control_3d)
            bias_cpu, _ = smooth_bias(voronoi, current_cpu, control_3d)
            bias = torch.as_tensor(bias_cpu, device=device)
        steps[f"three_d_{index + 1}_bias_seconds"] = time.perf_counter() - tick
        current = torch.where(bias == 0, current, current * 110.0 / bias)
        steps[f"three_d_{index + 1}_controls"] = detail
    result = torch.floor(current.clamp(0, 255) + 0.5).to(torch.uint8)
    return result.cpu().numpy(), {"steps": steps, "gentle_controls": details}


def normalize_t1_aseg(norm_file: str | Path, aseg_file: str | Path,
                      brainmask_file: str | Path, output_file: str | Path,
                      device: str | None = None, three_d_iterations: int = 2) -> dict:
    """Normalize a conformed T1 using aseg WM controls without FreeSurfer binaries."""
    if three_d_iterations not in (0, 1, 2):
        raise ValueError("three_d_iterations must be 0, 1, or 2")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    norm_file, aseg_file, brainmask_file = map(Path, (norm_file, aseg_file, brainmask_file))
    images = [nib.load(str(path)) for path in (norm_file, brainmask_file, aseg_file)]
    if any(image.shape != images[0].shape or not np.array_equal(image.affine, images[0].affine)
           for image in images[1:]):
        raise ValueError("norm, aseg, and brainmask must share a voxel grid")
    started = time.perf_counter()
    norm, brainmask, aseg = (np.asarray(image.dataobj) for image in images)
    masked, _ = prepare_aseg_source(norm, brainmask, aseg)
    tick = time.perf_counter()
    ridge, ridge_details = medial_ridge(aseg)
    controls, removed, wm_peak = filter_aseg_ridge(masked, ridge)
    ridge_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    initial = apply_initial_aseg_bias(masked, controls)
    initial_bias_seconds = time.perf_counter() - tick
    result, completion = _complete_from_initial(initial, device, three_d_iterations)
    save_same_dtype_mgh(norm_file, output_file, result)
    return {"device": device, "three_d_iterations": three_d_iterations,
            "ridge_seconds": ridge_seconds, "initial_bias_seconds": initial_bias_seconds,
            "ridge": ridge_details, "removed_controls": int(np.count_nonzero(removed)),
            "wm_peak": wm_peak, "completion": completion,
            "total_seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--aseg", type=Path, required=True)
    parser.add_argument("--brainmask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--three-d-iterations", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    print(json.dumps(normalize_t1_aseg(args.norm, args.aseg, args.brainmask,
                                       args.output, args.device, args.three_d_iterations)))


if __name__ == "__main__":
    main()
