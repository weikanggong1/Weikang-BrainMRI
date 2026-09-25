"""Source-matched mri_normalize through its 3D passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from ..mgh_compat import save_same_dtype_mgh
from .normalize_1d_source import normalize_1d
from .normalize_gentle_source import gentle_controls
from .normalize_voronoi_source import voronoi_fill, voronoi_fill_torch
from .normalize_gaussian_source import (smooth_bias, smooth_bias_torch,
                                        apply_gentle_bias, apply_gentle_bias_float)
from .normalize_3d_controls import controls_3d


def normalize_t1(input_file: str | Path, xfm_file: str | Path, output_file: str | Path,
                 device: str | None = None, three_d_iterations: int = 2,
                 diagnostic_dir: Path | None = None) -> dict:
    """Run 1D, gentle, and zero to two 3D normalization passes on a T1 MGH/MGZ."""
    if three_d_iterations not in (0, 1, 2):
        raise ValueError("three_d_iterations must be 0, 1, or 2")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    input_file, xfm_file, output_file = Path(input_file), Path(xfm_file), Path(output_file)
    started = time.perf_counter()
    image = nib.load(str(input_file))
    source = torch.as_tensor(np.asarray(image.dataobj).copy(), device=device)
    steps = {}

    tick = time.perf_counter()
    first, peaks = normalize_1d(source, image.affine, xfm_file)
    if source.is_cuda:
        torch.cuda.synchronize(source.device)
    steps["one_d_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    control, controls = gentle_controls(first)
    if source.is_cuda:
        torch.cuda.synchronize(source.device)
    steps["controls_seconds"] = time.perf_counter() - tick

    if source.is_cuda:
        tick = time.perf_counter()
        voronoi, propagation = voronoi_fill_torch(first, control)
        steps["voronoi_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        bias, smoothing = smooth_bias_torch(voronoi, first, control)
        steps["smoothing_seconds"] = time.perf_counter() - tick
    else:
        tick = time.perf_counter()
        first_cpu = first.cpu().numpy().astype(np.float32, copy=False)
        control_cpu = control.cpu().numpy()
        voronoi, propagation = voronoi_fill(first_cpu, control_cpu)
        steps["voronoi_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        bias, smoothing = smooth_bias(voronoi, first_cpu, control_cpu, strict=True)
        steps["smoothing_seconds"] = time.perf_counter() - tick

    tick = time.perf_counter()
    bias = torch.as_tensor(bias, device=device)
    passes = []
    if three_d_iterations:
        current = apply_gentle_bias_float(first, bias)
        if source.is_cuda:
            torch.cuda.synchronize(source.device)
        steps["gentle_apply_seconds"] = time.perf_counter() - tick
        for pass_index in range(three_d_iterations):
            if diagnostic_dir is not None:
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                nib.save(nib.MGHImage(current.cpu().numpy(), image.affine),
                         str(diagnostic_dir / f"src{pass_index}.mgh"))
            tick = time.perf_counter()
            control_3d, details_3d = controls_3d(current.cpu().numpy())
            steps[f"three_d_{pass_index + 1}_controls_seconds"] = time.perf_counter() - tick
            control_3d = torch.as_tensor(control_3d, device=device)
            if diagnostic_dir is not None:
                nib.save(nib.MGHImage(control_3d.cpu().numpy(), image.affine),
                         str(diagnostic_dir / f"ctrl{pass_index}.mgh"))
            tick = time.perf_counter()
            if source.is_cuda:
                voronoi_3d, propagation_3d = voronoi_fill_torch(current, control_3d)
                steps[f"three_d_{pass_index + 1}_voronoi_seconds"] = time.perf_counter() - tick
                tick = time.perf_counter()
                bias_3d, smoothing_3d = smooth_bias_torch(voronoi_3d, current, control_3d)
            else:
                current_cpu = current.cpu().numpy()
                control_cpu = control_3d.cpu().numpy()
                voronoi_3d, propagation_3d = voronoi_fill(current_cpu, control_cpu)
                steps[f"three_d_{pass_index + 1}_voronoi_seconds"] = time.perf_counter() - tick
                tick = time.perf_counter()
                bias_3d, smoothing_3d = smooth_bias(voronoi_3d, current_cpu, control_cpu)
            steps[f"three_d_{pass_index + 1}_smoothing_seconds"] = time.perf_counter() - tick
            bias_3d = torch.as_tensor(bias_3d, device=device)
            if diagnostic_dir is not None:
                nib.save(nib.MGHImage(bias_3d.cpu().numpy(), image.affine),
                         str(diagnostic_dir / f"bias{pass_index}.mgh"))
            tick = time.perf_counter()
            current = torch.where(bias_3d == 0, current, current * 110.0 / bias_3d)
            if source.is_cuda:
                torch.cuda.synchronize(source.device)
            steps[f"three_d_{pass_index + 1}_apply_seconds"] = time.perf_counter() - tick
            passes.append({"controls": details_3d, "propagation": propagation_3d,
                           "smoothing": smoothing_3d})
        tick = time.perf_counter()
        result = torch.floor(current.clamp(0, 255) + 0.5).to(torch.uint8)
    else:
        result = apply_gentle_bias(first, bias)
    save_same_dtype_mgh(input_file, output_file, result.cpu().numpy())
    steps["apply_write_seconds"] = time.perf_counter() - tick
    return {"device": device, "three_d_iterations": three_d_iterations,
            "steps": steps, "total_seconds": time.perf_counter() - started,
            "peaks": peaks, "controls": controls, "propagation": propagation,
            "smoothing": smoothing, "three_d_passes": passes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--xfm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--three-d-iterations", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--diagnostic-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(normalize_t1(args.input, args.xfm, args.output, args.device,
                                  args.three_d_iterations, args.diagnostic_dir)))


if __name__ == "__main__":
    main()
