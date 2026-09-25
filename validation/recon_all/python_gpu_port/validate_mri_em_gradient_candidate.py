"""Compare the independent Python EM gradient with the native final direction."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all.mri_em_register import (
    atlas_label_peak, estimate_image_white_matter_peak, find_all_samples,
    read_gca, read_masked_input, scale_input_intensity,
)
from fnit.recon_all.mri_em_register_em_candidate import EMObjective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("nu", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("reference_lta", type=Path)
    args = parser.parse_args()
    atlas = read_gca(args.atlas)
    masked = read_masked_input(args.nu, args.mask)
    peak, _, _ = estimate_image_white_matter_peak(atlas, masked)
    source = scale_input_intensity(masked, atlas_label_peak(atlas, 2), peak)
    objective = EMObjective(atlas, find_all_samples(atlas), source)
    pre_em = np.array([
        [1.1575514078, .0768136308, -.1292986125, -19.7544250488],
        [-.0608595274, 1.2991783619, .3140552044, -52.9928398132],
        [.0906093195, -.2865518630, 1.0190078020, 4.7752957344],
        [0., 0., 0., 1.],
    ], np.float32)
    lines = args.reference_lta.read_text().splitlines()
    position = lines.index("1 4 4")
    native = np.array([[float(value) for value in line.split()]
                       for line in lines[position + 1:position + 5]], np.float32)
    started = time.perf_counter()
    gradient = objective.gradient(pre_em)
    print("gradient_seconds", round(time.perf_counter() - started, 3))
    print("initial_cost", objective.cost(pre_em))
    print("gradient", gradient.tolist())
    delta = native[:3, :3] - pre_em[:3, :3]
    factor = -float(np.sum(delta * gradient[:, :3]) /
                    np.sum(gradient[:, :3] ** 2))
    residual = delta + factor * gradient[:, :3]
    print("native_step_factor", factor, "gradient_direction_max_error",
          float(np.max(np.abs(residual))), "translation_difference",
          (native[:3, 3] - pre_em[:3, 3]).tolist())
    assert np.max(np.abs(residual)) < 3e-7
    assert np.max(np.abs(native[:3, 3] - pre_em[:3, 3])) < 2e-5


if __name__ == "__main__":
    main()
