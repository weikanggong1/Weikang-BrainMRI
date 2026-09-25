"""Experimental Python replacement for the fixed T1 ``mri_em_register`` stage."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .mri_em_register import (
    atlas_label_peak, estimate_image_white_matter_peak, find_all_samples,
    find_stable_samples, gca_centroid, gca_mean_volume, read_gca,
    read_masked_input, scale_input_intensity,
)
from .mri_em_register_em_candidate import EMObjective
from .mri_em_register_lta import write_voxel_lta
from .mri_em_register_optimizer import first_em_line_search
from .mri_em_register_search_source import find_optimal_linear_transform_source
from .mri_em_register_translation_source import find_optimal_translation_source


def register_t1(nu_path: str | Path, atlas_path: str | Path,
                mask_path: str | Path, output_path: str | Path) -> dict:
    """Build a Talairach LTA without invoking a FreeSurfer executable."""

    timing = {}
    overall_started = time.perf_counter()
    started = time.perf_counter()
    atlas = read_gca(atlas_path)
    masked = read_masked_input(nu_path, mask_path)
    peak, _, _ = estimate_image_white_matter_peak(atlas, masked)
    source = scale_input_intensity(masked, atlas_label_peak(atlas, 2), peak)
    stable_samples = find_stable_samples(atlas)
    center = gca_centroid(gca_mean_volume(atlas))
    timing["prepare_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    translated, translation_history = find_optimal_translation_source(
        stable_samples, source, np.eye(4, dtype=np.float32))
    timing["translation_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    pre_em, linear_history = find_optimal_linear_transform_source(
        stable_samples, source, translated, center, translation_history[-1][0])
    timing["linear_search_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    objective = EMObjective(atlas, find_all_samples(atlas), source)
    matrix, cost, trials = first_em_line_search(objective, pre_em)
    timing["em_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    write_voxel_lta(output_path, matrix, nu_path, atlas_path, atlas, masked)
    timing["write_seconds"] = time.perf_counter() - started
    timing["total_seconds"] = time.perf_counter() - overall_started
    return {
        "output": str(output_path), "matrix": matrix.tolist(),
        "em_cost": cost, "em_trials": len(trials),
        "linear_iterations": len(linear_history), "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nu", type=Path)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register_t1(args.nu, args.atlas, args.mask, args.output), indent=2))


if __name__ == "__main__":
    main()
