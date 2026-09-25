"""Compare the isolated first EM optimizer pass with a native LTA."""

import argparse
import json
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
from fnit.recon_all.mri_em_register_optimizer import first_em_line_search
from validate_mri_em_first_divergence import NATIVE_PRE


def read_matrix(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    index = lines.index("1 4 4")
    return np.array([[float(v) for v in line.split()]
                     for line in lines[index + 1:index + 5]], np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    started = time.perf_counter()
    result, cost, history = first_em_line_search(objective, NATIVE_PRE)
    elapsed = time.perf_counter() - started
    reference = read_matrix(args.reference_lta)
    error = float(np.max(np.abs(result - reference)))
    print(json.dumps({
        "first_pass_trial_count": len(history),
        "first_pass_best_cost": cost,
        "native_final_matrix_max_abs_error": error,
        "first_pass_seconds": round(elapsed, 3),
        "result": result.tolist(),
    }, indent=2))
    return int(error > 1e-6)


if __name__ == "__main__":
    raise SystemExit(main())
