"""Require exact native float32 translation matrix on the fixed T1 subject."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all import mri_em_register as original
from fnit.recon_all.mri_em_register_translation_source import (
    find_optimal_translation_source,
)


NATIVE_TRANSLATED = np.eye(4, dtype=np.float32)
NATIVE_TRANSLATED[:3, 3] = [
    -4.934213161468506, 11.513154983520508, -21.381580352783203,
]
NATIVE_SCORES = np.array([
    -4.142121, -4.142121, -4.111981, -4.090111,
    -4.089818, -4.070169, -4.070169, -4.070169,
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("nu", type=Path)
    parser.add_argument("mask", type=Path)
    args = parser.parse_args()
    atlas = original.read_gca(args.atlas)
    masked = original.read_masked_input(args.nu, args.mask)
    peak, _, _ = original.estimate_image_white_matter_peak(atlas, masked)
    source = original.scale_input_intensity(masked, original.atlas_label_peak(atlas, 2), peak)
    samples = original.find_stable_samples(atlas)
    started = time.perf_counter()
    matrix, history = find_optimal_translation_source(
        samples, source, np.eye(4, dtype=np.float32))
    elapsed = time.perf_counter() - started
    print("translation_seconds", elapsed, flush=True)
    print("translation", matrix[:3, 3].tolist(), flush=True)
    print("history", [(score, best) for score, best, _ in history], flush=True)
    print("matrix_mismatches", int(np.count_nonzero(matrix != NATIVE_TRANSLATED)), flush=True)
    np.testing.assert_array_equal(matrix, NATIVE_TRANSLATED)
    np.testing.assert_allclose([h[0] for h in history], NATIVE_SCORES, rtol=0, atol=1e-6)


if __name__ == "__main__":
    main()
