"""Fixed-subject parity and timing gate for the isolated Numba search score."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all import mri_em_register as original
from fnit.recon_all.mri_em_register_search_jit import log_sample_probability_jit
from fnit.recon_all.mri_em_register_search_source import (
    find_optimal_linear_transform_source, search_linear_iteration_source,
)
from fnit.recon_all.mri_em_register_translation_source import (
    find_optimal_translation_source,
)


NATIVE_PRE = np.array([
    [1.1575514078140259, 0.07681363821029663, -0.1292986124753952, -19.75442886352539],
    [-0.0608595199882984, 1.2991782426834106, 0.3140552043914795, -52.99283218383789],
    [0.0906093269586563, -0.2865518629550934, 1.0190079212188721, 4.77530717849731],
    [0., 0., 0., 1.],
], dtype=np.float32)
NATIVE_ITERATION_0 = np.array([
    [1.1329857110977173, 0.10647270828485489, -0.13919870555400848, -16.2778377532959],
    [-0.08771470189094543, 1.189361333847046, 0.2907693684101105, -34.210716247558594],
    [0.11058612167835236, -0.31722772121429443, 1.036171793937683, 1.1534347534179688],
    [0., 0., 0., 1.],
], dtype=np.float32)
NATIVE_TRANSLATED = np.eye(4, dtype=np.float32)
NATIVE_TRANSLATED[:3, 3] = [
    -4.934213161468506, 11.513154983520508, -21.381580352783203,
]
EXPECTED_SCORES = np.array([
    -3.540358, -3.511400, -3.505222, -3.505222, -3.359223,
    -3.343485, -3.343485, -3.309276, -3.309276,
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("nu", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("--scores-only", action="store_true")
    parser.add_argument("--source-order", action="store_true")
    parser.add_argument("--first-iteration", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    atlas = original.read_gca(args.atlas)
    masked = original.read_masked_input(args.nu, args.mask)
    peak, _, _ = original.estimate_image_white_matter_peak(atlas, masked)
    source = original.scale_input_intensity(masked, original.atlas_label_peak(atlas, 2), peak)
    samples = original.find_stable_samples(atlas)
    centroid = original.gca_centroid(original.gca_mean_volume(atlas))
    print("prepare_seconds", round(time.perf_counter() - start, 3), flush=True)

    start = time.perf_counter()
    if args.source_order:
        translated, translation_history = find_optimal_translation_source(
            samples, source, np.eye(4, dtype=np.float32))
        np.testing.assert_array_equal(translated, NATIVE_TRANSLATED)
    else:
        translated, translation_history = original.find_optimal_translation(
            samples, source, np.eye(4, dtype=np.float32))
    print("translation_seconds", round(time.perf_counter() - start, 3), flush=True)

    # Compile before the timed search and check source-order score agreement.
    first = log_sample_probability_jit(samples, source, translated)
    reference = original.log_sample_probability(samples, source, translated)
    assert first == reference, (first, reference)
    rng = np.random.default_rng(94)
    exact = 0
    largest = 0.0
    for _ in range(100):
        matrix = translated.copy()
        matrix[:3, :3] += rng.normal(0, 0.1, (3, 3)).astype(np.float32)
        matrix[:3, 3] += rng.normal(0, 20, 3).astype(np.float32)
        score_jit = log_sample_probability_jit(samples, source, matrix)
        score_numpy = original.log_sample_probability(samples, source, matrix)
        exact += score_jit == score_numpy
        largest = max(largest, abs(score_jit - score_numpy))
    print("sampled_scores_exact", exact, "/100", "max_difference", largest, flush=True)
    if args.scores_only:
        assert exact == 100
        return
    if args.first_iteration:
        assert args.source_order
        start = time.perf_counter()
        first_matrix, first_score, reductions = search_linear_iteration_source(
            samples, source, translated, centroid, 1., .85, 1.15)
        print("first_iteration_seconds", time.perf_counter() - start, flush=True)
        print("first_iteration_score", first_score, flush=True)
        print("first_iteration_matrix", first_matrix.tolist(), flush=True)
        print("first_iteration_reductions", [(best, score) for best, score, _ in reductions], flush=True)
        print("first_iteration_mismatches",
              int(np.count_nonzero(first_matrix != NATIVE_ITERATION_0)), flush=True)
        np.testing.assert_array_equal(first_matrix, NATIVE_ITERATION_0)
        return

    if args.source_order:
        start = time.perf_counter()
        matrix, history = find_optimal_linear_transform_source(
            samples, source, translated, centroid, translation_history[-1][0])
        elapsed = time.perf_counter() - start
        for iteration, entry in enumerate(history):
            print("iteration", iteration, "scale", entry[0], "score", entry[1],
                  "matrix", entry[2].tolist(), flush=True)
            for reduction, (best, score, reduced_matrix) in enumerate(entry[3]):
                print("reduction", iteration, reduction, "best", best,
                      "score", score, "matrix", reduced_matrix.tolist(), flush=True)
    else:
        original_score = original.log_sample_probability
        original.log_sample_probability = log_sample_probability_jit
        try:
            start = time.perf_counter()
            matrix, history = original.find_optimal_linear_transform(
                samples, source, translated, centroid, translation_history[-1][0])
            elapsed = time.perf_counter() - start
        finally:
            original.log_sample_probability = original_score
    scores = np.array([entry[1] for entry in history])
    print("linear_seconds", round(elapsed, 3), flush=True)
    print("linear_scores", scores.tolist(), flush=True)
    print("matrix", matrix.tolist(), flush=True)
    print("native_pre_max_abs", float(np.max(np.abs(matrix - NATIVE_PRE))), flush=True)
    print("native_pre_mismatches", int(np.count_nonzero(matrix != NATIVE_PRE)), flush=True)
    if args.source_order:
        print("native_iteration0_mismatches",
              int(np.count_nonzero(history[0][2] != NATIVE_ITERATION_0)), flush=True)
    np.testing.assert_allclose(scores, EXPECTED_SCORES, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(matrix, NATIVE_PRE)
    if args.source_order:
        np.testing.assert_array_equal(history[0][2], NATIVE_ITERATION_0)
    assert len(history) == 9


if __name__ == "__main__":
    main()
