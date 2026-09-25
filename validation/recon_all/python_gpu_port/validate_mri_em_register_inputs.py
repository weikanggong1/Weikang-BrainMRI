"""Compare the isolated Python registration input stages with native snapshots."""

import argparse
import hashlib
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all.mri_em_register import (
    atlas_label_peak, estimate_image_white_matter_peak, find_all_samples,
    find_optimal_linear_transform,
    find_optimal_translation, find_stable_samples,
    gca_centroid, gca_mean_volume, log_sample_probability, read_gca, read_masked_input,
    regularize_covariance, scale_input_intensity, search_linear_iteration,
    search_translation_grid,
)


def compare(candidate: np.ndarray, native_path: Path) -> None:
    native = np.asarray(nib.load(str(native_path)).dataobj)
    indices = np.argwhere(candidate != native)
    first = tuple(indices[0]) if len(indices) else None
    print(native_path.name, "mismatch", len(indices), "first", first)
    if first is not None:
        print("first_values", float(candidate[first]), float(native[first]))
    assert candidate.shape == native.shape and not len(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("atlas", type=Path)
    parser.add_argument("nu", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--translation-grid", action="store_true")
    parser.add_argument("--full-translation", action="store_true")
    parser.add_argument("--first-linear-iteration", action="store_true")
    parser.add_argument("--all-linear", action="store_true")
    parser.add_argument("--all-sample-score", action="store_true")
    args = parser.parse_args()
    assert hashlib.sha256(args.atlas.read_bytes()).hexdigest() == (
        "2fcd276a39800f01f93a4c8828ae6d0a8cea3d8b8b9fe1599d4ee54e806be93e"
    )
    started = time.perf_counter()
    atlas = read_gca(args.atlas)
    print("read_gca_seconds", round(time.perf_counter() - started, 3))
    assert atlas.prior_shape == (128, 128, 128)
    assert atlas.node_shape == (64, 64, 64)
    assert atlas.volume_shape == (256, 256, 256)
    assert atlas.direction_cosines == (-1., 0., 0., 0., 0., -1., 0., 1., 0.)
    assert len(atlas.node_labels) == 324992
    assert len(atlas.prior_labels) == 2450897

    covariance = regularize_covariance(atlas)
    print("average_std", np.sqrt(covariance.average_variance),
          "min_determinant", covariance.minimum_determinant,
          "singular", covariance.singular,
          "ill_conditioned", covariance.ill_conditioned)
    assert covariance.singular == 0 and covariance.ill_conditioned == 884
    wm_peak = atlas_label_peak(atlas, 2)
    gm_peak = atlas_label_peak(atlas, 3)
    print("atlas_wm_peak", wm_peak, "atlas_gm_peak", gm_peak)
    assert wm_peak == 107 and gm_peak == 61

    masked = read_masked_input(args.nu, args.mask)
    compare(masked, args.diagnostics / "init_before_intensity.mgz")
    started = time.perf_counter()
    image_peak, threshold, box = estimate_image_white_matter_peak(atlas, masked)
    print("image_peak_seconds", round(time.perf_counter() - started, 3),
          "image_wm_peak", image_peak, "background_threshold", threshold,
          "white_matter_box", box)
    assert image_peak == 112 and abs(threshold - 16) < .05
    assert box == (92, 80, 104, 32, 31, 40)
    compare(scale_input_intensity(masked, wm_peak, image_peak),
            args.diagnostics / "init000.mgz")

    started = time.perf_counter()
    samples = find_stable_samples(atlas)
    print("stable_sample_seconds", round(time.perf_counter() - started, 3),
          "count", len(samples.labels), "unknown", int(np.count_nonzero(samples.labels == 0)))
    assert len(samples.labels) == 2841
    assert np.count_nonzero(samples.labels == 0) == 1017
    started = time.perf_counter()
    all_samples = find_all_samples(atlas)
    print("all_sample_seconds", round(time.perf_counter() - started, 3),
          "all_sample_sites", len(all_samples.labels), "unknown",
          int(np.count_nonzero(all_samples.labels == 0)))
    assert len(all_samples.labels) == 315638
    labels = np.zeros((256, 256, 256), np.uint8)
    means = np.zeros((256, 256, 256), np.float32)
    for coordinate, label, mean in zip(samples.coordinates, samples.labels, samples.means):
        x, y, z = (coordinate * 2).tolist()
        labels[x, y, z] = label if label else 29
        means[x, y, z] = mean
    compare(labels, args.diagnostics / "init000_fsamples.mgz")
    compare(means, args.diagnostics / "init000_means.mgz")
    mean_volume = gca_mean_volume(atlas)
    compare(mean_volume, args.diagnostics / "gca_mean.mgz")
    centroid = gca_centroid(mean_volume)
    print("gca_centroid", centroid.tolist())
    np.testing.assert_allclose(centroid, (126.80208924920316, 119.29025496465133,
                                          105.42089473318678), atol=1e-10)
    initial = log_sample_probability(samples, masked, np.eye(4, dtype=np.float32))
    print("initial_log_probability", initial)
    assert abs(initial - -4.382) < .0005
    if args.all_sample_score:
        # Fixed diagnostic matrix from the independent Python search run.
        pre_em = np.array([[1.1575514078, .0768136308, -.1292986125, -19.7544250488],
                           [-.0608595274, 1.2991783619, .3140552044, -52.9928398132],
                           [.0906093195, -.2865518630, 1.0190078020, 4.7752957344],
                           [0., 0., 0., 1.]], np.float32)
        started = time.perf_counter()
        all_score = log_sample_probability(
            all_samples, scale_input_intensity(masked, wm_peak, image_peak), pre_em)
        print("all_sample_score_seconds", round(time.perf_counter() - started, 3),
              "pre_em_log_probability", all_score)
        assert abs(all_score - -3.9) < .05
    if args.full_translation or args.first_linear_iteration or args.all_linear:
        started = time.perf_counter()
        scaled = scale_input_intensity(masked, wm_peak, image_peak)
        matrix, history = find_optimal_translation(
            samples, scaled, np.eye(4, dtype=np.float32))
        print("full_translation_seconds", round(time.perf_counter() - started, 3),
              "translation", matrix[:3, 3].tolist(), "history", history)
        expected_scores = (-4.142121, -4.142121, -4.111981, -4.090111,
                           -4.089818, -4.070169, -4.070169, -4.070169)
        np.testing.assert_allclose([row[0] for row in history], expected_scores, atol=2e-6)
        np.testing.assert_allclose(matrix[:3, 3], (-4.9, 11.5, -21.4), atol=.05)
        if args.first_linear_iteration:
            started = time.perf_counter()
            matrix, score = search_linear_iteration(samples, scaled, matrix, centroid, 1., .85, 1.15)
            print("first_linear_iteration_seconds", round(time.perf_counter() - started, 3),
                  "score", score, "matrix", matrix.tolist())
            expected = np.array([[1.13299, .10647, -.13920, -16.27784],
                                 [-.08771, 1.18936, .29077, -34.21072],
                                 [.11059, -.31723, 1.03617, 1.15343],
                                 [0., 0., 0., 1.]], np.float32)
            np.testing.assert_allclose(matrix, expected, atol=1e-5)
            assert abs(score - -3.540) < .0005
        if args.all_linear:
            started = time.perf_counter()
            matrix, linear_history = find_optimal_linear_transform(
                samples, scaled, matrix, centroid, history[-1][0])
            print("all_linear_seconds", round(time.perf_counter() - started, 3),
                  "history", linear_history, "matrix", matrix.tolist(), flush=True)
            assert len(linear_history) == 9
            np.testing.assert_allclose([item[1] for item in linear_history],
                                       (-3.540, -3.511, -3.505, -3.505, -3.359,
                                        -3.343, -3.343, -3.309, -3.309), atol=.0005)
            expected = np.array([[1.15755, .07681, -.12930, -19.75443],
                                 [-.06086, 1.29918, .31406, -52.99283],
                                 [.09061, -.28655, 1.01901, 4.77531],
                                 [0., 0., 0., 1.]], np.float32)
            np.testing.assert_allclose(matrix, expected, atol=2e-5)
            assert abs(linear_history[-1][1] - -3.309) < .0005
    elif args.translation_grid:
        started = time.perf_counter()
        scaled = scale_input_intensity(masked, wm_peak, image_peak)
        _, score, offset = search_translation_grid(
            samples, scaled, np.eye(4, dtype=np.float32), -200, 200)
        print("first_translation_grid_seconds", round(time.perf_counter() - started, 3),
              "offset", offset, "score", score)
        assert np.allclose(offset, (-200 / 19, 200 / 19, -200 / 19), atol=1e-6)
        assert abs(score - -4.142121) < .000002


if __name__ == "__main__":
    main()
