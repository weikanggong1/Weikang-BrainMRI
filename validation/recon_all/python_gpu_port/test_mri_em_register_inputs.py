"""Isolated input-stage tests for the fixed mri_em_register port."""

from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from fnit.recon_all.mri_em_register import (
    StableSamples, _vnl_affine_inverse, atlas_label_peak, find_all_sample_sites, find_all_samples,
    find_optimal_linear_transform,
    log_sample_probability, read_gca,
    read_masked_input, regularize_covariance, scale_input_intensity,
    search_translation_grid,
)


class EmRegisterInputTests(unittest.TestCase):
    def test_vnl_affine_inverse_matches_native_first_em_matrix(self):
        transform = np.array([
            [1.1575514078140259, .07681363821029663, -.1292986124753952, -19.754428863525391],
            [-.060859519988298416, 1.2991782426834106, .31405520439147949, -52.99283218383789],
            [.090609326958656311, -.28655186295509338, 1.0190079212188721, 4.7753071784973145],
            [0, 0, 0, 1],
        ], np.float32)
        native_prior_to_source = np.array([
            [1.7070131301879883, -.04977000132203102, .23193633556365967, 14.988024711608887],
            [.10923102498054504, 1.4382643699645996, -.4294087886810303, 40.21303176879883],
            [-.12106967717409134, .40887507796287537, 1.8213170766830444, 5.289217948913574],
            [0, 0, 0, 1],
        ], np.float32)
        prior_scale = np.diag(np.array([2, 2, 2, 1], np.float32))
        np.testing.assert_array_equal(_vnl_affine_inverse(transform) @ prior_scale,
                                      native_prior_to_source)

    def test_one_classifier_and_priors(self):
        header = struct.pack(">fff8i", 5., 2., 4., 2, 2, 2, 1, 1, 1, 1, 0)
        node = struct.pack(">iiiff", 1, 12, 2, 98., 4.) + struct.pack(">i", 0) * 6
        priors = struct.pack(">iiif", 1, 10, 2, .5) + struct.pack(">ii", 0, 0) * 7
        tags = struct.pack(">4i", 0xAB2C, 2, 1, 0)
        direction = struct.pack(">i12f3i3f", 3, -1, 0, 0, 0, 0, -1, 0, 1, 0,
                                0, 0, 0, 8, 8, 8, 1, 1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.gca"
            path.write_bytes(header + node + priors + tags + direction)
            gca = read_gca(path)
        self.assertEqual(gca.node_labels.tolist(), [2])
        self.assertEqual(gca.prior_labels.tolist(), [2])
        self.assertEqual(gca.volume_shape, (8, 8, 8))
        self.assertEqual(atlas_label_peak(gca, 2), 98)
        coordinates, labels, priors = find_all_sample_sites(gca)
        np.testing.assert_array_equal(coordinates, [[0, 0, 0]])
        np.testing.assert_array_equal(labels, [2])
        np.testing.assert_array_equal(priors, [.5])
        complete = find_all_samples(gca)
        np.testing.assert_array_equal(complete.means, [98.])
        np.testing.assert_array_equal(complete.variances, [4.])
        check = regularize_covariance(gca)
        self.assertEqual(check.singular, 0)
        self.assertEqual(check.ill_conditioned, 0)
        self.assertEqual(check.average_variance, 4.)

    def test_mask_values_zero_to_four(self):
        with tempfile.TemporaryDirectory() as directory:
            nu = Path(directory) / "nu.nii.gz"
            mask = Path(directory) / "brainmask.nii.gz"
            nib.save(nib.Nifti1Image(np.full((2, 3, 1), 100, np.uint8), np.eye(4)), nu)
            nib.save(nib.Nifti1Image(np.arange(6, dtype=np.uint8).reshape(2, 3, 1), np.eye(4)), mask)
            masked = read_masked_input(nu, mask)
        self.assertEqual(np.count_nonzero(masked), 1)
        self.assertEqual(masked[1, 2, 0], 100)

    def test_uint8_scalar_multiply_truncates(self):
        source = np.array([0, 1, 111, 112, 201], np.uint8)
        scaled = scale_input_intensity(source, 107, 112)
        np.testing.assert_array_equal(scaled, [0, 0, 106, 107, 192])

    def test_clamped_sample_likelihood_and_grid(self):
        samples = StableSamples(np.array([[1, 1, 1]], np.int32),
                                np.array([2], np.int32), np.array([100], np.float32),
                                np.array([4], np.float32), np.array([1], np.float32))
        source = np.zeros((5, 5, 5), np.uint8)
        source[2, 2, 2] = 100
        identity = np.eye(4, dtype=np.float32)
        self.assertAlmostEqual(log_sample_probability(samples, source, identity), -np.log(2), places=6)
        source[2, 2, 2] = 0
        self.assertEqual(log_sample_probability(samples, source, identity), -6.)
        source[2, 2, 2] = 100
        matrix, score, offset = search_translation_grid(samples, source, identity, -2, 2, 2)
        np.testing.assert_array_equal(matrix, identity)
        self.assertEqual(offset, (0., 0., 0.))
        self.assertAlmostEqual(score, -np.log(2), places=6)

    def test_linear_search_reductions(self):
        scores = iter((-3.540, -3.511, -3.505, -3.505, -3.359,
                       -3.343, -3.343, -3.309, -3.309))
        scales = []

        def iteration(_samples, _source, matrix, _origin, scale, minimum, maximum):
            scales.append((scale, minimum, maximum))
            return matrix, next(scores)

        with patch("fnit.recon_all.mri_em_register.search_linear_iteration",
                   side_effect=iteration):
            matrix, history = find_optimal_linear_transform(
                None, np.empty(0), np.eye(4, dtype=np.float32), np.zeros(3), -4.070)
        np.testing.assert_array_equal(matrix, np.eye(4, dtype=np.float32))
        self.assertEqual(len(history), 9)
        np.testing.assert_allclose([scale[0] for scale in scales],
                                   [1.] * 4 + [.25] * 3 + [.0625] * 2)
        np.testing.assert_allclose(scales[4][1:], (.9625, 1.0375))


if __name__ == "__main__":
    unittest.main()
