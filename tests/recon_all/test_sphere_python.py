"""Small source-operation checks; frozen-subject parity lives in validation/."""

import numpy as np

from fnit.recon_all.sphere_python import (
    inflate_before_quick_sphere,
    initial_scale,
    project_before_quick_sphere,
    project_radially,
    scale_about_bbox,
)
from fnit.recon_all.sphere_quick_python import _quadratic_minimum32


def test_native_first_sphere_line_fit_from_logged_bracket():
    dt = 1193.26928947152
    predicted = _quadratic_minimum32(
        (dt / 2, dt, dt * 1.5),
        (11566.981437, 11555.797034, 11548.594136),
    )
    assert predicted == float(np.float32(2204.0380859375))


def test_bbox_scale_uses_bbox_center():
    points = np.array([[1, 2, 3], [5, 6, 7]], np.float32)
    np.testing.assert_array_equal(
        scale_about_bbox(points, np.float32(0.5)),
        np.array([[2, 3, 4], [4, 5, 6]], np.float32),
    )


def test_projection_centers_and_sets_radius():
    points = np.array([[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0]], np.float32)
    result = project_radially(points)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 100, atol=1e-5)


def test_initial_bbox_dimension_truncates_like_pinned_cpp_abs():
    points = np.array([[0, 0, 0], [0, 215.832, 0]], np.float32)
    scaled = initial_scale(points)
    np.testing.assert_allclose(
        np.ptp(scaled[:, 1]), np.float32(215.832) * np.float32(75 / 215),
        rtol=1e-6,
    )


def test_zero_inflation_updates_equal_direct_projection():
    points = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    np.testing.assert_array_equal(
        inflate_before_quick_sphere(points, faces, iterations=0),
        project_before_quick_sphere(points),
    )
