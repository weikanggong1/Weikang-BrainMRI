import torch

from freesurfer_torch.fnirt.optimizer import (
    preconditioned_conjugate_gradient,
)


def test_pcg_matches_imlpp_iteration_and_step_oracle():
    matrix = torch.tensor(
        [
            [5.0, 1.0, 0.2, 0.0, 0.0, 0.0],
            [1.0, 4.0, 0.5, 0.1, 0.0, 0.0],
            [0.2, 0.5, 3.0, 0.4, 0.2, 0.0],
            [0.0, 0.1, 0.4, 2.5, 0.6, 0.1],
            [0.0, 0.0, 0.2, 0.6, 2.0, 0.7],
            [0.0, 0.0, 0.0, 0.1, 0.7, 1.8],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [1.0, -2.0, 0.5, 3.0, -1.0, 2.0], dtype=torch.float64
    )
    expected = torch.tensor(
        [
            0.3212305059173806,
            -0.6372311124824229,
            0.15372274822013607,
            1.504279645884813,
            -1.5353160436077122,
            1.6245466424440114,
        ],
        dtype=torch.float64,
    )

    actual, report = preconditioned_conjugate_gradient(
        lambda vector: matrix @ vector,
        rhs,
        diagonal=torch.diagonal(matrix),
        tolerance=1e-3,
        max_iterations=500,
    )

    assert report.iterations == 5
    assert report.converged
    assert abs(report.relative_residual - 0.0003584912872494894) < 1e-15
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
