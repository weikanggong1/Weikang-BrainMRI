"""Component oracles for the FSL FLIRT 2111.2 PyTorch target."""

import numpy as np
import pytest
import torch

import freesurfer_torch
from freesurfer_torch.flirt import TorchFLIRT
from freesurfer_torch.flirt.core import (
    _DefaultFLIRTEngine,
    _centre_of_gravity,
    _newimage_percentile,
    _resample_output,
    fsl_affine_from_parameters,
    fsl_coordinate_optimize,
    fsl_parameters_from_affine,
)


def _synthetic_pair():
    shape = (16, 17, 18)
    coordinates = np.indices(shape, dtype=np.float32)
    centre = (np.asarray(shape, dtype=np.float32) - 1) / 2
    reference = np.exp(
        -sum(
            (coordinates[axis] - centre[axis]) ** 2
            for axis in range(3)
        ) / 9
    ).astype(np.float32)
    moving = np.roll(reference, 1, axis=0)
    vox2world = np.array(
        [[-2, 0, 0, 30], [0, 2, 0, -10], [0, 0, 2, 5], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return moving, reference, vox2world


def test_torch_flirt_is_the_exact_target_public_api():
    assert freesurfer_torch.TorchFLIRT is TorchFLIRT
    assert TorchFLIRT(device="cpu").angular_search is True


def test_newimage_percentile_uses_order_statistic_without_interpolation():
    values = np.arange(64, dtype=np.float32)

    assert _newimage_percentile(values, .5) == 32
    assert np.percentile(values, 50) == 31.5


def test_fsl_parameter_composition_matches_flirt_2111_2_fixture():
    parameters = torch.tensor(
        [.05, -.04, .03, 1.2, -2.3, .7, 1.04, 1.04, 1.04, 0, 0, 0],
        dtype=torch.float64,
    )
    centre = torch.tensor([16.98656918, 16, 17], dtype=torch.float64)
    expected = np.array(
        [
            [1.038700492, .03117036336, .04158890696, -.6631258241],
            [-.03323397592, 1.038170644, .05193676256, -3.229124043],
            [-.03995912465, -.0532008727, 1.037869511, 1.586200716],
            [0, 0, 0, 1],
        ]
    )

    matrix = fsl_affine_from_parameters(parameters, centre, dof=7).numpy()

    np.testing.assert_allclose(matrix, expected, atol=7e-7, rtol=0)
    recovered = fsl_parameters_from_affine(matrix, centre.numpy())
    np.testing.assert_allclose(recovered[:7], parameters.numpy()[:7], atol=1e-7)


def test_correlation_ratio_matches_fsl_default_schedule_cost_oracles():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    expected = {
        8: 0.446353,
        4: 0.111078,
        2: 0.118574,
        1: 0.118574,
    }

    measured = {}
    for scale in expected:
        engine.set_scale(scale)
        measured[scale] = engine.cost(np.eye(4))

    np.testing.assert_allclose(
        [measured[scale] for scale in expected],
        list(expected.values()),
        atol=1e-6,
        rtol=0,
    )


def test_nonidentity_correlation_ratio_matches_fsl_fixture():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    matrix = np.array(
        [
            [1.03870052035, .0581360013514, .0208731747207, -.742409058073],
            [-.0332339779434, .967363767006, .06686295822, -2.34995929637],
            [-.0399591258423, -.0507381279885, 1.01791154105, 1.88608230552],
            [0, 0, 0, 1],
        ]
    )
    engine.set_scale(2)

    assert abs(engine.cost(matrix) - 0.356834) <= 1e-6


def test_correlation_ratio_bin_order_is_stable_within_each_bin():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    cost = engine.level.cost
    order = cost.bin_sort_order.cpu().numpy()
    bins = cost.bin_index.cpu().numpy()

    assert np.all(bins[order][1:] >= bins[order][:-1])
    for bin_index in np.unique(bins):
        positions = order[bins[order] == bin_index]
        assert np.all(positions[1:] > positions[:-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_correlation_ratio_reduction_is_bitwise_repeatable():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cuda",
        angular_search=False,
    )
    engine.set_scale(2)
    matrix = np.eye(4)

    measured = [engine.level.cost(matrix) for _ in range(3)]

    assert measured[0] == measured[1] == measured[2]


def test_angular_search_translation_accounts_for_initial_matrix(monkeypatch):
    moving, reference, vox2world = _synthetic_pair()
    initial = np.eye(4)
    initial[:3, 3] = (4.0, -3.0, 2.0)
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (2, 2, 2),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
        initial_matrix=initial,
    )
    moving_centre = _centre_of_gravity(
        engine.level.moving, np.diag([*engine.moving_sizes, 1.0])
    )
    reference_centre = _centre_of_gravity(
        engine.level.reference,
        np.diag([*engine.level.reference_sizes, 1.0]),
    )
    expected = reference_centre - (initial @ np.r_[moving_centre, 1])[:3]

    def capture(parameters, maximum_iterations=4):
        np.testing.assert_allclose(parameters[3:6], expected, atol=1e-12)
        raise RuntimeError("captured initial search parameters")

    monkeypatch.setattr(engine, "_optimize_search_subset", capture)
    with pytest.raises(RuntimeError, match="captured initial search parameters"):
        engine.angular_candidates()


def test_unavailable_finer_scale_refreshes_unblurred_moving_volume():
    moving, reference, vox2world = _synthetic_pair()
    engine = _DefaultFLIRTEngine(
        moving,
        reference,
        vox2world,
        vox2world,
        (.8, .8, .8),
        (2, 2, 2),
        device="cpu",
        angular_search=False,
    )
    engine.set_scale(2)
    blurred = engine.level.moving.clone()
    reference_at_two = engine.level.reference
    centre_at_two = engine.level.centre.copy()

    engine.set_scale(1)

    assert engine.level.moving is engine.moving_original
    assert engine.level.cost.moving is engine.moving_original
    assert engine.level.reference is reference_at_two
    assert engine.level.cost.bins == 128
    assert engine.level.cost.smooth_size == 1
    assert not torch.equal(blurred, engine.level.moving)
    np.testing.assert_array_equal(engine.level.centre, centre_at_two)


def test_coordinate_optimizer_limits_the_requested_parameter_prefix():
    initial = np.array([4.0, -3.0, 19.0])
    target = np.array([1.25, -.75, -11.0])

    fitted, value = fsl_coordinate_optimize(
        initial,
        np.array([.01, .01, .01]),
        lambda point: float(np.square(point - target).sum()),
        maximum_iterations=4,
        bound_guess=(10.0, 1.0),
        numopt=2,
    )

    np.testing.assert_allclose(fitted[:2], target[:2], atol=.02, rtol=0)
    assert fitted[2] == initial[2]
    assert value < 1e-3 + (initial[2] - target[2]) ** 2


def test_output_resampling_includes_flirt_default_antialias_blur():
    moving = np.zeros((9, 9, 9), dtype=np.float32)
    moving[4, 4, 4] = 1
    moving_fsl = np.diag([1, 1, 1, 1])
    fixed_fsl = np.diag([2, 2, 2, 1])

    output = _resample_output(
        moving,
        (5, 5, 5),
        moving_fsl,
        fixed_fsl,
        np.eye(4),
        (1, 1, 1),
        (2, 2, 2),
        torch.device("cpu"),
    )

    sigma = .85
    side = np.exp(-1 / (sigma * sigma * 4))
    centre_weight = (1 / (1 + 2 * side)) ** 3
    assert float(output[2, 2, 2]) < 1
    assert abs(float(output[2, 2, 2]) - centre_weight) < 1e-6
    assert torch.count_nonzero(output) == 1
