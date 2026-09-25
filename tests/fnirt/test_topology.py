import torch

from freesurfer_torch.fnirt.spline import (
    expand_coefficients,
    fit_field_coefficients,
    spline_bases,
)
from freesurfer_torch.fnirt.topology import (
    constrain_topology,
    jacobian_check,
)


def _identity(shape, voxel_sizes, dtype=torch.float32):
    axes = tuple(
        torch.arange(size, dtype=dtype) * voxel
        for size, voxel in zip(shape, voxel_sizes)
    )
    return torch.stack(torch.meshgrid(*axes, indexing="ij"))


def test_corner_jacobians_are_one_for_identity_warp():
    warp = _identity((7, 8, 6), (2.0, 2.5, 3.0))
    jacobians = jacobian_check(warp, (2.0, 2.5, 3.0))
    torch.testing.assert_close(jacobians, torch.ones_like(jacobians))


def test_topology_projection_leaves_valid_affine_warp_unchanged():
    grid = _identity((8, 7, 6), (2.0, 2.0, 2.0))
    linear = torch.tensor(
        [[1.1, 0.03, 0.0], [0.0, 0.95, 0.02], [0.0, 0.0, 1.05]],
        dtype=grid.dtype,
    )
    warp = torch.einsum("ij,jxyz->ixyz", linear, grid)
    projected, qc = constrain_topology(warp, (2.0, 2.0, 2.0), 0.2, 5.0)
    torch.testing.assert_close(projected, warp, atol=0, rtol=0)
    assert qc["iterations"] == 1
    assert qc["final"]["below"] == qc["final"]["above"] == 0


def test_spline_set_refits_a_dense_spline_field():
    shape = (15, 14, 13)
    spacing = (3, 3, 3)
    voxel_sizes = (2.0, 2.0, 2.0)
    bases = spline_bases(
        shape,
        spacing,
        voxel_sizes,
        device="cpu",
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(9)
    coefficients = torch.randn(
        (3, *(basis.shape[1] for basis in bases)),
        generator=generator,
        dtype=torch.float64,
    )
    field = expand_coefficients(coefficients, bases)
    fitted = fit_field_coefficients(field, spacing, voxel_sizes)
    reproduced = expand_coefficients(fitted, bases)
    # splinefield::Set deliberately adds its 0.005 end-coefficient penalty,
    # so even a representable field is refitted rather than copied exactly.
    assert float((reproduced - field).abs().max()) < 0.01
