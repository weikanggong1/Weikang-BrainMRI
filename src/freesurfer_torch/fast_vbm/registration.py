"""Affine plus selectable nonlinear registration for FastVBM."""

from dataclasses import dataclass

import numpy as np
import surfa as sf
import torch

from .legacy_registration import constrain_deformation, pull_jacobian, register
from .linear import (
    WORLD_FORWARD_CONVENTION,
    WORLD_PULL_CONVENTION,
    register_affine,
    resample_to_fixed,
)
from .fnirt_backend import PyTorchFNIRTRegistration
from .synthmorph_backend import SynthMorphDeformRegistration


@dataclass
class VBMRegistrationResult:
    """Gray-matter maps on the template grid and registration diagnostics."""

    warped_gm: sf.Volume
    jacobian: sf.Volume
    modulated_gm: sf.Volume
    pull_world_affine: np.ndarray
    fit_score: float
    maximum_displacement_mm: float
    qc: dict


def _array(volume, name):
    if not isinstance(volume, sf.Volume):
        raise TypeError(f"{name} must be a surfa.Volume")
    data = np.asarray(volume.data)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.any(data > 0):
        raise ValueError(f"{name} GM image is empty")
    return np.asarray(data, dtype=np.float32)


def _affine(volume, name):
    affine = np.asarray(volume.geom.vox2world.matrix, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name} affine must be a finite 4x4 matrix")
    if abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError(f"{name} affine must be invertible")
    return affine


def _initial_pull_world(initial_pull, moving, fixed, convention):
    """Validate an explicitly typed fixed-to-moving world-RAS affine."""
    if isinstance(initial_pull, sf.Affine):
        if convention not in (None, WORLD_PULL_CONVENTION):
            raise ValueError(
                f"initial_pull_convention must be {WORLD_PULL_CONVENTION!r}"
            )
        if not sf.transform.image_geometry_equal(
            fixed.geom, initial_pull.source, tol=1e-3
        ):
            raise ValueError("initial_pull source geometry must match fixed")
        if not sf.transform.image_geometry_equal(
            moving.geom, initial_pull.target, tol=1e-3
        ):
            raise ValueError("initial_pull target geometry must match moving")
        try:
            matrix = np.asarray(
                initial_pull.convert(space="world").matrix, dtype=np.float64
            )
        except RuntimeError as error:
            raise ValueError("initial_pull must define its coordinate space") from error
    else:
        if convention != WORLD_PULL_CONVENTION:
            raise ValueError(
                "a bare initial_pull matrix is coordinate-ambiguous; set "
                f"initial_pull_convention={WORLD_PULL_CONVENTION!r} only for a "
                "fixed-to-moving world-RAS matrix. An FSL FLIRT .mat is a "
                "moving-to-fixed scaled-mm matrix and must be converted first."
            )
        if isinstance(initial_pull, (str, bytes)):
            raise TypeError(
                "initial_pull does not accept matrix paths; load and convert an "
                "FSL FLIRT .mat with flirt_to_world_pull"
            )
        matrix = np.asarray(initial_pull, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("initial_pull must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("initial_pull must be a homogeneous affine matrix")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-8:
        raise ValueError("initial_pull must be invertible")
    return matrix


def _normalized_correlation(first, second):
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return float("nan")
    first = first[valid] - first[valid].mean()
    second = second[valid] - second[valid].mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first.dot(second) / denominator) if denominator > 0 else 0.0


def _displacement_qc(pull, fixed, pull_affine):
    displacement = np.asarray(pull.data)
    shape = tuple(fixed.shape[:3])
    if displacement.shape != (*shape, 3):
        raise ValueError("pull displacement must match the fixed grid")
    fixed_affine = np.asarray(fixed.geom.vox2world.matrix, dtype=np.float64)
    pull_affine = np.asarray(pull_affine, dtype=np.float64)
    pull_linear_delta = pull_affine[:3, :3] - np.eye(3)
    voxel_coefficients = pull_linear_delta @ fixed_affine[:3, :3]
    offset = (
        pull_linear_delta @ fixed_affine[:3, 3] + pull_affine[:3, 3]
    )
    y = np.arange(shape[1], dtype=np.float64)[None, :, None, None]
    z = np.arange(shape[2], dtype=np.float64)[None, None, :, None]
    maximum_total_squared = 0.0
    maximum_nonlinear_squared = 0.0
    for start in range(0, shape[0], 8):
        stop = min(start + 8, shape[0])
        x = np.arange(start, stop, dtype=np.float64)[:, None, None, None]
        affine_displacement = (
            offset
            + x * voxel_coefficients[:, 0]
            + y * voxel_coefficients[:, 1]
            + z * voxel_coefficients[:, 2]
        )
        chunk = np.asarray(displacement[start:stop], dtype=np.float64)
        nonlinear = chunk - affine_displacement
        maximum_total_squared = max(
            maximum_total_squared,
            float(np.einsum("...i,...i->...", chunk, chunk).max()),
        )
        maximum_nonlinear_squared = max(
            maximum_nonlinear_squared,
            float(np.einsum("...i,...i->...", nonlinear, nonlinear).max()),
        )
    return {
        "maximum_total_pull_displacement_mm": maximum_total_squared**0.5,
        "maximum_nonlinear_displacement_mm": maximum_nonlinear_squared**0.5,
    }


def _implementation_name(value):
    if isinstance(value, SynthMorphDeformRegistration):
        return "freesurfer_torch.synthmorph.SynthMorph"
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def register_gm(
    moving,
    fixed,
    *,
    device="cpu",
    initial_pull=None,
    initial_pull_convention=None,
    linear_strides=(4, 2, 1),
    linear_steps=(80, 60, 50),
    linear_learning_rates=(0.05, 0.025, 0.0125),
    synthmorph_weights=None,
    synthmorph_extent=256,
    synthmorph_hyper=0.5,
    synthmorph_steps=7,
    registration_backend="synthmorph",
    fnirt_strides=(4, 2, 1, 1),
    fnirt_steps=(20, 20, 30, 20),
    fnirt_learning_rates=(0.5, 0.25, 0.1, 0.05),
    fnirt_input_fwhm_mm=(6.0, 4.0, 2.0, 2.0),
    fnirt_reference_fwhm_mm=(4.0, 2.0, 0.0, 0.0),
    fnirt_warp_resolution_mm=10.0,
    fnirt_regularization=(150.0, 75.0, 50.0, 30.0),
    fnirt_jacobian_penalty=1.0,
    deform_model=None,
):
    """Register one GM PVE image to a GM template and compute modulation.

    The linear stage is an independent PyTorch FLIRT-compatible 12-DOF fit.
    The nonlinear stage is selected with ``registration_backend``. The default
    uses the official SynthMorph deform checkpoint with ``mid_space=False``.
    ``"fnirt"`` uses the package's cubic B-spline SSD optimizer on the selected
    PyTorch device. It follows FNIRT's transform role but is not numerically
    equivalent to FSL FNIRT.

    ``initial_pull``, when provided, is a fixed-to-moving world-RAS affine. A
    bare matrix requires ``initial_pull_convention='fixed-to-moving-world-ras'``
    because an FSL FLIRT ``.mat`` has incompatible direction and scaled-mm
    coordinates. A geometry-tagged :class:`surfa.Affine` needs no convention.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if registration_backend not in ("synthmorph", "fnirt"):
        raise ValueError("registration_backend must be 'synthmorph' or 'fnirt'")
    moving_data = _array(moving, "moving")
    fixed_data = _array(fixed, "fixed")
    moving_affine = _affine(moving, "moving")
    fixed_affine = _affine(fixed, "fixed")

    if initial_pull is None:
        if initial_pull_convention is not None:
            raise ValueError(
                "initial_pull_convention requires an initial_pull transform"
            )
        linear = register_affine(
            moving_data,
            fixed_data,
            moving_affine,
            fixed_affine,
            device=device,
            strides=linear_strides,
            steps=linear_steps,
            learning_rates=linear_learning_rates,
        )
        moving_to_fixed = linear.moving_to_fixed_world
        pull_affine = linear.fixed_to_moving_world
        linear_qc = linear.qc
    else:
        pull_affine = _initial_pull_world(
            initial_pull, moving, fixed, initial_pull_convention
        )
        moving_to_fixed = np.linalg.inv(pull_affine)
        moving_to_fixed[3] = (0, 0, 0, 1)
        linear_warped, valid = resample_to_fixed(
            moving_data,
            moving_affine,
            fixed_affine,
            fixed_data.shape,
            moving_to_fixed,
            device=device,
        )
        valid_array = valid.detach().cpu().numpy()
        linear_qc = {
            "backend": "supplied-fixed-to-moving-world-affine",
            "equivalent_to_fsl_flirt": None,
            "forward_transform_convention": WORLD_FORWARD_CONVENTION,
            "pull_transform_convention": WORLD_PULL_CONVENTION,
            "accepts_fsl_flirt_matrix_directly": False,
            "degrees_of_freedom": None,
            "correlation": _normalized_correlation(
                linear_warped.detach().cpu().numpy()[valid_array],
                fixed_data[valid_array],
            ),
            "overlap_fraction": float(valid.float().mean()),
        }

    initial = sf.Affine(
        moving_to_fixed,
        source=moving,
        target=fixed,
        space="world",
    )
    if deform_model is None:
        if registration_backend == "synthmorph":
            deform_model = SynthMorphDeformRegistration(
                weights=synthmorph_weights,
                device=device,
                extent=synthmorph_extent,
                hyper=synthmorph_hyper,
                steps=synthmorph_steps,
            )
        else:
            deform_model = PyTorchFNIRTRegistration(
                device=device,
                strides=fnirt_strides,
                steps=fnirt_steps,
                learning_rates=fnirt_learning_rates,
                input_fwhm_mm=fnirt_input_fwhm_mm,
                reference_fwhm_mm=fnirt_reference_fwhm_mm,
                warp_resolution_mm=fnirt_warp_resolution_mm,
                regularization=fnirt_regularization,
                jacobian_penalty=fnirt_jacobian_penalty,
            )
    package_synthmorph = isinstance(deform_model, SynthMorphDeformRegistration)
    package_fnirt = isinstance(deform_model, PyTorchFNIRTRegistration)
    if package_synthmorph and registration_backend != "synthmorph":
        raise ValueError(
            "a SynthMorph deform model requires registration_backend='synthmorph'"
        )
    if package_fnirt and registration_backend != "fnirt":
        raise ValueError(
            "a PyTorch FNIRT model requires registration_backend='fnirt'"
        )
    nonlinear = deform_model(moving, fixed, initial)
    displacement_qc = _displacement_qc(
        nonlinear.pull_transform, fixed, pull_affine
    )
    jacobian = np.asarray(nonlinear.nonlinear_jacobian.data)
    qc = {
        "linear": linear_qc,
        "nonlinear_backend": (
            "pytorch-synthmorph-deform"
            if package_synthmorph
            else (
                "pytorch-fnirt-style-cubic-bspline"
                if package_fnirt
                else "custom-deform-model"
            )
        ),
        "synthmorph_implementation": (
            _implementation_name(deform_model) if package_synthmorph else None
        ),
        "custom_deform_implementation": (
            None
            if package_synthmorph or package_fnirt
            else _implementation_name(deform_model)
        ),
        "synthmorph_warp_convention": (
            "fixed-grid target-to-source disp-ras: "
            "source_world(target)-target_world"
            if package_synthmorph
            else None
        ),
        "synthmorph_model": "deform" if package_synthmorph else None,
        "synthmorph_extent": synthmorph_extent if package_synthmorph else None,
        "synthmorph_hyper": synthmorph_hyper if package_synthmorph else None,
        "synthmorph_integration_steps": (
            synthmorph_steps if package_synthmorph else None
        ),
        "synthmorph_mid_space": False if package_synthmorph else None,
        "fnirt_implementation": (
            _implementation_name(deform_model) if package_fnirt else None
        ),
        "fnirt_style": package_fnirt,
        "fsl_fnirt_numerically_equivalent": False,
        "affine_jacobian_determinant": float(
            np.linalg.det(pull_affine[:3, :3])
        ),
        "jacobian_min": float(jacobian.min()),
        "jacobian_max": float(jacobian.max()),
        "nonpositive_jacobian_voxels": int((jacobian <= 0).sum()),
        **nonlinear.qc,
        **displacement_qc,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return VBMRegistrationResult(
        warped_gm=nonlinear.moved,
        jacobian=nonlinear.nonlinear_jacobian,
        modulated_gm=nonlinear.modulated_gm,
        pull_world_affine=pull_affine.copy(),
        fit_score=_normalized_correlation(nonlinear.moved.data, fixed_data),
        maximum_displacement_mm=displacement_qc[
            "maximum_nonlinear_displacement_mm"
        ],
        qc=qc,
    )


__all__ = [
    "VBMRegistrationResult",
    "constrain_deformation",
    "pull_jacobian",
    "register",
    "register_gm",
]
