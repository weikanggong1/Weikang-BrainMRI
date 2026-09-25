"""Coordinate, interpolation, header, and CLI tests for TorchApplyWarp."""

import numpy as np
import nibabel as nib
import pytest

import freesurfer_torch
from freesurfer_torch import cli
from freesurfer_torch.applywarp import ApplyWarpResult, TorchApplyWarp


def _image(data, affine):
    image = nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=float))
    image.set_qform(affine, 2)
    image.set_sform(affine, 4)
    return image


def _warp(field, affine, intent=0):
    image = _image(np.asarray(field, dtype=np.float32), affine)
    image.header["intent_code"] = intent
    return image


def _zero_field(shape):
    return np.zeros((*shape, 3), dtype=np.float32)


def _cubic_coefficients(field_shape, knot_spacing, values, embedded_affine=None):
    coefficient_shape = tuple(
        size if spacing == 1 else int(np.ceil((size + 1) / spacing)) + 2
        for size, spacing in zip(field_shape, knot_spacing)
    )
    data = np.zeros((*coefficient_shape, 3), dtype=np.float32)
    data[...] = values
    image = nib.Nifti1Image(data, np.eye(4))
    image.header["intent_code"] = 2007
    image.header["intent_p1"] = 1
    image.header["intent_p2"] = 1
    image.header["intent_p3"] = 1
    image.header["pixdim"][1:4] = knot_spacing
    image.header["qoffset_x"] = field_shape[0]
    image.header["qoffset_y"] = field_shape[1]
    image.header["qoffset_z"] = field_shape[2]
    image.set_sform(
        np.eye(4) if embedded_affine is None else embedded_affine, code=1
    )
    return image


@pytest.mark.parametrize(
    "affine,source_slice,target_slice",
    [
        (np.diag([-2.0, 3.0, 4.0, 1.0]), slice(None, -1), slice(1, None)),
        (np.diag([2.0, 3.0, 4.0, 1.0]), slice(1, None), slice(None, -1)),
    ],
)
def test_relative_x_displacement_obeys_fsl_storage_handedness(
    affine, source_slice, target_slice
):
    shape = (6, 5, 4)
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    field = _zero_field(shape)
    field[..., 0] = 2.0

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=_warp(field, affine, 2006),
    )

    expected = np.zeros(shape, dtype=np.float32)
    expected[source_slice] = ramp[target_slice]
    np.testing.assert_allclose(
        np.asarray(result.image.dataobj), expected, atol=1e-6, rtol=0
    )
    assert result.qc["warp_convention"] == "relative"
    assert result.qc["warp_convention_source"] == "FSL intent 2006"


def test_premat_and_postmat_use_fsl_forward_order():
    shape = (8, 5, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    translate = np.eye(4)
    translate[0, 3] = 1.0

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=_warp(_zero_field(shape), affine, 2006),
        premat=translate,
        postmat=translate,
    )

    expected = np.zeros(shape, dtype=np.float32)
    expected[2:] = ramp[:-2]
    np.testing.assert_allclose(
        np.asarray(result.image.dataobj), expected, atol=1e-6, rtol=0
    )


def test_nearest_uses_fsl_positive_half_up_ties():
    shape = (5, 4, 3)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    coordinates = np.indices(shape, dtype=np.float32)
    field = np.stack(
        (
            np.full(shape, 0.5, dtype=np.float32),
            coordinates[1],
            coordinates[2],
        ),
        axis=-1,
    )

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=_warp(field, affine),
        warp_convention="absolute",
        interpolation="nearest",
    )

    np.testing.assert_array_equal(np.asarray(result.image.dataobj), 1)


@pytest.mark.parametrize(
    "affine,source_slice,target_slice",
    [
        (np.diag([-1.0, 1.0, 1.0, 1.0]), slice(1, None), slice(None, -1)),
        (np.diag([1.0, 1.0, 1.0, 1.0]), slice(None, -1), slice(1, None)),
    ],
)
def test_cubic_coefficients_decode_residual_and_embedded_affine(
    affine, source_slice, target_slice
):
    shape = (8, 7, 6)
    ramp = np.broadcast_to(
        np.arange(shape[0], dtype=np.float32)[:, None, None], shape
    ).copy()
    embedded_affine = np.eye(4)
    embedded_affine[0, 3] = 1.0
    coefficients = _cubic_coefficients(
        shape, (4, 4, 4), (0.0, 0.0, 0.0), embedded_affine
    )

    result = TorchApplyWarp("cpu")(
        _image(ramp, affine),
        _image(np.zeros(shape, dtype=np.float32), affine),
        warp=coefficients,
    )

    expected = np.zeros_like(ramp)
    expected[source_slice] = ramp[target_slice]
    np.testing.assert_allclose(
        np.asarray(result.image.dataobj), expected, atol=1e-6, rtol=0
    )
    assert result.qc["warp_representation"] == "FNIRT cubic spline coefficients"
    assert result.qc["warp_convention_source"] == "FSL coefficient intent 2007"


def test_reference_header_and_input_dtype_define_output_contract():
    input_shape = (7, 6, 5)
    reference_shape = (5, 4, 3)
    input_affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    reference_affine = np.array(
        [[-2, 0, 0, 10], [0, 3, 0, -5], [0, 0, 4, 7], [0, 0, 0, 1]],
        dtype=float,
    )
    moving = _image(np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float64), input_affine)
    reference = _image(np.zeros(reference_shape, dtype=np.float32), reference_affine)
    field = _warp(_zero_field(reference_shape), reference_affine, 2006)

    result = TorchApplyWarp("cpu")(moving, reference, warp=field)

    assert isinstance(result, ApplyWarpResult)
    assert freesurfer_torch.TorchApplyWarp is TorchApplyWarp
    assert result.image.shape == reference_shape
    assert result.image.get_data_dtype() == np.dtype("float64")
    np.testing.assert_allclose(result.image.affine, reference.affine)
    np.testing.assert_allclose(result.image.get_qform(), reference.get_qform())
    np.testing.assert_allclose(result.image.get_sform(), reference.get_sform())
    assert int(result.image.header["qform_code"]) == 2
    assert int(result.image.header["sform_code"]) == 4


def test_small_integer_input_defaults_to_float_but_explicit_dtype_is_honoured():
    shape = (4, 4, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    moving = _image(np.ones(shape, dtype=np.uint8), affine)
    reference = _image(np.zeros(shape, dtype=np.float32), affine)

    default = TorchApplyWarp("cpu")(moving, reference)
    forced = TorchApplyWarp("cpu")(moving, reference, output_dtype="short")

    assert default.image.get_data_dtype() == np.dtype("float32")
    assert forced.image.get_data_dtype() == np.dtype("int16")


def test_explicit_integer_dtype_uses_fsl_truncation():
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    data = np.array([1.9, -1.9], dtype=np.float32)[:, None, None]

    result = TorchApplyWarp("cpu")(
        _image(data, affine), _image(np.zeros_like(data), affine), output_dtype="short"
    )

    np.testing.assert_array_equal(
        np.asarray(result.image.dataobj)[:, 0, 0], np.array([1, -1])
    )


def test_many_frame_integer_default_matches_fsl_range_rule():
    shape = (2, 2, 2, 11)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    data = np.zeros(shape, dtype=np.int16)
    data[..., 1] = 1
    data[..., 10] = 200

    result = TorchApplyWarp("cpu")(
        _image(data, affine), _image(np.zeros(shape[:3]), affine)
    )

    assert result.image.get_data_dtype() == np.dtype("float32")


@pytest.mark.parametrize("intent", [2008, 2009])
def test_fnirt_coefficients_are_explicitly_rejected(intent):
    shape = (4, 4, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    image = _image(np.zeros(shape, dtype=np.float32), affine)
    coefficient = _warp(_zero_field(shape), affine, intent)

    with pytest.raises(NotImplementedError, match="cubic"):
        TorchApplyWarp("cpu")(image, image, warp=coefficient)


@pytest.mark.parametrize("interpolation", ["sinc", "spline", "cubic"])
def test_unimplemented_image_interpolation_is_explicitly_rejected(interpolation):
    shape = (4, 4, 4)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    image = _image(np.zeros(shape, dtype=np.float32), affine)

    with pytest.raises(ValueError, match="trilinear or nearest"):
        TorchApplyWarp("cpu")(image, image, interpolation=interpolation)


def test_cli_writes_reference_grid_output(tmp_path):
    shape = (5, 4, 3)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    moving = tmp_path / "moving.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    warp = tmp_path / "warp.nii.gz"
    output = tmp_path / "warped.nii.gz"
    nib.save(_image(np.ones(shape, dtype=np.float32), affine), moving)
    nib.save(_image(np.zeros(shape, dtype=np.float32), affine), reference)
    nib.save(_warp(_zero_field(shape), affine, 2006), warp)

    cli.main([
        "applywarp", "--in", str(moving), "--ref", str(reference),
        "--warp", str(warp), "--out", str(output), "--device", "cpu",
        "--datatype", "float",
    ])

    saved = nib.load(output)
    assert saved.shape == shape
    assert saved.get_data_dtype() == np.dtype("float32")
    np.testing.assert_allclose(saved.affine, affine)
    np.testing.assert_array_equal(np.asarray(saved.dataobj), 1)
