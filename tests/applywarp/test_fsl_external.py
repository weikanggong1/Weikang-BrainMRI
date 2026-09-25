"""Optional external parity check against an installed FSL applywarp."""

import shutil
import subprocess

import nibabel as nib
import numpy as np
import pytest

from freesurfer_torch.applywarp import TorchApplyWarp


@pytest.mark.skipif(shutil.which("applywarp") is None, reason="FSL applywarp unavailable")
@pytest.mark.parametrize("interpolation", ["trilinear", "nn"])
def test_dense_relative_field_matches_fsl_applywarp(tmp_path, interpolation):
    shape = (13, 12, 11)
    affine = np.diag([-2.0, 2.5, 3.0, 1.0])
    x, y, z = np.indices(shape, dtype=np.float32)
    moving_data = x + 10 * y + 100 * z + 0.01 * x * y
    field_data = np.zeros((*shape, 3), dtype=np.float32)
    field_data[..., 0] = 0.35 + 0.08 * np.sin(y / 3)
    field_data[..., 1] = -0.22 + 0.05 * np.cos(z / 2)
    field_data[..., 2] = 0.18
    moving = tmp_path / "moving.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    field = tmp_path / "field.nii.gz"
    fsl_output = tmp_path / f"fsl-{interpolation}.nii.gz"
    torch_output = tmp_path / f"torch-{interpolation}.nii.gz"
    nib.save(nib.Nifti1Image(moving_data, affine), moving)
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), reference)
    nib.save(nib.Nifti1Image(field_data, affine), field)

    subprocess.run(
        [
            "applywarp", f"--in={moving}", f"--ref={reference}",
            f"--warp={field}", "--rel", f"--interp={interpolation}",
            "--datatype=float", f"--out={fsl_output}",
        ],
        check=True,
    )
    TorchApplyWarp("cpu").run(
        moving,
        reference,
        torch_output,
        warp=field,
        warp_convention="relative",
        interpolation=interpolation,
        output_dtype="float",
    )

    fsl = np.asarray(nib.load(fsl_output).dataobj, dtype=np.float32)
    candidate = np.asarray(nib.load(torch_output).dataobj, dtype=np.float32)
    tolerance = 2e-4 if interpolation == "trilinear" else 0
    np.testing.assert_allclose(candidate, fsl, atol=tolerance, rtol=0)
