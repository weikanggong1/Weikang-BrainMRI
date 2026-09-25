import gzip

import nibabel as nib
import numpy as np
import pytest

from fnit.recon_all.nifti_import import import_t1


def test_float_t1_import_preserves_voxels_and_sform(tmp_path):
    voxels = (np.arange(4 * 5 * 6, dtype=np.float32) / 7).reshape(4, 5, 6)
    affine = np.array([[0.95, -0.2, 0, -20],
                       [0.1, 1.25, 0, 30],
                       [0, 0, 1.1, -10],
                       [0, 0, 0, 1]], dtype=np.float32)
    source = nib.Nifti1Image(voxels, affine)
    source.set_sform(affine, code=1)
    source.header.set_xyzt_units("mm", "sec")
    source.header["pixdim"][4] = 0.009668
    input_file, output_file = tmp_path / "t1.nii.gz", tmp_path / "001.mgz"
    nib.save(source, str(input_file))

    import_t1(input_file, output_file)
    output = nib.load(str(output_file))
    np.testing.assert_array_equal(np.asarray(output.dataobj), voxels)
    np.testing.assert_allclose(output.affine, affine, atol=1e-5)
    assert output.get_data_dtype() == np.dtype(">f4")
    assert output.header["dof"] == 1
    assert output.header["tr"] == np.float32(9.668)
    assert gzip.decompress(output_file.read_bytes())[284:284 + voxels.size * 4] == \
        np.asarray(voxels, dtype=">f4").tobytes(order="F")


def test_scaled_nifti_is_rejected(tmp_path):
    source = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.float32), np.eye(4))
    source.header.set_xyzt_units("mm", "sec")
    source.header.set_slope_inter(2, 0)
    path = tmp_path / "scaled.nii.gz"
    nib.save(source, str(path))
    with pytest.raises(ValueError, match="scaled NIfTI"):
        import_t1(path, tmp_path / "out.mgz")
