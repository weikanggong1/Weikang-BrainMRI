"""Image geometry and public result checks for TorchFAST."""

import numpy as np
import pytest
import surfa as sf

from freesurfer_torch.fast import FASTResult, TorchFAST


def _volume(shape=(12, 11, 10), affine=None):
    if affine is None:
        affine = np.array([
            [0, -1.2, 0, 20], [1.0, 0, 0, -10], [0, 0, 1.5, 5], [0, 0, 0, 1],
        ], dtype=float)
    x = np.linspace(-1, 1, shape[0])[:, None, None]
    mask = np.broadcast_to(x * x < 0.9, shape)
    base = np.where(x < -0.2, 30, np.where(x < 0.3, 70, 110))
    data = np.broadcast_to(base * np.exp(0.2 * x), shape).astype(np.float32).copy()
    data[~mask] = 0
    geometry = sf.ImageGeometry(shape, vox2world=affine)
    return sf.Volume(data, geometry=geometry), sf.Volume(mask.astype(np.uint8), geometry=geometry)


def _model():
    return TorchFAST(
        init_iterations=3, bias_iterations=1, fixed_iterations=1,
        bias_fwhm_mm=6, pve_steps=10, mean_field_iterations=2,
        pve_chunk_size=4,
    )


def test_pipeline_preserves_geometry_and_returns_named_outputs(tmp_path):
    image, mask = _volume()
    result = _model()(image, mask)
    assert isinstance(result, FASTResult)
    fields = (
        result.pve_csf, result.pve_gm, result.pve_wm,
        result.hard_segmentation, result.pve_segmentation, result.mixel_type,
        result.bias_field, result.restored,
    )
    for output in fields:
        assert output.shape[:3] == image.shape[:3]
        np.testing.assert_allclose(output.geom.vox2world.matrix,
                                   image.geom.vox2world.matrix, atol=1e-6)
    assert result.pve_gm.data.dtype == np.float32
    assert result.hard_segmentation.data.dtype == np.int32

    path = tmp_path / "gm.nii.gz"
    result.pve_gm.save(path)
    loaded = sf.load_volume(path)
    np.testing.assert_allclose(loaded.geom.vox2world.matrix,
                               image.geom.vox2world.matrix, atol=1e-5)
    np.testing.assert_allclose(loaded.data, result.pve_gm.data, atol=1e-6)


def test_pipeline_rejects_mask_on_another_grid():
    image, _ = _volume()
    changed_affine = np.asarray(image.geom.vox2world.matrix).copy()
    changed_affine[0, 3] += 1
    _, mask = _volume(affine=changed_affine)
    with pytest.raises(ValueError, match="same shape and geometry"):
        _model()(image, mask)


def test_pipeline_rejects_bad_threads():
    with pytest.raises(ValueError, match="positive"):
        TorchFAST(threads=0)
