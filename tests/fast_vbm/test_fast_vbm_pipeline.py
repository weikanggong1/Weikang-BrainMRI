"""End-to-end API, validation, geometry, and completion-marker checks."""

import json

import numpy as np
import pytest
import surfa as sf

from freesurfer_torch.fast import FASTConfig, FASTResult
from freesurfer_torch.fast_vbm import FastVBM, FastVBMResult, OUTPUT_FILENAMES
from freesurfer_torch.fast_vbm import pipeline as pipeline_module


def _volume(shape=(8, 9, 10), affine=None):
    if affine is None:
        affine = np.array(
            [[0, -1.2, 0, 20], [1.0, 0, 0, -10],
             [0, 0, 1.5, 5], [0, 0, 0, 1]],
            dtype=float,
        )
    axes = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in shape], indexing="ij"
    )
    center = (np.asarray(shape, dtype=np.float32) - 1) / 2
    data = np.exp(
        -sum((axis - center[index]) ** 2 for index, axis in enumerate(axes)) / 8
    ).astype(np.float32)
    geometry = sf.ImageGeometry(shape, vox2world=affine)
    return sf.Volume(data, geometry=geometry)


class _FakeFAST:
    def __init__(self, **_):
        self.config = FASTConfig()

    def __call__(self, image, mask=None):
        data = np.asarray(image.data, dtype=np.float32)
        gm = data / max(float(data.max()), 1e-6)
        zeros = image.new(np.zeros_like(gm, dtype=np.float32))
        labels = image.new(np.ones_like(gm, dtype=np.int32))
        ones = image.new(np.ones_like(gm, dtype=np.float32))
        return FASTResult(
            pve_csf=zeros,
            pve_gm=image.new(gm),
            pve_wm=zeros,
            hard_segmentation=labels,
            pve_segmentation=labels,
            mixel_type=image.new(np.zeros_like(gm, dtype=np.int32)),
            bias_field=ones,
            restored=image.new(data.copy()),
            tissue_means=(1.0, 2.0, 3.0),
            tissue_variances=(0.1, 0.2, 0.3),
        )


def _pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_module, "TorchFAST", _FakeFAST)
    return FastVBM(
        device="cpu", affine_steps=0, deform_steps=0,
        smoothness=10.0, scales=(1,),
    )


def test_explicit_mask_pipeline_returns_input_and_template_grid_outputs(
        tmp_path, monkeypatch):
    image = _volume()
    mask = image.new(np.ones(image.shape[:3], dtype=np.uint8))
    template = image.new(np.asarray(image.data, dtype=np.float32).copy())
    pipeline = _pipeline(monkeypatch)

    result = pipeline(image, template, brain_mask=mask)

    assert isinstance(result, FastVBMResult)
    assert result.settings["mask_source"] == "explicit"
    assert result.settings["bias_correction"] is True
    assert result.settings["smoothness"] == 10.0
    assert result.settings["scales"] == [1]
    assert result.settings["fast"]["bias_fwhm_mm"] == 20.0
    for name in (
        "brain", "brain_mask", "pve_csf", "pve_gm", "pve_wm",
        "hard_segmentation", "pve_segmentation", "mixel_type",
        "bias_field", "restored",
    ):
        np.testing.assert_allclose(
            result.volumes()[name].geom.vox2world.matrix,
            image.geom.vox2world.matrix,
        )
    for name in ("warped_gm", "jacobian", "modulated_gm"):
        np.testing.assert_allclose(
            result.volumes()[name].geom.vox2world.matrix,
            template.geom.vox2world.matrix,
        )
    np.testing.assert_allclose(result.jacobian.data, 1, atol=1e-6)

    paths = result.save(tmp_path / "result")
    assert set(paths) == set(OUTPUT_FILENAMES)
    assert all((tmp_path / "result" / name).is_file()
               for name in OUTPUT_FILENAMES.values())
    report = json.loads((tmp_path / "result" / "fast_vbm_report.json").read_text())
    assert report["status"] == "experimental"
    assert report["fnirt_equivalent"] is False
    assert "global normalized correlation + 0.2 MSE" in report["method"]
    assert report["registration"]["jacobian_convention"].startswith(
        "nonlinear-only")
    assert report["fast"]["bias_range_inside_mask"] == [1.0, 1.0]


@pytest.mark.parametrize("kind", ["empty", "nonfinite", "grid"])
def test_explicit_mask_validation(kind, monkeypatch):
    image = _volume()
    template = image.copy()
    if kind == "empty":
        mask = image.new(np.zeros(image.shape[:3], dtype=np.uint8))
        message = "empty"
    elif kind == "nonfinite":
        mask = image.new(np.ones(image.shape[:3], dtype=np.float32))
        mask.data[0, 0, 0] = np.nan
        message = "NaN or infinity"
    else:
        affine = np.asarray(image.geom.vox2world.matrix).copy()
        affine[0, 3] += 1
        mask = _volume(shape=image.shape[:3], affine=affine)
        message = "same shape and geometry"

    with pytest.raises(ValueError, match=message):
        _pipeline(monkeypatch)(image, template, brain_mask=mask)


def test_template_must_contain_positive_gm(monkeypatch):
    image = _volume()
    mask = image.new(np.ones(image.shape[:3], dtype=np.uint8))
    template = image.new(np.zeros(image.shape[:3], dtype=np.float32))
    with pytest.raises(ValueError, match="positive GM"):
        _pipeline(monkeypatch)(image, template, brain_mask=mask)


def test_template_affine_must_be_invertible(monkeypatch):
    image = _volume()
    mask = image.new(np.ones(image.shape[:3], dtype=np.uint8))
    affine = np.eye(4)
    affine[2, 2] = 0
    with pytest.warns(RuntimeWarning):
        template = _volume(shape=image.shape[:3], affine=affine)
    with pytest.raises(ValueError, match="invertible"):
        _pipeline(monkeypatch)(image, template, brain_mask=mask)


def test_failed_overwrite_removes_completion_marker(tmp_path, monkeypatch):
    image = _volume()
    mask = image.new(np.ones(image.shape[:3], dtype=np.uint8))
    result = _pipeline(monkeypatch)(image, image.copy(), brain_mask=mask)
    output = tmp_path / "result"
    result.save(output)
    marker = output / "fast_vbm_report.json"
    assert marker.is_file()

    real_save = sf.Volume.save
    calls = 0

    def fail_during_stage(self, path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected write failure")
        return real_save(self, path, *args, **kwargs)

    monkeypatch.setattr(sf.Volume, "save", fail_during_stage)
    with pytest.raises(RuntimeError, match="injected"):
        result.save(output, overwrite=True)
    assert not marker.exists()


def test_saving_without_report_removes_an_old_completion_marker(tmp_path, monkeypatch):
    image = _volume()
    mask = image.new(np.ones(image.shape[:3], dtype=np.uint8))
    result = _pipeline(monkeypatch)(image, image.copy(), brain_mask=mask)
    output = tmp_path / "result"
    result.save(output)
    marker = output / "fast_vbm_report.json"
    assert marker.is_file()

    result.save(output, overwrite=True, report=False)
    assert not marker.exists()
