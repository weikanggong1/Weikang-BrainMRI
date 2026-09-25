"""The three neural launchers read from the selected external model directory."""

from freesurfer_torch.recon_all.gpu_tools import _models


def test_neural_model_directory_overrides_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    models = tmp_path / "weights"
    models.mkdir()
    monkeypatch.setenv("FREESURFER_HOME", str(bundle))
    monkeypatch.setenv("FS_TORCH_MODEL_DIR", str(models))
    assert _models() == models
