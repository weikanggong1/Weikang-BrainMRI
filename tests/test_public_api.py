"""Public feature imports resolve to their implementation objects."""
import importlib
import inspect
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module,names", [
    ("synthstrip", ("SynthStrip", "StripResult")),
    ("synthmorph", ("SynthMorph", "RegistrationResult", "apply_transform")),
    ("wmh_synthseg", ("WMHSynthSeg", "WMHResult")),
    ("synthseg_parc", ("SynthSeg", "SynthSegResult")),
    ("synthsr", ("SynthSR", "SynthSRResult", "SynthSRImage")),
    ("fast", ("TorchFAST", "FASTResult", "FASTConfig", "FASTTensorResult", "segment_t1")),
    ("flirt", ("FLIRTResult", "TorchFLIRT",
               "flirt_to_world_affine", "flirt_to_world_pull",
               "voxel_to_fsl_scaled_mm", "world_to_flirt_affine")),
    ("fnirt", ("TorchFNIRT", "TorchFNIRTResult", "GMFNIRTConfig")),
    ("fast_vbm", ("FastVBM", "FastVBMResult", "VBMRegistrationResult")),
])
def test_top_level_exports_are_feature_objects(module, names):
    package = importlib.import_module("freesurfer_torch")
    feature = importlib.import_module(f"freesurfer_torch.{module}")
    for name in names:
        assert getattr(package, name) is getattr(feature, name)


def test_removed_batch_api_is_absent():
    package = importlib.import_module("freesurfer_torch")
    for name in ("BatchRunner", "BatchResult", "run_batch"):
        assert not hasattr(package, name)

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("freesurfer_torch.batch")

    apply_transform = package.apply_transform
    assert "device" not in inspect.signature(apply_transform).parameters


def test_top_level_import_stays_lightweight():
    code = "import sys; import freesurfer_torch; assert not {'torch', 'surfa', 'h5py', 'nibabel', 'tensorflow'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
