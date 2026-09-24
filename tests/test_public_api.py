"""Public imports and old module paths must survive the feature layout change."""
import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module,names", [
    ("synthstrip", ("SynthStrip", "StripResult")),
    ("synthmorph", ("SynthMorph", "RegistrationResult", "apply_transform")),
    ("wmh_synthseg", ("WMHSynthSeg", "WMHResult")),
    ("synthsr", ("SynthSR", "SynthSRResult", "SynthSRImage")),
    ("fast", ("TorchFAST", "FASTResult", "FASTConfig", "FASTTensorResult", "segment_t1")),
    ("fast_vbm", ("FastVBM", "FastVBMResult", "LinearRegistrationResult",
                  "VBMRegistrationResult", "register_affine", "register_gm")),
    ("batch", ("BatchRunner", "BatchResult", "run_batch")),
])
def test_top_level_exports_are_feature_objects(module, names):
    package = importlib.import_module("freesurfer_torch")
    feature = importlib.import_module(f"freesurfer_torch.{module}")
    for name in names:
        assert getattr(package, name) is getattr(feature, name)


@pytest.mark.parametrize("old_module,new_module,names", [
    ("spatial", "synthmorph.spatial", (
        "grid", "square", "dense", "transform", "compose", "integrate", "affine_to_dense",
    )),
    ("synthmorph_models", "synthmorph.models", (
        "FeatureDetector", "barycenter", "fit_affine", "matrix_sqrt",
        "AffineNetwork", "DeformNetwork", "SynthMorphNetwork",
    )),
])
def test_legacy_module_exports_are_same_objects(old_module, new_module, names):
    old = importlib.import_module(f"freesurfer_torch.{old_module}")
    new = importlib.import_module(f"freesurfer_torch.{new_module}")
    for name in names:
        assert getattr(old, name) is getattr(new, name)


def test_top_level_import_stays_lightweight():
    code = "import sys; import freesurfer_torch; assert not {'torch', 'surfa', 'h5py', 'nibabel', 'tensorflow'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
