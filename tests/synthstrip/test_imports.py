"""SynthStrip keeps its model API and standalone module command."""
import subprocess
import sys

from freesurfer_torch.synthstrip import ConvBlock, StripModel, StripResult, SynthStrip, extend_sdt
from freesurfer_torch.synthstrip import model, pipeline


def test_feature_exports_are_original_objects():
    assert ConvBlock is model.ConvBlock
    assert StripModel is model.StripModel
    assert SynthStrip is pipeline.SynthStrip
    assert StripResult is pipeline.StripResult
    assert extend_sdt is pipeline.extend_sdt


def test_legacy_module_command_help():
    result = subprocess.run(
        [sys.executable, "-m", "freesurfer_torch.synthstrip", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--no-csf" in result.stdout
    assert "--model" in result.stdout
