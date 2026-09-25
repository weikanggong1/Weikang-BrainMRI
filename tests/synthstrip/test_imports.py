"""SynthStrip exports its model and inference API."""

from freesurfer_torch.synthstrip import ConvBlock, StripModel, StripResult, SynthStrip, extend_sdt
from freesurfer_torch.synthstrip import model, pipeline


def test_feature_exports_are_original_objects():
    assert ConvBlock is model.ConvBlock
    assert StripModel is model.StripModel
    assert SynthStrip is pipeline.SynthStrip
    assert StripResult is pipeline.StripResult
    assert extend_sdt is pipeline.extend_sdt
