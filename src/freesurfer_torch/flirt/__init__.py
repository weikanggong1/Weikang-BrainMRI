"""PyTorch FLIRT implementation with FSL image and matrix contracts."""

from .core import (
    FSLCorrelationRatio,
    FSLFLIRT,
    FSL_FLIRT_COMMIT,
    FSL_FLIRT_VERSION,
    TorchFLIRT,
    fsl_affine_from_parameters,
    fsl_coordinate_optimize,
    fsl_parameters_from_affine,
)
from .legacy import FLIRTResult, LegacyTorchFLIRT
from .standalone import run_flirt

__all__ = [
    "FLIRTResult",
    "FSLCorrelationRatio",
    "FSLFLIRT",
    "FSL_FLIRT_COMMIT",
    "FSL_FLIRT_VERSION",
    "LegacyTorchFLIRT",
    "TorchFLIRT",
    "fsl_affine_from_parameters",
    "fsl_coordinate_optimize",
    "fsl_parameters_from_affine",
    "run_flirt",
]
