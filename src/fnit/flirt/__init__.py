"""PyTorch FLIRT implementation with FSL image and matrix contracts."""

from .core import (
    FSLCorrelationRatio,
    FSL_FLIRT_COMMIT,
    FSL_FLIRT_VERSION,
    TorchFLIRT,
    fsl_affine_from_parameters,
    fsl_coordinate_optimize,
    fsl_parameters_from_affine,
)
from .coordinates import (
    flirt_to_world_affine,
    flirt_to_world_pull,
    voxel_to_fsl_scaled_mm,
    world_to_flirt_affine,
)
from .types import FLIRTResult
from .standalone import run_flirt

__all__ = [
    "FLIRTResult",
    "FSLCorrelationRatio",
    "FSL_FLIRT_COMMIT",
    "FSL_FLIRT_VERSION",
    "TorchFLIRT",
    "flirt_to_world_affine",
    "flirt_to_world_pull",
    "fsl_affine_from_parameters",
    "fsl_coordinate_optimize",
    "fsl_parameters_from_affine",
    "voxel_to_fsl_scaled_mm",
    "world_to_flirt_affine",
    "run_flirt",
]
