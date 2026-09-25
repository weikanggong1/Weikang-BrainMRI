"""Compatibility imports for the public FLIRT package.

New code should import from :mod:`freesurfer_torch.flirt`.
"""

from ..flirt.core import (
    FSLCorrelationRatio,
    FSLFLIRT,
    FSL_FLIRT_COMMIT,
    FSL_FLIRT_VERSION,
    TorchFLIRT,
    _DefaultFLIRTEngine,
    _centre_of_gravity,
    _newimage_percentile,
    _resample_output,
    fsl_affine_from_parameters,
    fsl_coordinate_optimize,
    fsl_parameters_from_affine,
)

__all__ = [
    "FSLCorrelationRatio",
    "FSLFLIRT",
    "FSL_FLIRT_COMMIT",
    "FSL_FLIRT_VERSION",
    "TorchFLIRT",
    "fsl_affine_from_parameters",
    "fsl_coordinate_optimize",
    "fsl_parameters_from_affine",
]
