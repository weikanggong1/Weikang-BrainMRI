"""Compatibility imports for the public FLIRT package.

New code should import from :mod:`freesurfer_torch.flirt`.
"""

from ..flirt import FLIRTResult, LegacyTorchFLIRT, TorchFLIRT
from ..flirt.legacy import _load_volume, _single_frame

__all__ = ["FLIRTResult", "LegacyTorchFLIRT", "TorchFLIRT"]
