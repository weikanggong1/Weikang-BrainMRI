"""Compatibility imports; implementations live in ``synthmorph.models``."""
from .synthmorph.models import (
    AffineNetwork,
    DeformNetwork,
    FeatureDetector,
    SynthMorphNetwork,
    barycenter,
    fit_affine,
    matrix_sqrt,
)
from .synthmorph.spatial import affine_to_dense, compose, integrate, transform

__all__ = [
    "AffineNetwork", "DeformNetwork", "FeatureDetector", "SynthMorphNetwork",
    "barycenter", "fit_affine", "matrix_sqrt", "affine_to_dense", "compose",
    "integrate", "transform",
]
