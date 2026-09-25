"""Legacy NCC/Adam affine registration retained for reproducibility.

The file and matrix interfaces follow FSL FLIRT's input/reference roles and
coordinate convention. The optimizer remains this package's NCC/Adam method;
it is not an implementation of FLIRT's correlation-ratio/Brent algorithm.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import surfa as sf
import torch



def _load_volume(value, name):
    if isinstance(value, (str, os.PathLike)):
        return sf.load_volume(str(value))
    if isinstance(value, sf.Volume):
        return value
    raise TypeError(f"{name} must be a path or surfa.Volume")


def _single_frame(volume, name):
    data = np.asarray(volume.data)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{name} must contain one 3D frame")
    if any(size < 2 for size in data.shape):
        raise ValueError(f"{name} dimensions must each contain at least two voxels")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(data, dtype=np.float32, copy=True)


@dataclass(frozen=True)
class FLIRTResult:
    """Reference-grid image and transforms returned by a FLIRT model."""

    moved: sf.Volume
    matrix: np.ndarray
    moving_to_fixed_world: np.ndarray
    fixed_to_moving_world: np.ndarray
    qc: dict

    @property
    def fsl_matrix(self):
        """Moving/input to fixed/reference transform in FSL scaled-mm."""
        return self.matrix

    def save(self, output=None, omat=None):
        """Save the moved image and/or FSL ``.mat`` transform."""
        if (
            output is not None
            and omat is not None
            and Path(output).resolve() == Path(omat).resolve()
        ):
            raise ValueError("output and omat must use different paths")
        if output is not None:
            self.moved.save(str(output))
        if omat is not None:
            np.savetxt(str(omat), self.matrix, fmt="%.12g")
        return self


class LegacyTorchFLIRT:
    """Legacy PyTorch NCC/Adam affine registration with FLIRT file contracts.

    ``moving``/``input`` is transformed to the ``fixed``/``reference`` grid.
    The moved volume therefore has the reference shape and geometry. ``matrix``
    follows an FSL ``flirt -omat`` file: moving/input to fixed/reference in FSL
    scaled-mm coordinates.

    This class preserves those I/O contracts but uses normalized correlation,
    Adam, and the package's fixed coarse-to-fine schedule. Its numerical result
    is not expected to equal the FSL FLIRT executable.
    """

    def __init__(
        self,
        device=None,
        *,
        strides=(4, 2, 1),
        steps=(80, 60, 50),
        learning_rates=(0.05, 0.025, 0.0125),
        cost="normcorr",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.strides = strides
        self.steps = steps
        self.learning_rates = learning_rates
        self.cost = cost

    def __call__(self, moving, fixed):
        from ..fast_vbm.linear import register_affine, world_to_flirt_affine

        moving = _load_volume(moving, "moving")
        fixed = _load_volume(fixed, "fixed")
        moving_data = _single_frame(moving, "moving")
        fixed_data = _single_frame(fixed, "fixed")
        moving_affine = np.array(
            moving.geom.vox2world.matrix, dtype=np.float64, copy=True
        )
        fixed_affine = np.array(
            fixed.geom.vox2world.matrix, dtype=np.float64, copy=True
        )

        registration = register_affine(
            moving_data,
            fixed_data,
            moving_affine,
            fixed_affine,
            device=self.device,
            strides=self.strides,
            steps=self.steps,
            learning_rates=self.learning_rates,
            cost=self.cost,
        )
        moved_data = registration.warped.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        moved = fixed.new(moved_data)
        matrix = world_to_flirt_affine(
            registration.moving_to_fixed_world,
            moving_affine,
            fixed_affine,
            moving_data.shape,
            fixed_data.shape,
            moving.geom.voxsize,
            fixed.geom.voxsize,
        )
        qc = dict(registration.qc)
        qc.update(
            {
                "image_output_grid": "fixed/reference",
                "matrix_coordinate_system": "FSL scaled-mm",
                "matrix_direction": "moving/input-to-fixed/reference",
                "equivalent_to_fsl_flirt_algorithm": False,
            }
        )
        return FLIRTResult(
            moved=moved,
            matrix=matrix,
            moving_to_fixed_world=registration.moving_to_fixed_world,
            fixed_to_moving_world=registration.fixed_to_moving_world,
            qc=qc,
        )

    def run(
        self,
        input,
        reference,
        *,
        output=None,
        omat=None,
    ):
        """Register FLIRT-style input/reference arguments and save outputs."""
        return self(input, reference).save(output=output, omat=omat)


__all__ = ["FLIRTResult", "LegacyTorchFLIRT"]
