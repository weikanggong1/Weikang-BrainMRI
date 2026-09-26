"""Final projection and overlap cleanup for conventional FreeSurfer spheres."""

from __future__ import annotations

import numpy as np
import torch

from .mris_register_overlap import remove_overlap_sphere
from .sphere_python import project_radially


@torch.no_grad()
def finish_standard_sphere(vertices: np.ndarray, faces: np.ndarray,
                           *, start_iteration: int, device: str = "cpu"
                           ) -> tuple[np.ndarray, list[int]]:
    """Continue from the last ``MRISunfold`` checkpoint to the final sphere."""
    projected = project_radially(vertices, already_sphere=True)
    xyz = torch.from_numpy(np.ascontiguousarray(projected)).to(device)
    triangles = torch.from_numpy(np.asarray(faces, np.int64)).to(device)
    result, negative_counts = remove_overlap_sphere(
        xyz, triangles, start_iteration=start_iteration)
    return result.cpu().numpy(), negative_counts
