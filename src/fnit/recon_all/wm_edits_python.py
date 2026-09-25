"""Standalone entorhinal and amygdala junction white-matter edits."""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from .mgh_compat import save_same_dtype_mgh


def amygdala_cortex_junction(aseg: np.ndarray) -> np.ndarray:
    """Return labels 7030/7031 at the 26-neighbor amygdala-cortex boundary."""
    if aseg.ndim != 3:
        raise ValueError("Expected a 3D aseg volume")
    cortex = (aseg == 3) | (aseg == 42)
    result = np.zeros(aseg.shape, dtype=np.int32)
    boundaries = []
    for amygdala, label in ((18, 7030), (54, 7031)):
        source = (aseg == amygdala).copy()
        source[[0, -1], :, :] = False
        source[:, [0, -1], :] = False
        source[:, :, [0, -1]] = False
        boundary = binary_dilation(source, structure=np.ones((3, 3, 3))) & cortex
        boundaries.append(boundary)
        result[boundary] = label
    for c, r, s in np.argwhere(boundaries[0] & boundaries[1]):
        neighbors = aseg[max(c - 1, 1):min(c + 2, aseg.shape[0] - 1),
                         max(r - 1, 1):min(r + 2, aseg.shape[1] - 1),
                         max(s - 1, 1):min(s + 2, aseg.shape[2] - 1)]
        last = neighbors.reshape(-1)[np.isin(neighbors.reshape(-1), (18, 54))][-1]
        result[c, r, s] = 7030 if last == 18 else 7031
    return result


def fix_ento_wm(input_file: str | Path, label_file: str | Path,
                output_file: str | Path, *, level: int,
                left_value: int, right_value: int,
                acj: bool = False) -> int:
    """Replay ``mri_edit_wm_with_aseg -sa-fix-ento-wm/-sa-fix-acj``."""
    source = nib.load(str(input_file))
    labels = nib.load(str(label_file))
    if (source.shape != labels.shape or len(source.shape) != 3 or
            not np.allclose(source.affine, labels.affine, rtol=0, atol=1e-4)):
        raise ValueError("Input and label volumes must share a 3D grid")
    voxels = np.asarray(source.dataobj).copy()
    segmentation = np.asarray(labels.dataobj)
    if acj:
        segmentation = amygdala_cortex_junction(segmentation)
    left = np.isin(segmentation, ([3006] if level in (1, 3) else []) +
                   ([7030 if acj else 3201] if level in (2, 3) else []))
    right = np.isin(segmentation, ([4006] if level in (1, 3) else []) +
                    ([7031 if acj else 4201] if level in (2, 3) else []))
    voxels[left] = left_value
    voxels[right] = right_value
    save_same_dtype_mgh(input_file, output_file, voxels)
    return int(left.sum() + right.sum())
