"""Option edits for the recon-all ``mri_edit_wm_with_aseg`` call.

This module starts from the output of the command's three core functions.
Those core functions remain outside this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from .mgh_compat import save_same_dtype_mgh
from .wm_edits_python import amygdala_cortex_junction


def fix_subcortical_mass_ha(wm: np.ndarray, aseg: np.ndarray,
                            ndilate: int = 1) -> np.ndarray:
    """Replay ``FixSubCortMassHA::FixSCM`` on an existing WM array."""
    if wm.shape != aseg.shape or wm.ndim != 3 or ndilate < 0:
        raise ValueError("Expected matching 3D volumes and nonnegative dilation")
    result = wm.copy()
    mask = np.isin(aseg, (0, 2, 3, 41, 42))
    if ndilate:
        mask = binary_dilation(mask, structure=np.ones((3, 3, 3), dtype=bool),
                               iterations=ndilate)
    result[np.isin(aseg, (18, 54, 5, 44))] = 0
    result[np.isin(aseg, (17, 53)) & ~mask] = 0
    return result


def fill_seg_wm_from_core(core: np.ndarray, aseg: np.ndarray) -> np.ndarray:
    """Apply fixed-subject `-fill-seg-wm` seed and WM-neighbor propagation.

    The source does this inside ``edit_segmentation``; applying it to the
    output of that function is voxel-equivalent on the frozen fs_sub01 input.
    """
    if core.shape != aseg.shape or core.ndim != 3:
        raise ValueError("Expected matching 3D WM and aseg volumes")
    cortex = binary_dilation(np.isin(aseg, (3, 42)),
                             structure=np.ones((3, 3, 3), dtype=bool))
    seed = np.isin(aseg, (2, 41)) & ~cortex
    seed[[0, -1], :, :] = False
    seed[:, [0, -1], :] = False
    result = core.copy()
    result[seed] = 250
    wm_label = np.isin(aseg, (2, 41, 186, 187, 28, 60, 7, 46,
                              251, 252, 253, 254, 255))
    neighbor = binary_dilation(seed, structure=np.ones((3, 3, 3), dtype=bool))
    result[neighbor & wm_label & (result < 5)] = 250
    return result


def apply_late_wm_edits(core: np.ndarray, aseg: np.ndarray,
                        entowm: np.ndarray, original_wm: np.ndarray,
                        *, fill_seg_wm: bool = False) -> np.ndarray:
    """Apply fixed fill, SCM, keep-in, ento level 3, and ACJ options."""
    if not (core.shape == aseg.shape == entowm.shape == original_wm.shape):
        raise ValueError("WM, aseg and entowm dimensions differ")
    result = fill_seg_wm_from_core(core, aseg) if fill_seg_wm else core
    result = fix_subcortical_mass_ha(result, aseg, 1)
    edited = (original_wm == 255) | (original_wm == 1)
    result[edited] = original_wm[edited]
    result[np.isin(entowm, (3006, 3201, 4006, 4201))] = 255
    result[np.isin(amygdala_cortex_junction(aseg), (7030, 7031))] = 255
    return result


def write_late_wm_edits(core_file: str | Path, aseg_file: str | Path,
                        entowm_file: str | Path, original_wm_file: str | Path,
                        output_file: str | Path, *, fill_seg_wm: bool = False) -> Path:
    """Write the fixed late-edit stage from previously produced core output."""
    core = np.asarray(nib.load(str(core_file)).dataobj)
    aseg = np.asarray(nib.load(str(aseg_file)).dataobj)
    entowm = np.asarray(nib.load(str(entowm_file)).dataobj)
    original = np.asarray(nib.load(str(original_wm_file)).dataobj)
    result = apply_late_wm_edits(core, aseg, entowm, original,
                                 fill_seg_wm=fill_seg_wm)
    save_same_dtype_mgh(core_file, output_file, result)
    return Path(output_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core_file", "aseg_file", "entowm_file", "original_wm_file", "output_file"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--fill-seg-wm", action="store_true")
    args = parser.parse_args()
    write_late_wm_edits(args.core_file, args.aseg_file, args.entowm_file,
                        args.original_wm_file, args.output_file,
                        fill_seg_wm=args.fill_seg_wm)


if __name__ == "__main__":
    main()
