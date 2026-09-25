"""The fixed `mri_surf2volseg --fix-presurf-with-ribbon` voxel step."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.spatial import cKDTree

from .mgh_compat import save_same_dtype_mgh


def fix_presurf_with_ribbon(
    aseg: np.ndarray, ribbon: np.ndarray, vox2ras_tkr: np.ndarray,
    surfaces: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Fix cortex/WM labels using the ribbon and the nearest cortex vertex."""
    if aseg.ndim != 3 or aseg.shape != ribbon.shape or np.shape(vox2ras_tkr) != (4, 4):
        raise ValueError("aseg and ribbon must share a 3D voxel grid")
    if len(surfaces) != 4:
        raise ValueError("expected left white/pial and right white/pial surfaces")
    out = aseg.copy()
    cortex = np.isin(aseg, (3, 42))
    white = np.isin(aseg, (2, 41))
    cerebellar_gray = np.isin(aseg, (8, 47))
    hypo = np.isin(aseg, (77, 78, 79, 87, 88, 89))
    ribbon_cortex = np.isin(ribbon, (3, 42))
    ribbon_white = np.isin(ribbon, (2, 41))
    out[cortex & (ribbon == 0)] = 0
    out[cortex & ribbon_white] = ribbon[cortex & ribbon_white]
    move = cerebellar_gray & (ribbon_cortex | ribbon_white)
    out[move] = ribbon[move]
    out[hypo & (ribbon == 0)] = 0
    out[hypo & ribbon_cortex] = ribbon[hypo & ribbon_cortex]
    out[white & ribbon_cortex] = ribbon[white & ribbon_cortex]

    # For WM outside the ribbon and unknown voxels inside it, the native
    # program uses the cortex marker of the closest white/pial vertex.
    candidate = ((aseg == 0) & (ribbon != 0)) | (white & (ribbon == 0))
    ijk = np.argwhere(candidate)
    if len(ijk):
        ras = ijk @ vox2ras_tkr[:3, :3].T + vox2ras_tkr[:3, 3]
        distances, indices, marks = [], [], []
        for xyz, cortex_mask in surfaces:
            if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(cortex_mask):
                raise ValueError("surface coordinates and cortex mask must align")
            distance, index = cKDTree(xyz).query(ras, workers=4)
            distances.append(distance)
            indices.append(index)
            marks.append(cortex_mask)
        closest = np.stack(distances, axis=1).argmin(axis=1)
        marked = np.array([marks[which][indices[which][i]]
                           for i, which in enumerate(closest)], dtype=bool)
        selected = ijk[marked]
        out[tuple(selected.T)] = ribbon[tuple(selected.T)]
    return out


def fix_presurf_volume(aseg_file: str | Path, ribbon_file: str | Path,
                       surface_dir: str | Path, label_dir: str | Path,
                       output_file: str | Path) -> None:
    source, ribbon_image = nib.load(str(aseg_file)), nib.load(str(ribbon_file))
    if (not isinstance(source, nib.MGHImage) or
            not isinstance(ribbon_image, nib.MGHImage) or
            source.shape != ribbon_image.shape or len(source.shape) != 3 or
            not np.allclose(source.affine, ribbon_image.affine, rtol=0, atol=1e-4)):
        raise ValueError("input segmentation and ribbon must share a 3D MGH grid")
    surface_dir, label_dir = Path(surface_dir), Path(label_dir)
    surfaces = []
    for hemi in ("lh", "rh"):
        cortex_vertices = fsio.read_label(str(label_dir / f"{hemi}.cortex.label"))
        for name in ("white", "pial"):
            xyz, _ = fsio.read_geometry(str(surface_dir / f"{hemi}.{name}"))
            mask = np.zeros(len(xyz), dtype=bool)
            mask[cortex_vertices] = True
            surfaces.append((xyz, mask))
    result = fix_presurf_with_ribbon(np.asarray(source.dataobj),
                                    np.asarray(ribbon_image.dataobj),
                                    source.header.get_vox2ras_tkr(), surfaces)
    save_same_dtype_mgh(aseg_file, output_file, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aseg_file", type=Path)
    parser.add_argument("ribbon_file", type=Path)
    parser.add_argument("surface_dir", type=Path)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args(argv)
    fix_presurf_volume(args.aseg_file, args.ribbon_file, args.surface_dir,
                       args.label_dir, args.output_file)


if __name__ == "__main__":
    main()
