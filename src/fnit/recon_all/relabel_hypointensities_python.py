"""Relabel cortex voxels inside white surfaces, as in recon-all."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.spatial import cKDTree
import torch

from .mgh_compat import save_same_dtype_mgh
from .surface_thickness_gpu import _normals


def relabel_hypointensities(aseg: np.ndarray, vox2ras_tkr: np.ndarray,
                           surfaces: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply both surface passes and two six-neighbor cortex recovery passes."""
    if aseg.ndim != 3 or np.shape(vox2ras_tkr) != (4, 4) or set(surfaces) != {"lh", "rh"}:
        raise ValueError("expected a 3D aseg, voxel-to-surface transform, and two surfaces")
    out = aseg.copy()
    out[np.isin(out, (78, 79))] = 77
    for hemi, cortex_label in (("lh", 3), ("rh", 42)):
        xyz, faces = surfaces[hemi]
        if xyz.ndim != 2 or xyz.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("each surface needs (vertices, 3) coordinates and faces")
        vertices = torch.as_tensor(np.asarray(xyz, dtype=np.float32))
        triangles = torch.as_tensor(np.asarray(faces, dtype=np.int64))
        normals = _normals(vertices, triangles).numpy()
        ijk = np.argwhere(out == cortex_label)
        ras = ijk @ vox2ras_tkr[:3, :3].T + vox2ras_tkr[:3, 3]
        distance, nearest = cKDTree(xyz).query(ras, workers=4)
        delta = (ras - xyz[nearest]).astype(np.float32)
        inside = (np.sum(delta * normals[nearest], axis=1) < 0) & (distance > 1)
        selected = ijk[inside]
        out[tuple(selected.T)] = 77
    six_neighbors = generate_binary_structure(3, 1)
    for _ in range(2):
        previous = out.copy()
        hypo = previous == 77
        left = hypo & binary_dilation(previous == 3, structure=six_neighbors)
        right = hypo & ~left & binary_dilation(previous == 42, structure=six_neighbors)
        out[left] = 3
        out[right] = 42
    return out


def relabel_volume(aseg_file: str | Path, surface_dir: str | Path,
                   output_file: str | Path) -> None:
    image = nib.load(str(aseg_file))
    if not isinstance(image, nib.MGHImage) or len(image.shape) != 3:
        raise ValueError("input must be a 3D MGH/MGZ image")
    surface_dir = Path(surface_dir)
    surfaces = {hemi: fsio.read_geometry(str(surface_dir / f"{hemi}.white"))
                for hemi in ("lh", "rh")}
    result = relabel_hypointensities(np.asarray(image.dataobj),
                                    image.header.get_vox2ras_tkr(), surfaces)
    save_same_dtype_mgh(aseg_file, output_file, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aseg_file", type=Path)
    parser.add_argument("surface_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args(argv)
    relabel_volume(args.aseg_file, args.surface_dir, args.output_file)


if __name__ == "__main__":
    main()
