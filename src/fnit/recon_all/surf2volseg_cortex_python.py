"""The fixed `mri_surf2volseg --label-cortex` atlas volume steps."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.spatial import cKDTree
import torch

from .mgh_compat import save_same_dtype_mgh
from .surface_thickness_gpu import _normals


def _nearest_with_dot(ras: np.ndarray, xyz: np.ndarray, faces: np.ndarray,
                      cortex_vertices: np.ndarray, sign: int) -> tuple[np.ndarray, np.ndarray]:
    normal = _normals(torch.as_tensor(np.asarray(xyz, dtype=np.float32)),
                      torch.as_tensor(np.asarray(faces, dtype=np.int64))).numpy()
    tree = cKDTree(xyz[cortex_vertices])
    distance, local = tree.query(ras, workers=4)
    vertex = cortex_vertices[local]
    dot = np.sum((ras - xyz[vertex]) * normal[vertex], axis=1)
    pending = np.flatnonzero(dot * sign < 0)
    count = min(256, len(cortex_vertices))
    while len(pending):
        near_distance, near_local = tree.query(ras[pending], k=count, workers=4)
        near_distance = np.reshape(near_distance, (len(pending), count))
        near_vertex = cortex_vertices[np.reshape(near_local, (len(pending), count))]
        delta = ras[pending, None, :] - xyz[near_vertex]
        valid = (np.sum(delta * normal[near_vertex], axis=2) * sign) >= 0
        found = valid.any(axis=1)
        first = valid.argmax(axis=1)
        matched = pending[found]
        distance[matched] = near_distance[found, first[found]]
        vertex[matched] = near_vertex[found, first[found]]
        pending = pending[~found]
        if count == len(cortex_vertices):
            distance[pending] = 1e10
            vertex[pending] = -1
            break
        count = min(2 * count, len(cortex_vertices))
    return distance, vertex


def label_cortex_voxels(
    aseg: np.ndarray, vox2ras_tkr: np.ndarray,
    hemispheres: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray, np.ndarray]],
    offsets: tuple[int, int] = (1000, 2000),
) -> np.ndarray:
    """Map cortex labels 3/42 to atlas IDs by nearest surface."""
    if aseg.ndim != 3 or np.shape(vox2ras_tkr) != (4, 4) or set(hemispheres) != {"lh", "rh"}:
        raise ValueError("expected a 3D aseg, voxel-to-surface transform, and two hemispheres")
    out = aseg.copy()
    for hemi, cortex_label, offset in (("lh", 3, offsets[0]), ("rh", 42, offsets[1])):
        white_xyz, white_faces, pial_xyz, pial_faces, cortex_vertices, annotation = hemispheres[hemi]
        if (len(white_xyz) != len(pial_xyz) or len(white_xyz) != len(annotation) or
                not np.array_equal(white_faces, pial_faces)):
            raise ValueError("white, pial, and annotation must share ordered vertices")
        ijk = np.argwhere(aseg == cortex_label)
        ras = ijk @ vox2ras_tkr[:3, :3].T + vox2ras_tkr[:3, 3]
        white_distance, white_vertex = _nearest_with_dot(
            ras, white_xyz, white_faces, cortex_vertices, +1)
        pial_distance, pial_vertex = _nearest_with_dot(
            ras, pial_xyz, pial_faces, cortex_vertices, -1)
        chosen = np.where(pial_distance < white_distance, pial_vertex, white_vertex)
        labeled = np.where(chosen >= 0, offset + np.maximum(annotation[chosen], 0), 0)
        out[tuple(ijk.T)] = labeled
    return out


def label_cortex_volume(aseg_file: str | Path, surface_dir: str | Path,
                        label_dir: str | Path, output_file: str | Path,
                        *, atlas: str = "aparc") -> None:
    offsets = {"aparc": (1000, 2000), "aparc.a2009s": (11100, 12100),
               "aparc.DKTatlas": (1000, 2000)}
    if atlas not in offsets:
        raise ValueError(f"unknown fixed atlas: {atlas}")
    source = nib.load(str(aseg_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("input must be a 3D MGH/MGZ image")
    surface_dir, label_dir = Path(surface_dir), Path(label_dir)
    hemispheres = {}
    for hemi in ("lh", "rh"):
        white_xyz, white_faces = fsio.read_geometry(str(surface_dir / f"{hemi}.white"))
        pial_xyz, pial_faces = fsio.read_geometry(str(surface_dir / f"{hemi}.pial"))
        cortex_vertices = fsio.read_label(str(label_dir / f"{hemi}.cortex.label"))
        annotation, _, _ = fsio.read_annot(str(label_dir / f"{hemi}.{atlas}.annot"))
        hemispheres[hemi] = (white_xyz, white_faces, pial_xyz, pial_faces,
                             cortex_vertices, annotation)
    result = label_cortex_voxels(np.asarray(source.dataobj),
                                 source.header.get_vox2ras_tkr(), hemispheres,
                                 offsets[atlas])
    save_same_dtype_mgh(aseg_file, output_file, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aseg_file", type=Path)
    parser.add_argument("surface_dir", type=Path)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--atlas", choices=("aparc", "aparc.a2009s", "aparc.DKTatlas"),
                        default="aparc")
    args = parser.parse_args(argv)
    label_cortex_volume(args.aseg_file, args.surface_dir, args.label_dir,
                        args.output_file, atlas=args.atlas)


if __name__ == "__main__":
    main()
