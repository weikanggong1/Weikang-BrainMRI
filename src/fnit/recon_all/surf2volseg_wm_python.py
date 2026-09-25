"""The fixed mri_surf2volseg --label-wm wmparc volume step."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from .mgh_compat import save_same_dtype_mgh
from .surf2volseg_cortex_python import _nearest_with_dot


def label_wm_voxels(
    aparc_aseg: np.ndarray, vox2ras_tkr: np.ndarray,
    hemispheres: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Assign 3000+/4000+ labels within cerebral WM and 5001/5002 beyond 5 mm."""
    if aparc_aseg.ndim != 3 or np.shape(vox2ras_tkr) != (4, 4) or set(hemispheres) != {"lh", "rh"}:
        raise ValueError("expected a 3D segmentation, voxel-to-surface transform, and both hemispheres")
    output = aparc_aseg.copy()
    def tkr_x(points: np.ndarray) -> np.ndarray:
        return points @ vox2ras_tkr[0, :3] + vox2ras_tkr[0, 3]

    for hemi, wm_label, hypo_label, offset, unknown in (
        ("lh", 2, 78, 3000, 5001), ("rh", 41, 79, 4000, 5002)
    ):
        xyz, faces, cortex_vertices, annotation = hemispheres[hemi]
        if len(xyz) != len(annotation):
            raise ValueError("surface and annotation vertex counts differ")
        points = np.argwhere((aparc_aseg == wm_label) | (aparc_aseg == hypo_label))
        unpaired = np.argwhere(aparc_aseg == 77)
        if len(unpaired):
            on_side = tkr_x(unpaired) <= 0 if hemi == "lh" else tkr_x(unpaired) > 0
            points = np.concatenate((points, unpaired[on_side]), axis=0)
        if len(points):
            ras = points @ vox2ras_tkr[:3, :3].T + vox2ras_tkr[:3, 3]
            distance, vertex = _nearest_with_dot(ras, xyz, faces, cortex_vertices, -1)
            label = np.full(len(points), unknown, dtype=np.int32)
            accepted = (vertex >= 0) & (distance <= 5.0)
            label[accepted] = np.where(annotation[vertex[accepted]] > 0,
                                       offset + annotation[vertex[accepted]], unknown)
            output[tuple(points.T)] = label

    # Future WMSA labels enter the native IS_HYPO branch, but no surface is
    # permitted for IDs 87-89; native mri_surf2volseg assigns side-specific unknown.
    unpaired = np.argwhere(np.isin(aparc_aseg, (87, 88, 89)))
    if len(unpaired):
        output[tuple(unpaired.T)] = np.where(tkr_x(unpaired) <= 0, 5001, 5002)
    return output


def label_wm_volume(aparc_aseg_file: str | Path, surface_dir: str | Path,
                    label_dir: str | Path, output_file: str | Path) -> None:
    """Read recon-all inputs and write wmparc.mgz without a FreeSurfer executable."""
    source = nib.load(str(aparc_aseg_file))
    if not isinstance(source, nib.MGHImage) or len(source.shape) != 3:
        raise ValueError("input must be a 3D MGH/MGZ image")
    surface_dir, label_dir = Path(surface_dir), Path(label_dir)
    hemispheres = {}
    for hemi in ("lh", "rh"):
        xyz, faces = fsio.read_geometry(str(surface_dir / f"{hemi}.white"))
        cortex_vertices = fsio.read_label(str(label_dir / f"{hemi}.cortex.label"))
        annotation, _, _ = fsio.read_annot(str(label_dir / f"{hemi}.aparc.annot"))
        hemispheres[hemi] = (xyz, faces, cortex_vertices, annotation)
    result = label_wm_voxels(np.asarray(source.dataobj), source.header.get_vox2ras_tkr(),
                             hemispheres)
    save_same_dtype_mgh(aparc_aseg_file, output_file, result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aparc_aseg_file", type=Path)
    parser.add_argument("surface_dir", type=Path)
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args(argv)
    label_wm_volume(args.aparc_aseg_file, args.surface_dir, args.label_dir, args.output_file)


if __name__ == "__main__":
    main()
