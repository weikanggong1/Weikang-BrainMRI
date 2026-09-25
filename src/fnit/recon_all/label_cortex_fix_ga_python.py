"""Python translation of recon-all's fixed ``label-cortex --fix-ga`` path.

The FreeSurfer script first creates a no-GA cortical label, selects medial
gyrus ambiens vertices from ``entowm.mgz``, then concatenates both labels.
The later ``cortex+hipamyg.label`` call is a separate recon-all step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from .inflate_python import vertex_normals
from .label_cortex_python import cortex_label_mask, _nearest, _voxel


def gyrus_ambiens_mask(vertices: np.ndarray, faces: np.ndarray,
                       entowm_image: nib.spatialimages.SpatialImage,
                       hemi: str) -> np.ndarray:
    """Return vertex IDs added by the fixed 1 mm inward GA search."""
    if hemi not in ("lh", "rh"):
        raise ValueError("hemi must be 'lh' or 'rh'")
    xyz = np.asarray(vertices, np.float32)
    normals = vertex_normals(xyz, np.asarray(faces, np.int32))
    volume = np.asanyarray(entowm_image.dataobj)
    ga_volume = volume == (3201 if hemi == "lh" else 4201)
    tk_to_vox = np.linalg.inv(entowm_image.header.get_vox2ras_tkr())
    on_ga = np.zeros(len(xyz), bool)
    for step in range(11):
        distance = -1.0 + 0.1 * step
        projected = xyz.astype(np.float64) + distance * normals.astype(np.float64)
        on_ga |= _nearest(ga_volume, _voxel(projected, tk_to_vox))

    tkr_to_scanner = entowm_image.affine @ tk_to_vox
    scanner_normals = normals @ tkr_to_scanner[:3, :3].T
    x_facing = (scanner_normals[:, 0] >= 0.01 if hemi == "lh"
                else scanner_normals[:, 0] <= -0.01)
    return on_ga & x_facing & (scanner_normals[:, 2] <= -0.01)


def label_cortex_fix_ga(surface_path: str | Path, aseg_path: str | Path,
                        entowm_path: str | Path, hemi: str,
                        output_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Write the native final label; return no-GA and appended GA vertex IDs."""
    vertices, faces = fsio.read_geometry(str(surface_path))
    aseg = nib.load(str(aseg_path))
    base = cortex_label_mask(vertices, faces, np.asanyarray(aseg.dataobj),
                             np.linalg.inv(aseg.header.get_vox2ras_tkr()))
    ga = gyrus_ambiens_mask(vertices, faces, nib.load(str(entowm_path)), hemi)
    base_ids, ga_ids = np.flatnonzero(base), np.flatnonzero(ga)
    with Path(output_path).open("w") as stream:
        stream.write("#!ascii label , from subject vox2ras=TkReg\n")
        stream.write(f"{len(base_ids) + len(ga_ids)}\n")
        for vertex in np.concatenate((base_ids, ga_ids)):
            x, y, z = vertices[vertex]
            stream.write(f"{vertex}  {x:.3f}  {y:.3f}  {z:.3f} 0.0000000000\n")
    return base_ids, ga_ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("aseg", type=Path)
    parser.add_argument("entowm", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    label_cortex_fix_ga(args.surface, args.aseg, args.entowm,
                        args.hemi, args.output)


if __name__ == "__main__":
    main()
