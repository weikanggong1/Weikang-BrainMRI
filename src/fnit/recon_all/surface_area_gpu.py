"""Compute a FreeSurfer surface's per-vertex triangle area on CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch


@torch.inference_mode()
def vertex_area(vertices: np.ndarray, faces: np.ndarray,
                device: str = "cuda:0") -> np.ndarray:
    """Assign one third of each adjacent triangle's area to every vertex."""
    xyz = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
    triangles = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    first = xyz[triangles[:, 0]]
    cross = torch.cross(xyz[triangles[:, 1]] - first,
                        xyz[triangles[:, 2]] - first, dim=1)
    share = torch.linalg.vector_norm(cross, dim=1).mul_(0.5 / 3.0)
    area = torch.zeros(len(xyz), dtype=torch.float32, device=device)
    for corner in range(3):
        area.index_add_(0, triangles[:, corner], share)
    return area.cpu().numpy()


def area_map(surface: str | Path, output: str | Path,
             device: str = "cuda:0") -> None:
    vertices, faces = fsio.read_geometry(str(surface))
    fsio.write_morph_data(str(output), vertex_area(vertices, faces, device=device))


@torch.inference_mode()
def mid_area_map(white_area: str | Path, pial_area: str | Path,
                 output: str | Path, device: str = "cuda:0") -> None:
    """Match recon-all's `mris_calc add` followed by `mris_calc div 2`."""
    white = fsio.read_morph_data(str(white_area))
    pial = fsio.read_morph_data(str(pial_area))
    if white.shape != pial.shape:
        raise ValueError("white and pial area maps have different vertex counts")
    mean = (torch.as_tensor(np.asarray(white, dtype=np.float32), device=device) +
            torch.as_tensor(np.asarray(pial, dtype=np.float32), device=device)).mul_(0.5)
    fsio.write_morph_data(str(output), mean.cpu().numpy())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    area_map(args.surface, args.output, device=args.device)


if __name__ == "__main__":
    main()
