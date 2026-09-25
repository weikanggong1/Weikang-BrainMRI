"""FreeSurfer ``mris_jacobian`` area ratios for an ordered surface pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch


def _area(xyz: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a, b, c = (xyz[faces[:, i]] for i in range(3))
    face_area = torch.linalg.vector_norm(torch.cross(b - a, c - a, dim=1), dim=1) * 0.5
    area = torch.zeros(len(xyz), dtype=torch.float32, device=xyz.device)
    share = face_area / 3.0
    for corner in range(3):
        area.index_add_(0, faces[:, corner], share)
    return area, face_area.sum(dtype=torch.float64)


@torch.inference_mode()
def jacobian_values(original: np.ndarray, mapped: np.ndarray, faces: np.ndarray,
                    *, device: str = "cuda:0") -> np.ndarray:
    """Return mapped/original vertex area, normalized by total surface area."""
    if original.shape != mapped.shape or original.ndim != 2 or original.shape[1] != 3:
        raise ValueError("surfaces must have the same (vertices, 3) shape")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (triangles, 3)")
    triangles = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    white = torch.as_tensor(np.asarray(original, dtype=np.float32), device=device)
    sphere = torch.as_tensor(np.asarray(mapped, dtype=np.float32), device=device)
    white_area, white_total = _area(white, triangles)
    sphere_area, sphere_total = _area(sphere, triangles)
    scale = (sphere_total / white_total).to(torch.float32)
    values = sphere_area / (white_area.clamp_min_(1e-5) * scale)
    return values.cpu().numpy()


def jacobian_map(original: str | Path, mapped: str | Path, output: str | Path,
                 *, device: str = "cuda:0") -> None:
    original_xyz, faces = fsio.read_geometry(str(original))
    mapped_xyz, mapped_faces = fsio.read_geometry(str(mapped))
    if not np.array_equal(faces, mapped_faces):
        raise ValueError("surfaces must have identical ordered faces")
    fsio.write_morph_data(str(output), jacobian_values(original_xyz, mapped_xyz,
                                                       faces, device=device))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("mapped", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    jacobian_map(args.original, args.mapped, args.output, device=args.device)


if __name__ == "__main__":
    main()
