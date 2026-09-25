"""Compute FreeSurfer's two-neighbour, ten-average mean curvature map."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import nibabel.freesurfer as fs
import numpy as np
from scipy.sparse import coo_matrix
import torch

from .surface_thickness_gpu import _normals


def _neighbours(faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.concatenate((edges, edges[:, ::-1]))
    adjacency = coo_matrix((np.ones(len(edges), dtype=np.int8),
                            (edges[:, 0], edges[:, 1])), shape=(size, size)).tocsr()
    adjacency.data[:] = 1
    second = adjacency + adjacency @ adjacency
    second.data[:] = 1
    second.setdiag(0)
    second.eliminate_zeros()

    def padded(matrix):
        counts = np.diff(matrix.indptr).astype(np.int32)
        indices = np.zeros((size, counts.max()), dtype=np.int64)
        valid = np.zeros(indices.shape, dtype=bool)
        rows = np.repeat(np.arange(size), counts)
        columns = np.arange(len(matrix.indices)) - np.repeat(matrix.indptr[:-1], counts)
        indices[rows, columns] = matrix.indices
        valid[rows, columns] = True
        return indices, valid

    first_indices, first_valid = padded(adjacency)
    second_indices, second_valid = padded(second)
    return first_indices, first_valid, second_indices, second_valid


@torch.inference_mode()
def curvature_values(vertices: np.ndarray, faces: np.ndarray,
                     *, device: str = "cuda:0") -> np.ndarray:
    xyz = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
    triangles = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    normal = _normals(xyz, triangles)
    first_np, first_mask_np, second_np, second_mask_np = _neighbours(faces, len(xyz))
    first = torch.as_tensor(first_np, device=device)
    first_mask = torch.as_tensor(first_mask_np, device=device)
    second = torch.as_tensor(second_np, device=device)
    second_mask = torch.as_tensor(second_mask_np, device=device)

    shifted = torch.stack((normal[:, 1], normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.cross(normal, shifted, dim=1)
    alternate = torch.stack((normal[:, 1], -normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.where(torch.linalg.vector_norm(e1, dim=1)[:, None] < .001,
                     torch.cross(normal, alternate, dim=1), e1)
    e1 = torch.nn.functional.normalize(e1, dim=1)
    e2 = torch.nn.functional.normalize(torch.cross(normal, e1, dim=1), dim=1)

    mean = torch.empty(len(xyz), dtype=torch.float32, device=device)
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        index = second[start:stop]
        delta = xyz[index] - xyz[start:stop, None, :]
        u = (delta * e1[start:stop, None, :]).sum(dim=2)
        v = (delta * e2[start:stop, None, :]).sum(dim=2)
        z = (delta * normal[start:stop, None, :]).sum(dim=2)
        rsq = u * u + v * v
        valid = second_mask[start:stop] & (rsq > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        heights = z * valid
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ heights[:, :, None]
        left, singular, right = torch.linalg.svd(gram)
        inverse = torch.where(singular >= 1e-4 * singular[:, :1],
                              singular.clamp_min(1e-30).reciprocal(), 0)
        coefficient = right.transpose(1, 2) @ (inverse[:, :, None] *
                                                 (left.transpose(1, 2) @ rhs))
        fitted = coefficient[:, 0, 0] + coefficient[:, 2, 0]
        k = torch.where(valid, z / rsq.clamp_min(1e-30), 0)
        largest = k.masked_fill(~valid, -torch.inf).max(dim=1).values
        smallest = k.masked_fill(~valid, torch.inf).min(dim=1).values
        fallback = (largest + smallest) / 2
        condition = singular[:, 0] / singular[:, -1].clamp_min(1e-30)
        mean[start:stop] = torch.where(condition >= 500000, fallback, fitted)

    degree = first_mask.sum(dim=1).to(torch.float32) + 1
    for _ in range(10):
        mean = (mean + (mean[first] * first_mask).sum(dim=1)) / degree
    return mean.cpu().numpy()


def curvature_map(surface: str | Path, output: str | Path,
                  *, device: str = "cuda:0") -> dict:
    start = time.perf_counter()
    vertices, faces = fs.read_geometry(str(surface))
    result = curvature_values(vertices, faces, device=device)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fs.write_morph_data(str(out), result)
    return {"vertices": len(vertices), "device": device,
            "total_seconds": time.perf_counter() - start}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface")
    parser.add_argument("output")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    print(curvature_map(args.surface, args.output, device=args.device))


if __name__ == "__main__":
    main()
