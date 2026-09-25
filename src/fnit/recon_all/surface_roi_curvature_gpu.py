"""FreeSurfer anatomical-stats curvature columns on an existing mesh."""

from __future__ import annotations

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from .surface_curvature_gpu import _neighbours
from .surface_thickness_gpu import _normals


@torch.inference_mode()
def principal_curvatures(vertices: np.ndarray, faces: np.ndarray,
                         device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray]:
    """Fit the two-hop quadratic used by MRIScomputeSecondFundamentalForm."""
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    xyz = torch.as_tensor(np.asarray(vertices, dtype=np.float32), device=device)
    tri = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    normal = _normals(xyz, tri)
    _, _, neighbours, mask = _neighbours(faces, len(xyz))
    neighbours = torch.as_tensor(neighbours, device=device)
    mask = torch.as_tensor(mask, device=device)
    shifted = torch.stack((normal[:, 1], normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.cross(normal, shifted, dim=1)
    alternate = torch.stack((normal[:, 1], -normal[:, 2], normal[:, 0]), dim=1)
    e1 = torch.where(torch.linalg.vector_norm(e1, dim=1)[:, None] < .001,
                     torch.cross(normal, alternate, dim=1), e1)
    e1 = torch.nn.functional.normalize(e1, dim=1)
    e2 = torch.nn.functional.normalize(torch.cross(normal, e1, dim=1), dim=1)
    principal = torch.empty((len(xyz), 2), dtype=torch.float32, device=device)
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        delta = xyz[neighbours[start:stop]] - xyz[start:stop, None, :]
        u = (delta * e1[start:stop, None, :]).sum(dim=2)
        v = (delta * e2[start:stop, None, :]).sum(dim=2)
        z = (delta * normal[start:stop, None, :]).sum(dim=2)
        rsq = u * u + v * v
        valid = mask[start:stop] & (rsq > 1e-12)
        design = torch.stack((u * u, 2 * u * v, v * v), dim=2) * valid[:, :, None]
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ (z * valid)[:, :, None]
        left, singular, right = torch.linalg.svd(gram)
        inverse = torch.where(singular >= 1e-4 * singular[:, :1],
                              singular.clamp_min(1e-30).reciprocal(), 0)
        coefficient = (right.transpose(1, 2) @
                       (inverse[:, :, None] * (left.transpose(1, 2) @ rhs)))[:, :, 0]
        a, b, c = (coefficient[:, i] for i in range(3))
        root = torch.sqrt((a - c).square() + 4 * b.square())
        high, low = a + c + root, a + c - root
        fitted = torch.where((high.abs() >= low.abs())[:, None],
                             torch.stack((high, low), dim=1),
                             torch.stack((low, high), dim=1))
        k = torch.where(valid, z / rsq.clamp_min(1e-30), 0)
        largest = k.masked_fill(~valid, -torch.inf).max(dim=1).values
        smallest = k.masked_fill(~valid, torch.inf).min(dim=1).values
        condition = singular[:, 0] / singular[:, -1].clamp_min(1e-30)
        fallback = torch.stack((largest, smallest), dim=1)
        principal[start:stop] = torch.where((condition >= 500000)[:, None],
                                            fallback, fitted)
    return principal[:, 0].cpu().numpy(), principal[:, 1].cpu().numpy()


def curvature_columns(surface: Path, area_map: Path, annotation: Path,
                      cortex_label: Path | None, device: str = "cuda:0") -> dict[str, tuple[float, ...]]:
    xyz, faces = fsio.read_geometry(str(surface))
    area = fsio.read_morph_data(str(area_map)).astype(np.float64)
    labels, _, names = fsio.read_annot(str(annotation))
    cortex = np.ones(len(xyz), dtype=bool) if cortex_label is None else np.zeros(len(xyz), dtype=bool)
    if cortex_label is not None:
        cortex[fsio.read_label(str(cortex_label))] = True
    k1, k2 = principal_curvatures(xyz, faces, device)
    k1, k2 = k1.astype(np.float64), k2.astype(np.float64)
    mean = np.abs((k1 + k2) / 2) * area
    gauss = np.abs(k1 * k2) * area
    fold = area * np.abs(k1) * (np.abs(k1) - np.abs(k2)) / (4 * np.pi)
    intrinsic = area * np.maximum(k1 * k2, 0) / (4 * np.pi)
    result = {}
    for index, name in enumerate(names):
        region = (labels == index) & cortex
        if not region.any() or name.decode() in {"corpuscallosum", "unknown", "Unknown", "Medial_wall"}:
            continue
        result[name.decode()] = (float(mean[region].sum() / region.sum()),
                                 float(gauss[region].sum() / region.sum()),
                                 float(fold[region].sum()),
                                 float(intrinsic[region].sum()))
    return result
