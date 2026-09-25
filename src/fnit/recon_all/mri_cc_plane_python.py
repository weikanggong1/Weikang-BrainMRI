"""The cutting-plane search used by FreeSurfer 8.2 ``mri_cc -aseg``.

This is an isolated first substage. It does not segment or label the callosum.
The source correspondence is ``mri_cc.cpp`` at FreeSurfer commit d932c45,
``find_cc_with_aseg`` and ``cc_cutting_plane_correct``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg, ndimage


def _midline_normal(aseg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match MRIprincipleComponents on voxels near both cortex labels."""
    near_left = ndimage.maximum_filter(aseg == 3, size=5, mode="nearest")
    near_right = ndimage.maximum_filter(aseg == 42, size=5, mode="nearest")
    points = np.argwhere(near_left & near_right)
    if len(points) == 0:
        raise ValueError("no voxels near both cortical hemispheres")
    # MRIprincipleComponents accumulates single-precision covariance in z,y,x
    # order. A vectorized float64 covariance changes the selected rotation.
    points = points[np.lexsort((points[:, 0], points[:, 1], points[:, 2]))]
    center = points.mean(axis=0)
    cov = np.zeros((3, 3), dtype=np.float32)
    for point in points:
        v = np.asarray(point - center, dtype=np.float32)
        cov += np.outer(v, v)
    cov *= np.float32(1.0 / len(points))
    _, vectors = linalg.eigh(cov, driver="evr")
    normal = vectors[:, 0].astype(np.float64)
    if normal[0] < 0:
        normal = -normal
    return center, normal


def _rotated_normal(yrot: float, zrot: float) -> np.ndarray:
    cy, sy = np.float32(math.cos(yrot)), np.float32(math.sin(yrot))
    cz, sz = np.float32(math.cos(zrot)), np.float32(math.sin(zrot))
    return np.array([float(np.float32(cy * cz)), float(-sz), float(np.float32(sy * cz))])


def _score(left: np.ndarray, right: np.ndarray, x0: float, y0: int, z0: int,
           normal: np.ndarray) -> int:
    center = np.array([x0, y0, z0])
    return int(np.count_nonzero((left - center) @ normal > 0.5)
               + np.count_nonzero((right - center) @ normal < -0.5))


def search_cutting_plane(aseg: np.ndarray) -> tuple[float, int, int, float, float, int]:
    """Return x/y/z center, y/z rotation in radians, and matched-voxel score."""
    if aseg.ndim != 3:
        raise ValueError("aseg must be a 3D volume")
    left_x = np.where((aseg == 2) | (aseg == 3))[0]
    right_x = np.where((aseg == 41) | (aseg == 42))[0]
    if len(left_x) == 0 or len(right_x) == 0:
        raise ValueError("aseg has no bilateral cerebral labels")
    xmin = min(int(left_x.min()), int(right_x.max())) - 1
    xmax = max(int(left_x.min()), int(right_x.max())) + 1
    region = aseg[max(0, xmin - 1):min(aseg.shape[0], xmax + 1)]
    x_offset = max(0, xmin - 1)
    left = np.argwhere((region >= 2) & (region <= 3))
    right = np.argwhere((region >= 41) & (region <= 42))
    left[:, 0] += x_offset
    right[:, 0] += x_offset
    center, normal = _midline_normal(aseg)
    x0 = float(center[0])
    y0, z0 = int(center[1]), int(center[2])
    zbest = -math.asin(float(normal[1]))
    ybest = math.asin(float(normal[2]) / math.cos(zbest))
    best = (x0, y0, z0, ybest, zbest,
            _score(left, right, x0, y0, z0, normal))
    delta, limit = math.radians(0.25), math.radians(7)
    zrot = zbest - limit
    while zrot <= zbest + limit:
        yrot = ybest - limit
        while yrot <= ybest + limit:
            n = _rotated_normal(yrot, zrot)
            # Sorting makes all integer slice scores available without a
            # per-voxel Python loop. Strict inequalities match the C++ code.
            lproj = np.sort(left @ n - y0*n[1] - z0*n[2])
            rproj = np.sort(right @ n - y0*n[1] - z0*n[2])
            for xi in range(xmin, xmax + 1):
                score = (len(left) - np.searchsorted(lproj, xi*n[0] + 0.5, side="right")
                         + np.searchsorted(rproj, xi*n[0] - 0.5, side="left"))
                if score > best[5]:
                    ybest, zbest = yrot, zrot
                    best = (float(xi), y0, z0, ybest, zbest, int(score))
            yrot += delta
        zrot += delta
    return best


def unadjusted_voxel_lta(plane: tuple[float, int, int, float, float, int]) -> np.ndarray:
    """Voxel-to-voxel rotation before the final best-slice x shift."""
    x0, y0, z0, yr, zr, _ = plane
    cy, sy = np.float32(math.cos(yr)), np.float32(math.sin(yr))
    cz, sz = np.float32(math.cos(zr)), np.float32(math.sin(zr))
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    a = rz @ ry
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = a
    out[:3, 3] = np.float32(128.0) - a @ np.asarray([x0, y0, z0], dtype=np.float32)
    return out
