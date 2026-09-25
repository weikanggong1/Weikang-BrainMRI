"""Fixed ``mris_sphere -q`` preprocessing and radial projection.

This implements the source operations surrounding ``MRISinflateToSphere``.
The 300 inflation updates and inherited momentum match the fixed native
LH diagnostic replay. The later nonlinear area optimizer is implemented in
``sphere_quick_python`` and remains separately validated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .inflate_center_python import center_vertices
from .inflate_python import add_spring, face_area_total, matrix, neighbor_spring, vertex_normals
from .smooth_surface_python import ordered_neighbors


RADIUS = np.float32(100)


def scale_about_bbox(vertices: np.ndarray, scale: np.float32) -> np.ndarray:
    """Translate ``MRISscaleBrain`` float32 vertex coordinate arithmetic."""
    xyz = np.asarray(vertices, dtype=np.float32)
    if scale == np.float32(1):
        return xyz.copy()
    low = xyz.min(axis=0)
    high = xyz.max(axis=0)
    center = np.float32(0.5) * (low.astype(np.float64) + high.astype(np.float64)).astype(np.float32)
    return (xyz - center) * scale + center


def project_radially(vertices: np.ndarray, radius: float = 100.0,
                     already_sphere: bool = False) -> np.ndarray:
    """Translate ``MRISprojectOntoSphereWkr`` including its status check."""
    xyz = np.asarray(vertices, np.float32) if already_sphere else center_vertices(vertices)
    wide = xyz.astype(np.float64)
    distance = np.sqrt(np.sum(wide * wide, axis=1))
    ratio = np.divide(radius, distance, out=np.zeros_like(distance), where=distance > 0)
    displacement = (1.0 - ratio)[:, None] * wide
    return (wide - displacement).astype(np.float32)


def initial_scale(vertices: np.ndarray) -> np.ndarray:
    """Apply the fixed input size cap preceding spherical inflation."""
    xyz = np.asarray(vertices, dtype=np.float32)
    # Pinned 8.2 C++ uses stdlib abs(int) on each float bbox dimension.  The
    # implicit conversion truncates toward zero (215.832 mm becomes 215 mm).
    dimension = np.float32(np.max(np.trunc(xyz.max(axis=0) - xyz.min(axis=0))))
    if dimension > np.float32(0.75) * RADIUS:
        scale = np.float32((np.float32(0.75) * RADIUS) / dimension)
        xyz = scale_about_bbox(xyz, scale)
    return xyz


def project_before_quick_sphere(vertices: np.ndarray) -> np.ndarray:
    """Evaluate the ``-q -in 0`` path up to the main radial projection."""
    xyz = initial_scale(vertices)
    xyz = center_vertices(xyz)
    xyz = center_vertices(xyz)
    xyz = scale_about_bbox(xyz, np.float32(0.5))
    xyz = project_radially(xyz)
    return xyz


def inflate_before_quick_sphere(vertices: np.ndarray, faces: np.ndarray,
                                iterations: int = 300,
                                return_momentum: bool = False
                                ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run the fixed sphere inflation loop through its pre-optimization projection.

    This follows the pinned `MRISinflateToSphere` sphere, convexity, normalized
    spring, and momentum updates. This fixed-count path was checked on the
    frozen subject, whose radial error does not trigger native early stopping.
    It deliberately stops before `MRISquickSphere`. The optimizer needs the
    per-vertex momentum carried from this loop, so the integrated stage requests
    that state with ``return_momentum=True``.
    """
    xyz = center_vertices(center_vertices(initial_scale(vertices)))
    faces = np.asarray(faces, np.int32)
    one, degree = matrix(ordered_neighbors(faces, len(xyz)))
    original_area = face_area_total(xyz, faces)
    current_area = original_area
    normals = vertex_normals(xyz, faces)
    previous = np.zeros_like(xyz)
    for _ in range(iterations):
        center = np.float32(0.5) * (xyz.min(axis=0).astype(np.float64)
                                    + xyz.max(axis=0).astype(np.float64)).astype(np.float32)
        centered = xyz - center
        length = np.sqrt(np.sum(centered * centered, axis=1, dtype=np.float32))
        unit = np.divide(centered, length[:, None], out=np.zeros_like(xyz),
                         where=length[:, None] > 0)
        radius_error = (np.float32(200) - length) / np.float32(200)
        # Native l_sphere stores 0.025f in a double, then evaluates r*l_sphere*x
        # in double before assigning the result to each float32 vertex gradient.
        gradient = (radius_error[:, None].astype(np.float64)
                    * float(np.float32(0.025))
                    * unit.astype(np.float64)).astype(np.float32)

        spring = neighbor_spring(xyz, one, degree)
        spring /= degree[:, None].astype(np.float32)
        projection = np.sum(spring * normals, axis=1, dtype=np.float32)
        gradient += np.maximum(projection, np.float32(0))[:, None] * normals
        gradient = add_spring(gradient, xyz, normals, one, degree,
                              original_area, current_area)

        previous = (gradient.astype(np.float64) * float(np.float32(0.9))
                    + (np.float32(0.9) * previous).astype(np.float64)).astype(np.float32)
        magnitude = np.sqrt(np.sum(previous.astype(np.float64) ** 2, axis=1))
        large = magnitude > 1
        previous[large] = (previous[large].astype(np.float64)
                           / magnitude[large, None]).astype(np.float32)
        xyz += previous
        normals = vertex_normals(xyz, faces)
        current_area = face_area_total(xyz, faces)

    xyz = scale_about_bbox(xyz, np.float32(0.5))
    projected = project_radially(xyz)
    return (projected, previous) if return_momentum else projected


def project_surface_before_quick_sphere(input_path: str | Path, output_path: str | Path) -> None:
    """Write the pre-optimization ``after`` diagnostic as a triangle surface."""
    raw = Path(input_path).read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError("expected FreeSurfer triangular surface")
    start = raw.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(raw[start:start + 4], "big")
    nfaces = int.from_bytes(raw[start + 4:start + 8], "big")
    xyz_start = start + 8
    xyz_end = xyz_start + 12 * nvertices
    faces_end = xyz_end + 12 * nfaces
    if len(raw) < faces_end:
        raise ValueError("truncated triangular surface")
    xyz = np.frombuffer(raw[xyz_start:xyz_end], dtype=">f4").reshape(-1, 3)
    projected = project_before_quick_sphere(xyz)
    with open(output_path, "wb") as stream:
        stream.write(b"\xff\xff\xfecreated by fnit\n\n")
        stream.write(raw[start:xyz_start])
        stream.write(np.asarray(projected, dtype=">f4").tobytes())
        stream.write(raw[xyz_end:])
