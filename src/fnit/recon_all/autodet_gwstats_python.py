"""FreeSurfer 8.2 gray/white border statistics from MRI and original surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_geometry
from numba import njit
from scipy import ndimage

from .place_surface_border import _sample, _voxel
from .place_surface_geometry import surface_ras_to_voxel
from .place_surface_normals import initial_vertex_normals


@njit(cache=True)
def _surface_modes(volume: np.ndarray, xyz: np.ndarray, normals: np.ndarray, affine: np.ndarray) -> tuple[int, int]:
    white = np.zeros(256, dtype=np.int32)
    gray = np.zeros(256, dtype=np.int32)
    for vertex in range(len(xyz)):
        x, y, z = float(xyz[vertex, 0]), float(xyz[vertex, 1]), float(xyz[vertex, 2])
        nx, ny, nz = float(normals[vertex, 0]), float(normals[vertex, 1]), float(normals[vertex, 2])
        for direction, histogram in ((-1.0, white), (1.0, gray)):
            i, j, k = _voxel(affine, x + direction * nx, y + direction * ny, z + direction * nz)
            value = _sample(volume, i, j, k)
            bin_index = int(np.floor(value + 0.5))
            if 0 <= bin_index < 256:
                histogram[bin_index] += 1
    white[0] = gray[0] = 0
    return int(np.argmax(white)), int(np.argmax(gray))


def _class_sigmas(brain: np.ndarray, wm: np.ndarray) -> tuple[np.float32, np.float32]:
    white = wm >= 5
    neighborhood_white = ndimage.maximum_filter(white, size=3, mode="nearest")
    neighborhood_nonwhite = ndimage.maximum_filter(~white, size=3, mode="nearest")
    clipped = brain.copy()
    clipped[white & (clipped > 110)] = 110
    white_values = clipped[white & neighborhood_nonwhite & (clipped >= 70)].astype(np.float64)
    gray_values = clipped[~white & neighborhood_white & (clipped >= 30) & (clipped <= 110)].astype(np.float64)
    white_std = np.float32(np.sqrt(np.mean(white_values * white_values) - np.mean(white_values) ** 2))
    gray_std = np.float32(np.sqrt(np.mean(gray_values * gray_values) - np.mean(gray_values) ** 2))
    return white_std, gray_std


def compute_autodet_stats(
    brain: np.ndarray,
    wm: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    volume_header,
    surface_metadata,
    *,
    hemisphere: str,
) -> dict[str, float | int]:
    """Compute the default 40 fields emitted by ``mris_autodet_gwstats``."""
    if hemisphere not in ("lh", "rh"):
        raise ValueError("hemisphere must be lh or rh")
    brain = np.asarray(brain, dtype=np.uint8)
    wm = np.asarray(wm, dtype=np.uint8)
    if brain.shape != wm.shape or brain.ndim != 3:
        raise ValueError("brain and wm must be matching 3D volumes")
    white_std, gray_std = _class_sigmas(brain, wm)
    clipped = brain.copy()
    clipped[(wm >= 5) & (clipped > 110)] = 110
    xyz = np.asarray(vertices, np.float32)
    normals = initial_vertex_normals(xyz, np.asarray(faces, np.int32))
    affine = surface_ras_to_voxel(volume_header, surface_metadata)
    white_mode, gray_mode = _surface_modes(clipped, xyz, normals, affine)
    white_mean, gray_mean = np.float32(white_mode), np.float32(gray_mode)
    min_gray_at_white_border = np.float32(gray_mean - gray_std)
    max_border_white = np.float32(white_mean + white_std)
    max_csf = np.float32(float(gray_mean) - 2.0 * float(gray_std))
    min_border_white = gray_mean
    max_gray = np.float32(white_mean - white_std)
    max_gray_at_csf_border = np.float32(gray_mean - gray_std)
    min_gray_at_csf_border = np.float32(gray_mean - np.float32(3.0 * gray_std))
    mid_gray = np.float32(np.float32(max_gray + min_gray_at_csf_border) / 2.0)
    pial_outside_hi = np.float32(np.float32(max_csf + max_gray_at_csf_border) / 2.0)
    return {
        "hemicode": 1 if hemisphere == "lh" else 2,
        "white_border_hi": float(max_border_white),
        "white_border_low": float(min_border_white),
        "white_outside_low": float(min_gray_at_white_border),
        "white_inside_hi": 120.0,
        "white_outside_hi": float(max_border_white),
        "pial_border_hi": float(max_gray_at_csf_border),
        "pial_border_low": float(min_gray_at_csf_border),
        "pial_outside_low": 10.0,
        "pial_inside_hi": float(max_gray),
        "pial_outside_hi": float(pial_outside_hi),
        "use_mode": 1,
        "variablesigma": 3.0,
        "std_scale": 1.0,
        "adWHITE_MATTER_MEAN": 110.0,
        "MAX_WHITE": 120.0,
        "MIN_BORDER_WHITE": 85.0,
        "MAX_BORDER_WHITE": 105.0,
        "MAX_GRAY": 95.0,
        "MID_GRAY": float(mid_gray),
        "MIN_GRAY_AT_CSF_BORDER": 40.0,
        "MAX_GRAY_AT_CSF_BORDER": 75.0,
        "MIN_CSF": 10.0,
        "adMAX_CSF": 40.0,
        "white_mean": float(white_mean),
        "white_mode": float(white_mode),
        "white_std": float(white_std),
        "gray_mean": float(gray_mean),
        "gray_mode": float(gray_mode),
        "gray_std": float(gray_std),
        "min_border_white": float(min_border_white),
        "max_border_white": float(max_border_white),
        "min_gray_at_white_border": float(min_gray_at_white_border),
        "max_gray": float(max_gray),
        "min_gray_at_csf_border": float(min_gray_at_csf_border),
        "max_gray_at_csf_border": float(max_gray_at_csf_border),
        "min_csf": 10.0,
        "max_csf": float(max_csf),
        "max_gray_scale": 0.0,
        "max_scale_down": 0.2,
    }


def format_autodet_stats(stats: dict[str, float | int]) -> str:
    """Serialize in the exact ordered FreeSurfer 8.2 text layout."""
    lines = []
    for name, value in stats.items():
        if name == "hemicode" or name in ("white_border_hi", "white_border_low", "white_outside_low", "white_inside_hi", "white_outside_hi"):
            label = f"{name:<19}"
        elif name.startswith("pial_"):
            label = f"{name:<18}"
        else:
            label = f"{name} "
        text = str(value) if name in ("hemicode", "use_mode") else f"{value:.6f}"
        lines.append(label + text)
    return "\n".join(lines) + "\n"


def write_autodet_stats(brain_path: Path, wm_path: Path, surface_path: Path, output_path: Path, hemisphere: str) -> dict[str, float | int]:
    image = nib.load(brain_path)
    wm_image = nib.load(wm_path)
    vertices, faces, metadata = read_geometry(surface_path, read_metadata=True)
    stats = compute_autodet_stats(
        np.asarray(image.dataobj), np.asarray(wm_image.dataobj), vertices, faces, image.header, metadata,
        hemisphere=hemisphere,
    )
    output_path.write_text(format_autodet_stats(stats))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", type=Path, required=True)
    parser.add_argument("--wm", type=Path, required=True)
    parser.add_argument("--surf", type=Path, required=True)
    parser.add_argument("--o", type=Path, required=True)
    args = parser.parse_args()
    hemisphere = args.surf.name[:2]
    write_autodet_stats(args.i, args.wm, args.surf, args.o, hemisphere)


if __name__ == "__main__":
    main()
