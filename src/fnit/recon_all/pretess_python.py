"""Source-ordered topology repair for FreeSurfer 8.2 ``mri_pretess``.

This stage is intentionally NumPy/CPU: the native algorithm edits voxels
in scan order, and later tests in the same pass see those edits.
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import nibabel as nib
import numpy as np

from .mgh_compat import save_same_dtype_mgh


_ZERO = (0, 0, 0)
# Offsets are (z, y, x); FreeSurfer scans x fastest, then y, then z.
_EDGE_AXES = (
    ((0, 1, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, -1)),
    ((0, 1, 0), (1, 0, 0)), ((0, 1, 0), (-1, 0, 0)),
    ((0, 0, 1), (1, 0, 0)), ((0, 0, -1), (1, 0, 0)),
)
_CORNER_SIGNS = ((1, 1), (-1, 1), (-1, -1), (1, -1))  # (dz, dy), dx=+1


def _add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(x + y for x, y in zip(a, b))


def _corners(dz: int, dy: int) -> tuple[tuple[int, int, int], ...]:
    x, y, z = (0, 0, 1), (0, dy, 0), (dz, 0, 0)
    return (_ZERO, _add(_add(x, y), z), x, y, z,
            _add(x, y), _add(x, z), _add(y, z))


def _origin_limits(shape: tuple[int, ...], offsets: tuple[tuple[int, ...], ...]):
    lower = tuple(max(0, -min(o[axis] for o in offsets)) for axis in range(3))
    upper = tuple(shape[axis] - max(o[axis] for o in offsets) for axis in range(3))
    return lower, upper


def _initial_candidates(mask: np.ndarray, offsets: tuple[tuple[int, ...], ...],
                        kind: str) -> np.ndarray:
    lower, upper = _origin_limits(mask.shape, offsets)

    def view(offset: tuple[int, ...]) -> np.ndarray:
        return mask[tuple(slice(lower[d] + offset[d], upper[d] + offset[d])
                          for d in range(3))]

    if kind == "edge":
        valid = view(offsets[0]) & view(offsets[1]) & ~view(offsets[2]) & ~view(offsets[3])
    elif kind == "foreground":
        a, b, x, y, z, xy, xz, yz = (view(o) for o in offsets)
        valid = a & b & ~((x & (xy | xz)) | (y & (yz | xy)) | (z & (xz | yz)))
    else:
        a, b, x, y, z, xy, xz, yz = (view(o) for o in offsets)
        valid = ~a & ~b & x & y & z & xy & xz & yz
    return np.argwhere(valid) + lower


def _run_pass(mask: np.ndarray, intensity: np.ndarray,
              offsets: tuple[tuple[int, ...], ...], kind: str) -> int:
    lower, upper = _origin_limits(mask.shape, offsets)
    depth, height, width = mask.shape
    total = 0
    while True:
        candidates = _initial_candidates(mask, offsets, kind)
        pending = [int((z * height + y) * width + x) for z, y, x in candidates]
        heapq.heapify(pending)
        queued = set(pending)
        changed = 0
        while pending:
            key = heapq.heappop(pending)
            queued.discard(key)
            z, rem = divmod(key, height * width)
            y, x = divmod(rem, width)
            origin = (z, y, x)
            points = [tuple(origin[d] + o[d] for d in range(3)) for o in offsets]
            if kind == "edge":
                a, b, c, d = points
                if not (mask[a] and mask[b] and not mask[c] and not mask[d]):
                    continue
                edits = [c if intensity[c] > intensity[d] else d]
            elif kind == "foreground":
                a, b, vx, vy, vz, vxy, vxz, vyz = points
                if not (mask[a] and mask[b]):
                    continue
                paths = ((vx, vxy), (vx, vxz), (vy, vyz),
                         (vy, vxy), (vz, vxz), (vz, vyz))
                if any(mask[first] and mask[second] for first, second in paths):
                    continue
                values = [int(intensity[first]) + int(intensity[second])
                          for first, second in paths]
                edits = paths[int(np.argmax(values))]
            else:
                a, b, *middle = points
                if mask[a] or mask[b] or not all(mask[point] for point in middle):
                    continue
                edits = [a if intensity[a] > intensity[b] else b]

            for point in edits:
                if mask[point]:
                    continue
                mask[point] = True
                changed += 1
                # A new configuration later in this scan is seen immediately;
                # an earlier one waits for the next pass, as in the C++ loops.
                for offset in offsets:
                    candidate = tuple(point[d] - offset[d] for d in range(3))
                    if not all(lower[d] <= candidate[d] < upper[d] for d in range(3)):
                        continue
                    next_key = (candidate[0] * height + candidate[1]) * width + candidate[2]
                    if next_key > key and next_key not in queued:
                        heapq.heappush(pending, next_key)
                        queued.add(next_key)
        total += changed
        if changed == 0:
            return total


def pretess_values(segmentation_xyz: np.ndarray, intensity_xyz: np.ndarray,
                   label: int | str) -> tuple[np.ndarray, int]:
    """Return repaired 3D UCHAR values and count of topology edits."""
    if (segmentation_xyz.shape != intensity_xyz.shape or
            segmentation_xyz.ndim != 3 or segmentation_xyz.dtype != np.uint8):
        raise ValueError("pretess inputs must be equally shaped 3D UCHAR volumes")
    original = np.ascontiguousarray(segmentation_xyz.transpose(2, 1, 0))
    intensity = np.ascontiguousarray(intensity_xyz.transpose(2, 1, 0))
    wm = label == "wm"
    target = original >= 5 if wm else original == int(label)
    changes = 0
    for _ in range(100):
        iteration = 0
        for first, second in _EDGE_AXES:
            offsets = (_ZERO, _add(first, second), first, second)
            iteration += _run_pass(target, intensity, offsets, "edge")
        for dz, dy in _CORNER_SIGNS:
            offsets = _corners(dz, dy)
            iteration += _run_pass(target, intensity, offsets, "foreground")
        for dz, dy in _CORNER_SIGNS:
            offsets = _corners(dz, dy)
            iteration += _run_pass(target, intensity, offsets, "background")
        changes += iteration
        if iteration == 0:
            break
    result = original.copy()
    if wm:
        result[target & (original < 5)] = 215  # PRETESS_FILL
    else:
        result[target] = int(label)
    return result.transpose(2, 1, 0), changes


def pretess_mgh(segmentation: str | Path, label: int | str,
                intensity: str | Path, output: str | Path) -> int:
    source = nib.load(str(segmentation))
    norm = nib.load(str(intensity))
    result, changes = pretess_values(np.asanyarray(source.dataobj),
                                     np.asanyarray(norm.dataobj), label)
    save_same_dtype_mgh(segmentation, output, result)
    return changes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segmentation", type=Path)
    parser.add_argument("label")
    parser.add_argument("intensity", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(f"topology edits: {pretess_mgh(args.segmentation, args.label, args.intensity, args.output)}")


if __name__ == "__main__":
    main()
