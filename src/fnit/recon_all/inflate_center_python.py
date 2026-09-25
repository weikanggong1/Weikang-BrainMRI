"""FreeSurfer's bounding-box centering used after surface inflation.

This only translates an already inflated surface. It does not perform
``MRISinflateBrain`` or the surrounding area rescaling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def center_vertices(vertices: np.ndarray) -> np.ndarray:
    """Match ``mrisComputeSurfaceDimensions`` + ``MRIScenter`` float rules."""
    xyz = np.asarray(vertices, dtype=np.float32)
    low = xyz.min(axis=0)
    high = xyz.max(axis=0)
    center = np.float32(0.5) * (low.astype(np.float64) + high.astype(np.float64)).astype(np.float32)
    return xyz - center


def center_surface(input_path: str | Path, output_path: str | Path) -> None:
    raw = Path(input_path).read_bytes()
    if raw[:3] != b"\xff\xff\xfe":
        raise ValueError("expected FreeSurfer triangular surface")
    start = raw.index(b"\n\n", 3) + 2
    nvertices = int.from_bytes(raw[start:start + 4], "big")
    nfaces = int.from_bytes(raw[start + 4:start + 8], "big")
    xyz_start = start + 8
    xyz_end = xyz_start + 12 * nvertices
    if len(raw) < xyz_end + 12 * nfaces:
        raise ValueError("truncated triangular surface")
    xyz = np.frombuffer(raw[xyz_start:xyz_end], dtype=">f4").reshape(-1, 3)
    centered = center_vertices(xyz)
    with open(output_path, "wb") as stream:
        stream.write(b"\xff\xff\xfecreated by fnit\n\n")
        stream.write(raw[start:xyz_start])
        stream.write(np.asarray(centered, dtype=">f4").tobytes())
        stream.write(raw[xyz_end:])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    center_surface(args.input, args.output)


if __name__ == "__main__":
    main()
