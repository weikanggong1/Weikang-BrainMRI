"""Fixed fsaverage-to-subject ``mri_label2label --regmethod surface`` call."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.spatial import cKDTree


def _scaled_sphere(path: str | Path) -> np.ndarray:
    xyz, _ = fsio.read_geometry(str(path))
    center = (xyz.min(axis=0).astype(np.float64)
              + xyz.max(axis=0).astype(np.float64)) / 2
    radius = np.linalg.norm(xyz.astype(np.float64) - center, axis=1).mean()
    return xyz * np.float32(100.0 / radius)


class SurfaceLabelMapper:
    """Reuse the bilateral sphere nearest-vertex maps across BA/FG labels."""

    def __init__(self, source_sphere: str | Path, target_sphere: str | Path,
                 target_white: str | Path, target_subject: str):
        source = _scaled_sphere(source_sphere)
        target = _scaled_sphere(target_sphere)
        self.white, _ = fsio.read_geometry(str(target_white))
        if len(self.white) != len(target):
            raise ValueError("target white and registration sphere vertex counts differ")
        self.source_count = len(source)
        self.forward = cKDTree(target).query(source)[1].astype(np.int32)
        self.reverse = cKDTree(source).query(target)[1].astype(np.int32)
        self.target_subject = target_subject

    def map_label(self, source_label: str | Path,
                  output_label: str | Path) -> np.ndarray:
        """Write the ordered target label and return its vertex IDs."""
        points = np.loadtxt(source_label, skiprows=2, ndmin=2)
        source_ids = points[:, 0].astype(np.int32)
        stats = points[:, 4].astype(np.float32)
        if np.any((source_ids < 0) | (source_ids >= self.source_count)):
            raise ValueError("source label has an invalid vertex ID")

        mapped = self.forward[source_ids]
        _, first = np.unique(mapped, return_index=True)
        first.sort()
        forward_ids = mapped[first]
        forward_stats = stats[first]
        already = np.zeros(len(self.white), bool)
        already[forward_ids] = True
        source_row = np.full(self.source_count, -1, np.int32)
        for row, vertex in enumerate(source_ids):
            if source_row[vertex] < 0:
                source_row[vertex] = row
        reverse_ids = np.flatnonzero(~already & (source_row[self.reverse] >= 0))
        reverse_stats = stats[source_row[self.reverse[reverse_ids]]]
        target_ids = np.concatenate((forward_ids, reverse_ids))
        target_stats = np.concatenate((forward_stats, reverse_stats))

        with Path(output_label).open("w") as stream:
            stream.write(f"#!ascii label  , from subject {self.target_subject} vox2ras=TkReg\n")
            stream.write(f"{len(target_ids)}\n")
            for vertex, stat in zip(target_ids, target_stats):
                x, y, z = self.white[vertex]
                stream.write(f"{vertex}  {x:.3f}  {y:.3f}  {z:.3f} {stat:.10f}\n")
        return target_ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_sphere", type=Path)
    parser.add_argument("target_sphere", type=Path)
    parser.add_argument("target_white", type=Path)
    parser.add_argument("target_subject")
    parser.add_argument("source_label", type=Path)
    parser.add_argument("output_label", type=Path)
    args = parser.parse_args(argv)
    mapper = SurfaceLabelMapper(args.source_sphere, args.target_sphere,
                                args.target_white, args.target_subject)
    mapper.map_label(args.source_label, args.output_label)


if __name__ == "__main__":
    main()
