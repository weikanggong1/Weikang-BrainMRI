"""Reproduce the fixed recon-all ``mri_segstats --annot ... --snr`` tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np


def surface_snr_rows(annotation: str | Path, intensity: str | Path,
                     area: str | Path, hemi: str) -> list[str]:
    labels, _, names = fsio.read_annot(str(annotation))
    values = np.asarray(nib.load(str(intensity)).dataobj).reshape(-1)
    vertex_area = fsio.read_morph_data(str(area))
    if len(labels) != len(values) or len(labels) != len(vertex_area):
        raise ValueError("Annotation, intensity and area vertex counts differ")
    if hemi not in ("lh", "rh"):
        raise ValueError("Hemisphere must be lh or rh")
    base = 1000 if hemi == "lh" else 2000
    rows = []
    for index in sorted(set(labels)):
        if index >= len(names):
            raise ValueError(f"Annotation index {index} has no color-table name")
        mask = labels == index
        data = values[mask].astype(np.float64, copy=False)
        count = len(data)
        if not count:
            continue
        total = data.sum(dtype=np.float64)
        mean = np.float32(total / count)
        minimum, maximum = np.float32(data.min()), np.float32(data.max())
        first = np.float32(np.float32(count) * mean * mean)
        second = float(np.float32(2 * mean)) * total
        std = np.float32(np.sqrt((float(first) - second +
                                  np.square(data).sum(dtype=np.float64)) / (count - 1))) if count > 1 else np.float32(0)
        if std == 0:
            snr = "      -nan" if mean == 0 else f"{np.copysign(np.inf, mean):10.4f}"
        else:
            snr = f"{np.float32(mean / std):10.4f}"
        name = names[0 if index == -1 else index].decode()
        native_area = np.cumsum(vertex_area[mask], dtype=np.float32)[-1]
        rows.append(f"{len(rows)+1:3d} {base + max(index, 0):3d}  {count:8d} "
                    f"{native_area:10.1f}  {name:<30s} "
                    f"{mean:10.4f} {std:10.4f} {minimum:10.4f} {maximum:10.4f} "
                    f"{np.float32(maximum - minimum):10.4f} {snr} ")
    return rows


def write_surface_snr_stats(subject: str | Path, hemi: str,
                            output: str | Path) -> Path:
    subject, output = Path(subject), Path(output)
    annotation = subject / "label" / f"{hemi}.aparc.annot"
    intensity = subject / "surf" / f"{hemi}.w-g.pct.mgh"
    area = subject / "surf" / f"{hemi}.area"
    rows = surface_snr_rows(annotation, intensity, area, hemi)
    column_names = ("Index", "SegId", "NVertices", "Area_mm2", "StructName",
                    "Mean", "StdDev", "Min", "Max", "Range", "SNR")
    lines = ["# Title Segmentation Statistics ", "# generating_program fnit",
             "# anatomy_type surface", f"# subjectname {subject.name}",
             f"# Annot {subject.name} {hemi} aparc", f"# InVolFile {intensity}",
             "# Only reporting non-empty segmentations"]
    for column, name in enumerate(column_names, 1):
        lines.append(f"# TableCol {column:2d} ColHeader {name}")
    lines.extend((f"# NRows {len(rows)} ", "# NTableCols 11 ",
                  "# ColHeaders  Index SegId NVertices Area_mm2 StructName "
                  "Mean StdDev Min Max Range SNR  ", *rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("hemi", choices=("lh", "rh"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_surface_snr_stats(args.subject, args.hemi, args.output)


if __name__ == "__main__":
    main()
