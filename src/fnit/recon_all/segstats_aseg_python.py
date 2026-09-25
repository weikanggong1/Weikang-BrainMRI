"""The fixed recon-all ``mri_segstats`` call for ``aseg.stats``."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from .anatomical_stats_global import read_brain_volume_stats
from .estimated_tiv import estimate_tiv
from .segstats_wmparc_python import _lut, _statistics


_MEASURES = (
    ("BrainSeg", "BrainSegVol", "Brain Segmentation Volume"),
    ("BrainSegNotVent", "BrainSegVolNotVent", "Brain Segmentation Volume Without Ventricles"),
    ("VentricleChoroidVol", "VentricleChoroidVol", "Volume of ventricles and choroid plexus"),
    ("lhCortex", "lhCortexVol", "Left hemisphere cortical gray matter volume"),
    ("rhCortex", "rhCortexVol", "Right hemisphere cortical gray matter volume"),
    ("Cortex", "CortexVol", "Total cortical gray matter volume"),
    ("lhCerebralWhiteMatter", "lhCerebralWhiteMatterVol", "Left hemisphere cerebral white matter volume"),
    ("rhCerebralWhiteMatter", "rhCerebralWhiteMatterVol", "Right hemisphere cerebral white matter volume"),
    ("CerebralWhiteMatter", "CerebralWhiteMatterVol", "Total cerebral white matter volume"),
    ("SubCortGray", "SubCortGrayVol", "Subcortical gray matter volume"),
    ("TotalGray", "TotalGrayVol", "Total gray matter volume"),
    ("SupraTentorial", "SupraTentorialVol", "Supratentorial volume"),
    ("SupraTentorialNotVent", "SupraTentorialVolNotVent", "Supratentorial volume"),
    ("Mask", "MaskVol", "Mask Volume"),
)


def _surface_holes(path: Path) -> int:
    vertices, faces = fsio.read_geometry(str(path))
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]])), axis=1)
    euler = len(vertices) - len(np.unique(edges, axis=0)) + len(faces)
    return 1 - euler // 2


def write_aseg_stats(subject: str | Path, aseg_lut: str | Path,
                     output: str | Path, *, brainvol_stats: str | Path | None = None,
                     stiv: str | Path | None = None) -> Path:
    """Write the numeric and table fields of the fixed ``aseg.stats`` call."""
    subject, aseg_lut, output = Path(subject), Path(aseg_lut), Path(output)
    image = nib.load(str(subject / "mri" / "aseg.mgz"))
    seg = np.asarray(image.dataobj).astype(np.int32, copy=False)
    intensity = np.asarray(nib.load(str(subject / "mri" / "norm.mgz")).dataobj)
    if seg.shape != intensity.shape or seg.min() < 0:
        raise ValueError("Expected matching nonnegative aseg and norm volumes")
    names = _lut(aseg_lut)
    ids = [label for label in sorted(names) if label and label not in (2, 3, 41, 42)
           and label <= seg.max()]
    voxel_volume = float(np.prod(image.header.get_zooms()[:3]))
    rows = {row[0]: row for row in _statistics(seg, intensity, ids, voxel_volume)}
    brainvol_stats = (Path(brainvol_stats) if brainvol_stats is not None else
                      subject / "stats" / "brainvol.stats")
    stiv = Path(stiv) if stiv is not None else subject / "stats" / "synthseg.tiv.dat"
    globals_ = read_brain_volume_stats(brainvol_stats)
    etiv = estimate_tiv(subject / "mri" / "transforms" / "talairach.xfm")
    holes = {hemi: _surface_holes(subject / "surf" / f"{hemi}.orig.nofix")
             for hemi in ("lh", "rh")}
    lines = ["# Title Segmentation Statistics ", "#",
             "# generating_program fnit", "# anatomy_type volume",
             f"# subjectname {subject.name}",
             "# BrainVolStatsFixed-NotNeeded because voxelvolume=1mm3"]
    lines.extend(f"# Measure {group}, {key}, {description}, {globals_[key]:.6f}, mm^3"
                 for group, key, description in _MEASURES)
    for group, key, description in (
        ("BrainSegVol-to-eTIV", "BrainSegVol-to-eTIV", "Ratio of BrainSegVol to eTIV"),
        ("MaskVol-to-eTIV", "MaskVol-to-eTIV", "Ratio of MaskVol to eTIV"),
    ):
        value = globals_["BrainSegVol" if group.startswith("BrainSeg") else "MaskVol"] / etiv
        lines.append(f"# Measure {group}, {key}, {description}, {value:.6f}, unitless")
    for group, value, description in (
        ("lhSurfaceHoles", holes["lh"], "Number of defect holes in lh surfaces prior to fixing"),
        ("rhSurfaceHoles", holes["rh"], "Number of defect holes in rh surfaces prior to fixing"),
        ("SurfaceHoles", holes["lh"] + holes["rh"], "Total number of defect holes in surfaces prior to fixing"),
    ):
        lines.append(f"# Measure {group}, {group}, {description}, {value}, unitless")
    lines.extend((
        "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, "
        f"{etiv:.6f}, mm^3",
        "# Measure SegmentedTotalIntraCranialVol, sTIV, Segmented Total Intracranial Volume, "
        f"{float(stiv.read_text().split()[0]):.6f}, mm^3",
        f"# SegVolFile {subject / 'mri' / 'aseg.mgz'}",
        f"# ColorTable {aseg_lut}",
        f"# InVolFile {subject / 'mri' / 'norm.mgz'}",
        f"# PVVolFile {subject / 'mri' / 'norm.mgz'}",
        "# Excluding Cortical Gray and White Matter", "# ExcludeSegId 0 2 3 41 42",
        f"# VoxelVolume_mm3 {voxel_volume:g}",
    ))
    columns = (("Index", "Index", "NA"), ("SegId", "Segmentation Id", "NA"),
               ("NVoxels", "Number of Voxels", "unitless"),
               ("Volume_mm3", "Volume", "mm^3"),
               ("StructName", "Structure Name", "NA"))
    columns += tuple((f"norm{suffix}", f"Intensity norm{suffix}", "MR")
                     for suffix in ("Mean", "StdDev", "Min", "Max", "Range"))
    for index, (header, field, units) in enumerate(columns, 1):
        lines.extend((f"# TableCol {index:2d} ColHeader {header}",
                      f"# TableCol {index:2d} FieldName {field}",
                      f"# TableCol {index:2d} Units     {units}"))
    lines.extend((f"# NRows {len(ids)}", "# NTableCols 10",
                  "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange"))
    for index, label in enumerate(ids, 1):
        _, count, volume, mean, std, lower, upper = rows.get(
            label, (label, 0, 0.0, 0.0, 0.0, 0.0, 0.0))
        lines.append(f"{index:3d} {label:3d}  {count:8d} {volume:10.1f}  "
                     f"{names[label]:<30s} {mean:10.4f} {std:10.4f} "
                     f"{lower:10.4f} {upper:10.4f} {upper-lower:10.4f} ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=Path)
    parser.add_argument("aseg_lut", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_aseg_stats(args.subject, args.aseg_lut, args.output)


if __name__ == "__main__":
    main()
