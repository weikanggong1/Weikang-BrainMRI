"""Global numeric headers for FreeSurfer cortical parcellation tables."""

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from .estimated_tiv import estimate_tiv


def read_brain_volume_stats(path: str | Path) -> dict[str, float]:
    """Read the 16 cached measures written by ``ComputeBrainVolumeStats``."""
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5 and fields[0].startswith("# Measure "):
            result[fields[1]] = float(fields[3])
    required = ("BrainSegVol", "BrainSegVolNotVent", "SupraTentorialVol",
                "SupraTentorialVolNotVent", "SubCortGrayVol", "lhCortexVol",
                "rhCortexVol", "CortexVol", "TotalGrayVol",
                "lhCerebralWhiteMatterVol", "rhCerebralWhiteMatterVol",
                "CerebralWhiteMatterVol", "MaskVol", "SupraTentorialVolNotVentVox",
                "BrainSegVolNotVentSurf", "VentricleChoroidVol")
    if any(name not in result for name in required):
        raise ValueError("Incomplete brainvol.stats cache")
    return result


def anatomical_stats_global_lines(area_map: str | Path, thickness: str | Path,
                                  annotation: str | Path, cortex_label: str | Path | None,
                                  brainvol_stats: str | Path | dict[str, float],
                                  talairach_xfm: str | Path,
                                  *, surface: str = "white",
                                  voxel_volume: float = 1.0) -> list[str]:
    """Return the numeric ``# Measure`` lines from ``mris_anatomical_stats``."""
    area = fsio.read_morph_data(str(area_map))
    thick = fsio.read_morph_data(str(thickness))
    labels, _, _ = fsio.read_annot(str(annotation))
    cortex = np.ones(len(area), dtype=bool) if cortex_label is None else np.zeros(len(area), dtype=bool)
    if cortex_label is not None:
        cortex[fsio.read_label(str(cortex_label))] = True
    if not (len(area) == len(thick) == len(labels)):
        raise ValueError("Area, thickness and annotation vertex counts differ")
    cortex &= labels >= 0
    count = int(cortex.sum())
    area_total = (np.float32(np.sum(area.astype(np.float64)))
                  if cortex_label is None else np.float32(0))
    thickness_total = np.float32(0)
    if cortex_label is not None:
        for vertex in np.flatnonzero(cortex):
            area_total = np.float32(area_total + area[vertex])
            thickness_total = np.float32(thickness_total + thick[vertex])
    mean_thickness = np.float32(thickness_total / count) if count else np.float32(0)
    surface_name = {"white": "WhiteSurfArea", "pial": "PialSurfArea"}.get(surface,
                                                                          "SurfArea")
    surface_description = {"white": "White Surface Total Area",
                           "pial": "Pial Surface Total Area"}.get(surface,
                                                                     "Surface Total Area")
    volume = (brainvol_stats if isinstance(brainvol_stats, dict)
              else read_brain_volume_stats(brainvol_stats))
    values = (("BrainSeg", "BrainSegVol", "Brain Segmentation Volume"),
              ("BrainSegNotVent", "BrainSegVolNotVent", "Brain Segmentation Volume Without Ventricles"),
              ("BrainSegNotVentSurf", "BrainSegVolNotVentSurf", "Brain Segmentation Volume Without Ventricles from Surf"),
              ("Cortex", "CortexVol", "Total cortical gray matter volume"),
              ("SupraTentorial", "SupraTentorialVol", "Supratentorial volume"),
              ("SupraTentorialNotVent", "SupraTentorialVolNotVent", "Supratentorial volume"))
    result = [
        f"# Measure Cortex, NumVert, Number of Vertices, {count}, unitless",
        f"# Measure Cortex, {surface_name}, {surface_description}, {area_total:g}, mm^2",
    ]
    if cortex_label is not None:
        result.append(f"# Measure Cortex, MeanThickness, Mean Thickness, {mean_thickness:g}, mm")
    result.extend([
        ("# BrainVolStatsFixed-NotNeeded because voxelvolume=1mm3"
         if abs(voxel_volume - 1) <= .01 else
         "# BrainVolStatsFixed see surfer.nmr.mgh.harvard.edu/fswiki/BrainVolStatsFixed"),
        *(f"# Measure {group}, {key}, {description}, {volume[key]:.6f}, mm^3"
          for group, key, description in values),
        ("# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, "
         f"{estimate_tiv(talairach_xfm):.6f}, mm^3"),
    ])
    return result
