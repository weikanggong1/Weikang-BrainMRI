"""Format the ten FreeSurfer cortical parcellation table columns."""

from pathlib import Path

from .surface_roi_curvature_gpu import curvature_columns
from .surface_roi_gpu import roi_area_thickness, roi_gray_volume


def anatomical_stats_rows(white: str | Path, pial: str | Path,
                          surface: str | Path, area_map: str | Path,
                          thickness: str | Path, annotation: str | Path,
                          cortex_label: str | Path | None, *, device: str = "cuda:0") -> list[str]:
    """Return native-formatted ROI lines for ``mris_anatomical_stats -no-th3``.

    ``surface`` is white or pial, and ``area_map`` must correspond to it.
    The calling pipeline must still compute the global header measures.
    """
    white, pial, surface = Path(white), Path(pial), Path(surface)
    area_map, thickness = Path(area_map), Path(thickness)
    annotation = Path(annotation)
    cortex_label = Path(cortex_label) if cortex_label is not None else None
    basic = roi_area_thickness(surface, annotation, thickness, device=device)
    volumes = roi_gray_volume(white, pial, thickness, annotation, device=device)
    curvature = curvature_columns(surface, area_map, annotation, cortex_label,
                                  device=device)
    if set(basic) != set(volumes) or set(basic) != set(curvature):
        raise ValueError("Area, volume and curvature ROI names differ")
    lines = []
    for name, (count, area, mean, std) in basic.items():
        cm, cg, fold, intrinsic = curvature[name]
        lines.append(f"{name:<40}  {count:5d}  {area:5.0f}  {volumes[name]:5.0f}"
                     f"  {mean:5.3f} {std:5.3f}  {cm:8.3f}  {cg:8.3f}"
                     f"  {fold:7.0f}  {intrinsic:6.1f}")
    return lines
