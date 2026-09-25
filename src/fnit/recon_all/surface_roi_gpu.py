"""Compute FreeSurfer surface ROI area, thickness and gray volume.

Curvature and folding columns are not implemented here.
"""

from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import torch

from .surface_area_gpu import vertex_area


@torch.inference_mode()
def vertex_th3_volume(white: str | Path, pial: str | Path,
                      cortex_label: str | Path, *, device: str = "cuda:0") -> np.ndarray:
    """Divide each white/pial triangular prism into the official three tetrahedra."""
    wxyz, faces = fsio.read_geometry(str(white))
    pxyz, pfaces = fsio.read_geometry(str(pial))
    if len(wxyz) != len(pxyz) or not np.array_equal(faces, pfaces):
        raise ValueError("White and pial surfaces must have identical topology")
    w = torch.as_tensor(np.asarray(wxyz, dtype=np.float32), device=device)
    p = torch.as_tensor(np.asarray(pxyz, dtype=np.float32), device=device)
    tri = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    a, b, c = (tri[:, i] for i in range(3))
    bw, cw, aw = w[b] - p[a], w[c] - p[a], w[a] - p[a]
    bp, cp = p[b] - p[a], p[c] - p[a]
    t1 = (aw * torch.cross(bw, cw, dim=1)).sum(1).abs()
    t2 = (bp * torch.cross(cp, bw, dim=1)).sum(1).abs()
    t3 = (cp * torch.cross(cw, bw, dim=1)).sum(1).abs()
    share = (t1 + t2 + t3).div_(18.0)
    result = torch.zeros(len(w), device=device, dtype=torch.float32)
    for corner in range(3):
        result.index_add_(0, tri[:, corner], share)
    cortex = np.zeros(len(wxyz), dtype=bool)
    cortex[fsio.read_label(str(cortex_label))] = True
    result *= torch.as_tensor(cortex, device=device)
    return result.cpu().numpy()


def vertex_volume_map(white: str | Path, pial: str | Path,
                      cortex_label: str | Path, output: str | Path,
                      *, device: str = "cuda:0") -> None:
    """Write the TH3 morphometry map used by `mris_convert --volume`."""
    fsio.write_morph_data(str(output), vertex_th3_volume(white, pial,
                                                         cortex_label, device=device))


@torch.inference_mode()
def roi_area_thickness(surface: str | Path, annotation: str | Path,
                       thickness: str | Path, *, device: str = "cuda:0") -> dict:
    """Return NumVert, SurfArea, ThickAvg and ThickStd for annotated regions."""
    xyz, faces = fsio.read_geometry(str(surface))
    labels, _, names = fsio.read_annot(str(annotation))
    values = torch.as_tensor(np.asarray(fsio.read_morph_data(str(thickness)),
                                        dtype=np.float64), device=device)
    area = torch.as_tensor(vertex_area(xyz, faces, device=device),
                           dtype=torch.float64, device=device)
    region = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    if len(values) != len(labels) or len(area) != len(labels):
        raise ValueError("Surface, annotation and thickness vertex counts differ")
    output = {}
    for index, raw_name in enumerate(names):
        name = raw_name.decode()
        if name in {"corpuscallosum", "unknown", "Unknown", "Medial_wall"}:
            continue
        selected = region == index
        if not bool(selected.any()):
            continue
        thick = values[selected]
        output[name] = (len(thick), float(area[selected].sum()),
                        float(thick.mean()), float(thick.std(unbiased=False)))
    return output


@torch.inference_mode()
def roi_gray_volume(white: str | Path, pial: str | Path,
                    thickness: str | Path, annotation: str | Path,
                    *, device: str = "cuda:0") -> dict[str, float]:
    """Match `mris_anatomical_stats -no-th3` gray volume by annotation."""
    wxyz, faces = fsio.read_geometry(str(white))
    pxyz, pfaces = fsio.read_geometry(str(pial))
    if len(wxyz) != len(pxyz) or not np.array_equal(faces, pfaces):
        raise ValueError("White and pial surfaces must have identical topology")
    w = torch.as_tensor(np.asarray(wxyz, dtype=np.float32), device=device)
    p = torch.as_tensor(np.asarray(pxyz, dtype=np.float32), device=device)
    tri = torch.as_tensor(np.asarray(faces, dtype=np.int64), device=device)
    thick = torch.as_tensor(np.asarray(fsio.read_morph_data(str(thickness)),
                                       dtype=np.float32), device=device)
    if len(thick) != len(w):
        raise ValueError("Surface and thickness vertex counts differ")
    areas = []
    for vertices in (w, p):
        v0, v1, v2 = (vertices[tri[:, i]] for i in range(3))
        areas.append(torch.linalg.vector_norm(torch.cross(v1 - v0, v2 - v0, dim=1),
                                               dim=1).mul_(0.5))
    mean_thick = thick[tri].to(torch.float64).mean(1)
    share = mean_thick * (areas[0].to(torch.float64) + areas[1].to(torch.float64)) / 6.0
    volume = torch.zeros(len(w), device=device, dtype=torch.float64)
    for corner in range(3):
        volume.index_add_(0, tri[:, corner], share)
    labels, _, names = fsio.read_annot(str(annotation))
    region = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    if len(volume) != len(region):
        raise ValueError("Surface and annotation vertex counts differ")
    return {name.decode(): float(volume[region == index].sum())
            for index, name in enumerate(names)
            if name.decode() not in {"corpuscallosum", "unknown", "Unknown", "Medial_wall"}
            and bool((region == index).any())}
