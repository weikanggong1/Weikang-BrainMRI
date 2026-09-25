"""Compare Python defectSmooth type 2 against the saved first-candidate s snapshot."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_defect_smooth import defect_smooth_type2


def _hist_mean(path: Path, bin_size: np.float32) -> tuple[float, float]:
    """Recover source float bins from the MGH width and the six-decimal plot."""
    printed, counts = np.loadtxt(path).T
    counts = counts.astype(np.float32)
    center = np.float32(printed[0])
    starts = [center]
    for direction in (np.float32(-np.inf), np.float32(np.inf)):
        value = center
        for _ in range(16):
            value = np.nextafter(value, direction)
            starts.append(value)
    matches = []
    for first in starts:
        bins = np.empty(len(printed), np.float32)
        value = first
        for index in range(len(bins)):
            bins[index] = value
            value = np.float32(value + bin_size)
        if all(f"{float(value):.6f}" == f"{float(saved):.6f}"
               for value, saved in zip(bins, printed)):
            matches.append((first, bins))
    if len(matches) != 1:
        raise ValueError(f"expected one unrounded histogram origin: {path}, got {len(matches)}")
    first, bins = matches[0]
    maximum = np.max(counts)
    mean = np.float32(0)
    total = np.float32(0)
    for value, count in zip(bins, counts):
        if count < np.float32(maximum / np.float32(100)):
            continue
        offset = np.float32(value - np.float32(bin_size / np.float32(2)))
        mean = np.float32(mean + np.float32(offset * count))
        total = np.float32(total + count)
    return float(np.float32(mean / total)), float(first)


def validate(patch_report: Path, base: Path, histograms: Path) -> dict:
    previous = json.loads(patch_report.read_text())
    inside = np.asarray(previous["first_candidate_inside_vertex_indices"], np.int32)
    neighbors = previous["first_candidate_inside_neighbor_rows"]
    xyz, faces = fsio.read_geometry(str(base))
    native, native_faces = fsio.read_geometry(str(Path(str(base) + "s")))
    if not np.array_equal(faces, native_faces):
        raise ValueError("native select0 and select0s faces differ")
    joint = histograms / "mri_k1_k2.mgh"
    sizes = nib.load(str(joint)).header.get_zooms()
    k1, k1_first = _hist_mean(histograms / "k1.plt", np.float32(sizes[0]))
    k2, k2_first = _hist_mean(histograms / "k2.plt", np.float32(sizes[1]))
    start = perf_counter()
    predicted = defect_smooth_type2(xyz, faces, inside, neighbors, k1, k2)
    seconds = perf_counter() - start
    difference = np.linalg.norm(predicted.astype(np.float64) - native.astype(np.float64), axis=1)
    mismatch = np.any(predicted.view(np.uint32) != np.asarray(native, np.float32).view(np.uint32), axis=1)
    outside = np.ones(len(xyz), np.bool_)
    outside[inside] = False
    return {
        "hemisphere": previous["hemisphere"],
        "scope": "25-iteration type-2 smoothing with native saved curvature plots and MGH bin widths; no independent histogram generation or MRI match",
        "inside_vertices": len(inside),
        "independent_first_patch_report_sha256": sha256(patch_report.read_bytes()).hexdigest(),
        "native_select0_sha256": sha256(base.read_bytes()).hexdigest(),
        "native_select0s_sha256": sha256(Path(str(base) + "s").read_bytes()).hexdigest(),
        "histogram_k1_mean_from_source_order_bins": k1,
        "histogram_k2_mean_from_source_order_bins": k2,
        "histogram_k1_first_bin": k1_first,
        "histogram_k2_first_bin": k2_first,
        "histogram_k1_bin_size": float(np.float32(sizes[0])),
        "histogram_k2_bin_size": float(np.float32(sizes[1])),
        "native_plot_and_mgh_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (histograms / "k1.plt", histograms / "k2.plt", joint)
        },
        "python_seconds_including_jit": seconds,
        "bitwise_equal_vertices": int(np.count_nonzero(~mismatch)),
        "inside_bitwise_equal_vertices": int(np.count_nonzero(~mismatch[inside])),
        "outside_bitwise_equal_vertices": int(np.count_nonzero(~mismatch[outside])),
        "outside_vertices": int(np.count_nonzero(outside)),
        "max_vertex_distance_mm": float(np.max(difference)),
        "inside_mean_distance_mm": float(np.mean(difference[inside])),
        "inside_median_distance_mm": float(np.median(difference[inside])),
        "inside_mismatch_vertices": inside[np.flatnonzero(mismatch[inside])].tolist(),
        "first_inside_difference": int(inside[np.flatnonzero(mismatch[inside])[0]])
        if np.any(mismatch[inside]) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-report", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--histograms", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    result = validate(args.patch_report, args.base, args.histograms)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
