"""Validate independent first-defect intensity histograms and MRI matching."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_defect_mri_match import defect_mri_match
from fnit.recon_all.topology_intensity_histogram import first_defect_intensity_histograms, histogram_mean
from fnit.recon_all.topology_preflight_python import (
    center_sphere, defect_component_labels, defect_regions,
    project_and_smooth_sphere,
)
from validation.recon_all.python_gpu_port.experimental.topology_edge_score_probe import (
    _border_values, _face_index, _median_twice, _vertex_normals,
)


def _hull(vertices: list[int], border: list[int], neighbors: list[list[int]]) -> list[int]:
    occupied = set(vertices) | set(border)
    result = border.copy()
    for vertex in border:
        for neighbor in neighbors[vertex]:
            if neighbor not in occupied:
                occupied.add(neighbor)
                result.append(neighbor)
    return result


def _compare_histogram(predicted: np.ndarray, path: Path) -> dict:
    rows = np.loadtxt(path)
    if not np.array_equal(rows[:, 0], np.arange(256)):
        raise ValueError(f"unexpected native histogram bins: {path}")
    native_text = [line.split()[1] for line in path.read_text().splitlines()]
    printed = [f"{float(value):.10f}" for value in predicted]
    diff = np.abs(predicted.astype(np.float64) - rows[:, 1])
    mismatch = [index for index, (left, right) in enumerate(zip(printed, native_text)) if left != right]
    return {"printed_10_decimal_equal": 256 - len(mismatch),
            "parsed_float32_equal": int(np.count_nonzero(predicted.view(np.uint32)
                                        == rows[:, 1].astype(np.float32).view(np.uint32))),
            "max_absolute_difference_from_plot": float(np.max(diff)),
            "first_printed_difference": int(mismatch[0]) if mismatch else None,
            "first_python_printed": printed[mismatch[0]] if mismatch else None,
            "first_native_printed": native_text[mismatch[0]] if mismatch else None}


def run(diagnostics: Path, hemisphere: str, patch_report: Path, base: Path,
        histograms: Path, report: Path) -> dict:
    started = perf_counter()
    surf = diagnostics / "fs_sub01" / "surf"
    orig_path = surf / f"{hemisphere}.orig.nofix"
    sphere_path = surf / f"{hemisphere}.qsphere.nofix"
    brain_path = diagnostics / "fs_sub01" / "mri" / "brain.mgz"
    orig, faces = fsio.read_geometry(str(orig_path))
    sphere, sphere_faces = fsio.read_geometry(str(sphere_path))
    orig = np.ascontiguousarray(orig, np.float32)
    faces = np.ascontiguousarray(faces, np.int32)
    if not np.array_equal(faces, sphere_faces):
        raise ValueError("original and sphere faces differ")
    sphere = center_sphere(project_and_smooth_sphere(sphere, faces))[0]
    labels = defect_component_labels(sphere, faces)
    vertices, border = defect_regions(labels, faces)[0]
    neighbors = ordered_neighbors(faces, len(orig))
    hull = _hull(vertices, border, neighbors)
    marked = labels != 0
    degree = np.fromiter((len(row) for row in neighbors), np.int32, count=len(orig))
    nbrs = np.zeros((len(orig), int(degree.max())), np.int32)
    for vertex, row in enumerate(neighbors):
        nbrs[vertex, :len(row)] = row

    mri = nib.load(str(brain_path))
    volume = np.ascontiguousarray(np.asarray(mri.dataobj))
    inv_tkr = np.linalg.inv(mri.header.get_vox2ras_tkr())
    offsets, face_ids, corners = _face_index(faces, len(orig))
    normal = _vertex_normals(orig, faces, offsets, face_ids, corners, marked)
    white, gray = _border_values(orig, normal, marked, volume, inv_tkr)
    white = _median_twice(white, marked, nbrs, degree)
    gray = _median_twice(gray, marked, nbrs, degree)
    hist = first_defect_intensity_histograms(white, gray, hull, neighbors, marked)
    histogram_seconds = perf_counter() - started

    patch = json.loads(patch_report.read_text())
    inside = np.asarray(patch["first_candidate_inside_vertex_indices"], np.int32)
    rows = patch["first_candidate_inside_neighbor_rows"]
    select_xyz, select_faces = fsio.read_geometry(str(base))
    match_started = perf_counter()
    predicted = defect_mri_match(select_xyz, select_faces, inside, rows, volume,
                                 inv_tkr, hist["white_mean"], hist["gray_mean"])
    match_seconds = perf_counter() - match_started
    artifact = report.with_suffix(".tsv")
    channels = ("white_raw", "gray_raw", "white_smooth", "gray_smooth")
    lines = ["bin\t" + "\t".join(name + "_float32_hex" for name in channels)]
    for bin_index in range(256):
        bits = (int(hist[name].view(np.uint32)[bin_index]) for name in channels)
        lines.append(str(bin_index) + "\t" + "\t".join(f"{value:08x}" for value in bits))
    artifact.write_text("\n".join(lines) + "\n")

    # Reference files are opened only after all independent outputs exist.
    native_labels = np.rint(fsio.read_morph_data(str(surf / f"{hemisphere}.defect_labels"))).astype(np.int32)
    native_xyz, native_faces = fsio.read_geometry(str(base) + "m")
    if not np.array_equal(select_faces, native_faces):
        raise ValueError("select0s and select0sm faces differ")
    bitwise = np.all(predicted.view(np.uint32) == native_xyz.astype(np.float32).view(np.uint32), axis=1)
    distance = np.linalg.norm(predicted.astype(np.float64) - native_xyz.astype(np.float64), axis=1)
    native_white = np.loadtxt(histograms / "w.plt")[:, 1].astype(np.float32)
    native_gray = np.loadtxt(histograms / "g.plt")[:, 1].astype(np.float32)
    white_mean_native = histogram_mean(native_white)
    gray_mean_native = histogram_mean(native_gray)
    comparisons = {name: _compare_histogram(hist[key], histograms / f"{name}.plt")
                   for name, key in (("wr", "white_raw"), ("gr", "gray_raw"),
                                     ("w", "white_smooth"), ("g", "gray_smooth"))}
    result = {
        "hemisphere": hemisphere,
        "scope": "independent first-defect white/gray intensity histograms and first select0s-to-select0sm transition",
        "source": "FreeSurfer 8.2 d932c45b7941662ea380a05efef580568b98d41a",
        "input_sha256": {name: sha256(path.read_bytes()).hexdigest() for name, path in {
            "original_surface": orig_path, "sphere": sphere_path, "brain": brain_path,
            "patch_report": patch_report, "select0s": base,
            "select0sm": Path(str(base) + "m"),
            "white_histogram_plot": histograms / "w.plt",
            "gray_histogram_plot": histograms / "g.plt"}.items()},
        "histogram_artifact": artifact.name,
        "histogram_artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
        "first_defect_vertices": len(vertices), "first_defect_border": len(border),
        "first_defect_hull": len(hull), "histogram_selected_vertices": len(hist["selected"]),
        "defect_label_differences": int(np.count_nonzero(labels != native_labels)),
        "histograms": comparisons,
        "white_mean_python": float(hist["white_mean"]),
        "gray_mean_python": float(hist["gray_mean"]),
        "white_mean_native_plot": float(white_mean_native),
        "gray_mean_native_plot": float(gray_mean_native),
        "white_mean_float32_bits_equal": bool(hist["white_mean"].view(np.uint32) == white_mean_native.view(np.uint32)),
        "gray_mean_float32_bits_equal": bool(hist["gray_mean"].view(np.uint32) == gray_mean_native.view(np.uint32)),
        "select0sm_inside_bitwise_equal": int(np.count_nonzero(bitwise[inside])),
        "select0sm_inside_vertices": len(inside),
        "select0sm_all_bitwise_equal": int(np.count_nonzero(bitwise)),
        "select0sm_all_vertices": len(bitwise),
        "select0sm_max_distance_mm": float(np.max(distance)),
        "histogram_python_seconds_including_jit_and_reads": histogram_seconds,
        "mri_match_python_seconds_including_jit": match_seconds,
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics", required=True, type=Path)
    ap.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    ap.add_argument("--patch-report", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--histograms", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(run(args.diagnostics, args.hemisphere, args.patch_report,
                         args.base, args.histograms, args.report), indent=2))


if __name__ == "__main__":
    main()
