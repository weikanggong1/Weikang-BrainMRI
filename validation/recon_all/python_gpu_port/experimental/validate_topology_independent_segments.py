"""Validate first-defect old-edge segmentation from MRI and surface inputs alone."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_edge_segments import segment_intersecting_edges
from fnit.recon_all.topology_preflight_python import (
    center_sphere, defect_component_labels, defect_regions,
    defect_retention_status, genetic_base_translation,
    genetic_candidate_edge_table, project_and_smooth_sphere,
)
from validation.recon_all.python_gpu_port.experimental.topology_edge_score_probe import (
    _CMP_TYPE, _border_values, _compare_edge_length, _face_index, _median_twice,
    _orig_normals, _scores, _smooth, _vertex_normals,
)
from validation.recon_all.python_gpu_port.experimental.topology_ga_segments_probe import _native_annotation
from validation.recon_all.python_gpu_port.validate_topology_edge_table import EDGE_DTYPE


def _ordered_scored_edges(diagnostics: Path, hemi: str):
    """Produce the first-defect EDGE table without native EDGE capture input."""
    surf = diagnostics / "fs_sub01" / "surf"
    sphere, faces = fsio.read_geometry(str(surf / f"{hemi}.qsphere.nofix"))
    faces = np.ascontiguousarray(faces, np.int32)
    sphere, _ = center_sphere(project_and_smooth_sphere(sphere, faces))
    orig, orig_faces = fsio.read_geometry(str(surf / f"{hemi}.orig.nofix"))
    orig = np.ascontiguousarray(orig, np.float32)
    orig_faces = np.ascontiguousarray(orig_faces, np.int32)
    if not np.array_equal(faces, orig_faces):
        raise ValueError("orig and qsphere faces differ")
    labels = defect_component_labels(sphere, faces)
    status = defect_retention_status(sphere, orig, faces, labels)
    table, all_pairs = genetic_candidate_edge_table(sphere, faces, labels, status, 0)
    forward, _ = genetic_base_translation(labels, faces)
    reverse = np.argsort(forward)
    source_ids = reverse[table[:, :2]]

    mri = nib.load(str(diagnostics / "fs_sub01" / "mri" / "brain.mgz"))
    volume = np.ascontiguousarray(np.asarray(mri.dataobj))
    inv_tkr = np.linalg.inv(mri.header.get_vox2ras_tkr())
    ripped = labels != 0
    neighbors = ordered_neighbors(faces, len(orig))
    degrees = np.fromiter((len(row) for row in neighbors), np.int32, count=len(orig))
    nbrs = np.zeros((len(orig), int(degrees.max())), np.int32)
    for vertex, row in enumerate(neighbors):
        nbrs[vertex, :len(row)] = row
    offsets, face_ids, corners = _face_index(faces, len(orig))
    normal = _vertex_normals(orig, faces, offsets, face_ids, corners, ripped)
    white, gray = _border_values(orig, normal, ripped, volume, inv_tkr)
    white = _median_twice(white, ripped, nbrs, degrees)
    gray = _median_twice(gray, ripped, nbrs, degrees)
    smooth = _smooth(orig, nbrs, degrees)
    active = np.unique(source_ids)
    normal_lookup = np.zeros((len(orig), 3), np.float32)
    normal_lookup[active] = _orig_normals(smooth, faces, offsets, face_ids, corners, active)
    edge_norm = np.concatenate((normal_lookup[source_ids[:, 0]],
                                normal_lookup[source_ids[:, 1]]), axis=1)
    score = _scores(table, source_ids, smooth, edge_norm, white, gray, volume, inv_tkr)
    rows = np.zeros(len(table), EDGE_DTYPE)
    rows["vno1"], rows["vno2"] = table[:, 0], table[:, 1]
    rows["used"] = table[:, 2]
    rows["length"] = score
    libc = ctypes.CDLL(None)
    libc.qsort.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, _CMP_TYPE]
    libc.qsort(rows.ctypes.data, len(rows), rows.dtype.itemsize, _compare_edge_length)
    corrected_sphere = np.ascontiguousarray(sphere[reverse], np.float32)
    return rows, table, score, corrected_sphere, labels, status, faces, forward, all_pairs


def _pairs(rows):
    return np.column_stack((rows["vno1"], rows["vno2"]))


def run(diagnostics: Path, captures: Path, annotation: Path, hemi: str) -> dict:
    # All prediction inputs are loaded and processed before opening reference files.
    rows, table, score, sphere, labels, status, faces, forward, all_pairs = _ordered_scored_edges(diagnostics, hemi)
    old = np.ascontiguousarray(_pairs(rows[rows["used"] == 2]), np.int32)
    vertices, border = defect_regions(labels, faces)[0]
    inside = forward[np.asarray(vertices, np.int32)]
    edges, groups, assigned = segment_intersecting_edges(sphere, old, inside)
    colors = np.asarray([-1974016, 5127885, 5127800], np.int32)
    predicted = np.zeros(len(labels), np.int32)
    reverse = np.argsort(forward)
    for index, (a, b) in enumerate(edges):
        group = int(assigned[index])
        if 0 <= group < len(colors):
            predicted[reverse[a]] = predicted[reverse[b]] = colors[group]

    surf = diagnostics / "fs_sub01" / "surf"
    native_labels = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_labels"))).astype(np.int32)
    native_status = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_status"))).astype(np.int8)
    before = np.fromfile(captures / f"capture_{hemi}" / f"{hemi}.edge.before.bin", EDGE_DTYPE)
    after = np.fromfile(captures / f"capture_{hemi}" / f"{hemi}.edge.after.bin", EDGE_DTYPE)
    native = _native_annotation(annotation, len(labels))
    relevant = np.asarray(vertices + border, np.int32)
    before_table = np.column_stack((_pairs(before), before["used"]))
    after_pair_diff = np.flatnonzero(np.any(_pairs(rows) != _pairs(after), axis=1))
    native_old = np.ascontiguousarray(_pairs(after[after["used"] == 2]), np.int32)
    old_diff = np.flatnonzero(np.any(old != native_old, axis=1))
    annotation_diff = np.flatnonzero(predicted != native)
    result = {
        "hemisphere": hemi,
        "scope": "first defect only; independent candidate edges, MRI scores, old-edge segmentation; no GA patch or final orig.premesh",
        "all_candidate_pairs": int(all_pairs),
        "scored_edges": len(rows),
        "defect_label_differences": int(np.count_nonzero(labels != native_labels)),
        "defect_status_differences": int(np.count_nonzero(status != native_status)),
        "candidate_table_differences": int(np.count_nonzero(np.any(table != before_table, axis=1))),
        "bitwise_score_equal": int(np.count_nonzero(score.view(np.uint32) == before["length"].view(np.uint32))),
        "max_score_absolute_error": float(np.max(np.abs(score - before["length"]))),
        "sorted_all_edge_order_differences": len(after_pair_diff),
        "first_sorted_all_edge_order_difference": int(after_pair_diff[0]) if len(after_pair_diff) else None,
        "old_edges": len(old),
        "sorted_old_edge_order_differences": len(old_diff),
        "first_sorted_old_edge_order_difference": int(old_diff[0]) if len(old_diff) else None,
        "cluster_sizes": [len(group) for group in groups if group],
        "unassigned_edges": int(np.count_nonzero(assigned < 0)),
        "annotation_differences_all_vertices": len(annotation_diff),
        "annotation_equal_defect_and_border": int(np.count_nonzero(predicted[relevant] == native[relevant])),
        "defect_and_border_vertices": len(relevant),
        "first_annotation_difference": int(annotation_diff[0]) if len(annotation_diff) else None,
    }
    if (len(rows) != len(before) or len(after) != len(rows) or len(old) != len(native_old)
            or result["defect_label_differences"] or result["defect_status_differences"]
            or result["candidate_table_differences"] or len(old_diff) or len(annotation_diff)):
        raise AssertionError(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", required=True, type=Path)
    ap.add_argument("--captures", required=True, type=Path)
    ap.add_argument("--annotation-lh", required=True, type=Path)
    ap.add_argument("--annotation-rh", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    result = {hemi: run(args.diagnostics, args.captures, annotation, hemi)
              for hemi, annotation in (("lh", args.annotation_lh), ("rh", args.annotation_rh))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
