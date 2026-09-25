"""Isolated first-candidate retessellation probe for mris_fix_topology -ga.

The independent mode derives sorted EDGE rows from MRI and surfaces. Both
modes stop before GA fitness and search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.topology_preflight_python import (
    center_sphere,
    defect_regions,
    genetic_base_translation,
    project_and_smooth_sphere,
)
from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_first_candidate import accept_first_candidate_edges
from fnit.recon_all.topology_fitness_search import invalid_modified_edges
from validation.recon_all.python_gpu_port.validate_topology_edge_table import EDGE_DTYPE
from validation.recon_all.python_gpu_port.experimental.topology_edge_score_probe import _smooth


def _edges(faces: np.ndarray) -> np.ndarray:
    result = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                             faces[:, [2, 0]]), axis=0).astype(np.int32)
    result.sort(axis=1)
    return np.unique(result, axis=0)


def _hull(vertices: list[int], border: list[int], neighbors: list[list[int]]) -> list[int]:
    # SMALL_CONVEX_HULL: start with border, then their first unmarked neighbors.
    occupied = set(vertices) | set(border)
    result = border.copy()
    for vertex in border:
        for neighbor in neighbors[vertex]:
            if neighbor not in occupied:
                occupied.add(neighbor)
                result.append(neighbor)
    return result


def _contains_another_on_sphere(canonical: np.ndarray, triangle: tuple[int, int, int],
                                neighbors: dict[int, set[int]]) -> bool:
    points = canonical[np.asarray(triangle)].astype(np.float64)
    normals = np.empty((3, 3), np.float64)
    origins = np.empty((3, 3), np.float64)
    signs = np.empty(3, np.float64)
    for i in range(3):
        opposite, first, second = points[i], points[(i + 1) % 3], points[(i + 2) % 3]
        origins[i] = (first + second) / 2.0
        normals[i] = np.cross(origins[i], second - first)
        signs[i] = np.dot(opposite - origins[i], normals[i])
    nearby = set().union(*(neighbors[v] for v in triangle)) - set(triangle)
    for vertex in nearby:
        point = canonical[vertex].astype(np.float64)
        if np.all(signs * np.einsum("ij,ij->i", point - origins, normals) >= 0):
            return True
    return False


def _triangle_faces(canonical: np.ndarray, base_edges: np.ndarray,
                    added_edges: set[tuple[int, int]]) -> set[tuple[int, int, int]]:
    neighbors: dict[int, set[int]] = {}
    for a, b in list(map(tuple, base_edges)) + list(added_edges):
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    triples = set()
    for a, b in added_edges:
        for c in neighbors[a] & neighbors[b]:
            triples.add(tuple(sorted((a, b, c))))
    return {row for row in triples if not _contains_another_on_sphere(canonical, row, neighbors)}


def _candidate_face_order(canonical: np.ndarray, base_faces: np.ndarray,
                          neighbors: list[list[int]],
                          candidate_edges: np.ndarray, accepted: np.ndarray,
                          active_vertices: list[int],
                          allowed: set[tuple[int, int, int]]) -> np.ndarray:
    for a, b in candidate_edges[accepted]:
        neighbors[a].append(int(b))
        neighbors[b].append(int(a))
    attached: dict[tuple[int, int], int] = {}
    for a, b, c in base_faces:
        for x, y in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(x), int(y))))
            attached[key] = attached.get(key, 0) + 1
    rows: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for vertex in active_vertices:
        for first in neighbors[vertex]:
            for second in neighbors[vertex]:
                if first == second:
                    continue
                key = tuple(sorted((vertex, first, second)))
                if key not in allowed or key in seen:
                    continue
                edges = [tuple(sorted(pair)) for pair in ((vertex, first), (vertex, second), (first, second))]
                if any(attached.get(pair, 0) >= 2 for pair in edges):
                    continue
                row = [vertex, first, second]
                a, b, c = canonical[row]
                if np.dot(np.cross(b - a, c - a), a + b + c) < 0:
                    row[1], row[2] = row[2], row[1]
                rows.append(row)
                seen.add(key)
                for pair in edges:
                    attached[pair] = attached.get(pair, 0) + 1
    return np.asarray(rows, np.int32).reshape(-1, 3)


def run(diagnostics: Path, captures: Path, patch: Path, hemi: str,
        final_patch: Path | None = None, independent: bool = False) -> dict:
    surf = diagnostics / "fs_sub01" / "surf"
    orig, faces = fsio.read_geometry(str(surf / f"{hemi}.orig.nofix"))
    if independent:
        from validation.recon_all.python_gpu_port.experimental.validate_topology_independent_segments import (
            _ordered_scored_edges,
        )
        capture, _, _, corrected_sphere, labels, status, scored_faces, forward, _ = (
            _ordered_scored_edges(diagnostics, hemi)
        )
        if not np.array_equal(faces, scored_faces):
            raise ValueError("independently scored and original face order differs")
    else:
        sphere, _ = fsio.read_geometry(str(surf / f"{hemi}.qsphere.nofix"))
        labels = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_labels"))).astype(np.int32)
        status = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_status"))).astype(np.int8)
        sphere, _ = center_sphere(project_and_smooth_sphere(sphere, faces))
        forward, _ = genetic_base_translation(labels, faces)
        corrected_sphere = np.ascontiguousarray(sphere[np.argsort(forward)], np.float32)
        capture = np.fromfile(captures / f"capture_{hemi}" / f"{hemi}.edge.after.bin", EDGE_DTYPE)
    base_faces = forward[faces[np.all(labels[faces] == 0, axis=1)]]
    # The correction surface retains links between intact vertices even when
    # the incident face is removed with a defect vertex. These dangling links
    # are part of the source's initial VT topology and candidate intersection.
    original_neighbors = ordered_neighbors(faces, len(orig))
    neighbors: list[list[int]] = [[] for _ in range(len(orig))]
    for old_vertex, row in enumerate(original_neighbors):
        if labels[old_vertex] == 0:
            neighbors[forward[old_vertex]] = [int(forward[v]) for v in row if labels[v] == 0]
    base_edges = np.asarray(sorted({tuple(sorted((v, n)))
                                    for v, row in enumerate(neighbors) for n in row}), np.int32)
    vertices, border = defect_regions(labels, faces)[0]
    hull = _hull(vertices, border, original_neighbors)
    local_vertices = forward[np.asarray(vertices + hull, np.int32)]
    local = np.any(np.isin(base_edges, local_vertices), axis=1)
    local_edges = np.ascontiguousarray(base_edges[local])
    candidates = np.ascontiguousarray(np.column_stack((capture["vno1"], capture["vno2"])), np.int32)
    used = np.ascontiguousarray(capture["used"], np.int16)
    accepted = accept_first_candidate_edges(corrected_sphere, candidates, used, local_edges)
    selected = set(map(tuple, np.sort(candidates[accepted], axis=1)))
    native_vertices, native_faces = fsio.read_geometry(str(patch))
    degrees = np.fromiter((len(row) for row in original_neighbors), np.int32,
                          count=len(orig))
    nbrs = np.zeros((len(orig), int(degrees.max())), np.int32)
    for vertex, row in enumerate(original_neighbors):
        nbrs[vertex, :len(row)] = row
    smooth = _smooth(np.ascontiguousarray(orig, np.float32), nbrs, degrees)
    predicted_vertices = np.ascontiguousarray(smooth[np.argsort(forward)], np.float32)
    native_coordinates = np.ascontiguousarray(native_vertices, np.float32)
    coordinate_equal = np.all(predicted_vertices.view(np.uint32) ==
                              native_coordinates.view(np.uint32), axis=1)
    coordinate_differences = np.abs(predicted_vertices - native_coordinates)
    if not np.array_equal(native_faces[:len(base_faces)], base_faces):
        raise ValueError("native patch does not preserve expected ordered base faces")
    native_new = set(map(tuple, _edges(native_faces))) - set(map(tuple, base_edges))
    native_face_set = set(map(tuple, np.sort(native_faces[len(base_faces):], axis=1)))
    predicted_face_set = _triangle_faces(corrected_sphere, base_edges, selected)
    selected_vertices = set(candidates[accepted].reshape(-1).tolist())
    active_vertices = [int(forward[v]) for v in vertices
                       if status[v] == 1 and int(forward[v]) in selected_vertices]
    patch_inside_vertices = len(active_vertices)
    active_vertices.extend(int(forward[v]) for v in border)
    face_order = _candidate_face_order(corrected_sphere, base_faces, neighbors, candidates,
                                       accepted, active_vertices, predicted_face_set)
    predicted_full_faces = np.concatenate((base_faces, face_order))
    predicted_topology_edges = np.concatenate((base_edges, candidates[accepted]))
    invalid_edges = invalid_modified_edges(predicted_topology_edges, predicted_full_faces,
                                           np.asarray(active_vertices, np.int32))
    native_added_faces = native_faces[len(base_faces):]
    nfaces = min(len(face_order), len(native_added_faces))
    order_diff = np.flatnonzero(np.any(face_order[:nfaces] != native_added_faces[:nfaces], axis=1))
    native_accept = np.fromiter((tuple(sorted(row)) in native_new for row in candidates),
                                np.bool_, count=len(candidates))
    mismatches = np.flatnonzero(accepted != native_accept)
    first = int(mismatches[0]) if len(mismatches) else None
    result = {
        "hemisphere": hemi,
        "candidate_source": "python surface and MRI inputs" if independent else "native EDGE capture",
        "native_snapshot": str(patch),
        "source_vertices": len(orig),
        "base_vertices": int(np.count_nonzero(labels == 0)),
        "base_faces": len(base_faces),
        "native_snapshot_vertices": len(native_vertices),
        "native_snapshot_faces": len(native_faces),
        "ordered_vertex_coordinate_exact_positions": int(np.count_nonzero(coordinate_equal)),
        "first_ordered_vertex_coordinate_difference": int(np.flatnonzero(~coordinate_equal)[0])
        if not np.all(coordinate_equal) else None,
        "max_ordered_vertex_coordinate_abs_difference_mm": float(coordinate_differences.max()),
        "native_added_faces": len(native_faces) - len(base_faces),
        "defect_vertices": len(vertices),
        "defect_border": len(border),
        "defect_hull": len(hull),
        "local_base_edges": len(local_edges),
        "candidate_edges": len(candidates),
        "native_new_edges": len(native_new),
        "native_added_faces_set_size": len(native_face_set),
        "python_triangle_faces_set_size": len(predicted_face_set),
        "python_only_triangle_faces": len(predicted_face_set - native_face_set),
        "native_only_triangle_faces": len(native_face_set - predicted_face_set),
        "python_generated_faces": len(face_order),
        "ordered_face_exact_positions": int(nfaces - len(order_diff)),
        "first_ordered_face_difference": int(order_diff[0]) if len(order_diff) else None,
        "first_ordered_face_native": native_added_faces[order_diff[0]].tolist() if len(order_diff) else None,
        "first_ordered_face_python": face_order[order_diff[0]].tolist() if len(order_diff) else None,
        "python_accepted_edges": len(selected),
        "first_candidate_patch_inside_vertices": patch_inside_vertices,
        "first_candidate_inside_vertex_indices": active_vertices[:patch_inside_vertices],
        "first_candidate_inside_neighbor_rows": [neighbors[v] for v in active_vertices[:patch_inside_vertices]],
        "first_candidate_patch_euler": patch_inside_vertices - len(selected) + len(face_order),
        "first_candidate_invalid_modified_edges": invalid_edges,
        "intersection_size": len(selected & native_new),
        "python_only_edges": len(selected - native_new),
        "native_only_edges": len(native_new - selected),
        "first_candidate_membership_mismatch": first,
        "first_candidate_corrected_pair": candidates[first].tolist() if first is not None else None,
        "first_candidate_used_flag": int(used[first]) if first is not None else None,
        "first_candidate_native_accept": bool(native_accept[first]) if first is not None else None,
        "first_candidate_python_accept": bool(accepted[first]) if first is not None else None,
        "first_candidate_ordered_surface_exact": bool(
            np.all(coordinate_equal) and len(face_order) == len(native_added_faces)
            and not len(order_diff) and len(mismatches) == 0),
    }
    if final_patch is not None:
        final_vertices, final_faces = fsio.read_geometry(str(final_patch))
        final_coordinates = np.ascontiguousarray(final_vertices, np.float32)
        shared_faces = min(len(final_faces), len(native_faces))
        face_equal = np.all(final_faces[:shared_faces] == native_faces[:shared_faces], axis=1)
        final_set = set(map(tuple, np.sort(final_faces[len(base_faces):], axis=1)))
        result["native_final_best"] = {
            "path": str(final_patch),
            "vertices": len(final_vertices),
            "faces": len(final_faces),
            "base_face_prefix_exact": bool(np.array_equal(final_faces[:len(base_faces)], base_faces)),
            "ordered_vertex_coordinate_exact_positions": int(np.count_nonzero(np.all(
                final_coordinates.view(np.uint32) == predicted_vertices.view(np.uint32), axis=1))),
            "ordered_face_exact_positions_against_initial_candidate": int(np.count_nonzero(face_equal)),
            "first_face_difference_against_initial_candidate": int(np.flatnonzero(~face_equal)[0])
            if not np.all(face_equal) else (shared_faces if len(final_faces) != len(native_faces) else None),
            "added_faces_set_size": len(final_set),
            "added_faces_set_overlap_with_initial_candidate": len(final_set & native_face_set),
            "matches_first_candidate_ordered_surface": bool(
                np.array_equal(final_faces, native_faces) and
                np.array_equal(final_coordinates.view(np.uint32), predicted_vertices.view(np.uint32))),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", required=True, type=Path)
    ap.add_argument("--captures", type=Path)
    ap.add_argument("--patch", required=True, type=Path)
    ap.add_argument("--final-patch", type=Path)
    ap.add_argument("--hemi", choices=("lh", "rh"), required=True)
    ap.add_argument("--independent", action="store_true")
    args = ap.parse_args()
    if not args.independent and args.captures is None:
        ap.error("--captures is required without --independent")
    print(json.dumps(run(args.diagnostics, args.captures or Path(), args.patch,
                         args.hemi, args.final_patch, args.independent), indent=2))


if __name__ == "__main__":
    main()
