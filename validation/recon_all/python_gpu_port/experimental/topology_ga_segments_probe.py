"""Check the source-order GA edge clusters against native saved annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit

from fnit.recon_all.topology_preflight_python import (
    _edge_intersects, center_sphere, defect_regions, genetic_base_translation,
    project_and_smooth_sphere,
)
from validation.recon_all.python_gpu_port.validate_topology_edge_table import EDGE_DTYPE


@njit
def _overlaps(sphere: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.zeros((len(edges), len(edges)), np.bool_)
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            result[i, j] = result[j, i] = _edge_intersects(
                sphere, edges[i, 0], edges[i, 1], edges[j, 0], edges[j, 1])
    return result


def _clusters(edges: np.ndarray, overlaps: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    incident: list[list[int]] = [[] for _ in range(int(edges.max()) + 1)]
    for index, (a, b) in enumerate(edges):
        incident[a].append(index)
        incident[b].append(index)
    groups: list[list[int]] = []
    exclusions: list[set[int]] = []
    assigned = np.full(len(edges), -1, np.int32)

    def compatible(group: int, edge: int) -> bool:
        return (group >= 0 and edge not in exclusions[group] and
                not any(overlaps[edge, member] for member in groups[group]))

    def add(group: int, edge: int) -> None:
        groups[group].append(edge)
        assigned[edge] = group
        exclusions[group].update(np.flatnonzero(overlaps[edge]).tolist())

    for edge, (a, b) in enumerate(edges):
        nearby: list[int] = []
        for vertex in (a, b):
            for neighbor in incident[vertex]:
                group = int(assigned[neighbor])
                if group not in nearby and compatible(group, edge):
                    nearby.append(group)
        if not nearby:
            try:
                target = next(i for i, row in enumerate(groups) if not row)
            except StopIteration:
                target = len(groups)
                groups.append([])
                exclusions.append(set())
            add(target, edge)
        elif len(nearby) == 1:
            add(nearby[0], edge)
        else:
            for _ in range(1, len(nearby)):
                if len(groups[nearby[1]]) < len(groups[nearby[0]]):
                    nearby[0], nearby[1] = nearby[1], nearby[0]
            target = nearby[0]
            add(target, edge)
            for other in nearby[1:]:
                if any(member in exclusions[other] for member in groups[target]) or \
                        any(member in exclusions[target] for member in groups[other]):
                    continue
                for member in groups[other]:
                    assigned[member] = target
                groups[target].extend(groups[other])
                exclusions[target].update(exclusions[other])
                groups[other].clear()
                exclusions[other].clear()

    active = [index for index, row in enumerate(groups) if row]
    active.sort(key=lambda index: -len(groups[index]))
    remap = {old: new for new, old in enumerate(active)}
    reordered = [groups[old] for old in active]
    assigned = np.asarray([remap[int(x)] for x in assigned], np.int32)
    for group in range(3, len(reordered)):
        if group >= 10 or len(reordered[group]) < 5:
            for edge in reordered[group]:
                assigned[edge] = -1
            reordered[group] = []
    changed = True
    while changed:
        changed = False
        for edge, (a, b) in enumerate(edges):
            if assigned[edge] >= 0:
                continue
            target = -1
            size = -1
            for vertex in (a, b):
                for neighbor in incident[vertex]:
                    group = int(assigned[neighbor])
                    if group < 0:
                        continue
                    if size == -1:
                        target, size = group, len(reordered[group])
                    elif len(reordered[group]) < size:
                        target, size = group, len(reordered[target])
            if target >= 0:
                assigned[edge] = target
                reordered[target].append(edge)
                changed = True
    return reordered, assigned


def _native_annotation(path: Path, nvertices: int) -> np.ndarray:
    raw = path.read_bytes()
    if int.from_bytes(raw[:4], "big") != nvertices:
        raise ValueError("annotation vertex count differs from frozen surface")
    return np.frombuffer(raw, dtype=">i4", count=2 * nvertices, offset=4).reshape(-1, 2)[:, 1]


def run(diagnostics: Path, captures: Path, annotation: Path, hemi: str) -> dict:
    surf = diagnostics / "fs_sub01" / "surf"
    sphere, faces = fsio.read_geometry(str(surf / f"{hemi}.qsphere.nofix"))
    labels = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_labels"))).astype(np.int32)
    forward, _ = genetic_base_translation(labels, faces)
    sphere, _ = center_sphere(project_and_smooth_sphere(sphere, faces))
    sphere = np.ascontiguousarray(sphere[np.argsort(forward)], np.float32)
    capture = np.fromfile(captures / f"capture_{hemi}" / f"{hemi}.edge.after.bin", EDGE_DTYPE)
    old_edges = np.column_stack((capture["vno1"][capture["used"] == 2],
                                 capture["vno2"][capture["used"] == 2])).astype(np.int32)
    vertices, border = defect_regions(labels, faces)[0]
    inside = set(forward[np.asarray(vertices, np.int32)].tolist())
    first = [row for row in old_edges if int(row[0]) in inside and int(row[1]) in inside]
    second = [row for row in old_edges if int(row[0]) not in inside or int(row[1]) not in inside]
    edges = np.ascontiguousarray(first + second, np.int32)
    groups, assigned = _clusters(edges, _overlaps(sphere, edges))
    # First three saveSegmentation RGB colors after signed 24-bit annotation packing.
    colors = np.asarray([-1974016, 5127885, 5127800], np.int32)
    predicted = np.zeros(len(labels), np.int32)
    reverse = np.argsort(forward)
    for index, (a, b) in enumerate(edges):
        group = assigned[index]
        if 0 <= group < len(colors):
            predicted[reverse[a]] = predicted[reverse[b]] = colors[group]
    native = _native_annotation(annotation, len(labels))
    relevant = np.asarray(vertices + border, np.int32)
    differences = relevant[predicted[relevant] != native[relevant]]
    return {
        "hemisphere": hemi,
        "source_old_edges": len(edges),
        "cluster_sizes": [len(row) for row in groups if row],
        "unassigned_edges": int(np.count_nonzero(assigned < 0)),
        "native_annotation_nonzero": int(np.count_nonzero(native)),
        "predicted_annotation_nonzero": int(np.count_nonzero(predicted)),
        "defect_and_border_annotation_exact_positions": int(len(relevant) - len(differences)),
        "defect_and_border_annotation_positions": len(relevant),
        "first_annotation_difference": int(differences[0]) if len(differences) else None,
        "first_annotation_native": int(native[differences[0]]) if len(differences) else None,
        "first_annotation_python": int(predicted[differences[0]]) if len(differences) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", required=True, type=Path)
    ap.add_argument("--captures", required=True, type=Path)
    ap.add_argument("--annotation", required=True, type=Path)
    ap.add_argument("--hemi", required=True, choices=("lh", "rh"))
    args = ap.parse_args()
    print(json.dumps(run(args.diagnostics, args.captures, args.annotation, args.hemi), indent=2))


if __name__ == "__main__":
    main()
