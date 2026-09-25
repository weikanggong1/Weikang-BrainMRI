"""Trace selected pial candidate moves against source and installed RAM meshes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

STATE = np.dtype([("floats", "<f4", 9), ("flags", "<i4", 3)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--vertices", type=int, nargs="+", required=True)
    parser.add_argument("--installed-previous", type=Path, required=True)
    parser.add_argument("--installed-current", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from fnit.recon_all.place_surface_collision import (
        _candidate_collision, _moved_face_geometry, _project_close_neighbors,
        asynchronous_first_step, subvolume_assignment,
    )
    from fnit.recon_all.place_surface_smoothing import _ordered_neighbors
    from fnit.recon_all.place_surface_step import unconstrained_step_with_offsets

    prefix = args.source_run / f"lh.gradient.step{args.step:02d}"
    clear = np.fromfile(str(prefix) + ".clear", dtype=STATE)
    final = np.fromfile(str(prefix) + ".tangential_spring", dtype=STATE)
    source_after = np.fromfile(str(prefix) + ".after_collision", dtype=STATE)
    start = clear["floats"][:, :3].copy()
    gradient = final["floats"][:, 6:9]
    ripped = clear["flags"][:, 0].astype(bool)
    _, faces = nib.freesurfer.read_geometry(args.subject / "surf/lh.white")
    faces = np.asarray(faces, dtype=np.int32)
    installed_previous, installed_previous_faces = nib.freesurfer.read_geometry(args.installed_previous)
    installed_current, installed_current_faces = nib.freesurfer.read_geometry(args.installed_current)
    if not np.array_equal(faces, installed_previous_faces) or not np.array_equal(faces, installed_current_faces):
        raise ValueError("installed/source ordered faces differ")
    proposed, offsets = unconstrained_step_with_offsets(start, gradient, ripped, dt=0.5)
    geometry, _, svi = subvolume_assignment(start, faces, proposed, ripped)
    order = np.concatenate([np.flatnonzero(svi == region) for region in range(65)])
    neighbors, neighbor_valid, _ = _ordered_neighbors(faces, len(start))
    triangles = start[faces]
    centers = triangles.mean(axis=1, dtype=np.float64)
    radii = np.linalg.norm(triangles.astype(np.float64) - centers[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centers)
    max_radius = float(radii.max())
    report = {
        "reference_class": "source-order Python candidate and collision trace from pinned-source gradient; installed candidate gradient is not captured, only installed accepted RAM coordinates",
        "step": args.step,
        "dt": 0.5,
        "vertices": [],
    }
    for vertex in args.vertices:
        position = int(np.flatnonzero(order == vertex)[0])
        before, _ = asynchronous_first_step(
            start, faces, proposed, ripped, limit=position, fast=True,
            offsets=offsets, ordered_neighbors=(neighbors, neighbor_valid),
        )
        projected, valid = _project_close_neighbors(
            before, vertex, neighbors, neighbor_valid, offsets[vertex],
            geometry, int(svi[vertex]), np.float32(0.01),
        )
        endpoint = np.float32(start[vertex] + projected)
        near_neighbors = []
        for slot in range(neighbors.shape[1]):
            if not neighbor_valid[vertex, slot]:
                continue
            other = int(neighbors[vertex, slot])
            gap = float(np.linalg.norm(before[other].astype(np.float64) - before[vertex].astype(np.float64)))
            if gap <= 0.01:
                near_neighbors.append({"vertex": other, "distance_mm": gap})
        incident = np.flatnonzero(np.any(faces == vertex, axis=1))
        collision_face = None
        broadphase_count = 0
        for face_id in (incident if valid and np.any(proposed[vertex] != start[vertex]) else []):
            moved, center, radius, low, high = _moved_face_geometry(
                before, faces, int(face_id), vertex, endpoint,
            )
            nearby = np.asarray(tree.query_ball_point(center, radius + max_radius + 1.0), dtype=np.int32)
            broadphase_count += len(nearby)
            for other in nearby:
                if _candidate_collision(
                    before, faces, moved, faces[face_id], low, high,
                    np.asarray([other], dtype=np.int32),
                ):
                    collision_face = {"moving_face": int(face_id), "other_face": int(other)}
                    break
            if collision_face is not None:
                break
        installed_delta = installed_current[vertex].astype(np.float64) - installed_previous[vertex].astype(np.float64)
        source_delta = source_after["floats"][vertex, :3].astype(np.float64) - start[vertex].astype(np.float64)
        predicted_accept = bool(valid and collision_face is None and np.any(proposed[vertex] != start[vertex]))
        predicted_source = endpoint if predicted_accept else start[vertex]
        report["vertices"].append({
            "vertex": vertex,
            "source_order_position": position,
            "source_subvolume": int(svi[vertex]),
            "source_start_xyz": start[vertex].tolist(),
            "source_gradient": gradient[vertex].tolist(),
            "source_clipped_offset": offsets[vertex].tolist(),
            "source_clipped_endpoint": proposed[vertex].tolist(),
            "source_close_neighbors_mm": near_neighbors,
            "source_projection_valid": bool(valid),
            "source_projected_offset": projected.tolist(),
            "source_projected_endpoint": endpoint.tolist(),
            "source_broadphase_faces_visited": broadphase_count,
            "source_first_triangle_collision": collision_face,
            "source_predicted_accept": predicted_accept,
            "source_prediction_matches_native_checkpoint": bool(np.array_equal(
                predicted_source, source_after["floats"][vertex, :3],
            )),
            "source_accepted_delta_mm": source_delta.tolist(),
            "source_moved": bool(np.any(source_delta != 0)),
            "installed_previous_xyz": installed_previous[vertex].tolist(),
            "installed_current_xyz": installed_current[vertex].tolist(),
            "installed_accepted_delta_mm": installed_delta.tolist(),
            "installed_moved": bool(np.any(installed_delta != 0)),
            "pre_step_source_installed_distance_mm": float(np.linalg.norm(
                start[vertex].astype(np.float64) - installed_previous[vertex].astype(np.float64),
            )),
            "post_step_source_installed_distance_mm": float(np.linalg.norm(
                source_after["floats"][vertex, :3].astype(np.float64)
                - installed_current[vertex].astype(np.float64),
            )),
        })
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
