"""Isolated source-order probe for pinned mris_fix_topology EDGE.len.

This is a validation probe, not a recon-all stage.  It stops before qsort/GA.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from numba import njit

from fnit.recon_all.smooth_surface_python import ordered_neighbors
from fnit.recon_all.topology_preflight_python import genetic_base_translation
from validation.recon_all.python_gpu_port.validate_topology_edge_table import EDGE_DTYPE


@njit
def _unit(v):
    d = np.float32(np.sqrt(float(np.float32(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))))
    if d > 0:
        v[0] = np.float32(v[0] / d)
        v[1] = np.float32(v[1] / d)
        v[2] = np.float32(v[2] / d)
    return v


@njit
def _vertex_normals(xyz, faces, offsets, face_ids, corners, ripped):
    result = np.zeros_like(xyz)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        norm = np.zeros(3, np.float32)
        for slot in range(offsets[vertex], offsets[vertex + 1]):
            f = faces[face_ids[slot]]
            corner = corners[slot]
            v0 = xyz[f[(corner + 2) % 3]]
            v1 = xyz[f[(corner + 1) % 3]]
            center = xyz[vertex]
            a = _unit((center - v0).copy())
            b = _unit((v1 - center).copy())
            contribution = np.empty(3, np.float32)
            contribution[0] = np.float32(-b[1] * a[2] + a[1] * b[2])
            contribution[1] = np.float32(b[0] * a[2] - a[0] * b[2])
            contribution[2] = np.float32(-b[0] * a[1] + a[0] * b[1])
            contribution = _unit(contribution)
            norm[0] = np.float32(norm[0] + contribution[0])
            norm[1] = np.float32(norm[1] + contribution[1])
            norm[2] = np.float32(norm[2] + contribution[2])
        result[vertex] = _unit(norm)
    return result


@njit
def _smooth(xyz, nbrs, degrees):
    current = xyz.copy()
    for _ in range(2):
        nxt = np.empty_like(current)
        for vertex in range(len(current)):
            x, y, z = current[vertex]
            for slot in range(degrees[vertex]):
                neighbor = nbrs[vertex, slot]
                x = np.float32(x + current[neighbor, 0])
                y = np.float32(y + current[neighbor, 1])
                z = np.float32(z + current[neighbor, 2])
            d = np.float32(1 + degrees[vertex])
            nxt[vertex, 0] = np.float32(x / d)
            nxt[vertex, 1] = np.float32(y / d)
            nxt[vertex, 2] = np.float32(z / d)
        current = nxt
    return current


@njit
def _sample(volume, ras, inv_tkr):
    # mriSurfaceRASToVoxel casts RAS and its matrix result to float before
    # MRIsampleVolume receives voxel coordinates as double arguments.
    xr, yr, zr = np.float32(ras[0]), np.float32(ras[1]), np.float32(ras[2])
    x = float(np.float32(inv_tkr[0, 0] * xr + inv_tkr[0, 1] * yr + inv_tkr[0, 2] * zr + inv_tkr[0, 3]))
    y = float(np.float32(inv_tkr[1, 0] * xr + inv_tkr[1, 1] * yr + inv_tkr[1, 2] * zr + inv_tkr[1, 3]))
    z = float(np.float32(inv_tkr[2, 0] * xr + inv_tkr[2, 1] * yr + inv_tkr[2, 2] * zr + inv_tkr[2, 3]))
    if x < -0.5 or y < -0.5 or z < -0.5 or x > volume.shape[0] - 0.5 or y > volume.shape[1] - 0.5 or z > volume.shape[2] - 0.5:
        return 0.0
    x = min(max(x, 0.0), volume.shape[0] - 1.0)
    y = min(max(y, 0.0), volume.shape[1] - 1.0)
    z = min(max(z, 0.0), volume.shape[2] - 1.0)
    i, j, k = int(x), int(y), int(z)
    ip, jp, kp = min(i + 1, volume.shape[0] - 1), min(j + 1, volume.shape[1] - 1), min(k + 1, volume.shape[2] - 1)
    dx, dy, dz = x - float(np.float32(i)), y - float(np.float32(j)), z - float(np.float32(k))
    ax, ay, az = 1.0 - dx, 1.0 - dy, 1.0 - dz
    return (ax * ay * az * volume[i, j, k] + ax * ay * dz * volume[i, j, kp]
            + ax * dy * az * volume[i, jp, k] + ax * dy * dz * volume[i, jp, kp]
            + dx * ay * az * volume[ip, j, k] + dx * ay * dz * volume[ip, j, kp]
            + dx * dy * az * volume[ip, jp, k] + dx * dy * dz * volume[ip, jp, kp])


@njit
def _border_values(xyz, normal, ripped, volume, inv_tkr):
    white = np.zeros(len(xyz), np.float32)
    gray = np.zeros(len(xyz), np.float32)
    for vertex in range(len(xyz)):
        if ripped[vertex]:
            continue
        point = xyz[vertex].astype(np.float64)
        direction = normal[vertex].astype(np.float64)
        white[vertex] = np.float32(_sample(volume, point - 0.5 * direction, inv_tkr))
        gray[vertex] = np.float32(_sample(volume, point + 0.5 * direction, inv_tkr))
    return white, gray


@njit
def _median_twice(values, ripped, nbrs, degrees):
    current = values.copy()
    for _ in range(2):
        nxt = current.copy()
        for vertex in range(len(values)):
            if ripped[vertex]:
                continue
            buf = np.empty(degrees[vertex] + 1, np.float32)
            buf[0] = current[vertex]
            n = 1
            for slot in range(degrees[vertex]):
                neighbor = nbrs[vertex, slot]
                if not ripped[neighbor]:
                    buf[n] = current[neighbor]
                    n += 1
            sorted_buf = np.sort(buf[:n])
            if n % 2:
                nxt[vertex] = sorted_buf[n // 2]
            else:
                nxt[vertex] = np.float32((sorted_buf[n // 2] + sorted_buf[n // 2 - 1]) / np.float32(2))
        current = nxt
    return current


@njit
def _orig_normals(xyz, faces, offsets, face_ids, corners, vertices):
    result = np.zeros((len(vertices), 3), np.float32)
    for index in range(len(vertices)):
        vertex = vertices[index]
        norm = np.zeros(3, np.float32)
        for slot in range(offsets[vertex], offsets[vertex + 1]):
            f = faces[face_ids[slot]]
            corner = corners[slot]
            v0 = xyz[vertex] - xyz[f[(corner + 2) % 3]]
            v1 = xyz[f[(corner + 1) % 3]] - xyz[vertex]
            a = _unit(v0.copy())
            b = _unit(v1.copy())
            norm[0] = np.float32(norm[0] + np.float32(-b[1] * a[2] + a[1] * b[2]))
            norm[1] = np.float32(norm[1] + np.float32(b[0] * a[2] - a[0] * b[2]))
            norm[2] = np.float32(norm[2] + np.float32(-b[0] * a[1] + a[0] * b[1]))
        result[index] = _unit(norm)
    return result


@njit
def _scores(edges, source_ids, orig, normals, white, gray, volume, inv_tkr):
    score = np.empty(len(edges), np.float32)
    for index in range(len(edges)):
        a, b = source_ids[index]
        norm = np.empty(3, np.float32)
        for axis in range(3):
            norm[axis] = np.float32((normals[index, axis] + normals[index, axis + 3]) / np.float32(2))
        length = np.sqrt(float(norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]))
        if length == 0:
            length = 1.0
        norm = (norm.astype(np.float64) / length).astype(np.float32)
        wval = float(np.float32((white[a] + white[b]) / np.float32(2)))
        gval = float(np.float32((gray[a] + gray[b]) / np.float32(2)))
        delta = (orig[b] - orig[a]).astype(np.float64)
        total = 0.0
        d = 0.1
        for step in range(11):
            if step == 0:
                point = orig[a].astype(np.float64)
            elif step == 1:
                point = orig[b].astype(np.float64)
            else:
                point = orig[a].astype(np.float64) + d * delta
                d += 0.1
            total += abs(_sample(volume, point + 0.5 * norm, inv_tkr) - gval)
            total += abs(_sample(volume, point - 0.5 * norm, inv_tkr) - wval)
        score[index] = np.float32(total / 22.0)
        if edges[index, 2] == 0:
            score[index] = np.float32(score[index] + np.float32(100))
    return score


def _face_index(faces, nvertices):
    slots = [[] for _ in range(nvertices)]
    for face, row in enumerate(faces):
        for corner, vertex in enumerate(row):
            slots[int(vertex)].append((face, corner))
    offsets = np.zeros(nvertices + 1, np.int32)
    for vertex, row in enumerate(slots):
        offsets[vertex + 1] = offsets[vertex] + len(row)
    face_ids = np.fromiter((face for row in slots for face, _ in row), np.int32)
    corners = np.fromiter((corner for row in slots for _, corner in row), np.int32)
    return offsets, face_ids, corners


class _Edge(ctypes.Structure):
    _fields_ = [("vno1", ctypes.c_int), ("vno2", ctypes.c_int),
                ("length", ctypes.c_float), ("used", ctypes.c_short),
                ("padding", ctypes.c_short)]


_CMP_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)


@_CMP_TYPE
def _compare_edge_length(left, right):
    a = ctypes.cast(left, ctypes.POINTER(_Edge)).contents
    b = ctypes.cast(right, ctypes.POINTER(_Edge)).contents
    if a.length > b.length:
        return 1
    if a.length < b.length:
        return -1
    if a.vno1 > b.vno1:
        return 1
    return -1


def _native_qsort(edges, lengths):
    rows = edges.copy()
    rows["length"] = lengths
    libc = ctypes.CDLL(None)
    libc.qsort.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_size_t, _CMP_TYPE]
    libc.qsort(rows.ctypes.data, len(rows), rows.dtype.itemsize,
               _compare_edge_length)
    return np.column_stack((rows["vno1"], rows["vno2"]))


def run(diagnostics: Path, captures: Path, hemi: str, limit: int) -> dict:
    surf = diagnostics / "fs_sub01" / "surf"
    mri = nib.load(str(diagnostics / "fs_sub01" / "mri" / "brain.mgz"))
    volume = np.ascontiguousarray(np.asarray(mri.dataobj))
    inv_tkr = np.linalg.inv(mri.header.get_vox2ras_tkr())
    orig, faces = fsio.read_geometry(str(surf / f"{hemi}.orig.nofix"))
    orig = np.ascontiguousarray(orig.astype(np.float32))
    faces = np.ascontiguousarray(faces.astype(np.int32))
    labels = np.rint(fsio.read_morph_data(str(surf / f"{hemi}.defect_labels"))).astype(np.int32)
    ripped = labels != 0
    capture = np.fromfile(captures / f"capture_{hemi}" / f"{hemi}.edge.before.bin", EDGE_DTYPE)
    if limit:
        capture = capture[:limit]
    neighbors = ordered_neighbors(faces, len(orig))
    degrees = np.fromiter((len(row) for row in neighbors), np.int32, count=len(orig))
    nbrs = np.zeros((len(orig), int(degrees.max())), np.int32)
    for vertex, row in enumerate(neighbors):
        nbrs[vertex, :len(row)] = row
    offsets, face_ids, corners = _face_index(faces, len(orig))
    normal = _vertex_normals(orig, faces, offsets, face_ids, corners, ripped)
    white, gray = _border_values(orig, normal, ripped, volume, inv_tkr)
    white, gray = _median_twice(white, ripped, nbrs, degrees), _median_twice(gray, ripped, nbrs, degrees)
    smooth = _smooth(orig, nbrs, degrees)
    forward, _ = genetic_base_translation(labels, faces)
    reverse = np.argsort(forward)
    source_ids = reverse[np.column_stack((capture["vno1"], capture["vno2"]))]
    active = np.unique(source_ids)
    orig_norm = _orig_normals(smooth, faces, offsets, face_ids, corners, active)
    normal_lookup = np.zeros((len(orig), 3), np.float32)
    normal_lookup[active] = orig_norm
    edge_norm = np.concatenate((normal_lookup[source_ids[:, 0]], normal_lookup[source_ids[:, 1]]), axis=1)
    edge_data = np.column_stack((capture["vno1"], capture["vno2"], capture["used"]))
    score = _scores(edge_data, source_ids, smooth, edge_norm, white, gray, volume, inv_tkr)
    native = capture["length"]
    diff = np.abs(score - native)
    first = int(np.flatnonzero(diff != 0)[0]) if np.any(diff != 0) else None
    first_large = int(np.flatnonzero(diff > 0.01)[0]) if np.any(diff > 0.01) else None
    max_at = int(np.argmax(diff))
    worst = [int(i) for i in np.argsort(diff)[-5:][::-1]]
    details = []
    for i in dict.fromkeys([index for index in (first_large, max_at, *worst) if index is not None]):
        a, b = source_ids[i]
        details.append({"index": i, "native": float(native[i]), "python": float(score[i]),
                        "source_pair": [int(a), int(b)], "corrected_pair": edge_data[i, :2].tolist(),
                        "used": int(edge_data[i, 2]), "ripped": [bool(ripped[a]), bool(ripped[b])],
                        "white_gray": [[float(white[v]), float(gray[v])] for v in (a, b)]})
    native_after = np.fromfile(captures / f"capture_{hemi}" / f"{hemi}.edge.after.bin", EDGE_DTYPE)
    if limit:
        native_after = native_after[:limit]
    predicted_order = np.lexsort((capture["vno1"], score))
    if not limit:
        actual_pairs = np.column_stack((native_after["vno1"], native_after["vno2"]))
        predicted_pairs = np.column_stack((capture["vno1"][predicted_order], capture["vno2"][predicted_order]))
        mismatch = np.flatnonzero(np.any(actual_pairs != predicted_pairs, axis=1))
        first_order_difference = int(mismatch[0]) if len(mismatch) else None
        exact_order_count = int(len(score) - len(mismatch))
        native_lex_order = np.lexsort((capture["vno1"], native))
        native_lex_pairs = np.column_stack((capture["vno1"][native_lex_order], capture["vno2"][native_lex_order]))
        native_lex_mismatch = int(np.count_nonzero(np.any(actual_pairs != native_lex_pairs, axis=1)))
        native_qsort_pairs = _native_qsort(capture, native)
        python_qsort_pairs = _native_qsort(capture, score)
        native_qsort_mismatch = int(np.count_nonzero(np.any(actual_pairs != native_qsort_pairs, axis=1)))
        python_qsort_mismatch = int(np.count_nonzero(np.any(actual_pairs != python_qsort_pairs, axis=1)))
        key_to_before = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(capture["vno1"], capture["vno2"]))}
        sort_details = []
        for rank in mismatch[:10]:
            actual_pair = actual_pairs[rank].tolist()
            predicted_pair = predicted_pairs[rank].tolist()
            actual_index = key_to_before[tuple(actual_pair)]
            predicted_index = key_to_before[tuple(predicted_pair)]
            sort_details.append({"rank": int(rank), "native_pair": actual_pair,
                                 "native_pair_score": float(native[actual_index]),
                                 "python_pair": predicted_pair,
                                 "python_pair_native_score": float(native[predicted_index]),
                                 "python_pair_python_score": float(score[predicted_index]),
                                 "native_score_gap": float(abs(native[actual_index] - native[predicted_index]))})
        max_mismatch_native_gap = float(max((abs(native[key_to_before[tuple(actual_pairs[rank])]]
                                                 - native[key_to_before[tuple(predicted_pairs[rank])]])
                                             for rank in mismatch), default=0))
    else:
        first_order_difference = None
        exact_order_count = None
        native_lex_mismatch = None
        native_qsort_mismatch = None
        python_qsort_mismatch = None
        sort_details = []
        max_mismatch_native_gap = None
    return {"hemisphere": hemi, "edges": len(score), "first_difference": first,
            "first_native": float(native[first]) if first is not None else None,
            "first_python": float(score[first]) if first is not None else None,
            "max_abs_difference": float(diff.max(initial=0)),
            "mean_abs_difference": float(diff.mean()),
            "bitwise_equal_scores": int(np.count_nonzero(diff == 0)),
            "within_1e_5": int(np.count_nonzero(diff <= 1e-5)),
            "within_1e_3": int(np.count_nonzero(diff <= 1e-3)),
            "within_1e_2": int(np.count_nonzero(diff <= 1e-2)),
            "first_difference_over_0_01": first_large,
            "largest_difference_index": max_at,
            "difference_details": details,
            "predicted_sort_first_difference": first_order_difference,
            "predicted_sort_exact_positions": exact_order_count,
            "native_lexsort_disagreements_with_native_qsort": native_lex_mismatch,
            "native_replay_qsort_disagreements": native_qsort_mismatch,
            "python_score_qsort_disagreements": python_qsort_mismatch,
            "max_native_score_gap_at_mismatched_ranks": max_mismatch_native_gap,
            "sort_difference_details": sort_details,
            "predicted_sort_head": [[int(capture["vno1"][i]), int(capture["vno2"][i]), float(score[i])]
                                    for i in predicted_order[:10]],
            "top_10": [[int(i), float(native[i]), float(score[i])] for i in range(min(10, len(score)))],
            "source_first": source_ids[0].tolist(),
            "white_gray_first": [[float(white[v]), float(gray[v])] for v in source_ids[0]],
            "normal_first": edge_norm[0].tolist(),
            "smooth_first": smooth[source_ids[0]].tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", type=Path, required=True)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--hemi", choices=("lh", "rh"), default="lh")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(run(args.diagnostics, args.captures, args.hemi, args.limit), indent=2))


if __name__ == "__main__":
    main()
