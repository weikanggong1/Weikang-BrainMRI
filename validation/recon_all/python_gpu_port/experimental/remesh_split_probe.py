"""Isolated source-order mris_remesh split-phase probe; not a full stage."""

import argparse
import heapq
import json
import math
from pathlib import Path
import struct

import nibabel.freesurfer as fs
import numpy as np


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def edge_length(points, a, b):
    u, v = points[a], points[b]
    x = u[0] - v[0]
    y = u[1] - v[1]
    z = u[2] - v[2]
    return math.sqrt(x * x + y * y + z * z)


def initial_topology(faces):
    edge_index = {}
    edge_vertices = []
    edge_faces = []
    face_edges = []
    for ti, (a, b, c) in enumerate(faces):
        row = []
        for u, v in ((a, b), (b, c), (c, a)):
            key = edge_key(u, v)
            ei = edge_index.get(key)
            if ei is None:
                ei = len(edge_vertices)
                edge_index[key] = ei
                edge_vertices.append([key[0], key[1]])
                edge_faces.append([ti])
            else:
                edge_faces[ei].append(ti)
            row.append(ei)
        face_edges.append(row)
    return edge_index, edge_vertices, edge_faces, face_edges


def split_edge(points, faces, edge_index, edge_vertices, edge_faces, face_edges, ei):
    a, b = edge_vertices[ei]
    v1, v2 = edge_key(a, b)
    assert edge_index[(v1, v2)] == ei
    ni = len(points)
    points.append(tuple(v1c * 0.5 + v2c * (1.0 - 0.5)
                        for v1c, v2c in zip(points[v1], points[v2])))
    en = len(edge_vertices)
    edge_vertices.append([ni, v2])
    edge_index[edge_key(ni, v2)] = en
    if edge_vertices[ei][0] == v2:
        edge_vertices[ei][0] = ni
    else:
        assert edge_vertices[ei][1] == v2
        edge_vertices[ei][1] = ni
    edge_index[edge_key(v1, ni)] = ei
    del edge_index[(v1, v2)]
    edge_faces.append([])

    replacement = []
    for ti in edge_faces[ei]:
        face = faces[ti]
        other = next(k for k in range(3) if face[k] != v1 and face[k] != v2)
        reverse = face[(other + 1) % 3] == v2
        assert face[(other + 2) % 3] == (v1 if reverse else v2)
        if not reverse:
            assert face[(other + 1) % 3] == v1
        eint = len(edge_vertices)
        edge_vertices.append([face[other], ni])
        edge_index[edge_key(face[other], ni)] = eint
        t2i = len(faces)
        faces.append([ni, face[other], face[(other + 1) % 3]])
        face_edges.append([eint, face_edges[ti][other], en if reverse else ei])
        tte = face_edges[ti][other]
        neighbours = edge_faces[tte]
        neighbours[neighbours.index(ti)] = t2i
        edge_faces.append([ti, t2i])
        face[(other + 1) % 3] = ni
        face_edges[ti][other] = eint
        face_edges[ti][(other + 1) % 3] = ei if reverse else en
        if reverse:
            edge_faces[en].append(t2i)
            replacement.append(ti)
        else:
            edge_faces[en].append(ti)
            replacement.append(t2i)
    edge_faces[ei] = replacement


def split_pass(points, faces, edge_index, edge_vertices, edge_faces, face_edges, threshold):
    queue = []
    for ei, (a, b) in enumerate(edge_vertices):
        length = edge_length(points, a, b)
        heapq.heappush(queue, (-length, -ei))
    inserted = 0
    while queue and -queue[0][0] > threshold:
        old_neg_length, neg_ei = heapq.heappop(queue)
        ei = -neg_ei
        if not edge_faces[ei]:
            continue
        a, b = edge_vertices[ei]
        current = edge_length(points, a, b)
        old = -old_neg_length
        if current < old:
            heapq.heappush(queue, (-current, -ei))
        else:
            split_edge(points, faces, edge_index, edge_vertices, edge_faces, face_edges, ei)
            inserted += 1
    return inserted


def read_native_dump(path):
    with path.open('rb') as stream:
        nv, nf = struct.unpack('<II', stream.read(8))
        vertices = np.frombuffer(stream.read(nv * 3 * 8), dtype='<f8').reshape(nv, 3)
        faces = np.frombuffer(stream.read(nf * 3 * 4), dtype='<i4').reshape(nf, 3)
    return vertices, faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('reference', type=Path)
    args = parser.parse_args()
    original_vertices, original_faces = fs.read_geometry(args.input)
    points = [tuple(row) for row in original_vertices]
    faces = original_faces.tolist()
    edge_index, edge_vertices, edge_faces, face_edges = initial_topology(faces)
    total_length = 0.0
    for a, b in edge_vertices:
        total_length += edge_length(points, a, b)
    threshold = (total_length / len(edge_vertices)) * 0.8 * 4.0 / 3.0
    counts = []
    while True:
        inserted = split_pass(points, faces, edge_index, edge_vertices, edge_faces, face_edges, threshold)
        counts.append(inserted)
        print(json.dumps({'pass': len(counts), 'inserted': inserted,
                          'vertices': len(points), 'faces': len(faces)}), flush=True)
        if inserted == 0:
            break
    reference_points, reference_faces = read_native_dump(args.reference)
    candidate_points = np.asarray(points, dtype=np.float64)
    candidate_faces = np.asarray(faces, dtype=np.int32)
    report = {
        'threshold': threshold,
        'pass_insertions': counts,
        'candidate_vertices': len(points),
        'reference_vertices': len(reference_points),
        'candidate_faces': len(faces),
        'reference_faces': len(reference_faces),
        'vertex_values_equal': bool(np.array_equal(candidate_points, reference_points)),
        'face_indices_equal': bool(np.array_equal(candidate_faces, reference_faces)),
    }
    if candidate_points.shape == reference_points.shape:
        report['vertex_values_different'] = int(np.count_nonzero(candidate_points != reference_points))
        report['max_coordinate_error'] = float(np.max(np.abs(candidate_points - reference_points)))
    if candidate_faces.shape == reference_faces.shape:
        report['face_indices_different'] = int(np.count_nonzero(candidate_faces != reference_faces))
    print(json.dumps(report, indent=2), flush=True)
    return int(not report['vertex_values_equal'] or not report['face_indices_equal'])


if __name__ == '__main__':
    raise SystemExit(main())
