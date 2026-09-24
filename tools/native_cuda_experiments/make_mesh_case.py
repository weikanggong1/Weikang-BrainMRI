#!/usr/bin/env python3
"""Build an ordered CSR smoothing case from a FreeSurfer triangle surface."""

import argparse
import hashlib
import json
import struct
from pathlib import Path


MAGIC = b"FSGRAD1\0"
TRIANGLE_MAGIC = 16777214


def read_surface(path):
    with path.open("rb") as stream:
        magic = int.from_bytes(stream.read(3), "big")
        if magic != TRIANGLE_MAGIC:
            raise ValueError("expected a FreeSurfer triangular surface")
        stream.readline()
        stream.readline()
        nvertices, nfaces = struct.unpack(">ii", stream.read(8))
        if nvertices <= 0 or nfaces <= 0:
            raise ValueError("invalid surface dimensions")
        coords = struct.unpack(">" + "f" * (3 * nvertices), stream.read(12 * nvertices))
        faces = list(struct.iter_unpack(">iii", stream.read(12 * nfaces)))
        if len(faces) != nfaces:
            raise ValueError("truncated face table")
    return coords, faces


def ordered_neighbors(nvertices, faces):
    """Match the default mrisCompleteTopology_old first-ring face walk."""
    neighbors = [[] for _ in range(nvertices)]
    seen = [set() for _ in range(nvertices)]
    for face in faces:
        if any(vertex < 0 or vertex >= nvertices for vertex in face):
            raise ValueError("face contains an invalid vertex")
        for corner, vertex in enumerate(face):
            for adjacent in (face[(corner - 1) % 3], face[(corner + 1) % 3]):
                if adjacent not in seen[vertex]:
                    seen[vertex].add(adjacent)
                    neighbors[vertex].append(adjacent)
    return neighbors


def make_case(surface, output, iterations, gradient_bin=None, rip_vertices=None):
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    coords, neighbors_faces = read_surface(surface)
    nvertices = len(coords) // 3
    neighbors = ordered_neighbors(nvertices, neighbors_faces)
    ripped = set()
    if rip_vertices:
        ripped = {int(line) for line in rip_vertices.read_text().splitlines() if line.strip()}
        if any(vertex < 0 or vertex >= nvertices for vertex in ripped):
            raise ValueError("rip list contains an invalid vertex")
    active = [vertex for vertex in range(nvertices) if vertex not in ripped]
    active_index = {vertex: index for index, vertex in enumerate(active)}
    offsets = [0]
    indices = []
    for vertex in active:
        indices.extend(active_index[adjacent] for adjacent in neighbors[vertex] if adjacent not in ripped)
        offsets.append(len(indices))

    if gradient_bin:
        raw = gradient_bin.read_bytes()
        if len(raw) != 12 * nvertices:
            raise ValueError("gradient-bin must contain 3 float32 values per original vertex")
        gradients = struct.unpack("<" + "f" * (3 * nvertices), raw)
    else:
        gradients = coords  # representative values from real geometry, not FreeSurfer optimization gradients
    values = [value for vertex in active for value in gradients[3 * vertex:3 * vertex + 3]]

    with output.open("wb") as stream:
        stream.write(struct.pack("<8sIIII", MAGIC, len(active), len(indices), iterations, nvertices))
        stream.write(struct.pack("<" + "I" * len(active), *active))
        stream.write(struct.pack("<" + "I" * len(offsets), *offsets))
        stream.write(struct.pack("<" + "I" * len(indices), *indices))
        stream.write(struct.pack("<" + "f" * len(values), *values))
    meta = {
        "surface": str(surface.resolve()),
        "surface_sha256": hashlib.sha256(surface.read_bytes()).hexdigest(),
        "gradient_source": str(gradient_bin.resolve()) if gradient_bin else "surface_coordinates",
        "original_vertices": nvertices,
        "active_vertices": len(active),
        "neighbors": len(indices),
        "iterations": iterations,
        "case_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path, help="FreeSurfer triangular surface, e.g. surf/lh.inflated")
    parser.add_argument("output", type=Path, help="output .bin case")
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--gradient-bin", type=Path, help="optional little-endian float32 [original_vertex, xyz]")
    parser.add_argument("--rip-vertices", type=Path, help="optional text file with one ripped vertex ID per line")
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error("iterations must be nonnegative")
    print(json.dumps(make_case(args.surface, args.output, args.iterations, args.gradient_bin, args.rip_vertices)))


if __name__ == "__main__":
    main()
