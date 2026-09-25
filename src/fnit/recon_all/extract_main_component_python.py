"""Convert old FreeSurfer quads and retain the largest surface component."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def read_quad_core(path: str | Path) -> tuple[np.ndarray, np.ndarray, bytes]:
    raw = Path(path).read_bytes()
    if raw[:3] != b"\xff\xff\xfd":
        raise ValueError("expected FreeSurfer new-quad surface")
    vertices = int.from_bytes(raw[3:6], "big")
    quads = int.from_bytes(raw[6:9], "big")
    core_end = 9 + 12 * (vertices + quads)
    if len(raw) < core_end:
        raise ValueError("truncated quad surface")
    xyz = np.frombuffer(raw, dtype=">f4", count=3 * vertices, offset=9).reshape(-1, 3)
    packed = np.frombuffer(raw, dtype=np.uint8, count=12 * quads,
                           offset=9 + 12 * vertices).reshape(-1, 3)
    indices = ((packed[:, 0].astype(np.int32) << 16) |
               (packed[:, 1].astype(np.int32) << 8) | packed[:, 2]).reshape(-1, 4)
    geometry_start = raw.index(b"valid = ", core_end)
    geometry_end = raw.index(b"\n", raw.index(b"cras   = ", geometry_start)) + 1
    return np.asarray(xyz, dtype=np.float32), indices, raw[geometry_start:geometry_end]


def quad_triangles(quads: np.ndarray) -> np.ndarray:
    """Use FreeSurfer's WHICH_FACE_SPLIT(v0, v1) choice, not nibabel's."""
    choose = np.floor(np.sqrt(1.9 * quads[:, 0]) +
                      np.sqrt(3.5 * quads[:, 1]) + 0.5).astype(np.int64) % 2
    triangles = np.empty((len(quads), 2, 3), dtype=np.int32)
    even = choose == 0
    triangles[even, 0] = quads[even][:, [0, 1, 3]]
    triangles[even, 1] = quads[even][:, [2, 3, 1]]
    triangles[~even, 0] = quads[~even][:, [0, 1, 2]]
    triangles[~even, 1] = quads[~even][:, [0, 2, 3]]
    flat = triangles.reshape(-1, 3)
    valid = ((flat[:, 0] != flat[:, 1]) &
             (flat[:, 0] != flat[:, 2]) &
             (flat[:, 1] != flat[:, 2]))
    return flat[valid]


def largest_component(vertices: np.ndarray,
                      faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Preserve source vertex and face order; tied sizes choose the first."""
    nvertices = len(vertices)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(edges), dtype=np.uint8),
                        (edges[:, 0], edges[:, 1])),
                       shape=(nvertices, nvertices)).tocsr()
    count, component = connected_components(graph, directed=False)
    sizes = np.bincount(component, minlength=count)
    first_vertex = np.full(count, nvertices, dtype=np.int64)
    np.minimum.at(first_vertex, component, np.arange(nvertices))
    selected_component = np.lexsort((first_vertex, -sizes))[0]
    keep = component == selected_component
    translation = np.full(nvertices, -1, dtype=np.int32)
    translation[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    return vertices[keep], translation[faces[np.all(keep[faces], axis=1)]], count


def _surface_transform(geometry: bytes) -> tuple[bytes, np.ndarray, bytes]:
    lines = {}
    for row in geometry.decode().splitlines():
        key, value = row.split("=", 1)
        lines[key.strip()] = value.split("#", 1)[0].strip()
    if not int(lines["valid"]):
        return b"NIFTI_XFORM_UNKNOWN", np.eye(4, dtype=np.float32), b"NIFTI_XFORM_UNKNOWN"
    shape = np.fromstring(lines["volume"], sep=" ", dtype=np.float32)
    spacing = np.fromstring(lines["voxelsize"], sep=" ", dtype=np.float32)
    direction = np.stack([np.fromstring(lines[key], sep=" ", dtype=np.float32)
                          for key in ("xras", "yras", "zras")], axis=1)
    center = np.fromstring(lines["cras"], sep=" ", dtype=np.float32)
    scanner = np.eye(4, dtype=np.float32)
    scanner[:3, :3] = direction * spacing
    scanner[:3, 3] = center - scanner[:3, :3] @ (shape / 2)
    tkregister = np.array([[-spacing[0], 0, 0, spacing[0] * shape[0] / 2],
                           [0, 0, spacing[2], -spacing[2] * shape[2] / 2],
                           [0, -spacing[1], 0, spacing[1] * shape[1] / 2],
                           [0, 0, 0, 1]], dtype=np.float32)
    return (b"NIFTI_XFORM_UNKNOWN", scanner @ np.linalg.inv(tkregister),
            b"NIFTI_XFORM_SCANNER_ANAT")


def write_triangle_surface(path: str | Path, vertices: np.ndarray,
                           faces: np.ndarray, geometry: bytes) -> None:
    def tag(stream, number: int, payload: bytes) -> None:
        stream.write(number.to_bytes(4, "big"))
        stream.write(len(payload).to_bytes(8, "big"))
        stream.write(payload)

    dataspace, transform, transformedspace = _surface_transform(geometry)
    matrix = "Matrix" + "".join(f" {value:10f}" for value in transform.ravel())
    with open(path, "wb") as stream:
        stream.write(b"\xff\xff\xfecreated by fnit\n\n")
        stream.write(np.asarray([len(vertices), len(faces)], dtype=">i4").tobytes())
        stream.write(np.asarray(vertices, dtype=">f4").tobytes())
        stream.write(np.asarray(faces, dtype=">i4").tobytes())
        stream.write((2).to_bytes(4, "big"))  # TAG_OLD_USEREALRAS
        stream.write((0).to_bytes(4, "big"))
        stream.write((20).to_bytes(4, "big"))  # TAG_OLD_SURF_GEOM
        stream.write(geometry)
        tag(stream, 22, dataspace)
        tag(stream, 23, matrix.encode())
        tag(stream, 24, transformedspace)


def extract_main_component(input_path: str | Path, output_path: str | Path) -> int:
    vertices, quads, geometry = read_quad_core(input_path)
    triangles = quad_triangles(quads)
    vertices, triangles, components = largest_component(vertices, triangles)
    write_triangle_surface(output_path, vertices, triangles, geometry)
    return components


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(f"connected components: {extract_main_component(args.input, args.output)}")


if __name__ == "__main__":
    main()
