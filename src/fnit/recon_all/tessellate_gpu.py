"""Reproduce the default FreeSurfer ``mri_tessellate`` quad geometry.

The six directed voxel boundaries and the vertex/face scan order follow
FreeSurfer 8.2.0's ``mri_tessellate.cpp``.  This module returns the raw quad
geometry, before ``mris_extract_main_component`` changes the surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


@torch.inference_mode()
def tessellate_quads(
    volume_xyz: np.ndarray,
    value: int,
    vox2ras_tkr: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return FreeSurfer-ordered RAS vertices and ordered quad indices."""
    voxels = torch.as_tensor(np.ascontiguousarray(volume_xyz.transpose(2, 1, 0)), device=device)
    target = voxels == value
    depth, height, width = target.shape
    lattice = torch.zeros((depth + 1, height + 1, width + 1),
                          dtype=torch.bool, device=device)

    z_edge = target[1:] != target[:-1]
    y_edge = target[:, 1:] != target[:, :-1]
    x_edge = target[:, :, 1:] != target[:, :, :-1]
    for dy in (0, 1):
        for dx in (0, 1):
            lattice[1:depth, dy:height + dy, dx:width + dx] |= z_edge
    for dz in (0, 1):
        for dx in (0, 1):
            lattice[dz:depth + dz, 1:height, dx:width + dx] |= y_edge
    for dz in (0, 1):
        for dy in (0, 1):
            lattice[dz:depth + dz, dy:height + dy, 1:width] |= x_edge

    lattice_coords = lattice.nonzero()  # z, y, x scan order
    vertex_numbers = torch.full(lattice.shape, -1, dtype=torch.int32, device=device)
    vertex_numbers[lattice_coords[:, 0], lattice_coords[:, 1], lattice_coords[:, 2]] = (
        torch.arange(len(lattice_coords), device=device, dtype=torch.int32))

    # The order within one lattice location is the order of n=0 check_face
    # calls in mri_tessellate.cpp: f=0, 2, 4, 1, 3, 5.
    directed = (
        (0, (target[1:] & ~target[:-1]), (1, 0, 0)),
        (2, (target[:, 1:] & ~target[:, :-1]), (0, 1, 0)),
        (4, (target[:, :, 1:] & ~target[:, :, :-1]), (0, 0, 1)),
        (1, (target[:-1] & ~target[1:]), (1, 0, 0)),
        (3, (target[:, :-1] & ~target[:, 1:]), (0, 1, 0)),
        (5, (target[:, :, :-1] & ~target[:, :, 1:]), (0, 0, 1)),
    )
    starts = []
    face_types = []
    for priority, (_, boundary, offset) in enumerate(directed):
        zyx = boundary.nonzero() + torch.tensor(offset, device=device)
        starts.append(zyx)
        face_types.append(torch.full((len(zyx),), priority,
                                     dtype=torch.int64, device=device))
    starts = torch.cat(starts)
    face_types = torch.cat(face_types)
    key = (((starts[:, 0] * (height + 1) + starts[:, 1]) * (width + 1)
            + starts[:, 2]) * 6 + face_types)
    order = torch.argsort(key)
    starts = starts[order]
    face_types = face_types[order]

    # Corner offsets in the original face.v[0:4] order.
    offsets = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],  # f0
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # f2
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],  # f4
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # f1
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],  # f3
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # f5
    ], device=device)
    corners = starts[:, None, :] + offsets[face_types]
    quads = vertex_numbers[corners[..., 0], corners[..., 1], corners[..., 2]]
    if bool((quads < 0).any()):
        raise RuntimeError("a quad corner has no tessellated vertex")

    xyz1 = torch.stack((lattice_coords[:, 2] - 0.5,
                        lattice_coords[:, 1] - 0.5,
                        lattice_coords[:, 0] - 0.5,
                        torch.ones(len(lattice_coords), device=device)), dim=1).to(torch.float64)
    affine = torch.as_tensor(vox2ras_tkr, dtype=torch.float64, device=device)
    ras = (xyz1 @ affine.T)[:, :3].to(torch.float32)
    return ras.cpu().numpy(), quads.cpu().numpy()


def tessellate_mgh(input_path: str | Path, value: int,
                   device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(input_path))
    volume = np.asanyarray(image.dataobj)
    if volume.dtype != np.uint8:
        raise ValueError("current implementation requires MRI_UCHAR input")
    return tessellate_quads(volume, value, image.header.get_vox2ras_tkr(), device)


def write_quad_surface(path: str | Path, vertices: np.ndarray,
                       quads: np.ndarray, image: nib.spatialimages.SpatialImage,
                       input_path: str | Path) -> None:
    """Write FreeSurfer's old quad format, including volume geometry tags."""
    counts = np.array([len(vertices), len(quads)], dtype=np.int64)
    if np.any(counts >= 2**24):
        raise ValueError("quad surface exceeds the FreeSurfer 3-byte count limit")
    packed = np.asarray(quads, dtype=np.int64).reshape(-1)
    with open(path, "wb") as stream:
        stream.write(b"\xff\xff\xfd")
        for count in counts:
            stream.write(int(count).to_bytes(3, "big"))
        stream.write(np.asarray(vertices, dtype=">f4").tobytes())
        for index in packed:
            stream.write(int(index).to_bytes(3, "big"))
        # TAG_USEREALRAS uses a native 32-bit int payload.  The default
        # mri_tessellate path writes the conformed surface RAS (zero here).
        stream.write((4).to_bytes(4, "big"))
        stream.write((4).to_bytes(8, "big"))
        stream.write((0).to_bytes(4, "little"))
        stream.write((20).to_bytes(4, "big"))  # TAG_OLD_SURF_GEOM, no length
        header = image.header
        dims = image.shape[:3]
        spacing = header["delta"]
        direction = header["Mdc"]
        center = header["Pxyz_c"]

        def values(items: np.ndarray) -> str:
            return " ".join(f"{float(item):.15e}" for item in items)

        geometry = (
            f"valid = {int(header['goodRASFlag'])}  # volume info valid\n"
            f"filename = {input_path}\n"
            f"volume = {dims[0]} {dims[1]} {dims[2]}\n"
            f"voxelsize = {values(spacing)}\n"
            f"xras   = {values(direction[0])}\n"
            f"yras   = {values(direction[1])}\n"
            f"zras   = {values(direction[2])}\n"
            f"cras   = {values(center)}\n"
        )
        stream.write(geometry.encode("utf-8"))
        command = f"ProgramName: mri_tessellate_gpu  Input: {input_path}  Output: {path}\n"
        payload = command.encode("utf-8") + b"\x00"
        stream.write((3).to_bytes(4, "big"))  # TAG_CMDLINE
        stream.write(len(payload).to_bytes(8, "big"))
        stream.write(payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("value", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    image = nib.load(str(args.input))
    volume = np.asanyarray(image.dataobj)
    if volume.dtype != np.uint8:
        raise ValueError("current implementation requires MRI_UCHAR input")
    vertices, quads = tessellate_quads(volume, args.value,
                                      image.header.get_vox2ras_tkr(), args.device)
    write_quad_surface(args.output, vertices, quads, image, args.input)


if __name__ == "__main__":
    main()
