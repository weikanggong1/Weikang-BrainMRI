"""Create a private intensity-gradient fixture from one completed FreeSurfer subject."""

import argparse
import struct
from pathlib import Path

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    image = nib.load(str(args.subject_dir / "mri/mrisps.wpa.mgz"))
    volume = np.ascontiguousarray(np.asarray(image.dataobj), dtype=np.uint8)
    vertices, faces = nib.freesurfer.read_geometry(str(args.subject_dir / "surf/lh.orig"))
    vertices = vertices.astype(np.float32)
    triangle = vertices[faces]
    face_normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    normal = np.zeros_like(vertices)
    np.add.at(normal, faces[:, 0], face_normal)
    np.add.at(normal, faces[:, 1], face_normal)
    np.add.at(normal, faces[:, 2], face_normal)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
    voxel = nib.affines.apply_affine(np.linalg.inv(image.header.get_vox2ras_tkr()), vertices)
    nearest = np.rint(voxel).astype(int)
    if np.any(nearest < 0) or np.any(nearest >= 256):
        raise ValueError("surface leaves the sub-01 MRI fixture")
    sampled = volume[nearest[:, 0], nearest[:, 1], nearest[:, 2]].astype(np.float32)
    offset = np.where(np.arange(len(vertices)) % 2, 3.0, -3.0)
    target = np.clip(sampled + offset, 0, 255).astype(np.float32)[:, None]
    sigma = np.full((len(vertices), 1), 1.0, dtype=np.float32)
    records = np.ascontiguousarray(np.concatenate((vertices, normal, target, sigma), axis=1))

    if volume.shape != (256, 256, 256):
        raise ValueError(f"expected sub-01 256^3 fixture, got {volume.shape}")
    if not np.allclose(image.header.get_vox2ras_tkr(),
                       np.array([[-1, 0, 0, 128], [0, 0, 1, -128],
                                 [0, -1, 0, 128], [0, 0, 0, 1]])):
        raise ValueError("fixture CUDA transform only supports this sub-01 geometry")
    with args.output.open("wb") as out:
        out.write(struct.pack("<8sII", b"FSITERM\0", len(vertices), len(volume.tobytes())))
        out.write(volume.tobytes())
        out.write(records.tobytes())
    print(f"vertices={len(vertices)} voxels={volume.size} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
