"""Preserve MGH/MGZ metadata when only same-type voxel values change."""

from __future__ import annotations

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np


def save_same_dtype_mgh(source_file: str | Path, output_file: str | Path,
                        values: np.ndarray) -> None:
    """Replace voxel bytes; retain the source header and trailing FreeSurfer tags."""
    source_path, output_path = Path(source_file), Path(output_file)
    if source_path.suffix not in (".mgh", ".mgz") or output_path.suffix not in (".mgh", ".mgz"):
        raise ValueError("MGH/MGZ paths are required")
    image = nib.load(str(source_path))
    if not isinstance(image, nib.MGHImage) or tuple(values.shape) != image.shape or len(image.shape) != 3:
        raise ValueError("values must match a 3D MGH/MGZ input")
    dtype = image.get_data_dtype()
    if values.dtype.newbyteorder("=") != dtype.newbyteorder("="):
        raise ValueError("voxel dtype changes require a different MGH writer")
    raw = gzip.decompress(source_path.read_bytes()) if source_path.suffix == ".mgz" else source_path.read_bytes()
    offset = int(image.header.get_data_offset())
    end = offset + values.size * dtype.itemsize
    if len(raw) < end:
        raise ValueError("MGH voxel payload is truncated")
    data = np.asarray(values, dtype=dtype).tobytes(order="F")
    payload = raw[:offset] + data + raw[end:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(gzip.compress(payload, mtime=0) if output_path.suffix == ".mgz" else payload)
