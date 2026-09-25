"""GPU implementation of recon-all's label-match mri_binarize call."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import struct

import nibabel as nib
import numpy as np
import torch


def _canonicalize_synthseg_footer(footer: bytes) -> bytes:
    """Match FreeSurfer's MGH tag order and empty color-table filename."""
    def integer(offset: int) -> int:
        return struct.unpack_from(">i", footer, offset)[0]

    if integer(20) != 1 or integer(24) != -2:
        raise ValueError("expected the fixed SynthSeg MGH color-table footer")
    filename_length = integer(32)
    position = 36 + filename_length
    entry_count = integer(position)
    position += 4
    entry_start = position
    entries = []
    for _ in range(entry_count):
        start = position
        index = integer(position)
        name_length = integer(position + 4)
        position += 8 + name_length + 16
        entries.append((index, footer[start:position]))
    colortable = footer[20:entry_start] + b"".join(row for _, row in sorted(entries))
    if filename_length == 0:
        colortable = colortable[:12] + struct.pack(">i", 1) + b"\0" + colortable[16:]
    return footer[:20] + footer[position:] + colortable


def binarize_labels(values: torch.Tensor, labels: tuple[int, ...], *,
                    invert: bool = False) -> torch.Tensor:
    """Return FreeSurfer's default int32 mask for ``--match`` and ``--inv``."""
    if values.ndim != 3 or not labels:
        raise ValueError("expected a 3D volume and at least one matching label")
    selected = torch.zeros_like(values, dtype=torch.bool)
    for label in labels:
        selected |= values == label
    return (~selected if invert else selected).to(torch.int32)


def binarize_volume(input_file: str | Path, output_file: str | Path,
                    labels: tuple[int, ...], *, invert: bool = False,
                    device: str = "cuda:0") -> None:
    """Apply label matching to a FreeSurfer MGH/MGZ volume on ``device``."""
    image = nib.load(str(input_file))
    if not isinstance(image, nib.MGHImage) or len(image.shape) != 3:
        raise ValueError("expected a 3D MGH/MGZ input volume")
    dtype = image.get_data_dtype().newbyteorder("=")
    if dtype != np.dtype("int32"):
        raise ValueError("this recon-all label-match call requires int32 MGH input")
    voxels = np.asarray(image.dataobj).astype(dtype, copy=True)
    result = binarize_labels(torch.from_numpy(voxels).to(device), labels,
                             invert=invert).cpu().numpy()
    source = Path(input_file)
    output = Path(output_file)
    if source.suffix not in (".mgh", ".mgz") or output.suffix not in (".mgh", ".mgz"):
        raise ValueError("MGH/MGZ paths are required")
    raw = gzip.decompress(source.read_bytes()) if source.suffix == ".mgz" else source.read_bytes()
    offset = int(image.header.get_data_offset())
    end = offset + voxels.nbytes
    data = np.asarray(result, dtype=image.get_data_dtype()).tobytes(order="F")
    payload = raw[:offset] + data + _canonicalize_synthseg_footer(raw[end:])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(payload, mtime=0) if output.suffix == ".mgz" else payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", required=True, dest="input_file")
    parser.add_argument("--o", required=True, dest="output_file")
    parser.add_argument("--match", required=True, nargs="+", type=int)
    parser.add_argument("--inv", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    binarize_volume(args.input_file, args.output_file, tuple(args.match),
                    invert=args.inv, device=args.device)


if __name__ == "__main__":
    main()
