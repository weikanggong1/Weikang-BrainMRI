"""Convert the fixed recon-all cortical labels into a FreeSurfer annotation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct

import nibabel.freesurfer.io as fsio
import numpy as np


def _color_table(path: Path) -> dict[int, tuple[str, tuple[int, int, int, int]]]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) < 6:
            continue
        try:
            index = int(columns[0])
            colors = tuple(int(value) for value in columns[2:6])
        except ValueError:
            continue
        entries.setdefault(index, (columns[1], colors))
    if not entries:
        raise ValueError("color table has no entries")
    return entries


def _binary_color_table(path: Path, entries: dict) -> bytes:
    pack = lambda value: struct.pack(">i", value)
    filename = os.fsencode(path) + b"\0"
    result = bytearray(pack(1) + pack(-2) + pack(max(entries) + 1))
    result += pack(len(filename)) + filename + pack(len(entries))
    for index, (name, colors) in sorted(entries.items()):
        encoded = name.encode("utf-8") + b"\0"
        result += pack(index) + pack(len(encoded)) + encoded
        result += b"".join(pack(value) for value in colors)
    return bytes(result)


def write_label_annotation(surface_file: str | Path, color_table_file: str | Path,
                           label_files: list[str | Path], output_file: str | Path) -> None:
    """Apply ``--maxstatwinner`` to ordered labels and write an ``.annot`` file."""
    surface_file, color_table_file = Path(surface_file), Path(color_table_file)
    vertices, _ = fsio.read_geometry(str(surface_file))
    entries = _color_table(color_table_file)
    by_name = {name: colors for name, colors in entries.values()}
    hemi = surface_file.name.split(".", 1)[0]
    unknown = entries.get(0, ("", (0, 0, 0, 0)))[1]
    annotation = np.zeros(len(vertices), np.int32)
    max_stat = np.zeros(len(vertices), np.float32)
    hit = np.zeros(len(vertices), bool)
    for label_file in label_files:
        label_file = Path(label_file)
        name = label_file.name.removesuffix(".label").removeprefix(f"{hemi}.")
        if name not in by_name:
            raise ValueError(f"{name} is missing from {color_table_file}")
        red, green, blue, _ = by_name[name]
        color = red + (green << 8) + (blue << 16)
        indices, stats = fsio.read_label(str(label_file), read_scalars=True)
        for index, stat in zip(indices, stats):
            if not 0 <= index < len(vertices):
                raise ValueError(f"vertex {index} is outside {surface_file}")
            value = np.float32(stat)
            if value >= max_stat[index]:
                annotation[index] = color
                max_stat[index] = value
                hit[index] = True
    annotation[~hit] = unknown[0] + (unknown[1] << 8) + (unknown[2] << 16)
    pairs = np.empty((len(vertices), 2), dtype=">i4")
    pairs[:, 0] = np.arange(len(vertices))
    pairs[:, 1] = annotation
    payload = struct.pack(">i", len(vertices)) + pairs.tobytes()
    payload += _binary_color_table(color_table_file, entries)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_file", type=Path)
    parser.add_argument("color_table_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("label_files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    write_label_annotation(args.surface_file, args.color_table_file,
                           args.label_files, args.output_file)


if __name__ == "__main__":
    main()
