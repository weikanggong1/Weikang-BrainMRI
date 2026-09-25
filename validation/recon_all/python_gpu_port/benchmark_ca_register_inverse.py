"""Compare first inverse-GCAM kernels with pinned native intermediate volumes."""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.ca_register_inverse import (
    read_warp_geometries,
    splat_inverse_coordinate_sums,
    splat_inverse_counts,
    warp_to_source_voxels,
)


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    if actual.dtype.kind == "u":
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    else:
        difference = np.abs(actual - expected)
    return {
        "elements": int(difference.size),
        "different": int(np.count_nonzero(difference)),
        "max_abs": float(difference.max()),
        "p99_abs": float(np.quantile(difference, 0.99)),
    }


def read_native_absolute_warp(path: Path) -> np.ndarray:
    # FreeSurfer warp MGZ uses a custom MGH version rejected by nibabel.
    with gzip.open(path, "rb") as source:
        header = source.read(284)
        _, width, height, depth, frames, mri_type = struct.unpack(">6i", header[:24])
        if frames != 3 or mri_type != 3:
            raise ValueError("expected a three-frame float32 FreeSurfer warp")
        data = source.read(width * height * depth * frames * 4)
    return np.frombuffer(data, dtype=">f4").reshape(
        (width, height, depth, frames), order="F"
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp", type=Path, required=True)
    parser.add_argument("--native-absolute", type=Path, required=True)
    parser.add_argument("--native-counts", type=Path, required=True)
    parser.add_argument("--native-x-sum", type=Path)
    parser.add_argument("--native-control", type=Path)
    parser.add_argument("--native-x-final", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.native_control or args.native_x_final) and not args.native_x_sum:
        parser.error("--native-control and --native-x-final require --native-x-sum")
    if args.native_x_final and not args.native_control:
        parser.error("--native-x-final requires --native-control")

    input_warp = nib.load(args.warp)
    displacement = np.asarray(input_warp.dataobj, dtype=np.float32)[:, :, :, 0, :]
    started = time.perf_counter()
    source_vox2ras, atlas_vox2ras, source_shape = read_warp_geometries(input_warp)
    geometry_seconds = time.perf_counter() - started
    started = time.perf_counter()
    coordinates = warp_to_source_voxels(displacement, atlas_vox2ras, source_vox2ras)
    coordinate_seconds = time.perf_counter() - started
    native_coordinates = read_native_absolute_warp(args.native_absolute)
    coordinate_comparison = compare(coordinates, native_coordinates)

    native_counts = np.asarray(nib.load(args.native_counts).dataobj, dtype=np.float32)
    started = time.perf_counter()
    counts = splat_inverse_counts(coordinates, source_shape)
    splat_seconds = time.perf_counter() - started
    count_comparison = compare(counts, native_counts)
    started = time.perf_counter()
    native_coordinate_counts = splat_inverse_counts(native_coordinates, native_counts.shape)
    native_coordinate_splat_seconds = time.perf_counter() - started
    native_coordinate_count_comparison = compare(native_coordinate_counts, native_counts)
    report = {
        "warp_shape": list(displacement.shape),
        "source_shape": list(native_counts.shape),
        "geometry_read_seconds": geometry_seconds,
        "coordinate_conversion": {"seconds": coordinate_seconds, **coordinate_comparison},
        "count_splat": {"seconds_including_jit": splat_seconds, **count_comparison},
        "count_splat_from_native_coordinates": {
            "seconds": native_coordinate_splat_seconds,
            **native_coordinate_count_comparison,
        },
    }
    if args.native_x_sum:
        native_x_sum = np.asarray(nib.load(args.native_x_sum).dataobj, dtype=np.float32)
        started = time.perf_counter()
        coordinate_sums = splat_inverse_coordinate_sums(coordinates, source_shape)
        report["x_coordinate_splat"] = {
            "seconds_including_jit_and_yz": time.perf_counter() - started,
            **compare(coordinate_sums[0], native_x_sum),
        }
        if args.native_control:
            control = (counts >= np.float32(0.1)).astype(np.uint8)
            native_control = np.asarray(nib.load(args.native_control).dataobj, dtype=np.uint8)
            report["control_mask"] = {
                "marked_voxels": int(np.count_nonzero(control)),
                **compare(control, native_control),
            }
            if args.native_x_final:
                average_x = np.zeros_like(counts)
                np.divide(coordinate_sums[0], counts, out=average_x, where=control.astype(bool))
                native_x_final = np.asarray(nib.load(args.native_x_final).dataobj, dtype=np.float32)
                report["control_x_average"] = compare(
                    average_x[control.astype(bool)], native_x_final[control.astype(bool)]
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out.read_text())
    checks = ("coordinate_conversion", "count_splat", "count_splat_from_native_coordinates")
    if args.native_x_sum:
        checks += ("x_coordinate_splat",)
    if args.native_control:
        checks += ("control_mask",)
    if args.native_x_final:
        checks += ("control_x_average",)
    if any(report[name]["different"] for name in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
