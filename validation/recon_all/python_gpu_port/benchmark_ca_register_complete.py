"""Run the isolated fixed inverse-GCAM numerical path against native fields."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
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
from fnit.recon_all.ca_register_inverse_fill import soap_bubble_float, voronoi_fill
from fnit.recon_all.ca_register_inverse_output import (
    inverse_coordinates_to_displacement_ras,
    write_inverse_warp_nifti,
)


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    difference = np.abs(actual - expected)
    return {
        "elements": int(difference.size),
        "different": int(np.count_nonzero(difference)),
        "max_abs": float(difference.max()),
        "p99_abs": float(np.quantile(difference, 0.99)),
    }


def file_comparison(actual: Path, expected: Path) -> dict[str, object]:
    raw_hashes = [hashlib.sha256(), hashlib.sha256()]
    different = 0
    first_difference = None
    length = 0
    with gzip.open(actual, "rb") as left, gzip.open(expected, "rb") as right:
        while True:
            a, b = left.read(1 << 20), right.read(1 << 20)
            if not a and not b:
                break
            raw_hashes[0].update(a)
            raw_hashes[1].update(b)
            if len(a) != len(b):
                raise ValueError("decompressed NIfTI files differ in length")
            mismatch = np.flatnonzero(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8))
            if mismatch.size:
                different += int(mismatch.size)
                if first_difference is None:
                    first_difference = length + int(mismatch[0])
            length += len(a)
    return {
        "decompressed_bytes": length,
        "different_bytes": different,
        "first_difference": first_difference,
        "written_raw_sha256": raw_hashes[0].hexdigest(),
        "native_raw_sha256": raw_hashes[1].hexdigest(),
        "written_compressed_sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
        "native_compressed_sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp", type=Path, required=True)
    parser.add_argument("--native-x", type=Path, required=True)
    parser.add_argument("--native-y", type=Path, required=True)
    parser.add_argument("--native-z", type=Path, required=True)
    parser.add_argument("--native-inverse", type=Path, required=True)
    parser.add_argument("--inverse-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    total_start = time.perf_counter()
    warp = nib.load(args.warp)
    source_geometry, atlas_geometry, source_shape = read_warp_geometries(warp)
    displacement = np.asarray(warp.dataobj, dtype=np.float32)[:, :, :, 0, :]
    coordinates = warp_to_source_voxels(displacement, atlas_geometry, source_geometry)
    counts = splat_inverse_counts(coordinates, source_shape)
    sums = splat_inverse_coordinate_sums(coordinates, source_shape)
    control = counts >= np.float32(0.1)
    fields = np.empty((*source_shape, 3), dtype=np.float32)
    report: dict[str, object] = {"input_and_splat_seconds": time.perf_counter() - total_start, "fields": {}}

    for axis, name, native_path in zip(range(3), "xyz", (args.native_x, args.native_y, args.native_z)):
        started = time.perf_counter()
        average = np.zeros_like(counts)
        np.divide(sums[axis], counts, out=average, where=control)
        expanded, expansion_iterations = voronoi_fill(average, control)
        smoothed, smoothing_iterations = soap_bubble_float(expanded, control)
        fields[:, :, :, axis] = smoothed
        native = np.asarray(nib.load(native_path).dataobj, dtype=np.float32)
        report["fields"][name] = {
            "seconds": time.perf_counter() - started,
            "voronoi_iterations": expansion_iterations,
            "soap_iterations": smoothing_iterations,
            **comparison(smoothed, native),
        }
        print(name, report["fields"][name], flush=True)
        del average, expanded, smoothed, native

    started = time.perf_counter()
    result = inverse_coordinates_to_displacement_ras(fields, atlas_geometry, source_geometry)
    native_image = nib.load(args.native_inverse)
    native_result = np.asarray(native_image.dataobj, dtype=np.float32)[:, :, :, 0, :]
    report["displacement"] = {"seconds": time.perf_counter() - started, **comparison(result, native_result)}
    if args.inverse_out:
        started = time.perf_counter()
        write_inverse_warp_nifti(args.warp, result, args.inverse_out)
        report["written_file"] = {"seconds": time.perf_counter() - started, **file_comparison(args.inverse_out, args.native_inverse)}
    report["total_seconds_including_io_jit_comparison"] = time.perf_counter() - total_start
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out.read_text(), flush=True)
    if (any(value["different"] for value in report["fields"].values())
        or report["displacement"]["different"]
        or ("written_file" in report and report["written_file"]["different_bytes"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
