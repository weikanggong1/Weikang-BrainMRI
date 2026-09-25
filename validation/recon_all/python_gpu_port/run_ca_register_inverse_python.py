"""Run the isolated fixed inverse-GCAM stage without native checkpoint reads."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("warp", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timing-json", type=Path)
    args = parser.parse_args()

    start = time.perf_counter()
    warp = nib.load(args.warp)
    source, atlas, source_shape = read_warp_geometries(warp)
    displacement = np.asarray(warp.dataobj, dtype=np.float32)[:, :, :, 0, :]
    coordinates = warp_to_source_voxels(displacement, atlas, source)
    counts = splat_inverse_counts(coordinates, source_shape)
    sums = splat_inverse_coordinate_sums(coordinates, source_shape)
    control = counts >= np.float32(0.1)
    report = {"load_convert_splat_seconds": time.perf_counter() - start, "fields": {}}
    fields = np.empty((*source_shape, 3), dtype=np.float32)
    for axis, name in enumerate("xyz"):
        field_start = time.perf_counter()
        average = np.zeros_like(counts)
        np.divide(sums[axis], counts, out=average, where=control)
        expanded, voronoi_iterations = voronoi_fill(average, control)
        fields[:, :, :, axis], soap_iterations = soap_bubble_float(expanded, control)
        report["fields"][name] = {
            "seconds": time.perf_counter() - field_start,
            "voronoi_iterations": voronoi_iterations,
            "soap_iterations": soap_iterations,
        }
        del average, expanded
    conversion_start = time.perf_counter()
    result = inverse_coordinates_to_displacement_ras(fields, atlas, source)
    report["displacement_seconds"] = time.perf_counter() - conversion_start
    write_start = time.perf_counter()
    write_inverse_warp_nifti(args.warp, result, args.output)
    report["write_seconds"] = time.perf_counter() - write_start
    report["total_seconds"] = time.perf_counter() - start
    if args.timing_json:
        args.timing_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
