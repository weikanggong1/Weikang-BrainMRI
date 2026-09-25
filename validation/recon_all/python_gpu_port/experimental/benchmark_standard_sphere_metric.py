"""Compare a fixed conventional-sphere metric matrix with native diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.sphere_standard_metric import (
    average_standard_metric, sample_standard_metric_matrix,
)


def _log_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pairs = [row.split(": ")[1].split(", ")
             for row in path.read_text().splitlines()]
    return (np.asarray([int(pair[0]) for pair in pairs], np.int32),
            np.asarray([float(pair[1]) for pair in pairs], np.float64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--native-log", action="append", default=[],
                        help="vertex:path to native vN.log after reciprocal averaging")
    parser.add_argument("--native-table", type=Path,
                        help="native FS_MEASURE_DISTANCES distance.log, four decimals")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    xyz, faces = fsio.read_geometry(str(args.smoothwm))
    read_seconds = perf_counter() - start
    offsets, indices, before, report = sample_standard_metric_matrix(
        xyz, faces, limit=args.limit)
    report["read_seconds"] = read_seconds
    report["first_row_count"] = int(offsets[1])
    report["last_row_count"] = int(offsets[-1] - offsets[-2])
    report["count_min"] = int(np.diff(offsets).min())
    report["count_max"] = int(np.diff(offsets).max())
    full = len(offsets) == len(xyz) + 1
    if full:
        after, matched, seconds = average_standard_metric(offsets, indices, before)
        report["reciprocal_matches"] = matched
        report["reciprocal_seconds_including_jit"] = seconds
        report["changed_distances"] = int(np.count_nonzero(after != before))
    else:
        after = None
    for entry in args.native_log:
        vertex_text, filename = entry.split(":", 1)
        vertex = int(vertex_text)
        native_ids, native_values = _log_rows(Path(filename))
        ids = indices[offsets[vertex]:offsets[vertex + 1]]
        values = (before if after is None else after)[
            offsets[vertex]:offsets[vertex + 1]]
        if len(values) != len(native_values):
            raise AssertionError((vertex, len(values), len(native_values)))
        delta = np.abs(values.astype(np.float64) - native_values)
        report[f"native_vertex_{vertex}"] = {
            "ordered_ids_exact": bool(np.array_equal(ids, native_ids)),
            "exact_ids": int(np.count_nonzero(ids == native_ids)),
            "entries": len(native_ids),
            "printed_six_decimal_exact": int(np.count_nonzero(
                [f"{value:.6f}" == f"{native:.6f}"
                 for value, native in zip(values, native_values)])),
            "max_abs_error_mm": float(delta.max()),
            "mean_abs_error_mm": float(delta.mean()),
        }
    if args.native_table:
        if not full:
            raise ValueError("native table comparison needs a complete surface")
        maximum = 0.0
        total = 0.0
        exact = 0
        count = 0
        with args.native_table.open() as stream:
            for count, row in enumerate(stream, start=1):
                native = row.split()[-1]
                if count > len(after):
                    raise AssertionError("native metric has more entries")
                value = float(after[count - 1])
                delta = abs(value - float(native))
                maximum = max(maximum, delta)
                total += delta
                exact += f"{value:.4f}" == native
        if count != len(after):
            raise AssertionError(("native metric entry count", count, len(after)))
        report["native_full_matrix"] = {
            "entries": count,
            "four_decimal_exact": exact,
            "max_abs_error_mm": maximum,
            "mean_abs_error_mm": total / count,
        }
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
