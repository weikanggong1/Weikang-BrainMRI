"""Compare a first-pass border-search checkpoint with pinned source."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


STATE = np.dtype([("floats", "<f4", 16), ("flags", "<i4", 2)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--hemisphere", choices=("lh", "rh"), default="lh")
    parser.add_argument("--surface", choices=("white", "pial"), default="white")
    parser.add_argument("--prefix", default="lh.border_probe")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    from place_surface_border import compute_border_values_first_pass

    before = np.fromfile(args.probe / f"{args.prefix}.before", dtype=STATE)
    after = np.fromfile(args.probe / f"{args.prefix}.after", dtype=STATE)
    n = args.limit or len(before)
    before, after = before[:n], after[:n]
    volume_image = nib.load(args.probe / f"{args.prefix}.volume.mgz")
    volume = np.asarray(volume_image.dataobj)
    seg = np.asarray(nib.load(args.subject / "mri/aseg.presurf.mgz").dataobj)
    sras2vox = np.fromfile(args.probe / f"{args.prefix}.transform", dtype="<f4").reshape(4, 4)
    stats = dict(
        line.split()[:2]
        for line in (args.subject / f"surf/autodet.gw.stats.{args.hemisphere}.dat").read_text().splitlines()
        if len(line.split()) >= 2
    )
    thresholds = np.array([
        float(stats[f"{args.surface}_{name}"])
        for name in ("inside_hi", "border_hi", "border_low", "outside_low", "outside_hi")
    ])
    start = time.perf_counter()
    result = compute_border_values_first_pass(
        volume, seg, before["floats"][:, :3], before["floats"][:, 3:6],
        before["floats"][:, 6:9], before["flags"][:, 0], before["floats"][:, 9],
        sras2vox, thresholds, hemisphere=args.hemisphere, surface=args.surface,
    )
    elapsed = time.perf_counter() - start
    reference = (
        after["floats"][:, 9], after["floats"][:, 10], after["floats"][:, 11],
        after["floats"][:, 12:15], after["flags"][:, 1], after["floats"][:, 15],
    )
    names = ("value", "distance", "gradient", "target_xyz", "marked", "sigma")
    report = {"hemisphere": args.hemisphere, "surface": args.surface, "vertices": n, "seconds_including_jit": elapsed, "comparisons": {}}
    for name, actual, expected in zip(names, result, reference):
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        summary = {
            "exact_elements": int(np.count_nonzero(actual == expected)),
            "total_elements": int(expected.size),
            "max_abs_error": float(error.max(initial=0)),
        }
        if name == "value":
            mismatch = np.flatnonzero(actual != expected)
            summary["first_mismatch_vertices"] = mismatch[:20].tolist()
            summary["first_mismatch_actual"] = actual[mismatch[:20]].tolist()
            summary["first_mismatch_native"] = expected[mismatch[:20]].tolist()
        report["comparisons"][name] = summary
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and any(
        item["exact_elements"] != item["total_elements"]
        for item in report["comparisons"].values()
    ):
        raise SystemExit("border-search checkpoint differs from pinned source")


if __name__ == "__main__":
    main()
