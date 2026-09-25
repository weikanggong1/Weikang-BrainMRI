"""Probe three mris_anatomical_stats ROI columns on frozen surfaces."""

import argparse
import json
from pathlib import Path

import numpy as np

from fnit.recon_all.surface_roi_gpu import roi_area_thickness, roi_gray_volume


def reference_rows(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) == 10:
            rows[fields[0]] = (int(fields[1]), float(fields[2]),
                               float(fields[4]), float(fields[5]), float(fields[3]))
    return rows


def formatted(values):
    result = (f"{values[1]:.0f}", f"{values[2]:.3f}", f"{values[3]:.3f}")
    return result + ((f"{values[4]:.0f}",) if len(values) == 5 else ())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--atlas", default="aparc")
    parser.add_argument("--surface", default="white", choices=("white", "pial"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--volume", action="store_true")
    args = parser.parse_args()
    hemi = args.hemi
    subject = args.subject
    stats_name = f"{hemi}.{args.atlas}"
    if args.surface == "pial":
        stats_name += ".pial"
    expected = reference_rows(subject / "stats" / f"{stats_name}.stats")
    actual = roi_area_thickness(subject / "surf" / f"{hemi}.{args.surface}",
                                subject / "label" / f"{hemi}.{args.atlas}.annot",
                                subject / "surf" / f"{hemi}.thickness",
                                device=args.device)
    if args.volume:
        volumes = roi_gray_volume(subject / "surf" / f"{hemi}.white",
                                  subject / "surf" / f"{hemi}.pial",
                                  subject / "surf" / f"{hemi}.thickness",
                                  subject / "label" / f"{hemi}.{args.atlas}.annot",
                                  device=args.device)
        actual = {name: (*values, volumes[name]) for name, values in actual.items()}
    else:
        expected = {name: values[:4] for name, values in expected.items()}
    names = sorted(set(expected) | set(actual))
    report = {"hemi": hemi, "atlas": args.atlas, "surface": args.surface,
              "reference_rows": len(expected), "candidate_rows": len(actual),
              "missing": sorted(set(expected) - set(actual)),
              "extra": sorted(set(actual) - set(expected))}
    if not report["missing"] and not report["extra"]:
        array = np.array([actual[name] for name in names])
        target = np.array([expected[name] for name in names])
        columns = ("NumVert", "SurfArea", "ThickAvg", "ThickStd", "GrayVol")
        report["max_error"] = dict(zip(columns,
                                       np.abs(array - target).max(axis=0).tolist()))
        report["mismatched_numvert"] = [name for name in names
                                        if actual[name][0] != expected[name][0]]
        report["area_outliers"] = [name for name in names
                                   if abs(actual[name][1] - expected[name][1]) > 1.0]
        report["thickness_outliers"] = [name for name in names
                                        if max(abs(actual[name][i] - expected[name][i])
                                               for i in (2, 3)) > 0.01]
        if args.volume:
            report["volume_outliers"] = [name for name in names
                                         if abs(actual[name][4] - expected[name][4]) >
                                         1.0 + 0.001 * abs(expected[name][4])]
            report["volume_examples"] = {name: [actual[name][4], expected[name][4]]
                                         for name in names[:5]}
        report["formatted_differences"] = [name for name in names
                                           if formatted(actual[name]) != formatted(expected[name])]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
