"""Compare four anatomical-stats curvature columns on one frozen surface."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fnit.recon_all.surface_roi_curvature_gpu import curvature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--surface", choices=("white", "pial"), default="white")
    parser.add_argument("--atlas", default="aparc")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prefix = args.subject / "surf" / args.hemi
    stats = args.subject / "stats" / f"{args.hemi}.{args.atlas}"
    if args.surface == "pial":
        stats = stats.with_name(stats.name + ".pial")
    target = {}
    for line in Path(str(stats) + ".stats").read_text().splitlines():
        fields = line.split()
        if len(fields) == 10 and not line.startswith("#"):
            target[fields[0]] = tuple(float(value) for value in fields[6:10])
    area = Path(str(prefix) + (".area.pial" if args.surface == "pial" else ".area"))
    began = time.perf_counter()
    actual = curvature_columns(Path(str(prefix) + "." + args.surface), area,
                               args.subject / "label" / f"{args.hemi}.{args.atlas}.annot",
                               args.subject / "label" / f"{args.hemi}.cortex.label",
                               args.device)
    elapsed = time.perf_counter() - began
    names = sorted(set(target) & set(actual))
    delta = np.array([np.array(actual[name]) - target[name] for name in names])
    formatted = []
    for name in names:
        candidate = actual[name]
        reference = target[name]
        if (f"{candidate[0]:.3f}", f"{candidate[1]:.3f}",
            f"{candidate[2]:.0f}", f"{candidate[3]:.1f}") != (
            f"{reference[0]:.3f}", f"{reference[1]:.3f}",
            f"{reference[2]:.0f}", f"{reference[3]:.1f}"):
            formatted.append({"name": name, "candidate": candidate, "reference": reference})
    print(json.dumps({"hemi": args.hemi, "surface": args.surface, "atlas": args.atlas,
                      "device": args.device, "elapsed_seconds": elapsed,
                      "rows": len(names), "missing": sorted(set(target) - set(actual)),
                      "extra": sorted(set(actual) - set(target)),
                      "max_abs": np.abs(delta).max(axis=0).tolist(),
                      "formatted_mismatches": formatted,
                      "formatted_mismatch_count": len(formatted)}, indent=2))


if __name__ == "__main__":
    main()
