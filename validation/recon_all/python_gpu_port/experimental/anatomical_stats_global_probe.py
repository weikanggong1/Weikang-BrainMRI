"""Compare the numeric/global cortical stats header against native output."""

import argparse
import json
from pathlib import Path

from fnit.recon_all.anatomical_stats_global import anatomical_stats_global_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--surface", choices=("white", "pial"), default="white")
    parser.add_argument("--atlas", default="aparc")
    args = parser.parse_args()
    subject, hemi = args.subject, args.hemi
    surf = subject / "surf" / hemi
    label = subject / "label"
    table = subject / "stats" / f"{hemi}.{args.atlas}"
    if args.surface == "pial":
        table = table.with_name(table.name + ".pial")
    expected = [line for line in Path(str(table) + ".stats").read_text().splitlines()
                if line.startswith("# Measure ") or line.startswith("# BrainVolStatsFixed")]
    actual = anatomical_stats_global_lines(
        Path(str(surf) + (".area.pial" if args.surface == "pial" else ".area")),
        Path(str(surf) + ".thickness"), label / f"{hemi}.{args.atlas}.annot",
        label / f"{hemi}.cortex.label", subject / "stats" / "brainvol.stats",
        subject / "mri" / "transforms" / "talairach.xfm", surface=args.surface)
    mismatches = [{"index": i, "native": a, "python": b}
                  for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    print(json.dumps({"hemi": hemi, "surface": args.surface, "atlas": args.atlas,
                      "native_lines": len(expected), "python_lines": len(actual),
                      "mismatch_count": len(mismatches),
                      "first_mismatches": mismatches[:4]}, indent=2))


if __name__ == "__main__":
    main()
