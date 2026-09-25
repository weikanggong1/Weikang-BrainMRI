"""Compare complete ten-column ROI rows with a frozen native stats table."""

import argparse
import json
from pathlib import Path
import time

from fnit.recon_all.anatomical_stats_rows import anatomical_stats_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--surface", choices=("white", "pial"), default="white")
    parser.add_argument("--atlas", default="aparc")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    surf = args.subject / "surf" / args.hemi
    label = args.subject / "label"
    table = args.subject / "stats" / f"{args.hemi}.{args.atlas}"
    if args.surface == "pial":
        table = table.with_name(table.name + ".pial")
    saved = table.with_suffix(table.suffix + ".stats").read_text().splitlines()
    start = next(i for i, line in enumerate(saved) if line.startswith("# ColHeaders "))
    expected = [line for line in saved[start + 1:] if line.strip()]
    began = time.perf_counter()
    actual = anatomical_stats_rows(
        Path(str(surf) + ".white"), Path(str(surf) + ".pial"),
        Path(str(surf) + "." + args.surface),
        Path(str(surf) + (".area.pial" if args.surface == "pial" else ".area")),
        Path(str(surf) + ".thickness"), label / f"{args.hemi}.{args.atlas}.annot",
        label / f"{args.hemi}.cortex.label", device=args.device)
    elapsed = time.perf_counter() - began
    mismatches = [{"index": i, "native": a, "python": b}
                  for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    print(json.dumps({"hemi": args.hemi, "atlas": args.atlas,
                      "surface": args.surface, "device": args.device,
                      "reference_rows": len(expected), "python_rows": len(actual),
                      "exact_line_mismatches": len(mismatches),
                      "first_mismatches": mismatches[:3],
                      "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
