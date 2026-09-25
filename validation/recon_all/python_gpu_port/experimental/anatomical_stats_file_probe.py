"""Validate one complete Python cortical stats file's numerical content."""

import argparse
import json
from pathlib import Path

from fnit.recon_all.anatomical_stats_file import write_anatomical_stats
from fnit.recon_all.brain_volume_stats_python import compute_brain_volume_stats


def values(path: Path) -> tuple[dict[str, float], list[str]]:
    lines = path.read_text().splitlines()
    measures = {fields[1]: float(fields[3]) for line in lines
                if line.startswith("# Measure ")
                for fields in [[item.strip() for item in line.split(",")]]}
    start = next(i for i, line in enumerate(lines) if line.startswith("# ColHeaders "))
    return measures, lines[start + 1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--aseg-lut", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    brainvol = compute_brain_volume_stats(args.subject, args.aseg_lut)
    output = write_anatomical_stats(args.subject, "lh", "aparc", "white",
                                    brainvol, args.output, device="cpu")
    native_values, native_rows = values(args.subject / "stats" / "lh.aparc.stats")
    python_values, python_rows = values(output)
    differences = {key: python_values[key] - native_values[key] for key in native_values}
    print(json.dumps({"reference_rows": len(native_rows), "python_rows": len(python_rows),
                      "exact_row_mismatches": sum(a != b for a, b in zip(native_rows, python_rows)),
                      "header_measure_differences": differences}, indent=2))


if __name__ == "__main__":
    main()
