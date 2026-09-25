"""Compare Python brain volume measures with the native cache."""

import argparse
import json
from pathlib import Path
import time

from fnit.recon_all.anatomical_stats_global import read_brain_volume_stats
from fnit.recon_all.brain_volume_stats_python import compute_brain_volume_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--aseg-lut", type=Path, required=True)
    args = parser.parse_args()
    reference = read_brain_volume_stats(args.subject / "stats" / "brainvol.stats")
    start = time.perf_counter()
    actual = compute_brain_volume_stats(args.subject, args.aseg_lut)
    elapsed = time.perf_counter() - start
    differences = {key: actual[key] - reference[key] for key in actual}
    print(json.dumps({"elapsed_seconds": elapsed,
                      "max_absolute_difference_mm3": max(abs(value) for value in differences.values()),
                      "differences_mm3": differences}, indent=2))


if __name__ == "__main__":
    main()
