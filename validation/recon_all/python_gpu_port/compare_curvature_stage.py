"""Compare one FreeSurfer curvature map against an independent implementation."""

from __future__ import annotations

import argparse
import json

import nibabel.freesurfer as fs
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = fs.read_morph_data(args.reference)
    candidate = fs.read_morph_data(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError("vertex counts differ")
    difference = np.abs(reference - candidate)
    worst = np.argsort(difference)[-10:][::-1]
    print(json.dumps({
        "vertices": int(reference.size),
        "reference_range": [float(reference.min()), float(reference.max())],
        "candidate_range": [float(candidate.min()), float(candidate.max())],
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.quantile(difference, .99)),
        "maximum_abs": float(difference.max()),
        "outliers_0_005_plus_0_1pct": int(np.count_nonzero(
            difference > .005 + .001 * np.abs(reference))),
        "correlation": float(np.corrcoef(reference, candidate)[0, 1]),
        "worst": [[int(i), float(reference[i]), float(candidate[i])] for i in worst],
    }))


if __name__ == "__main__":
    main()
