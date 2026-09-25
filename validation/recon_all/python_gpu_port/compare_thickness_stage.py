"""Compare standalone thickness maps against a FreeSurfer subject's maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer as fs
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-surf-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    for hemi in ("lh", "rh"):
        reference_file = args.reference_surf_dir / f"{hemi}.thickness"
        candidate_file = args.candidate_dir / f"{hemi}.thickness"
        reference = fs.read_morph_data(str(reference_file))
        candidate = fs.read_morph_data(str(candidate_file))
        if reference.shape != candidate.shape:
            raise ValueError(f"{hemi}: vertex counts differ")
        delta = np.abs(reference - candidate)
        print(json.dumps({
            "hemi": hemi,
            "vertices": int(reference.size),
            "outliers": int(np.count_nonzero(delta > .005 + .001 * np.abs(reference))),
            "maximum_abs_mm": float(delta.max()),
            "p99_abs_mm": float(np.quantile(delta, .99)),
            "binary_identical": reference_file.read_bytes() == candidate_file.read_bytes(),
        }), flush=True)


if __name__ == "__main__":
    main()
