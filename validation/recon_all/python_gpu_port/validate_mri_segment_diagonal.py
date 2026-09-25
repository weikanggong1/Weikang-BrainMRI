"""Check the fixed diagonal WM filter against a native diagnostic volume."""

import argparse
import json
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from fnit.recon_all.mri_segment import filter_diagonal_morphology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mgz")
    parser.add_argument("native_output_mgz")
    args = parser.parse_args()
    source = np.asarray(nib.load(args.input_mgz).dataobj).copy()
    native = np.asarray(nib.load(args.native_output_mgz).dataobj)
    start = perf_counter()
    candidate = filter_diagonal_morphology(torch.from_numpy(source)).numpy()
    report = {
        "seconds": perf_counter() - start,
        "shape": list(source.shape),
        "changed_voxels": int(np.count_nonzero(candidate != source)),
        "native_mismatches": int(np.count_nonzero(candidate != native)),
    }
    print(json.dumps(report, indent=2))
    if report["native_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
