"""Compare TH3 per-vertex cortical volume with an official lh/rh.volume map."""

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

from fnit.recon_all.surface_roi_gpu import vertex_th3_volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--hemi", choices=("lh", "rh"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    hemi = args.hemi
    reference = np.asarray(fsio.read_morph_data(str(args.subject / "surf" /
                                                  f"{hemi}.volume")), dtype=np.float32)
    actual = vertex_th3_volume(args.subject / "surf" / f"{hemi}.white",
                               args.subject / "surf" / f"{hemi}.pial",
                               args.subject / "label" / f"{hemi}.cortex.label",
                               device=args.device)
    if actual.shape != reference.shape:
        raise ValueError("Vertex count differs")
    error = np.abs(actual - reference)
    print(json.dumps({"hemi": hemi, "vertices": len(actual),
                      "reference_sum": float(reference.sum()),
                      "candidate_sum": float(actual.sum()),
                      "max_abs_mm3": float(error.max()),
                      "mean_abs_mm3": float(error.mean()),
                      "outliers": int((error > 0.005 + 0.001 * np.abs(reference)).sum())},
                     indent=2))


if __name__ == "__main__":
    main()
