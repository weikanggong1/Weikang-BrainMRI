#!/usr/bin/env python3
"""Compare exact surface and auxiliary volume output for two isolated replays."""

import json
import pathlib
import sys

import nibabel as nib
import numpy as np


root = pathlib.Path(__file__).resolve().parent
left, right = sys.argv[1:3]
lv, lf = nib.freesurfer.read_geometry(root / f"replay_{left}/surf/lh.white.preaparc")
rv, rf = nib.freesurfer.read_geometry(root / f"replay_{right}/surf/lh.white.preaparc")
distance = np.linalg.norm(lv.astype(np.float64) - rv.astype(np.float64), axis=1)
lm = np.asarray(nib.load(root / f"replay_{left}/mri/mrisps.wpa.mgz").dataobj)
rm = np.asarray(nib.load(root / f"replay_{right}/mri/mrisps.wpa.mgz").dataobj)
result = {
    "left": left,
    "right": right,
    "vertices": int(len(lv)),
    "faces": int(len(lf)),
    "faces_equal": bool(np.array_equal(lf, rf)),
    "vertices_exact": bool(np.array_equal(lv, rv)),
    "vertices_different": int(np.count_nonzero(distance)),
    "vertex_max_mm": float(distance.max()),
    "vertex_p99_mm": float(np.percentile(distance, 99)),
    "volume_equal": bool(np.array_equal(lm, rm)),
    "volume_different": int(np.count_nonzero(lm != rm)),
}
print(json.dumps(result, indent=2))
