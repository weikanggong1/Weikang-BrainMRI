"""Compare standalone entorhinal and ACJ edits with native replay."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fnit.recon_all.wm_edits_python import (
    amygdala_cortex_junction, fix_ento_wm)


def voxels(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mri = args.subject / "mri"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    acj = amygdala_cortex_junction(voxels(mri / "aseg.presurf.mgz"))
    native_mask = voxels(args.native_dir / "acj_mask_native.mgz")
    reports = [{"case": "acj_mask", "mismatched_voxels":
                int(np.count_nonzero(acj != native_mask)),
                "labeled_voxels": int(np.count_nonzero(native_mask))}]
    cases = (
        ("ento_brain", "brain.finalsurfs.mgz", "entowm.mgz", 2, False),
        ("ento_wmseg", "wm.seg.mgz", "entowm.mgz", 3, False),
        ("acj_brain", "brain.finalsurfs.mgz", "aseg.presurf.mgz", 3, True),
        ("acj_wmseg", "wm.seg.mgz", "aseg.presurf.mgz", 3, True),
    )
    for name, input_name, label_name, level, use_acj in cases:
        input_file = mri / input_name
        output_file = args.output_dir / f"{name}_python.mgz"
        affected = fix_ento_wm(input_file, mri / label_name, output_file,
                                level=level, left_value=255, right_value=255,
                                acj=use_acj)
        native = voxels(args.native_dir / f"{name}_native.mgz")
        candidate = voxels(output_file)
        reports.append({"case": name, "labeled_voxels": affected,
                        "changed_from_input": int(np.count_nonzero(native != voxels(input_file))),
                        "mismatched_voxels": int(np.count_nonzero(candidate != native))})
    print(json.dumps({"cases": reports, "volume_voxels": native.size}, indent=2))


if __name__ == "__main__":
    main()
