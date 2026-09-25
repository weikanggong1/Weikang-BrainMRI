"""Measure the experimental normalization passes against frozen sub-01 outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from normalize_gpu import run


def compare(source: Path, candidate: Path, reference: Path) -> dict:
    src = np.asarray(nib.load(str(source)).dataobj)
    got_image = nib.load(str(candidate))
    ref_image = nib.load(str(reference))
    got, ref = np.asarray(got_image.dataobj), np.asarray(ref_image.dataobj)
    region = (src > 0) | (ref > 0)
    delta = np.abs(got.astype(np.float32) - ref)
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "shape": list(map(int, ref.shape)),
        "dtype_equal": got.dtype == ref.dtype,
        "affine_equal": bool(np.array_equal(got_image.affine, ref_image.affine)),
        "foreground_voxels": int(region.sum()),
        "voxel_mismatches": int(np.count_nonzero(got != ref)),
        "foreground_mae": float(delta[region].mean()),
        "foreground_p95_absolute_error": float(np.quantile(delta[region], 0.95)),
        "foreground_max_absolute_error": float(delta[region].max()),
        "foreground_exact_fraction": float(np.mean(got[region] == ref[region])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    mri = args.subject / "mri"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"device": args.device, "reference_log_seconds": {"first": 109.49,
                                                               "second": 145.07}}
    for phase, source_name, target_name in (("first", "nu", "T1"),
                                            ("second", "norm", "brain")):
        source = mri / f"{source_name}.mgz"
        target = mri / f"{target_name}.mgz"
        output = args.output_dir / f"{phase}.mgz"
        extra = {"aseg_file": mri / "aseg.presurf.mgz",
                 "mask_file": mri / "brainmask.mgz"} if phase == "second" else {}
        start = time.perf_counter()
        details = run(source, output, phase=phase, device=args.device, **extra)
        elapsed = time.perf_counter() - start
        report[phase] = dict(details, wall_seconds=elapsed,
                             **compare(source, output, target))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
