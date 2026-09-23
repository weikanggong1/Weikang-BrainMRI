"""Recreate the three defaced CC0 examples from original OpenNeuro files.

Requires FreeSurfer 8.2, the official SynthStrip weight, nibabel and scipy.
Input files must be downloaded separately using examples/data/SOURCES.json.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--freesurfer-home", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True,
                        help="directory containing synthstrip.1.pt")
    args = parser.parse_args()
    records = json.loads((Path(__file__).resolve().parents[1] / "examples/data/SOURCES.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case in records["cases"]:
        source = args.raw_dir / case["file"]
        output = args.out_dir / case["file"]
        if output.exists():
            raise FileExistsError(output)
        if hashlib.sha256(source.read_bytes()).hexdigest() != case["source_sha256"]:
            raise ValueError(f"source checksum mismatch: {source}")
        with TemporaryDirectory() as temporary:
            mask_file = Path(temporary) / "brain_mask.nii.gz"
            command = [str(args.freesurfer_home / "bin/mri_synthstrip"),
                       "-i", str(source), "-m", str(mask_file), "-t", "4",
                       "--model", str(args.weights / "synthstrip.1.pt")]
            subprocess.run(command, check=True,
                           env={**os.environ, "FREESURFER_HOME": str(args.freesurfer_home)})
            image = nib.load(source)
            mask = nib.load(mask_file).get_fdata() > 0.5
        assert mask.shape == image.shape
        distance = distance_transform_edt(~mask, sampling=image.header.get_zooms()[:3])
        data = image.get_fdata(dtype=np.float32)
        data[distance > 12.0] = 0
        header = image.header.copy()
        for name in ("descrip", "aux_file", "db_name", "intent_name"):
            header[name] = b""
        header.extensions.clear()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(data, image.affine, header), output)
        sha = hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"{output}: {sha}", flush=True)
        if sha != case["published_sha256"]:
            print("Byte checksum differs; verify FreeSurfer, weight and NIfTI library versions.", flush=True)


if __name__ == "__main__":
    main()
