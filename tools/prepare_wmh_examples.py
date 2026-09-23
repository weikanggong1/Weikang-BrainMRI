"""Make three brain-limited CC0 FLAIR examples from OpenNeuro ds003592.

Download the original files named sub-02/03/04_FLAIR.nii.gz into --raw-dir
first. This script uses the original FreeSurfer SynthStrip and its official
checkpoint to remove intensities farther than 6 mm from the predicted brain.
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


SOURCES = {
    "sub-02": "6784c9822415f701dd25a6009e68bee2dfd11427a04d99617c8fc0859473e7d9",
    "sub-03": "cf4f77d52f5b39f6631cb002b684f8c0571ca123f139a48efc4ee6e139d4bb6d",
    "sub-04": "49de51a875ade818485c239d37632dce4aafc042f7bf241188460ff9d989c622",
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--freesurfer-home", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--gpu-python", type=Path,
                        help="CUDA-capable Python to run the unmodified original SynthStrip script")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for subject, source_hash in SOURCES.items():
        source = args.raw_dir / f"{subject}_FLAIR.nii.gz"
        output = args.out_dir / source.name
        if digest(source) != source_hash:
            raise ValueError(f"Source SHA-256 mismatch: {source}")
        if output.exists():
            raise FileExistsError(output)
        with TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / "mask.nii.gz"
            launcher = ([str(args.gpu_python), str(args.freesurfer_home / "python/scripts/mri_synthstrip"), "-g"]
                        if args.gpu_python else [str(args.freesurfer_home / "bin/mri_synthstrip")])
            subprocess.run(launcher + [
                "-i", str(source), "-m", str(mask_path), "-t", "4",
                "--model", str(args.weights / "synthstrip.1.pt"),
            ], check=True, env={**os.environ, "FREESURFER_HOME": str(args.freesurfer_home)})
            mask = nib.load(mask_path).get_fdata() > 0.5
        image = nib.load(source)
        if mask.shape != image.shape:
            raise ValueError(f"Mask geometry mismatch: {subject}")
        distance = distance_transform_edt(~mask, sampling=image.header.get_zooms()[:3])
        data = image.get_fdata(dtype=np.float32)
        data[distance > 6.0] = 0
        header = image.header.copy()
        for field in ("descrip", "aux_file", "db_name", "intent_name"):
            header[field] = b""
        header.extensions.clear()
        header.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(data, image.affine, header), output)
        records.append({
            "file": output.name,
            "source_url": (f"https://s3.amazonaws.com/openneuro.org/ds003592/"
                           f"{subject}/ses-1/anat/{subject}_ses-1_FLAIR.nii.gz"),
            "source_sha256": source_hash,
            "published_sha256": digest(output),
            "published_bytes": output.stat().st_size,
            "shape": list(image.shape),
            "voxel_size_mm": [float(value) for value in image.header.get_zooms()[:3]],
        })
        print(f"{output}: {records[-1]['published_sha256']}", flush=True)
    manifest = {
        "dataset": "OpenNeuro ds003592",
        "source_dataset_url": "https://openneuro.org/datasets/ds003592",
        "source_doi": "10.18112/openneuro.ds003592.v1.0.13",
        "source_license": "CC0",
        "source_license_metadata": "https://raw.githubusercontent.com/OpenNeuroDatasets/ds003592/master/dataset_description.json",
        "derivative_note": ("FLAIR examples, retaining image intensities no farther than 6 mm "
                            "from the brain mask predicted by original FreeSurfer mri_synthstrip "
                            "using synthstrip.1.pt. Other voxels are zeroed; geometry is preserved, "
                            "NIfTI header identifiers removed. These are derived test inputs, not "
                            "clinical ground truth."),
        "preparation_script": "tools/prepare_wmh_examples.py",
        "cases": records,
    }
    (args.out_dir / "SOURCES.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
