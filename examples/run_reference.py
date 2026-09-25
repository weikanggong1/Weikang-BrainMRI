"""Run official FreeSurfer CPU commands for the documented figure pair."""

import os
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parent
data = root / "data"
output = root / "results/reference"
output.mkdir(parents=True, exist_ok=True)
fs = Path(os.environ["FREESURFER_HOME"])
weights = Path(os.environ["FNIT_WEIGHTS"])
env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NVIDIA_TF32_OVERRIDE": "0"}

for case in ("sub-01", "sub-02"):
    subprocess.run([
        str(fs / "bin/mri_synthstrip"), "-i", str(data / f"{case}_T1w.nii.gz"),
        "-o", str(output / f"{case}_brain.nii.gz"),
        "-m", str(output / f"{case}_mask.nii.gz"),
        "-t", "4", "--model", str(weights / "synthstrip.1.pt"),
    ], check=True, env=env)

subprocess.run([
    str(fs / "bin/mri_synthmorph"), "register", "-m", "joint", "-e", "256",
    "-r", "0.5", "-n", "7", "-j", "8",
    "-w", str(weights / "synthmorph.affine.2.h5"),
    "-w", str(weights / "synthmorph.deform.3.h5"),
    "-o", str(output / "sub-02_in_sub-01.nii.gz"),
    "-t", str(output / "sub-02_to_sub-01.mgz"),
    str(data / "sub-02_T1w.nii.gz"), str(data / "sub-01_T1w.nii.gz"),
], check=True, env=env)
