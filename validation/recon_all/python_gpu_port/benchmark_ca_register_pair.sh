#!/usr/bin/env bash
set -euo pipefail

warp=${1:?fixed input warp required}
output=${2:?isolated output directory required}
python_bin=${3:?Python with NumPy, Numba, SciPy, and nibabel required}
source_dir=${4:?isolated Python source directory required}
export FS_LICENSE=${5:?FreeSurfer license file required}

mkdir -p "$output/native" "$output/python"
module load freesurfer
uptime > "$output/load_before.txt"
/usr/bin/time -p mri_ca_register -invert-and-save "$warp" "$output/native/inverse.nii.gz" \
  > "$output/native/run.log" 2>&1
uptime > "$output/load_between.txt"
/usr/bin/time -p env PYTHONPATH="$source_dir" "$python_bin" \
  "$source_dir/run_ca_register_inverse_python.py" "$warp" "$output/python/inverse.nii.gz" \
  --timing-json "$output/python/timing.json" > "$output/python/run.log" 2>&1
uptime > "$output/load_after.txt"
sha256sum "$warp" "$output/native/inverse.nii.gz" "$output/python/inverse.nii.gz" \
  > "$output/sha256.txt"
cmp "$output/native/inverse.nii.gz" "$output/python/inverse.nii.gz"
touch "$output/PARITY_PASS"
