#!/usr/bin/env bash
set -euo pipefail

warp=${1:?input FreeSurfer displacement NIfTI required}
output=${2:?isolated output directory required}
mkdir -p "$output"
cd "$output"

module load freesurfer
/usr/bin/time -p mri_warp_convert --infswarp "$warp" --outfswarp "$output/input.abs-crs.mgz" --out-interp abs-crs \
  > "$output/absolute_reference.log" 2>&1
/usr/bin/time -p env DIAG=0x8 DIAG_VERBOSE=1 mri_ca_register -invert-and-save "$warp" "$output/inverse.nii.gz" \
  > "$output/inverse_reference.log" 2>&1
test -s "$output/c.mgz"
