#!/usr/bin/env bash
set -euo pipefail

warp=${1:?input FreeSurfer displacement NIfTI required}
output=${2:?isolated output directory required}
mkdir -p "$output"
cd "$output"
module load freesurfer

(
  while [[ ! -s c.mgz ]]; do sleep 0.2; done
  cp xi.mgz xi.raw.mgz
) &
watcher=$!
/usr/bin/time -p env DIAG=0x8 DIAG_VERBOSE=1 mri_ca_register -invert-and-save "$warp" "$output/inverse.nii.gz" \
  > "$output/inverse_reference.log" 2>&1
wait "$watcher"
gzip -t xi.raw.mgz
