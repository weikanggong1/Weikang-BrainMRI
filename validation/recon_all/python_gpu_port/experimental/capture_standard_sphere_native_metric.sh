#!/usr/bin/env bash
# Capture the first nonquick MRISsampleDistances result without running unfolding.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 INPUT.inflated OUTPUT_DIR" >&2
  exit 2
fi
if [ -z "${FS_LICENSE:-}" ]; then
  echo "FS_LICENSE must point to the existing private license" >&2
  exit 2
fi
module load freesurfer
input=$(realpath "$1")
output=$2
mkdir -p "$output"
cd "$output"
export FS_MEASURE_DISTANCES=1
set +e
/usr/bin/time -f 'native_elapsed_seconds=%e\nnative_peak_kb=%M' -o native_time.txt \
  mris_sphere -threads 1 -seed 1234 -a 0 -n 1 -remove_negative 0 \
  "$input" "$PWD/sphere.metric.probe" > native.log 2>&1
status=$?
set -e
# FreeSurfer's diagnostic deliberately calls exit(1) immediately after dump.
if [ "$status" -ne 1 ] || [ ! -s distance.log ]; then
  echo "native diagnostic failed; exit=$status" >&2
  exit 1
fi
wc -l distance.log
cat native_time.txt
