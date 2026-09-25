#!/usr/bin/env bash
# One fixed-subject CPU EntoWM diagnostic; no FreeSurfer executable is called.
set -euo pipefail

work=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925
stage="$work/connected_entowm_cpu_20260926"
snapshot="$work/fnit_reconall_snapshot_20260926c"
official=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official
python=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_main_20260924/venv/bin/python
weights=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/weights

mkdir -p "$stage"
cd "$snapshot"
ulimit -v $((48 * 1024 * 1024))
set +e
env -u FREESURFER_HOME -u FS_LICENSE -u SUBJECTS_DIR \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src:. \
  timeout 600 "$python" \
  validation/recon_all/python_gpu_port/experimental/compare_connected_entowm.py \
  "$work/connected_input_ca_chain_cpu_20260926/mri/nu.mgz" \
  "$official/mri/entowm.mgz" "$weights" "$stage/entowm.mgz" \
  --device cpu --official-stats "$official/stats/entowm.stats" \
  --report "$stage/comparison.json" > "$stage/run.log" 2>&1
code=$?
printf '%s\n' "$code" > "$stage/exit"
exit "$code"
