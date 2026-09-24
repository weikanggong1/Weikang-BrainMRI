# Recon-all paired end-to-end timing record

Run A (official FreeSurfer 8.2) and C (verified GPU bundle) serially on the
same node and T1. The default order is A→C; pass `--order C-A` with a **new**
`--output` directory for a second pair to counter run-order and cache effects.
Both commands use `-all -parallel -openmp 4 -itkthreads 1`;
C uses logical `cuda:1`. The benchmark script prints a dry-run plan by default.
It never removes an existing output directory. Use a new `--output` path for
each pair.

```bash
module load freesurfer
python tools/benchmark_recon_all_pair.py \
  --t1 examples/data/sub-01_T1w.nii.gz \
  --official-home "$FREESURFER_HOME" \
  --candidate-python /absolute/path/to/runtime/bin/python \
  --bundle /absolute/path/to/verified/fs820-bundle \
  --license /absolute/path/to/freesurfer/license.txt \
  --output work/reconall_pair_001
```

Review the printed plan and confirm that the node is free of competing jobs.
Add `--execute` to the same command to run the selected order. The script
refuses an existing output path, starts the second run only after the first
completes successfully, and records
`plan.json`, `summary.json`, each run's `/usr/bin/time -v` output, recon-all
stdout, and 10-second host CPU/GPU samples. The production C command does not
pass `--development-bundle`; its manifest and runtime profile must pass.

The script records the T1 SHA-256 before A and checks it again before C. It
also records host name, CPU affinity, CUDA visibility, GPU UUID/index, official
build stamp and binary hash, bundle manifest hash, effective command lines and
run exit status. `cuda:1` is a **logical CUDA index**; inspect the recorded
`CUDA_VISIBLE_DEVICES` and GPU inventory when assigning the physical GPU.

## Result to fill after a completed pair

| Check | Value |
|---|---|
| T1 SHA-256 |  |
| Node / CPU affinity / GPU UUID |  |
| Official build / bundle manifest SHA-256 |  |
| Competing jobs or resource contention |  |
| Order (`A-C` or `C-A`) |  |
| A wall seconds from `/usr/bin/time -v` |  |
| C wall seconds from `/usr/bin/time -v` |  |
| C/A time ratio; A/C throughput speedup |  |
| A and C exit 0, required outputs present |  |
| Numerical comparison JSON and mandatory checks |  |
| Largest per-stage regressions |  |

Report the observed pair as one run. A claim that the GPU workflow is faster
requires the numerical comparison to pass and repeated isolated pairs with
the same settings. The host load CSV describes contention; it does not by
itself attribute elapsed-time changes to a specific implementation stage.
