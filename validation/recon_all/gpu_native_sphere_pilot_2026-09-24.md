# CUDA native sphere pilot: one left hemisphere

This is an isolated `mris_sphere` experiment on `gpucw1`, not a change to the
validated recon-all bundle. All four runs consumed the same frozen
`lh.inflated` from the official sub-01 reconstruction (SHA-256
`a4153ecb43c37a538bed92d6da91ae9d095becf29e4211ef5fd96b4c84baa8fb`),
used four threads and seed 1234, and exited 0. The CUDA run printed
`MRISaverageGradients: CUDA active`.

| Final `inflated` to `sphere` run | Wall time | Maximum coordinate difference from official |
|---|---:|---:|
| Official FreeSurfer 8.2 binary | 329.6 s | 0 mm; also matches the original recon-all sphere exactly |
| Clean rebuilt CPU binary | 330.0 s | 7.968 mm |
| Patched binary, CUDA disabled | 388.5 s | 7.968 mm |
| Patched binary, CUDA enabled | 356.9 s | 7.968 mm |

The clean rebuilt, patched CPU and patched CUDA surfaces have identical
ordered faces and float32 coordinates. The 7.968 mm mismatch therefore occurs
**before** the CUDA bridge: the official and clean rebuilt programs diverge in
the `MRISunfold` optimization. Their first printed post-fold-removal distance
errors are 19.55% and 19.50%. The official executable was built with GCC 4.8.5;
the rebuilt objects used GCC 11.2 and C++17. Compiler or native-library
differences are hypotheses, not yet established causes. A targeted
`FS_MEASURE_DISTANCES=1` replay generated 8,268,920 identical distance records
in both binaries (SHA-256
`869b4d474f844a33757f1b62d01b3d104f55d1176a309a83f715eac9821c5afd`).
Snapshots from iterations 0–6 also had identical ordered coordinates and
faces. The divergence occurs later in optimization. The experiment fails the
1e-5 mm official-coordinate gate and cannot enter the recon-all bundle.

The sequential timings were measured while the shared node's one-minute load
started between 60.12 and 62.52. They do not establish a repeatable CUDA
speedup: the patched CPU and CUDA runs differ in wall time even though the
unpatched and patched CPU programs produce identical surfaces. The earlier
`inflated.nofix` to `qsphere.nofix` left-hemisphere test passed exact
official/CPU/CUDA coordinate and face agreement, but its patched CPU and CUDA
stage times were about 76.0 and 76.7 seconds. Thus it showed no stage gain.

As an independent CPU speed check, setting `FS_FASTER_MP=1` on the **official**
binary reduced this final left-hemisphere stage to 269.55 s, but its output
differed from the official default by up to 7.537 mm. It also fails the
per-vertex gate. In three fixture trials per hemisphere, the CUDA bridge's
warm call was approximately 9 ms (one 148 ms outlier) versus roughly
1.2–1.4 s for a one-thread ordered CPU calculation. All fixture float32
outputs matched bitwise. These fixture vectors come from surface geometry,
not the optimizer's actual gradients, and do not predict the full stage time.

The source patch, C ABI bridge and benchmark driver are under
`tools/native_cuda_experiments/`. The full final-stage report is retained on
shared storage at
`work/gpu_native_pilot_results/stage_lh_final_compat_20260924/report.json`
(SHA-256 `3effcb9f602c7f57dbda983ccb3c792698a2b205382253c2fc7ea7986bdbe030`).
Further native work needs a compiler/runtime-matched clean rebuild and an
official-coordinate replay before testing CUDA downstream.
