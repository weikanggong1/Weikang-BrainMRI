# Anatomical-stats ROI curvature columns

`surface_roi_curvature_gpu.py` replays the four remaining curvature columns of
the fixed `mris_anatomical_stats -no-th3` tables on already matched meshes and
annotations: `MeanCurv`, `GausCurv`, `FoldInd`, and `CurvInd`. It fits the
two-neighbor quadratic from FreeSurfer 8.2's
`MRIScomputeSecondFundamentalForm` and applies the area-weighted region
formulas in the pinned `mris_anatomical_stats.cpp` and
`mrisurf_metricProperties.cpp` sources. The fit is computed with PyTorch;
adjacency construction uses SciPy on CPU. CUDA matrix multiplication enables
TF32; half-precision types are not used.

The principal-curvature order matters for `FoldInd`. The regular eigen fit
orders the larger absolute curvature first. FreeSurfer's ill-conditioned
fallback keeps its raw `kmax,kmin` order, even when `abs(kmin)>abs(kmax)`.
Always sorting by absolute magnitude produced one wrong displayed ROI value
in the left pial aparc table. Replaying that fallback made it match. The
native diagnostic `curv.dat` exposed 106,615 fitted vertices; the seven
early-exit vertices were checked separately against the source branch.

On the frozen `fs_sub01` subject, eight tables covered bilateral aparc,
aparc.a2009s and aparc.DKTatlas white surfaces plus bilateral aparc pial
surfaces. All **346/346** region rows matched FreeSurfer's displayed values
in all four columns. Maximum unrounded differences against the *rounded*
reference table were 0.000499, 0.000499, 0.497 and 0.0499 respectively;
these are below each column's display half-unit. Every row name also matched.
The [CPU report](roi_curvature_cpu_report.json) retains each table's row count,
error and elapsed Python compute time. A separate H100 GPU 1 run with TF32
matrix operations also matched **346/346** displayed rows in all four columns;
its [report](roi_curvature_gpu_tf32_report.json) records all eight cases and
their timings. Each CUDA timing is from a fresh process with first CUDA use
included, but has no paired native run. The earlier five
area/thickness/volume columns are validated separately in
[the stage summary](README.md).

This test used the official subject's area maps, annotations, cortex labels
and surfaces as inputs. A Python area-map port has independently passed its
vertex-level check, but the complete Python stats file writer and global
header measures are still pending. Neither curvature report contains a
same-host native timing. The full reconstruction is not yet native-free.
