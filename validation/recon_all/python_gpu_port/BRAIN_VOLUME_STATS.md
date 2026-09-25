# Brain volume statistics

`brain_volume_stats_python.py` implements the `ComputeBrainVolumeStats2`
branch used by this FreeSurfer 8.2.0 `recon-all` run. It reads final
`aseg.mgz`, `brainmask.mgz`, four white/pial surfaces and the small
`ASegStatsLUT.txt` data file. Voxel measures use NumPy counts; the four
surface-related measures use ordered mesh geometry. It does not execute or
load FreeSurfer binaries.

For the frozen `fs_sub01` inputs, **12/16 measures were numerically exact**
against the official `brainvol.stats` cache. The other four differed by at
most **0.000295 mm³** due to floating-point surface-volume accumulation.
The [report](brain_volume_stats_cpu_report.json) gives every signed
difference and a 1.96 s Python CPU compute time on headcw. No paired native
timing is available yet. The source code branch was confirmed in the subject
log: `mri_brainvol_stats` ran `ComputeBrainVolumeStats2` with `KeepCSF=1`.

The reference inputs here are official final segmentation and surfaces. This
port still needs those files from the Python reconstruction. Aseg label 77
requires MNI305 lateralization and is rejected explicitly; that label is
absent from the validated subject. This stage therefore has a known
generalization gap until a subject containing label 77 is tested.
