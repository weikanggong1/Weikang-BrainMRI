# LH independent smoothwm input and first continuation boundary

The frozen `fs_sub01` LH `smoothwm`, original sphere, exact Python sulc
`debug0056` seed and atlas have the SHA-256 values in the paired JSON reports.
This is a bounded FreeSurfer 8.2 comparison on stored native checkpoints,
not an end-to-end reconstruction.

The production Python/PyTorch `smoothwm_mean_curvature` computes raw H from
the frozen surface without reading the native raw H. Its
[all-vertex report](mris_register_lh_smoothwm_production_raw_headcw.json)
matches **106,622/106,622 float32 values bitwise** and has the same raw-array
SHA-256 `319c755b667a04331ef0c545714aaf5897c9a04e0c08a882b69b48542f0ab39c`.
The measured 14.05 s is one CPU run that includes Numba startup, not a
paired native speed comparison.

Using that independent raw H, the [first-update report](mris_register_lh_default_epoch0057_independent_fixed_headcw.json)
selects `dt=2.721714973449707` and matches the native saved LH
`debug0057` surface at **106,622/106,622 ordered vertices**, maximum
coordinate error 0 mm. It starts from the previously verified exact Python
`debug0056` sulc seed. Setup, force, averaging, spring and line search took
22.51, 18.25, 6.91, 0.008 and 0.55 s respectively in that shared-host CPU run;
these exclude file I/O and are not a native speed ratio.

The [continuous two-update probe](probe_mris_register_sno2_epoch.py) retains
its own first-update coordinates for the next iteration and reads native
`debug0058` only for comparison. The [report](mris_register_lh_default_epoch0057_0058_continuous_headcw.json)
again matches `debug0057` completely, then selects `dt=3.4400434494018555`
for `debug0058`, versus native printed `3.448`. Only **18/106,622 ordered
vertices** match exactly; 842 are within 0.00001 mm, with maximum coordinate
error **0.00157928466796875 mm**. This is the first measured failure in the
current independent smoothwm continuation.

The source calls `mrisProjectSurface` at the end of each `MRISintegrate`
iteration. A diagnostic with an extra projection at the start of the second
iteration also failed (110 exact vertices, 0.0005035400390625 mm maximum
error). Removing the extra projection follows the source loop and gives the
current result above. The stored surface comparison does not locate the
second-step internal force or objective difference. Later epochs, final
`sphere.reg` and downstream vertex metrics are unaccepted.
