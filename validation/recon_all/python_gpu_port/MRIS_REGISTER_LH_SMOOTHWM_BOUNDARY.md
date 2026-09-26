# LH independent smoothwm curvature and 45 exact normal updates

The frozen `fs_sub01` LH `smoothwm`, original sphere, exact Python sulc
`debug0056` seed and atlas have SHA-256 values in the paired JSON reports.
This is a bounded FreeSurfer 8.2 comparison on stored native checkpoints,
not an end-to-end reconstruction.

The production Python/PyTorch `smoothwm_mean_curvature` computes raw H from
the frozen surface without reading the native raw H. Its
[all-vertex report](mris_register_lh_smoothwm_production_raw_headcw.json)
matches **106,622/106,622 float32 values bitwise** and has the same raw-array
SHA-256 `319c755b667a04331ef0c545714aaf5897c9a04e0c08a882b69b48542f0ab39c`.
The measured 14.05 s is one CPU run including Numba startup, not a paired
native speed comparison.

Using that independent raw H, the [first-update report](mris_register_lh_default_epoch0057_independent_fixed_headcw.json)
selects `dt=2.721714973449707` and matches the native saved LH
`debug0057` surface at **106,622/106,622 ordered vertices**, maximum
coordinate error 0 mm. It starts from the previously verified exact Python
`debug0056` sulc seed. Setup, force, averaging, spring and line search took
22.51, 18.25, 6.91, 0.008 and 0.55 s respectively in that shared-host CPU run;
these exclude file I/O and are not a native speed ratio.

The [continuous probe](probe_mris_register_sno2_epoch.py) retains its own
coordinates between updates and reads native snapshots only for comparison.
The sigma-4, 2, 1 and 0.5 normal smoothwm passes match **all 45 saved
`debug0057`–`debug0101` surfaces**, 106,622/106,622 ordered vertices and all
faces at each checkpoint, maximum coordinate error 0 mm. Five bounded runs
cover the path without gaps:

| Report | Native epochs checked | Exact vertices at every epoch |
| --- | --- | ---: |
| [sigma 4 first segment](mris_register_lh_default_epoch0057_0068_continuous_headcw.json) | 0057–0068 | 106,622 / 106,622 |
| [sigma 4 second segment](mris_register_lh_default_epoch0069_0080_resume_headcw.json) | 0069–0080 | 106,622 / 106,622 |
| [sigma 2](mris_register_lh_default_epoch0081_0087_resume_headcw.json) | 0081–0087 | 106,622 / 106,622 |
| [sigma 1](mris_register_lh_default_epoch0088_0094_resume_headcw.json) | 0088–0094 | 106,622 / 106,622 |
| [sigma 0.5](mris_register_lh_default_epoch0095_0101_resume_headcw.json) | 0095–0101 | 106,622 / 106,622 |

Each resumed run verifies that its starting native checkpoint has the same
coordinate-array SHA-256 as the preceding independently computed Python
state and the same file SHA-256 as the preceding report. Native intermediate
coordinates are not injected during a run. The probe receives the averaging
schedule observed in the native log; independent stopping decisions and the
following negative-face repair remain unverified. The sum of measured CPU
update times across these segments is about 763 s, excluding each segment's
repeated setup and file I/O. This is not a matched native or GPU benchmark.

The previous [failed diagnostic](mris_register_lh_default_epoch0057_0058_continuous_headcw.json)
reprojected coordinates inside both distance and area gradient helpers at
the second integration step. It matched 18/106,622 vertices at `debug0058`
with 0.00157928466796875 mm maximum error. The FreeSurfer source calls
`mrisProjectSurface` at the end of each integration iteration and once at
the start of a new averaging stage. Passing `project=False` to both helpers
within an existing stage removes the duplicate projections and restores
exact second-step parity. Starting the 256-average stage with one projection
then matches `debug0059`; the same one-projection rule also matches every later normal pass through `debug0101`.

The first continuation reports evaluated reloaded native vertices as
float64 because that is the `nibabel` reader default. Their
`native_reference_sse` fields therefore do **not** measure the float32
registration objective and should not be compared with the native log.
The [corrected probe](probe_mris_register_sno2_epoch.py) casts reference
vertices to float32. Without rerunning the exact 45-step geometry, the
[hash-linked score audit](mris_register_lh_default_epoch0057_0101_selected_score_audit_headcw.json)
compares each selected float32 trial SSE with the
[native one-decimal status log](mris_register_lh_default_native_run_headcw.log): the largest absolute difference is 0.3441 across 45 updates. For
`debug0058`, the Python selected-trial SSE is 1,418,705.2775 versus the
native printed 1,418,705.2. These comparisons do not establish bitwise SSE
parity. The smoothwm raw-curvature fit currently uses CPU Numba for
source-order matrix operations.
