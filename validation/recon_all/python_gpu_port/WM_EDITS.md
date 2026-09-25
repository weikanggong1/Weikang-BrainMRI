# Entorhinal and amygdala junction WM edits

`wm_edits_python.py` implements the standalone `mri_edit_wm_with_aseg`
`-sa-fix-ento-wm` and `-sa-fix-acj` calls. The ACJ mask marks cortex voxels
within the source's 26-neighbor amygdala boundary; entorhinal labels
3006/3201 and 4006/4201 are replaced at the selected level. This is
NumPy/SciPy CPU code and does not invoke FreeSurfer binaries.

On the fixed `fs_sub01` volumes, the 598-voxel ACJ mask matched the native
`-label-acj` output at all 16,777,216 voxels. Four paired same-input
standalone edit replays also had **zero voxel mismatches** each. Two tests
made real changes to `wm.seg.mgz`: 1,453 entorhinal voxels and 598 ACJ
voxels. The `brain.finalsurfs.mgz` replay was idempotent because those values
had already been edited by the official run. See the
[report](wm_edits_cpu_report.json) and
[probe](experimental/wm_edits_probe.py).

The combined `mri_edit_wm_with_aseg` call after `mri_segment` now has a
fixed-input Python/Numba CPU replay with zero voxel mismatches against a fresh
native run. Its input-hash gate prevents use on unvalidated subjects. See
[`WM_ASEGEDIT_FIXED.md`](WM_ASEGEDIT_FIXED.md). The final end-to-end volume
chain has not yet been validated from the T1 input.
