# Connected T1-to-brainmask Python replay

The [`run_input_brainmask_chain`](../../../src/fnit/recon_all/input_brainmask_chain.py)
API ran once in a new subject directory on headcw CPU from the public
`sub-01_T1w.nii.gz`. It calls the already validated import, conform,
PyTorch SynthStrip, PyTorch SynthMorph, SimpleITK N4, first-pass PyTorch/CPU
normalization, and PyTorch mask stages in sequence. Model weights and the
MNI305 template came from external directories. It invokes no FreeSurfer
executable. The [runner](experimental/run_connected_brainmask_headcw.py),
[call report](connected_input_brainmask_api_cpu_20260926.json), and
[comparison script](experimental/compare_connected_brainmask.py) record the
fixed paths, stages and numerical checks.

| Output | Previous Python output, differing voxels | Archived official, differing voxels |
| --- | ---: | ---: |
| `nu.mgz` | 0 / 16,777,216 | 34 / 16,777,216 |
| `T1.mgz` | 0 / 16,777,216 | 112 / 16,777,216 |
| `brainmask.mgz` | no previous connected output | 49 / 16,777,216 |

The matching previous Python `nu.mgz` and `T1.mgz` were each compared to a
fresh same-input native stage in earlier checks: both had zero voxel
differences. In this run, all compared MGH affines and 284-byte headers
matched exactly. The connected `nu.mgz` and `T1.mgz` also matched their
previous Python output's MGH header and voxel payload bytes.

The connected and archived SynthStrip masks have zero voxel differences.
Exactly 49 of the 112 archived `T1.mgz` differences fall inside that mask;
all 49 `brainmask.mgz` differences equal the corresponding masked `T1.mgz`
differences, leaving zero unexplained differing voxels. The archive's
earlier N4 run differs from a fresh native replay by 34 voxels; this
comparison does not establish the cause of those historical differences.
See the [structured comparison](connected_input_brainmask_comparison_20260926.json).

The single connected call took 175.86 s: N4 95.60 s, first normalization
57.84 s, SynthMorph affine 12.02 s, SynthStrip 6.06 s, conform/tag 2.49 s,
and initial mask 0.46 s, with the remaining time in import, wrapper and
process overhead. These are CPU measurements from one run, not a paired
native benchmark. An installed native `mri_mask` same-input reference could
not be run because its configured FreeSurfer license was absent on headcw;
the earlier independent frozen-input mask validation remains the direct
native stage comparison. This chain stops at the initial `brainmask.mgz`.
