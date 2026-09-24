# PyTorch SynthSeg and full recon-all batch integration

This record covers the fixed `fs820-single-t1-v8-all-v1` profile on `gpucw1`
on 2026-09-24. The candidate uses the package's PyTorch/CUDA SynthSeg and
other neural replacements together with bundled FreeSurfer 8.2 native programs.
The input `examples/data/sub-01_T1w.nii.gz` has SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`.
The reference is the completed installed FreeSurfer 8.2 recon-all output for
this input. The native bundle and personal license are not published in Git.

## Feature snapshot: completed single-subject certification

The frozen feature snapshot had Python package-tree SHA-256
`29d2509974eecc02afdfb1923d34fe225f8468df4321b2e176fc5a20d046f92f`.
A fresh 308-file bundle preflight passed. A clean `env -i` launch with file and
execution tracing finished sub-01 on `cuda:1` in **5,249.754 s** with exit code
zero, all mandatory outputs, and the expected configuration SHA-256. The
SynthSeg launcher recorded `FS_TORCH_NEURAL_TOOL mri_synthseg device=cuda:1`.

| Numerical gate against official recon-all | Result |
|---|---:|
| Surface, vertex, atlas, volume and statistics checks | 52/52 pass |
| Ribbon and wmparc voxel checks | 2/2 pass |
| Aggregate checks | 19/19 pass |
| Aseg, aparc+aseg, ribbon and wmparc voxel mismatches | 0 each |
| White and pial surface coordinate maximum error | 0 mm |
| Compared per-vertex thickness, area, volume and curvature maximum error | 0 |

SynthSeg soft volumes meet the prespecified field tolerances but are not
bitwise equal: eTIV differs by +464.8 mm³ and CSF by +285.7475 mm³. The
hard segmentation and the checked downstream surfaces, vertex maps and atlas
results match the official reference on this input. The bundle was promoted
to `standalone_verified=true` after the file and execution trace audit and all
numerical gates passed. Raw evidence is in
`work/reconall_synthseg_batch_20260924/evidence/` on shared project storage:

| Evidence | SHA-256 |
|---|---|
| 52-check comparison JSON | `6004400dd4a6b0ac376a0dfadd111e44f599b2288d43f7e9d0a1614217b34ae3` |
| 2-check auxiliary volume JSON | `dafbe4c36f3c43236b22c14401567c11b46a57e99243a0bf22870d6b9f741095` |
| 19-check aggregate JSON | `ba05a4fdc3bba9762794a502732082fbf90d4562e23933dd5a43c5ab4b80d272` |
| Frozen code manifest JSON | `5a0426148fd5d2afa662319dacb9241c242e9c776bc66e401c3ded776076b946` |

## Two-GPU full-workflow batch

The public Python `run_recon_all_batch` API ran two independent complete
recon-all workflows concurrently from 12:44:50 to 14:04:51 UTC, **4,801.138 s
(80.02 min)** of batch wall time. Sub-02 was job 0 on `cuda:0` and finished in
4,795.419 s; sub-01 was job 1 on `cuda:1` and finished in 4,711.816 s. Both
started at 12:44:55 UTC, returned exit code zero, had no missing mandatory
outputs, matched the fixed configuration hash, and recorded the corresponding
`FS_TORCH_NEURAL_TOOL mri_synthseg device=cuda:N` marker in their logs. The
reports were returned in input job order. The batch used the promoted feature
snapshot bundle with `standalone_verified=true`.

The batch sub-01 output independently passed **52/52** official recon-all
checks, **2/2** ribbon/wmparc voxel checks, and **19/19** aggregate gates.
The batch sub-02 output passed the completeness and configuration checks; a
full official reference is unavailable for sub-02. Batch evidence is in the
same shared work directory:

| Evidence | SHA-256 |
|---|---|
| Ordered two-subject batch report | `6ad85752757e98d1e96e9b6e5d323f228bbadb7f5fc90b70ee6eac3a5856f8a3` |
| Batch sub-01 52-check comparison | `b46346d2558c2d706923c60a6a31d1f501a9851972b1a5e1618a80b88e6ed4c5` |
| Batch sub-01 auxiliary volumes | `4c32216564ea759f5b29166f210ede462ad424567e51b22e12986e347a3cc8c8` |
| Batch sub-01 19-check aggregate | `6fdf55d6e485cfa3b351bb1819e7f382752b570ea50965316f81e52cd3da1e9e` |

The batch demonstrates overlapping full workflows and correct device
assignment, not a controlled speedup over two sequential runs. Both jobs also
competed for shared CPU and storage resources.

## Merged main package

The merge preserves the repository's FAST/VBM modules and the newer Python
parallel inference functions. Its whole-package source SHA-256 is
`3d7713bcb016724b7645d9b53a5bd4af89880d138cbc578e33e65b21a442dcb1`,
different from the feature snapshot. The `0.6.0` merged package passed
**81 tests, with 4 skips** across recon-all, SynthSeg and batch suites on
headcw. Its built wheel has SHA-256
`9f912a59d9709c444a3c84a902282d671ec31cf99b5c01fc5737ced4997b31a8`.
Independent full-run certification against this exact source hash is pending.

## Scope

The full numerical comparison uses only sub-01. Sub-02 has no completed
official full recon-all reference in this validation set. The existing
SynthSeg-only sub-02 audit reported a CSF soft-volume difference outside the
provisional tolerance, even though its hard segmentation matched. These
results do not establish numerical agreement for untested inputs. No speedup
against official recon-all is claimed: the reference run used different
parallel flags and node conditions. Surface processing and statistics still
run in bundled native CPU programs.
