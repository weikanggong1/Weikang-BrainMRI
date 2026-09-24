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

## Merged 0.6.0 package

The merge preserves the repository's FAST/VBM modules and the newer Python
parallel inference functions. Its whole-package source SHA-256 is
`3d7713bcb016724b7645d9b53a5bd4af89880d138cbc578e33e65b21a442dcb1`,
different from the feature snapshot. The `0.6.0` merged package passed
**81 tests, with 4 skips** across recon-all, SynthSeg and batch suites on
headcw. Its built wheel has SHA-256
`9f912a59d9709c444a3c84a902282d671ec31cf99b5c01fc5737ced4997b31a8`.
An independent clean-environment sub-01 run using this exact source hash took
**4,602.741 s** on `cuda:1`, exited zero, produced all mandatory outputs, and
matched the fixed configuration hash. The same official reference passed
**52/52** surface, vertex, atlas, volume and statistics checks, **2/2**
ribbon/wmparc voxel checks, and **19/19** aggregate gates. The 308-file bundle
preflight and trace audit passed; the bundle was promoted to
`standalone_verified=true` after the comparison gates. Evidence is in
`work/reconall_main_20260924/evidence/` on shared project storage:

| Evidence | SHA-256 |
|---|---|
| 0.6 source manifest | `7e587a9917c80745ddeb5f0492c2b0ed19ac174ecb102470b07df8a2803a9c9b` |
| 52-check comparison | `ef09115439f3c773ccd75308b5c8d93c5230f18aa4e80a098699fec5340fb578` |
| 2-check auxiliary volumes | `9dc16fc6d51780aee52f0db367fbd8bbf17277a15c50d1b31d0f16e90f667312` |
| 19-check aggregate | `4bfe6350e3aada06dc8c9a52166c27be78dca929ddb4b1ca96475667999bfebc` |
| Promoted bundle manifest | `c6fa70a5fe326a50f2d4aa182b84853a249b376ba16bf252aec7c56027bb2230` |

## 0.7.0 release equivalence

The newer `main` adds a FastVBM SynthMorph backend and increments the package
version. Its source-tree hash is
`d34fc3e7a7c7bb205c710614c4a09d5ea7cbdd6da408f06a1f6320e15db257c1`.
The 0.7.0 snapshot passed **132 tests, with 4 skips** in the recon-all,
SynthSeg, batch, FastVBM and public-API suites on headcw. Its built wheel has
SHA-256 `1f0a80d0d22d4de914be7764d61c79d63b05f178907d07c9cbde389c679c4603`.
No duplicate full reconstruction was performed for the unchanged pipeline code.
The [source-equivalence audit](v06_to_v07_runtime_equivalence.json) checked
24 pinned recon-all runtime/comparison modules, both frozen code manifests,
the 0.7.0 project version and the 0.6.0 certified bundle link. Of the pinned
modules, only the root initializer and weight registry changed: the former
increments the version and adds lazy FastVBM exports; the latter adds a
FastVBM-only checkpoint. Other changed Python modules are confined to the
generic CLI and FastVBM. The recon-all entry, neural launchers, segmentation
code, surface and statistics comparison code are byte-identical. The audit
script also rejects unexpected runtime or unrelated package changes.

The certified 0.6.0 bundle was copied, and
[`derive_equivalent_bundle.py`](../../tools/recon_all_native/derive_equivalent_bundle.py)
linked its original verification record and the source-equivalence audit to
the 0.7.0 frozen code snapshot. The original certified bundle manifest stayed
unchanged. On `gpucw1`, the 0.7.0 default `_check_bundle` entry accepted the
derived copy with CUDA available and all **308** inventoried files intact;
the check exited zero. This checks release compatibility and bundle integrity,
without rerunning T1 reconstruction or numerical comparison in 0.7.0.

| Release-equivalence evidence | SHA-256 |
|---|---|
| 0.7 source manifest | `3faa8b5c55606971baaf15cd024c959b3b9558a7919a6c29668678a09981cae1` |
| 0.6→0.7 source-equivalence JSON | `8bd69608112b600c25c49d5faf501f0ebb3570085749914d2d4ee2588816f31f` |
| 0.7 derived bundle manifest | `089b5f1a3cd61d3fa96d3888239d1916af3354942202a12e74f5d5f21c053c99` |

## Scope

The full numerical comparison uses only sub-01. Sub-02 has no completed
official full recon-all reference in this validation set. The existing
SynthSeg-only sub-02 audit reported a CSF soft-volume difference outside the
provisional tolerance, even though its hard segmentation matched. These
results do not establish numerical agreement for untested inputs. No speedup
against official recon-all is claimed: the reference run used different
parallel flags and node conditions. Surface processing and statistics still
run in bundled native CPU programs.
