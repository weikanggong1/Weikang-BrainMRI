# Python/CUDA recon-all stage validation

This is an **isolated stage replay**, not an end-to-end Python reconstruction. The
reference is FreeSurfer 8.2.0 (`d932c45b7941662ea380a05efef580568b98d41a`) on
one completed `fs_sub01` T1 reconstruction. The replay uses that subject's frozen
surface files as inputs. No replacement stage is accepted solely because its
command exits successfully: its complete output must be compared with the native
stage, then both implementations are timed on the same host and inputs.
The translation priority is matched input and matched output. Slower short
stages are acceptable; long stages should be optimized only after their
numerical gates pass. CPU NumPy/Numba or SimpleITK is used where it reproduces
the source operation more directly. No float16/bfloat16 path is used.

The fixed workflow's **102/102 non-model data files** now have verified
official download sources and per-file size/SHA-256 checks. They install
outside the Python package, without a FreeSurfer binary or license file;
see [`ASSETS_VALIDATION.md`](ASSETS_VALIDATION.md). Two official archives
make the first transfer 796.17 MiB even though the used files total
328.73 MiB on disk.

## Input conversion, conform and N4

The first `mri_convert` T1 NIfTI import matched all 10,223,616 float32 voxels,
the MGH header, scan parameters and affine of the official `orig/001.mgz`.
Three paired fresh-process headcw CLI trials, without clearing the OS cache,
had native/Python medians of 1.925/0.594 s. FreeSurfer's command-history
footer was not reproduced. This container conversion runs on CPU. See
[`NIFTI_IMPORT.md`](NIFTI_IMPORT.md) and the
[`paired report`](nifti_import_headcw_report.json).

The fixed `mri_convert rawavg.mgz orig.mgz --conform` translation matched
FreeSurfer's 256³ output voxel by voxel on CPU and H100 CUDA 1. The 284-byte
MGH header and affine also matched exactly. Its subsequent
`mri_add_xform_to_header` replacement wrote the same Talairach XFORM tag;
native `mri_info` loaded the transform successfully. Native command-history
tags remain different, so the complete MGH file is not byte-identical.
Three paired headcw CPU CLI trials had native/PyTorch medians of
1.483/3.064 s. A single gpucw1 CUDA 1/native CLI pair took 5.73/4.06 s.
See [`CONFORM.md`](CONFORM.md) and the
[`CUDA report`](conform_cuda1_gpucw1_report.json).

The new `run_input_chain` API connects import, the single-run rawavg copy,
conform, and the XFORM path tag in one Python process. From the original
`fs_sub01` T1 on headcw CPU, its three files `orig/001.mgz`, `rawavg.mgz`,
and `orig.mgz` each matched the official 284-byte MGH header and every
voxel payload byte; their affines were exact. Footer command history or
XFORM path bytes differed for the isolated output directory.
One run took 0.429 s import, 0.002 s copy, and 1.730 s conform plus tag.
These are unpaired timings, not a full recon-all benchmark. The future
`talairach.xfm` still requires SynthMorph. See the
[`connected report`](input_chain_fs_sub01_report.json) and
[`replay script`](experimental/compare_input_chain.py).

The connected Python `orig.mgz` was then fed directly to the package's
PyTorch SynthStrip API with the externally configured official weight. Its
`synthstrip.mgz` matched the archived official result in all 16,777,216
voxels, the 284-byte MGH header, payload and affine. One headcw CPU inference
and output write took 6.449 s; the reference was the earlier gpucw1 GPU run,
so this is not a paired speed comparison. See the
[`connected SynthStrip report`](connected_synthstrip_fs_sub01_report.json)
and [`replay script`](experimental/compare_connected_synthstrip.py).

The connected 33-class SynthSeg path was then run on H100 CUDA 1 from the
Python-generated `orig.mgz`. Full float32 cuDNN inference matched the true
official segmentation at all 16,777,216 voxels, including dtype and MGH
header/payload. With TF32 enabled, 166 labels differed. After matching the
official NumPy posterior reduction order, the 33 soft-volume columns differ
by at most 0.04 mm³ after an isolated foreground-threshold correction and
source-style float32 CSV rendering; six exceed the existing 0.005 mm³ stats
tolerance. The CSV rendering check reused saved values without new inference.
The strict CSV gate remains open; the correction has been tested on this one
input only. The independent headcw CPU attempt had
previously failed or consumed excessive RAM; it was not repeated. See the
[connected GPU report](CONNECTED_SYNTHSEG_GPU_20260926.md) and
[CPU diagnostic](connected_synthseg_cpu_headcw_20260926.json).

The Python/SimpleITK N4 correction plus Python wrapper reproduced a **fresh**
official `orig.mgz → nu.mgz` run on headcw, including every voxel, header,
and complete decompressed MGH bytes. One full-stage native/Python pair took
158.10/125.56 s; three same-input post-N4 pairs had medians of
14.71/0.99 s and complete byte identity. This N4 implementation runs on CPU.
The historical gpucw1 `nu.mgz` differs from both fresh runs in 34 voxels;
the original intermediate N4 output was removed, so the cause is unresolved.
See [the N4 wrapper validation](../../../docs/recon_all/N4_WRAPPER_VALIDATION.md)
and [its paired report](n4_wrapper_headcw_report.json).

The following `AntsDenoiseImageFs` replacement uses the `antspyx==0.6.3`
Python API and compiled ANTs/ITK on CPU. It matched all 16,777,216 voxels,
the affine, and the MGH header on the frozen input. One same-host native/Python
pair took 27.58/28.25 s. The trailing MGH metadata differs by one byte.
See [`ANTS_DENOISE_STATUS.md`](ANTS_DENOISE_STATUS.md). The optional
`recon-all-python-stages` dependency group lists its tested ANTsPy version
and the separately tested SimpleITK N4 version; the combined environment and
end-to-end Python runner have not yet been validated.

The two standalone T1 intensity normalization translations now match their
respective native stages on frozen `fs_sub01` inputs. The first `mri_normalize
-g 1 -seed 1234 -mprage` pass matched all 16,777,216 final voxels on headcw
CPU and H100 CUDA 1; see [`NORMALIZE_FIRST_PASS.md`](experimental/NORMALIZE_FIRST_PASS.md).
The second `-seed 1234 -mprage -aseg -mask` pass independently generated its
45,325-voxel WM ridge, filtered controls, outlier map, and float32 initial
bias with zero mismatches. Its complete Python CPU `brain.mgz` matched a fresh
native run in all 16,777,216 voxels and the 284-byte MGH header and voxel
payload. The native run appended 996 more trailing metadata bytes, so whole-file
hashes differ. One sequential same-host CLI pair took 75.37 s Python and
115.55 s native; this is a single timing observation. See
[`NORMALIZE_SECOND_PASS.md`](NORMALIZE_SECOND_PASS.md) and the
[`paired report`](normalize_second_fs_sub01_report.json). The second CUDA path
and other subjects remain unvalidated, and neither stage is connected to the
full Python recon-all entry point.

The separate atlas-based `mri_ca_normalize` stage now matches a fresh native
run for all 16,777,216 `norm.mgz` voxels and all six 256³ `ctrl_pts.mgz` frames,
including the 284-byte MGH headers. The Python/Numba CPU call took 18.00 s;
the fresh native full stage took 42.68 s. MGH trailer metadata differs.
See [`CA_NORMALIZE.md`](CA_NORMALIZE.md) and the
[`file-level report`](ca_normalize_fs_sub01_report.json).

The isolated `mri_ca_register -invert-and-save` Python/Numba CPU stage now
reproduces the complete compressed 256³×3 inverse NIfTI byte for byte on the
same frozen warp input. A single sequential headcw pair took 84.68 s native
and 204.82 s Python amid changing system load; this stage has no speed gain.
It is not connected to the recon-all entry point. See
[`CA_REGISTER_INVERSE_KERNELS.md`](CA_REGISTER_INVERSE_KERNELS.md) and the
[`paired hashes`](ca_register_pair_benchmark/sha256.txt).

The isolated `mri_em_register` Python/Numba CPU stage now passes a same-input
fixed-subject gate. Eight translation grids and the final pre-EM matrix match
all 16 native float32 matrix elements exactly; all 31 captured EM objective
values also match. Its independently generated final LTA has maximum matrix
error `7.45e-9`, equal nonmatrix metadata and geometry, and **0/315,638**
atlas samples mapping to a different source voxel. A full Python stage run
took 230.20 s wall time versus a separate 236.52 s native reference run on
headcw; these are not controlled paired timings. The previous NumPy Python
stage took 448.82 s and failed parity. See
[`MRI_EM_REGISTER_VALIDATION.md`](MRI_EM_REGISTER_VALIDATION.md) and
[`MRI_EM_SEARCH_SOURCE_VALIDATION.md`](MRI_EM_SEARCH_SOURCE_VALIDATION.md).
This CPU stage has not been connected to the recon-all entry point; other
subjects and a GPU implementation remain unvalidated.

## Aligned-volume masks

The Python/Torch `mri_mask` replacement replays four actual recon-all calls in
order, feeding each preceding output to the next call. On the same 256³ subject
input, all four CUDA 1 outputs had zero voxel mismatches, identical affines,
and complete **decompressed MGH byte identity** with the FreeSurfer 8.2
outputs. A separate three-repeat CPU replay passed the same byte check.

| Call | Native (s) | CUDA 1 (s) | Native median (s) | Python CPU median (s) |
| --- | ---: | ---: | ---: | ---: |
| `brainmask` | 0.896 | 0.904 | 0.894 | 0.834 |
| `finalsurfs_threshold` | 0.985 | 0.671 | 1.168 | 0.731 |
| `finalsurfs_mcadura` | 2.585 | 0.996 | 2.376 | 0.983 |
| `finalsurfs_vsinus` | 2.356 | 0.904 | 2.315 | 0.965 |

The CUDA columns are **one** same-host paired trial on gpucw1 H100 GPU 1;
the CPU columns are three paired trials on gpucw1. Native times include binary
startup, while candidate times use a resident Python process with import and
CUDA initialization excluded. All times include file reads and writes. See
[`benchmark_mask.py`](benchmark_mask.py),
[`mask_cuda1_shared_report.json`](mask_cuda1_shared_report.json), and
[`mask_cpu_shared_report.json`](mask_cpu_shared_report.json). These calls are
not yet connected to the Python recon-all entry point. A later attempt to
collect three CUDA trials stopped at initial context allocation with a shared
GPU out-of-memory error; it supplied no additional timing data.

Frozen external input SHA-256 values were `T1.mgz`
`38649d61806ed5faf4cdc4d7888f6957bec16da740f1c4f3bb2558aaf112a5be`,
`synthstrip.mgz`
`3152962e99b326e960bce57c728c3f249e51821ffd7895b711746e3e98005ad9`,
`brain.mgz`
`1b7360d069b76c8a5a63f93f3c296dddeb4db725c4e2625e11ebfc156279f401`,
`mca-dura.mgz`
`08e4137f81a91faaa4e33e1133848f94127658ef3a52885754d00e9441ffb3f0`,
and `vsinus.mgz`
`cee69ac211f27f9e577ca8d40f3a30bc33db1a220f692fdc986956d0e256307a`.

The fixed `mri_binarize --match 3 42 --inv` cortical mask call also produced
byte-identical decompressed MGH on both CPU and CUDA; its command scope and
validation are recorded in [`BINARIZE.md`](BINARIZE.md). The isolated
SimpleITK N4 correction matched every output voxel but had different MGH
footer metadata; see
[`N4_SITK_VALIDATION.md`](../../../docs/recon_all/N4_SITK_VALIDATION.md).

The standalone entorhinal and ACJ `mri_edit_wm_with_aseg` calls were also
replayed in Python on real aligned volumes. The 598-voxel ACJ mask and four
16,777,216-voxel edited outputs all matched native voxel by voxel, including
two edits that changed 1,453 and 598 voxels. See
[`WM_EDITS.md`](WM_EDITS.md). The larger combined WM-editing call also
matched a fresh native replay at every voxel on the frozen subject. Its
Python/Numba CPU implementation has an input-hash gate, so general-subject
parity remains open. See [`WM_ASEGEDIT_FIXED.md`](WM_ASEGEDIT_FIXED.md).

The fixed `mri_segment -wsizemm 13 -mprage` call independently produced the
same 256³ `wm.seg.mgz` voxel values and MGH header from the frozen
`antsdn.brain.mgz` input. A scan-order Numba port reduced the isolated
diagonal morphology from 33.48 s to 0.46 s while keeping all voxels exact.
The updated complete Python file-output run took 90.40 s on headcw; one
native call without diagnostic writes took 45.93 s. Broader subject parity
and further speed work remain open. See
[`MRI_SEGMENT_VALIDATION.md`](MRI_SEGMENT_VALIDATION.md).

The later `mri_relabel_hypointensities` call matched the archived and fresh
native outputs across all 16,777,216 voxels and in decompressed MGH bytes.
A same-input headcw CLI pair took 11.14 s native and 5.15 s Python CPU. See
[`RELABEL_HYPOINTENSITIES.md`](RELABEL_HYPOINTENSITIES.md).

The following `mri_surf2volseg --fix-presurf-with-ribbon` branch generated
`aseg.mgz` with 0/16,777,216 voxel differences and an identical MGH header
and voxel payload. Its trailing color table/history differs; the two
annotation-based `mri_surf2volseg` branches are checked separately. See
[`SURF2VOLSEG_FIX.md`](SURF2VOLSEG_FIX.md).

The three `--label-cortex` branches generated `aparc+aseg.mgz`,
`aparc.a2009s+aseg.mgz`, and `aparc.DKTatlas+aseg.mgz` with all 16,777,216
voxels and the archived full decompressed MGH bytes matching official output
for each atlas. The aparc branch's 15,033 normal-direction checks also
matched the fresh native log. Same-input headcw CLI pairs took
27.20/6.14 s, 24.43/5.96 s, and 22.34/5.25 s native/Python CPU. See
[`SURF2VOLSEG_CORTEX.md`](SURF2VOLSEG_CORTEX.md).

The `--label-wm` branch then generated `wmparc.mgz` with zero voxel
differences and complete decompressed MGH byte identity on the frozen
inputs. One same-input headcw pair took 9.69 s native and 5.70 s Python CPU.
See [`SURF2VOLSEG_WM.md`](experimental/SURF2VOLSEG_WM.md).

The two `mri_segstats --annot ... --snr` white/gray contrast tables also matched
all 70 data lines byte for byte, including when fed Python-generated contrast maps.
Same-host left/right CLI calls took 0.37/0.35 s native and 0.20/0.23 s Python
CPU. See [`SURFACE_SNR_STATS.md`](SURFACE_SNR_STATS.md).
The preceding PyTorch surface sampling now matches all 424,326 frozen
white/gray intermediate values and both final percentage maps vertex for
vertex; see
[`VOL2SURF_CONTRAST_STATUS.md`](VOL2SURF_CONTRAST_STATUS.md).

A chained replay fed the Python hypointensity and ribbon outputs through
the three Python `mri_surf2volseg` branches in order. `aseg.mgz`,
`aparc+aseg.mgz`, and `wmparc.mgz` each retained 0/16,777,216 voxel
differences against official results. The surfaces, cortex labels, and aparc
annotations remained frozen official inputs; this is therefore a volume-chain
check, not a complete native-free subject run. See the
[`chain report`](cortical_volume_chain_headcw_report.json).

The preceding `mris_volmask` stage generated the combined and two unilateral
ribbon volumes with **zero voxel differences** against freshly rerun native
outputs. With the verified external color table, all three complete
decompressed MGH files also match byte for byte. Same-host headcw calls took
72.36 s native and 3.46 s Python/Numba CPU, including the color-table
serialization. See [`VOLMASK.md`](VOLMASK.md).

The isolated `mri_cc` port generated `aseg.auto.mgz` with zero differences
across all 16,777,216 voxels and an exact MGH header/voxel payload on this
subject. Its LTA matrix differs by at most `7.63e-6`; same-host single calls
took 89.67 s native and 17.41 s Python CPU. See
[`MRI_CC_PYTHON.md`](MRI_CC_PYTHON.md). It is not yet connected to the main
runner.

The raw `mri_tessellate` quad mesh matched ordered vertex and face core bytes
for both hemispheres on CPU and H100 GPU 1. Three paired gpucw1 trials had
native/CUDA medians of 1.398/0.255 s (left) and 1.437/0.162 s (right), with
imports and CUDA context creation excluded. The upstream `mri_pretess` input
was freshly generated by the native program. See [`TESSELLATE.md`](TESSELLATE.md)
and [`the GPU report`](tessellate_cuda1_report.json).

The adjacent Python `mri_pretess → mri_tessellate →
mris_extract_main_component → mris_smooth` replay also matched the archived
bilateral `smoothwm.nofix` ordered vertices and faces exactly from frozen
`filled.mgz`/`norm.mgz` inputs. The three upstream stages were tested on
headcw CPU with paired native runs; `mri_pretess` was slower in Python, while
component extraction was faster. See [`PRETESS.md`](PRETESS.md),
[`EXTRACT_MAIN_COMPONENT.md`](EXTRACT_MAIN_COMPONENT.md), and
[`SMOOTH_SURFACE.md`](SMOOTH_SURFACE.md). The optional smoothing CUDA path
still needs its own real-subject comparison. Direct bilateral six-stage
replays ([LH](six_stage_surface_chain_lh_report.json),
[RH](six_stage_surface_chain_rh_report.json)) from those same frozen
volume inputs through `qsphere.nofix` matched the official `orig`,
`smoothwm`, `inflated`, and `qsphere` ordered vertices, faces, and
volume geometry exactly. Each stage consumed the preceding Python file.
Six Python CPU CLI calls took 186.99 s LH and 202.27 s RH in separate
unpaired runs. This is not full recon-all or a paired speed ratio.
An optimized Numba update of the last two stages retained bilateral exact
ordered geometry and volume tags: single headcw calls took 30.31/24.25 s
LH/RH for inflation and 56.75/62.27 s for quick sphere. These were separate
shared-host observations, not a controlled comparison with native. See the
[optimized report](inflate_qsphere_numba_optimized_report.json).
The new `run_initial_surface_chain` Python API and module CLI also join these
six stages in one process for both hemispheres without invoking a FreeSurfer
binary. On the same frozen volumes, both pretess voxel arrays/affines and all
eight output meshes matched the saved official ordered vertices, faces, and
volume geometry exactly. The single headcw CPU call took 84.31 s LH and
85.37 s RH for its six timed stages; these are shared-host observations, not
paired native benchmarks. See the [API comparison report](initial_surface_chain_api_fs_sub01_report.json)
and [reproducible comparison script](experimental/compare_initial_surface_chain_api.py).
The output still stops at `qsphere.nofix` and is not a native-free full recon-all.

The isolated CPU Python `mris_remesh --remesh --iters 3` stage matches both
hemispheres' final ordered coordinates, faces, and volume geometry against
fresh native runs on the same frozen `orig.premesh` inputs. All three
iterations' split, collapse, and smoothing checkpoints are exact. Separate
native/Python calls took 14.46/100.87 s LH and 19.53/97.16 s RH; this stage
is slower in Python. See [`REMESH_VALIDATION.md`](REMESH_VALIDATION.md).
The following [`mris_remove_intersection`](MRIS_REMOVE_INTERSECTION_STATUS.md)
Python stage detects the fixed subject's zero-intersection branch and
preserves both ordered meshes exactly; five focused tests including a
synthetic intersection control passed. Its actual repair branch remains open
for subjects with intersecting faces.

The corrected CPU NumPy [`mris_inflate`](INFLATE_STATUS.md) translation now
reproduces both final `inflated.nofix` files from the original bilateral
`smoothwm.nofix` inputs at **every ordered vertex and face**, with volume
geometry metadata exact. A native RH diagnostic also matched all 60 raw
Python inflation updates vertex for vertex. One same-input LH native/Python
CLI pair took 6.986/41.710 s; Python CPU is slower. The concurrent RH Python
call took 46.44 s and has no paired native timing.

The independent NumPy CPU `mris_sphere -q` translation takes those
**Python-generated** `inflated` surfaces through 300 sphere-inflation
updates, inherited momentum, and nonlinear-area optimization. The connected
`smoothwm → inflated → qsphere` CLI outputs match both official quick spheres
exactly at all 102,764/101,454 ordered vertices, 205,560/202,936 faces, and
volume geometry fields. The [bilateral file report](inflate_to_qsphere_chain_exact_report.json)
records both boundaries. A clean same-input LH native/Python CLI pair took 41.866/135.866 s,
so Python CPU is slower; the RH stage has no clean paired speed trial.
Earlier diagnostic replays with extra snapshot I/O cannot be compared directly. This stage has no CUDA path, is not in a
native-free recon-all runner, and has only one-subject evidence. See
[`SPHERE_QUICK_STATUS.md`](SPHERE_QUICK_STATUS.md).

The later conventional [`mris_sphere`](SPHERE_STANDARD_STATUS.md) is a
separate, longer stage. Its bilateral initial scale and projection match
native checkpoints exactly. The full post-averaging distance tables match
native four-decimal diagnostics at all 8,268,920 LH and 8,183,354 RH ordered
entries. Three sampled vertex rows also match every neighbor ID and native
six-decimal distance; the full native dump does not include all IDs.
The independent [Python stage API](../../../src/fnit/recon_all/sphere_standard_run.py)
now runs from the original `inflated` and `smoothwm` inputs through all 243 LH
and 134 RH source-scheduled updates to final `sphere` files, without native
checkpoints. Both final outputs match the official ordered vertex bytes, face
bytes, and volume geometry exactly on the frozen subject. The standalone
headcw CPU runs took 372.82 s LH and 220.17 s RH including I/O; their final
fold cleanup also passed separately on H100 CUDA. Gradients, metric sampling,
and line search still use CPU/Numba, and upstream topology repair remains open.
See the [LH API audit](standard_sphere_lh_api_headcw_file_audit.json),
[RH API audit](standard_sphere_rh_api_headcw_file_audit.json), and
[stage status](SPHERE_STANDARD_STATUS.md).

In [`mris_fix_topology`](experimental/TOPOLOGY_FITNESS_SEARCH.md),
the first candidate's structural patch, smoothing, and MRI coordinate
matching have separate bilateral checks. Given native `select0s` snapshots
and independently generated white/gray intensity histograms, the
Python/Numba 40-step MRI matcher generated `select0sm` coordinates bitwise
equal at all 102,764 LH and 101,454 RH vertices. All four 256-bin
histograms match the native text plots at their ten-decimal precision, and
the two truncated means match float32 bits. Curvature histograms, later
candidates and defects, and final `orig.premesh` remain open.

The separate [`mris_autodet_gwstats`](AUTODET_GWSTATS_STATUS.md) Python/Numba
CPU implementation now generates both gray/white threshold files from the
original MRI, WM, and `orig.premesh` inputs **byte for byte** equal to native
(40/40 fields per hemisphere). It is not yet connected to the runner.

In `mris_place_surface`, isolated Python/PyTorch/Numba calculations regenerate
the first LH/RH white and pial border targets from original MRI, meshes,
geometry, labels, and frozen autodet thresholds. Every first-pass target and
intermediate input matches the pinned-source checkpoints exactly. The first
LH pial collision-constrained optimizer step also matches a copied-source
snapshot at all 106,622 vertices; the installed binary's RAM-only diagnostic
agrees at 105,195 vertices with maximum difference 7.63e-6 mm. The second
LH pial gradient now matches all six native force checkpoints on all
106,622 vertices after carrying the first collision step's cropped state;
the first three LH pial steps match the copied-source mesh at all 106,622
vertices and 213,240 faces after each step. Their six force components and
complete SSE decisions also match the pinned source. The optimized Numba
collision and cached-neighbor path retains this three-step exact result
([report](place_surface_three_steps_fast_report.json)); its separate CPU
run took 73.63 s versus 265.54 s for the earlier unoptimized run under
changing shared-host load. This is not a controlled speed benchmark.
The same optimized path then matched **ten** LH pial optimizer checkpoints
for an isolated copied-source diagnostic using `--n_averages 2`: six gradient
arrays, ordered mesh coordinates and faces, cropped count, and area at every
step. The maximum complete SSE discrepancy was `3.73e-9`; see the
[ten-step report](place_surface_ten_steps_projection_report.json). The
original `recon-all` LH pial first pass uses `n_averages=16`, so this
diagnostic does not certify that pass. A same-input, same-parameter normal CLI
control found that the installed binary and unmodified pinned-source rebuild
make identical ordered faces but diverge at the final LH pial vertices:
median 0.01577 mm, P99 0.22936 mm and maximum 1.66927 mm. The installed
binary repeats the archived LH `pial.T1` ordered coordinates exactly.
Therefore source-build diagnostic agreement cannot certify installed-binary
parity; see the [paired final-file report](place_surface_official_final_cli_comparison.json).
Applying the independently validated CPU vertex-area kernel to these two
same-input LH pial meshes finds 79,656/106,622 vertex-area outliers under the
existing `0.001 + 0.001 × reference` mm² rule (maximum 0.62124 mm²).
This is an intermediate installed-versus-source comparison, not the final
`area.pial` map or a Python optimizer result; see the
[area impact report](pial_source_vs_installed_vertex_area_20260926.json) and
[comparison script](experimental/compare_pial_intermediate_area.py).
The later independent LH pial replay now computes its own dt/reject schedule
through all 41 steps from frozen official white/MRI inputs. All 41 decisions
and 14 installed RAM checkpoints match; final pial ordered vertices and faces
are exact, and every area, thickness, vertex-volume and pial-curvature map has
zero outliers. Three of 12 rejected-trial SSE log values differ by 0.1 at
printed precision without changing decisions. RH final geometry matches only
under a native dt/reject diagnostic schedule. Independent white placement,
RH pial decisions and connected orchestration remain open; see
[`PLACE_SURFACE_WHITE_PIAL.md`](PLACE_SURFACE_WHITE_PIAL.md) and the
[LH full summary](place_surface_installed_lh_independent_decision_full_summary.json).

For spherical registration, the independent PyTorch sigma-4 rigid search
matches every ordered vertex and face on both hemispheres. The default sulc
nonlinear epochs match every saved ordered vertex checkpoint through LH
`debug0056` and RH `debug0055`. Both independently computed raw smoothwm
curvature arrays match native bitwise: 106,622/106,622 LH and 105,541/105,541
RH values. The LH normal smoothwm path matches all 45 consecutive saved
surfaces, `debug0057`–`debug0101`, at every ordered vertex and face. The
RH `debug0058` line-search discrepancy came from the float32 area ratio
in the spring SSE. A later `debug0083` discrepancy came from using double
`0.05` for a native float32 correlation weight. With both precision fixes,
all 42 consecutive RH smoothwm checkpoints `debug0056`–`debug0097` match
105,541/105,541 ordered vertices and all faces at every update. The native
RH final `sphere.reg` has the same vertices, faces and volume geometry as
`debug0097`; its negative-face repair loop was a no-op. On LH, the six
fold-cleanup surfaces `debug0102`–`debug0107` also match every ordered vertex
and face. A PyTorch repair from frozen native `debug0107` matched all 102
negative-triangle counts and the official final sphere at all 106,622
vertices on both CPU and H100 CUDA. The prior bounded geometry replays use
the native averaging schedule. A source-rule audit predicts all 91 saved
bilateral smoothwm decisions, and a short LH runtime check reproduces the
first four meshes with Python-selected averages. A continuous bilateral
source-scheduled invocation and connected full reconstruction remain open; see
[`MRIS_REGISTER_STATUS.md`](MRIS_REGISTER_STATUS.md), the
[RH SVD correction](MRIS_REGISTER_RH_RAW_H_SVD_MATCH.md), and the
[LH continuation boundary](MRIS_REGISTER_LH_SMOOTHWM_BOUNDARY.md).

The complete isolated `mris_ca_label` Python CPU stage now reads all six fixed
GCS atlases and produces LH/RH DK, Destrieux and DKT annotation files byte
for byte equal to native on frozen subject meshes, sphere registrations,
aseg and cortex labels. Six same-input native calls independently reproduced
the official files. Python took 21.49–33.71 s per call versus 5.17–8.48 s
native, so this stage is slower. It is not connected to the main runner;
upstream `sphere.reg` generation remains open. See
[`MRIS_CA_LABEL_STATUS.md`](MRIS_CA_LABEL_STATUS.md) and the
[`six-atlas report`](gcsa_label_six_atlas_headcw.json).

## Surface area maps

The ordered white-to-sphere `mris_jacobian` replacement matched a fresh
FreeSurfer run for all 212,163 bilateral vertices within `1e-5` on CPU and
H100 CUDA 1; the maximum H100 error was `2.86e-6`. This short step is slower
as a separate Python CLI. See [`SURFACE_JACOBIAN.md`](SURFACE_JACOBIAN.md).

`surface_area_gpu.py` computes one third of each incident face's area per
vertex. Four real surface maps were checked against the saved native outputs
using the existing `area` and `area.pial` criterion
(`abs_error <= 0.001 + 0.001 * abs(reference)` mm²). All 424,326 compared
vertex values passed. The native replay matched the saved maps exactly.

| Map | Vertices | Max CUDA error (mm²) | Native median (s) | CUDA median (s) |
| --- | ---: | ---: | ---: | ---: |
| `lh.area` | 106,622 | 4.77e-7 | 0.754 | 0.033 |
| `lh.area.pial` | 106,622 | 9.54e-7 | 0.671 | 0.011 |
| `rh.area` | 105,541 | 4.77e-7 | 0.673 | 0.011 |
| `rh.area.pial` | 105,541 | 9.54e-7 | 0.693 | 0.011 |

The `area.mid` step reads the two maps for each hemisphere and matches the
native `mris_calc add` then `mris_calc div 2` output exactly (0 differing
vertex values in both hemispheres).

| Map | Native median (s) | CUDA median (s) |
| --- | ---: | ---: |
| `lh.area.mid` | 0.060 | 0.017 |
| `rh.area.mid` | 0.049 | 0.007 |

These are three alternating paired trials per map on `gpucw1` GPU 1, an H100.
Each timed call reads the files and writes its output. The native side launches
the FreeSurfer binary; the CUDA side calls a Python function in one already
loaded process. Python/Torch imports and initial CUDA context creation are
excluded. A separate cold CUDA CLI area-map call took about 5.5 s, so these
figures do **not** establish single-call CLI or full recon-all speedup. The
shared GPU is subject to other users' load and occasional CUDA context OOM.

Reproduction scripts and full individual times are in
[`benchmark_area.py`](benchmark_area.py),
[`area_cuda1_report.json`](area_cuda1_report.json),
[`benchmark_mid_area.py`](benchmark_mid_area.py), and
[`mid_area_cuda1_report.json`](mid_area_cuda1_report.json). The native programs
are `mris_place_surface --area-map` and `mris_calc` from the pinned FreeSurfer
bundle. The reference input SHA-256 values are recorded below to prevent an
unpaired comparison:

| Input | SHA-256 |
| --- | --- |
| `lh.white` | `9c88a786c4a571c07d830b83fe647461b09fdacb63ff143670771340ef50b51c` |
| `lh.pial` | `f7c61a103d48497d38fb46ff0495984624eba726f384473f5634a8d7102d66d5` |
| `rh.white` | `5c7fda4367dd3624de0ef265303dab49579aa603aee75f1cce2f9c30bec965b9` |
| `rh.pial` | `0e0ac9a32a6e3d8087d63be2f3a68a237bef6331e1e867d33531611dd11e1eac` |
| `lh.area` | `f5a8545e6656938f7019308453b31e7a964abc9ae0730fbe40172aa8e2365a65` |
| `lh.area.pial` | `b3a1b4a64138e0776263f4b0943dbf363686c7295a27afd3119dd1ea2bfac851` |
| `rh.area` | `5d91003ad60b11aa1f5b0c47825edf4c70675e0f54e1eb70faf4caf159ebfa0b` |
| `rh.area.pial` | `44ae6cbc8497cdfb5c8df2f581ec7449eeec9634468c30d531f0a6f208210f34` |

## Cortical thickness and vertex volume

The isolated `mris_place_surface --thickness` replacement passed the existing
per-vertex thickness rule on both hemispheres of this subject: 106,622 left and
105,541 right vertices, zero outliers, and maximum absolute error below
`1e-6 mm`. A matched *single cold CLI call* on the right hemisphere took
27.62 s for the native program and 7.84 s for the updated Python/CUDA code on
the same gpucw1 H100. The left CUDA output also passed, but its updated cold
CLI rerun encountered an intermittent CUDA context OOM. The Python 20-hop
topology reachability test still runs on CPU. See
[`thickness_stage_report.json`](thickness_stage_report.json) for the exact
command, versions, and failure record. The stage is not connected to the main
reconstruction entry point.

The `mris_convert --volume` replacement calculates three tetrahedra per
white/pial face and assigns one third of each face volume to each vertex in
the cortex label. Both CUDA output maps matched the saved official maps with
zero outliers across 212,163 vertices; the largest error was
`1.91e-6 mm³`. See [`vertex_volume_cuda0_report.json`](vertex_volume_cuda0_report.json).
After two CUDA-context OOM attempts, a later three-repeat same-host
gpucw1 CUDA/native timing succeeded:

| Vertex volume map | Native median (s) | Python/Torch CUDA median (s) |
| --- | ---: | ---: |
| `lh.volume` | 1.439 | 0.033 |
| `rh.volume` | 1.438 | 0.031 |

These measurements include the surface and label reads and map write, but
exclude Python/Torch import and first CUDA context creation. Shared GPU load
and first-trial variation remain relevant. Individual times are in
[`vertex_volume_cuda0_benchmark.json`](vertex_volume_cuda0_benchmark.json).
A same-host gpucw1 *CPU* comparison also completed:

| Vertex volume map | Native median (s) | Python/Torch CPU median (s) |
| --- | ---: | ---: |
| `lh.volume` | 1.541 | 0.208 |
| `rh.volume` | 1.536 | 0.097 |

These are also three paired file-to-file trials per hemisphere after Python
import. Individual times and source replay are in
[`vertex_volume_cpu_report.json`](vertex_volume_cpu_report.json) and
[`benchmark_vertex_volume.py`](benchmark_vertex_volume.py).

## Atlas label projection and annotation

The fixed `mri_label2label --regmethod surface` branch generated all 72
bilateral BA/FG/V1/MT/exvivo labels byte for byte from downloaded fsaverage
surface/label assets and the frozen subject sphere registration. A subsequent
Python `mris_label2annot --maxstatwinner` replacement consumed those 72 Python
labels and wrote all six bilateral BA and VPnL annotation files byte for byte
against the official outputs. The ordered subject spheres remain official
inputs. See [`LABEL2LABEL_SURFACE_STATUS.md`](LABEL2LABEL_SURFACE_STATUS.md)
and [`LABEL2ANNOT.md`](LABEL2ANNOT.md).

## ROI columns

The Python/Torch ROI functions calculate `NumVert`, `SurfArea`,
`GrayVol`, `ThickAvg`, and `ThickStd`. The actual recon-all command uses
`mris_anatomical_stats -no-th3`, so ROI `GrayVol` is the average of the white
and pial face-area times mean face thickness; it is **not** the TH3 vertex
volume map above. On the frozen white and pial meshes, 346 table rows from
bilateral aparc, aparc.a2009s, and aparc.DKTatlas outputs matched the official
five columns after FreeSurfer's integer/three-decimal formatting. All rows
also passed the existing numerical tolerances. The headcw CPU comparison is
[`roi_cpu_volume_report.json`](roi_cpu_volume_report.json); one H100 CUDA replay
of 34 left aparc rows also had zero formatted differences in
[`roi_cuda0_lh_aparc_report.json`](roi_cuda0_lh_aparc_report.json). This is a
partial `mris_anatomical_stats` implementation.

The four remaining table curvature columns (`MeanCurv`, `GausCurv`, `FoldInd`,
`CurvInd`) matched their displayed official values in all 346 rows from the
same eight tables on headcw CPU and gpucw1 H100 GPU 1 with TF32 enabled.
The singular-vertex principal-curvature ordering is reproduced. See
[`ROI_CURVATURE.md`](ROI_CURVATURE.md), the
[`CPU report`](roi_curvature_cpu_report.json), and the
[`H100 report`](roi_curvature_gpu_tf32_report.json). The isolated stats writer
and global header measures are assessed below.

The combined Python CPU formatter then matched the **complete text** of all
346 data lines across the same eight tables and 56 bilateral BA/exvivo data
lines: names, order, whitespace and all ten numeric columns. See
[`ANATOMICAL_STATS_ROWS.md`](ANATOMICAL_STATS_ROWS.md)
and its [report](anatomical_stats_rows_cpu_report.json). Global header
measures remain outside this substage. One header measure, eTIV, has a
separate exact six-decimal Python port; see
[`ESTIMATED_TIV.md`](ESTIMATED_TIV.md). The complete set of 11 numeric/global
header lines also matched the native text in all eight tables when supplied
with the official `brainvol.stats` cache: 88/88 exact lines. See
[`ANATOMICAL_STATS_GLOBAL.md`](ANATOMICAL_STATS_GLOBAL.md). Recomputing that
cache has a separate Python CPU implementation: 12/16 measures were exact
and the largest of the remaining four errors was 0.000295 mm³ on this
subject. See [`BRAIN_VOLUME_STATS.md`](BRAIN_VOLUME_STATS.md). Combining those
Python stages into a stats-style file reproduced all 34 left aparc white
data lines exactly, with only a 0.000295 mm³ displayed `CortexVol` header
difference; see [`ANATOMICAL_STATS_FILE.md`](ANATOMICAL_STATS_FILE.md).
With the frozen official brain-volume cache, the four BA/exvivo tables also
matched 36/36 numeric Measure lines and 56/56 complete data lines. Their
uncensored white-surface area uses the full vertex-area sum and omits the
`MeanThickness` header, matching the native no-`-cortex` command.

The fixed `mri_segstats` branches now also match all 70 `wmparc.stats` and
45 `aseg.stats` table rows and their printed numeric/global measures on the
frozen inputs. The aseg branch computes the 9/8 uncorrected-surface holes
directly from mesh topology. Same-host isolated CLI calls took 100.34/9.35 s
native/Python for wmparc (warm Python repeat) and 22.66/10.34 s for aseg.
See [`SEGSTATS_WMPARC.md`](experimental/SEGSTATS_WMPARC.md) and
[`ASEG_STATS.md`](ASEG_STATS.md). These stages still consume frozen upstream
segmentations, meshes, and the brain-volume cache.

## White and pial curvature maps

The isolated `mris_place_surface --curv-map surface 2 10 output` replacement
matched all 424,326 vertex values across `lh/rh.curv` and
`lh/rh.curv.pial` under the existing curvature tolerance; the largest
absolute difference was `4.58e-4`. A single same-host H100 comparison of
all four file-to-file maps took 6.646 s with native executables and 1.733 s
with a resident Python/CUDA process. Each time includes surface read, data
transfer and morphometry write; Python import and first CUDA context creation
preceded timing. This is one paired observation, not an end-to-end result.
The two-hop neighborhood is constructed with SciPy on CPU; PyTorch/CUDA fits
and smooths the curvature. Per-map values, commands and raw records are in
[`curvature_stage_report.json`](curvature_stage_report.json).

Other native stages still need equivalent Python output. All results here
assume identical input meshes and vertex order. The final acceptance gate
remains a full single-T1 reconstruction: voxel labels, surface
topology/coordinates, every vertex-level thickness/area/volume/curvature map,
and atlas and global statistics.
