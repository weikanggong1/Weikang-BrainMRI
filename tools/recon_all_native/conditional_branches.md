# FreeSurfer 8.2.0 single T1 branch audit

Source: `work/fs_source/scripts/recon-all`, commit `d932c45` (the reference
build-stamp commit). Scope: a new subject, one T1 input, `-all -parallel
-openmp 4 -itkthreads 1`, stock V8 configuration, and no additional expert
options or edits. The reference `recon-all.cmd` confirms which branches ran for
this T1. This is review evidence, **not** a scanner ignore list or a standalone
bundle pass.

The script's help prose starts at the literal `BEGINHELP` on line 8978 and
continues to EOF. Line 8975 exits before it; line 8973 prints that prose only
for `-help`. Do not parse commands or resources from the help body.

**A branch can be inactive yet its executable still required.** On the normal
path, lines 689–795 call `-all-info` on every member of `allinfocmds`, aborting
if any call fails. Line 777 also calls `mri_motion_correct.fsl -version`.
`RECONALL_MAKE_SUBJECT` skips this inventory at line 645, but it is absent from
the reference run. An unchanged upstream script therefore needs the inventory
binaries, including `mri_gcut`, `mri_remove_neck`, `mri_robust_template`, and
`mris_topo_fixer`, even when their processing branches are inactive. A modified
runner may remove nonessential inventory calls only with an explicit code
change and end-to-end validation.

| Inactive processing branch under this scope | Source evidence | Packaging implication |
| --- | --- | --- |
| Longitudinal timepoint `mri_add_new_tp` | `longitudinal=0` at 244; command guarded by `if($longitudinal && ! $LongSamseg)` at 1195–1206. | No processing call for a new cross-sectional subject. Other scripts may call it. |
| Defacing `mri_deface` | `DoDeface=0` at 328; guarded command at 1720–1734; `-all` block 7198–7249 does not enable it. | Face/brain defacing templates are not required by this branch. |
| Multiple-run `mri_robust_template` | The one-run path copies `001.mgz` at 1437–1453; robust template runs only when `RunList > 1` at 1458–1494. | Processing is skipped for one T1, but `mri_robust_template` remains in the unconditional `-all-info` inventory at 728. |
| Neck removal `mri_remove_neck` | `-all` explicitly sets `DoRemoveNeck=0` at 7206; guard and command at 2815–2832. | Processing is skipped; executable remains in the inventory at 727. |
| Graph-cut skull refinement `mri_gcut` | `DoGcut=0` at 125; command guarded at 2575–2591. | Processing is skipped; executable remains in the inventory at 715. |
| V1 prediction `predict_v1.sh` | `DoLabelV1=0` at 196; guarded at 5603–5631; `-all` does not enable it. | `V1_average` is not required by this branch. Do not infer whether other branches need the same resource. |
| Ex vivo EC label `mris_spherical_average` | `DoLabelExvivoEC=0` at 380; guarded at 5642–5689; `-all` does not enable it. | EC-average subject resources are not required by this branch. Check other scripts independently. |
| Local gyrification `mris_compute_lgi` | `DoLocalGyriIndex=0` at 378; guarded at 5696–5720; `-all` does not enable it. | No processing call in this route. |
| Qdec cache `mris_preproc` | `DoQdecCache=0` at 181; guarded at 5727–5848; `-all` does not enable it. | Do not exempt `fsaverage` globally: other default label steps may use it. |
| Extra subfields scripts | `DoSubfields=0` at 238; guarded at 5882–5894; `-all` does not enable it. | This branch does not call `segmentHA_T1.sh`, `segmentThalamicNuclei.sh`, or `segmentBS.sh`. |

Further candidates from the same source, conditioned on a **fresh**,
unmodified subject:

| Candidate | Guard and observed route | Limit of the conclusion |
| --- | --- | --- |
| `mri_compile_edits` | `DoShowEdits=0` at 80; command inside `if($DoShowEdits)` at 1100–1186. `-all` does not enable that flag. | `-show-edits` enables it. |
| `mri_seg_diff` | Subcortical label edits at 2904–2959 run only with `DoCALabel && !UseSynthSeg` and relevant existing files. On the reference SynthSeg route, the later ASeg merge at 3087–3108 copies `aseg.auto.mgz` when `aseg.manedit.mgz` is absent. | Check `UseSynthSeg` and absence of manual edits for each run; a rerun or altered configuration can execute it. |
| `mris_reposition_surface` | White and T1 pial commands require `repos.$hemi.white.json` or `repos.$hemi.pial.json` to exist at 4463–4483 and 4525–4546; the T2/FLAIR path has another guarded use at 4718–4735. | A fresh T1 subject without these files skips it; a user-created reposition file enables it. |
| `seg2recon` | Its only direct `recon-all` call is inside `if(0 && ! $CblumFromSynthSeg && ! $SynthSegForSurf)` at 1650–1669. | This proves only the direct call is unreachable in this commit; another script could invoke it. |
| `mris_apply_reg` | `UseHighMyelin=0` at 191; the command is inside `if($UseHighMyelin)` at 4377–4392; `-all` does not enable the flag. | High-myelin options or expert configuration change the route. |
| `rca-base-init`, `rca-long-tp-init` | Base creation is guarded by `DoCreateBaseSubj` at 1339–1355; longitudinal initialization is guarded by `longitudinal` at 1195–1206 and `LongSamseg` at 1227–1232. These modes default to false for a new cross-sectional subject. | Base/longitudinal processing needs these scripts and their own dependency closure. |

The remaining helper-script observations use the same pinned source:

| Helper dependency | Source and reference evidence | Limit of the conclusion |
| --- | --- | --- |
| `fs_temp_dir` | `defect2seg:64–66` and `fscalc:38–40` call it when no temporary directory is supplied. Both scripts run in the reference. | This is a required dependency; the packager and repair tool include it. |
| `fscalc` → `mri_volsynth` | `fscalc:57–84` uses `mri_convert` when its first operand is an existing volume and `mri_volsynth` in the constant-input branch. The default caller at `label-cortex:195` passes the previously generated `$nxmask`; reference log lines 4220 and 4476 show the two mask-volume calls. | A constant first operand needs `mri_volsynth` and `fname2ext`; the general-purpose `fscalc` command is not covered by this scoped observation. |
| `mri_motion_correct.fsl` → `mri_average` | `recon-all:777` requests only `-version`. `mri_motion_correct.fsl:45–51` responds using `mri_convert -all-info` and exits before processing. The single-T1 image path copies its one run at `recon-all:1437–1453`. | Actual motion-correction use is outside this scope and additionally needs the FSL-dependent processing branch. |

The GUI helper scripts `tkmeditfv` and `tkregisterfv` entered the candidate
through startup-resource expansion. `FreeSurferEnv.csh:398–402` only adds an
existing macOS `freeview.app` path to `PATH`; it does not launch a GUI. The
reference log's `tkregisterfv` lines 319/509 follow “To check affine
registration” prompts, and `tkmeditfv` line 3086 comes from a literal `echo`
at `defect2seg:155` (another occurrence is after that script's `BEGINHELP`).
No `tk*` command occurs in the reference `recon-all.cmd`. The separately
transferred installed `fs-synthmorph-reg` script confirms that its lines 393
and 475 also use `echo` for these optional QC instructions. These observations
alone do not prove full GUI reachability. In the explicit fixed profile below,
the two GUI launchers are replaced with unconditional exit 64; their originals
are preserved, hashed and no longer dispatched. An unexpected actual GUI call
therefore fails the reconstruction visibly. Without that profile, the original
`isargflag`/`mri_coreg` static findings remain unresolved.

Configuration evidence must use the **resolved** subject configuration. The
reference log's first line records V8-injected options, including `-synthstrip`,
`-synthseg`, and `-synthmorph`. `recon-all:438–455` combines the V8 expert file,
subject-directory expert options and supplied CLI before `rca-config` resolves
the YAML. A base YAML `value: False` does not alone establish a disabled branch.

Two counterexamples prevent name-based exemptions: `mri_convert` appears in
the inactive T2/FLAIR conversion branch (1279–1323) **and** in the required T1
input path; `mris_topo_fixer` is a fallback if the old topology fixer fails
(3764–3823), besides being in the version inventory. Keep runtime preflight
strict until the exact config, CLI, input multiplicity, and actual process
trace establish the package closure.

## Implemented fixed-profile audit

The resolved reference YAML was subsequently obtained and has SHA256
`892990ac5603bae47e22fbdac414c080178d5d53e47f4ede48906fbf56de03de`.
It sets `UseSynthSeg=True`, `DoSynthSR=False` and `UseStopMaskSCM=False`.
The latter guards `mri_stopmask` at `recon-all:3896–3923`; the former SynthSR
value guards `mri_synthsr` at `1513–1534`.

`single_t1_scope.py` implements 16 specific `(script, command)` rules for
`fs820-single-t1-v8-all-v1`, including these two now-resolved branches. It
requires exact hashes of the upstream sources, base YAML, V8 expert options,
configuration resolver and resolved YAML. The runner must enforce a fresh
single T1, empty output directory, clean environment, and exactly
`-all -parallel -openmp 4 -itkthreads 1`, with no other flags or manual edits.
Each manifest rule carries its source lines and applicable conditions.
Preflight rechecks the evidence and disables all exclusions on any mismatch.
These rules do not establish a general-purpose recon-all closure or mark a
candidate standalone; an isolated full run remains required.
