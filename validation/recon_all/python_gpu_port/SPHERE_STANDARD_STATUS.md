# Conventional `mris_sphere` parity boundary

The frozen `fs_sub01` commands are `mris_sphere -threads 4 -seed 1234
../surf/{lh,rh}.inflated ../surf/{lh,rh}.sphere` from FreeSurfer 8.2.0,
source commit `d932c45`. This nonquick stage reads the matching `smoothwm`
surface for its original metric. The Python projection is in
`sphere_standard_python.py`; the full sparse original metric sampler is in
`sphere_standard_metric.py`. The bounded first `MRISunfold` epoch is in
`sphere_standard_unfold.py` and `sphere_standard_line_search.py`. The first
one-ring nonlinear-area fold-repair update is in `sphere_standard_nonlinear.py`.

| Fixed projection checkpoint | LH | RH |
| --- | ---: | ---: |
| `before`: exact ordered vertices | 106,622 / 106,622 | 105,541 / 105,541 |
| `after`: exact ordered vertices | 106,622 / 106,622 | 105,541 / 105,541 |
| `after`: exact ordered faces | 213,240 / 213,240 | 211,078 / 211,078 |
| Python scale + projection, headcw | 0.0206 s | 0.0148 s |

The metric sampler constructs the source-order 1–7 ring candidates on the
frozen `smoothwm`, carries FreeSurfer's seeded VNL random stream through all
vertices, keeps the first three complete rings, samples eight entries per
ring from rings 3–7, and performs FreeSurfer's serial reciprocal-distance
averaging. The third ring intentionally occurs both in the complete prefix
and in the sampled suffix. The CSR rows preserve those duplicates.

An isolated native `FS_MEASURE_DISTANCES` diagnostic wrote the complete
**post-averaging** distance table at four decimal places immediately after
the first `MRISsampleDistances` call, before any `MRISunfold` optimization.
The Python and native tables have the same number of entries, and every
entry agrees at all four printed decimal places in source row order:

| Whole metric validation on headcw | LH | RH |
| --- | ---: | ---: |
| Vertices | 106,622 | 105,541 |
| Sampled entries | 8,268,920 | 8,183,354 |
| Native printed distances matching | 8,268,920 / 8,268,920 | 8,183,354 / 8,183,354 |
| Max absolute error against printed values | 0.000050 mm | 0.000050 mm |
| Native isolated sample + text dump | 14.33 s | 14.44 s |
| Python topology / sample incl. JIT / average incl. JIT | 4.49 / 10.98 / 0.66 s | 3.42 / 8.52 / 0.51 s |

The native diagnostic logs use four decimal places, so the complete table
check establishes agreement at that precision rather than bitwise identity.
Separate native `-v` logs include neighbor IDs and six-decimal distances:
LH vertex 0 and vertex 1 each match all 79 ordered IDs and 79 printed
distances; RH vertex 0 matches all 76 ordered IDs and 76 printed distances.
The whole-table dump omits IDs, so ordered IDs have only been checked for
those three queried vertices. The native time includes writing 8 million
text lines, whereas Python time excludes that write; these are not paired
sampler speed measurements. The sampler currently runs on CPU through
Numba, not GPU.

Structured reports are [LH](standard_sphere_full_metric_lh.json) and
[RH](standard_sphere_full_metric_rh.json). Earlier candidate-ring diagnostics
([LH](standard_sphere_metric_rings_lh.json),
[RH](standard_sphere_metric_rings_rh.json)), seeded neighbor comparisons
([LH](standard_sphere_vnl_seed_lh.json),
[RH](standard_sphere_vnl_seed_rh.json)), and the
[LH continued RNG](standard_sphere_vnl_continuity_lh.json) record the
pre-averaging checks. Native large text tables remain under
`$W/tessellate/sphere/standard_{lh,rh}/full_metric_native/distance.log`.

## Bounded first `MRISunfold` epoch

Separate isolated native probes used copied `fs_sub01` inputs, one thread,
seed 1234, `-a 0 -n 1 -p 1`, and one saved coordinate snapshot per update.
The initial negative-area repair took five updates on LH and zero on RH;
Python starts from the native checkpoint **after** that repair. The original
`smoothwm` copies have the same SHA256 as the read-only official inputs.
Each [LH](standard_sphere_first_epoch_lh.json) and
[RH](standard_sphere_first_epoch_rh.json) gradient report includes SHA256 for
the `smoothwm`, before/after snapshots, and native verbose log.

The current spherical distance used in the force has its own six-decimal
native log check ([LH](standard_sphere_current_arc_lh.json),
[RH](standard_sphere_current_arc_rh.json)): LH vertex 0 matches 79/79
ordered distances and RH vertex 0 matches 76/76. The Numba implementation must accumulate the norm and dot
product in double precision, then perform the C++ float division before the
angular approximation. This is a CPU float32/float64 arithmetic match, not a
GPU benchmark.

| First epoch, whole surface | LH | RH |
| --- | ---: | ---: |
| Vertices / negative faces | 106,622 / 2,305 | 105,541 / 491 |
| Python gradient L2 / native printed | 449.855171 / 449.855 | 10.984243 / 10.984 |
| Python maximum force / native printed | 147.603461 / 147.603 | 2.153895 / 2.154 |
| Python starting SSE / native printed | 2,627,487.011 / 2,627,487.05 | 2,660,370.600 / 2,660,370.64 |
| Python and native selected candidate | 2 / 2 | 4 / 4 |
| Independent selected `dt` / native printed | 0.014806829 / 0.015 | 17.633028 / 17.633 |
| Exact updated coordinate components | 319,863 / 319,866 | 316,581 / 316,623 |
| Updated vertices within 0.00001 mm | 106,622 / 106,622 | 105,541 / 105,541 |
| Maximum vertex error | 0.00000381 mm | 0.00000769 mm |
| Matrix / gradient / warm line search CPU time | 16.09 / 2.71 / 0.76 s | 16.23 / 2.71 / 0.66 s |

The independent search evaluates all native-form trial scores. Its float32
3×3 quadratic fit reproduces the native printed `(a,b,c)` exactly: LH
`(21495808,-306176,2627632)`, RH `(436,-7688,2661536)`. The chosen trial
SSE differs from native printed values by 0.051 mm² LH and 0.041 mm² RH.
The selected dt above comes only from Python scores and fit; the earlier
coordinate-fitted dt in the gradient reports is diagnostic. The independent
search and every updated vertex satisfy the 0.00001 mm first-step checkpoint
criterion. Reports are [LH](standard_sphere_line_search_lh.json) and
[RH](standard_sphere_line_search_rh.json). The focused float32 fit regression
passes on headcw (`1 passed`). Timing includes Numba JIT for the gradient,
while line-search timing is warm; no native matched-time claim follows.

## Paired repair and next-epoch checkpoints

The isolated native run reports a 0.005% negative-area threshold. The projected
LH input has 2,185 negative faces (0.0221117%) and enters five repair updates;
RH has 491 (0.00331086%) and skips repair. For each listed update, Python
starts from that update's **native saved input**, independently computes the
whole-surface gradient and line search, and compares every ordered output
vertex with the next native snapshot. These checks therefore establish
paired-checkpoint parity; they do not yet establish a continuously propagated
Python `mris_sphere` run.

| Hemisphere and update | Distance weight | Native / Python selected trial | Python `dt` | Exact xyz components | Vertices within 0.00001 mm | Max vertex error (mm) | Python gradient / search CPU time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LH repair 0 | 0.000001 | 4 / 4 | 0.967522740 | 319,856 / 319,866 | 106,622 / 106,622 | 0.000007629 | 2.347 / 1.058 |
| LH repair 1 | 0.00001 | 0 / 0 | 0.013019243 | 319,866 / 319,866 | 106,622 / 106,622 | 0 | 1.349 / 0.834 |
| LH repair 2 | 0.001 | 3 / 3 | 0 | 319,866 / 319,866 | 106,622 / 106,622 | 0 | 1.426 / 0.821 |
| LH repair 3 | 0.01 | 2 / 2 | 0.250300967 | 319,861 / 319,866 | 106,622 / 106,622 | 0.000008530 | 1.498 / 0.796 |
| LH repair 4 | 0.1 | 4 / 4 | 6.571062565 | 319,810 / 319,866 | 106,622 / 106,622 | 0.000008741 | 1.337 / 0.793 |
| LH first unfold 5 | 0.1 | 2 / 2 | 0.014806829 | 319,863 / 319,866 | 106,622 / 106,622 | 0.000003815 | 1.360 / 0.733 |
| LH next unfold 6 | 1 | 4 / 4 | 0.008593750 | 319,866 / 319,866 | 106,622 / 106,622 | 0 | 1.501 / 0.791 |
| RH first unfold 0 | 0.1 | 4 / 4 | 17.633028030 | 316,581 / 316,623 | 105,541 / 105,541 | 0.000007689 | 2.367 / 1.164 |
| RH next unfold 1 | 1 | 4 / 4 | 0.004401409 | 316,623 / 316,623 | 105,541 / 105,541 | 0 | 1.379 / 0.764 |

The original-distance matrix takes 16.070 s LH and 15.714 s RH on CPU,
including Numba compilation/cache loading. The per-update times are measured
inside the Python probe and exclude input loading and the matrix build. Native
logs give only whole isolated-command time (0.0111 h LH, 0.0085 h RH), so
these timings do not support a full-stage speed ratio.

A first LH repair-3 comparison missed the 0.00001 mm threshold at the three
vertices of face 167119 (largest error 0.000195576 mm). Its cross-product
normal has length about 8.37e-8, below float32 epsilon. Native `V3_NORMALIZE`
keeps that short cross product unscaled; the former Python implementation
unitized it and changed the local area force. Applying the native threshold
reduces the maximum error to 0.000008530 mm across all 106,622 vertices.
The [before-correction diagnostic](standard_sphere_repair_step3_diagnostic_lh.json)
and [corrected step-3 report](standard_sphere_repair_step3_epsilon_corrected_lh.json)
record this first divergence and its resolution. The corrected full
[LH](standard_sphere_repair_next_corrected_lh.json) and
[RH](standard_sphere_next_corrected_rh.json) reports include SHA256 of each
input, native verbose log, native before/after surface, implementation module,
and ordered step specification, plus selected scores and timings. The focused
near-degenerate-face regression passes (`1 passed`).

## Continuous state propagation from original input

The [continuous probe](experimental/probe_standard_sphere_continuous.py)
starts from the copied original `inflated` and `smoothwm` files, samples the
original metric once, and carries its own coordinates through each update.
Native snapshots are read only for comparison; no native state is injected.
A force-accumulation-order correction made all seven LH and both RH earlier
updates exact in every coordinate component. The older
[pre-correction LH](standard_sphere_continuous_lh.json) failure at repair 3
is historical; the current [LH](standard_sphere_continuous_nonlinear_lh.json)
and [RH](standard_sphere_continuous_nonlinear_rh.json) runs have no divergence
through the first nonlinear-area fold-repair update.

| Continuous chain from original input | LH | RH |
| --- | ---: | ---: |
| Initial projection exact components | 319,866 / 319,866 | 316,623 / 316,623 |
| Earlier repair / main-unfold updates | 7 / 7 exact | 2 / 2 exact |
| First nonlinear update index | 7 | 2 |
| First nonlinear input exact components | 319,866 / 319,866 | 316,623 / 316,623 |
| First nonlinear output exact components | 319,866 / 319,866 | 316,623 / 316,623 |
| First nonlinear maximum vertex error | 0 mm | 0 mm |
| First nonlinear Python selected trial / native | 4 / 4 | 2 / 2 |
| First nonlinear Python selected `dt` | 0.01120199170 | 0.00147362326 |

The native switch before this nonlinear stage calls
`MRISresetNeighborhoodSize(mris, 1)`. The Python implementation takes the
first-ring rows from the full original metric (639,720 LH and 633,234 RH
entries) while retaining native `avg_nbrs` from the prior full matrix
(77.553604 LH and 77.537201 RH). The first fold-repair coefficient has
`l_nlarea=1` and `l_dist=1e-6`. Its full-surface SSE agrees with native
`FREESURFER_logSSE=1` to all six printed decimals:

| First nonlinear starting SSE | LH native / Python | RH native / Python |
| --- | ---: | ---: |
| Nonlinear area | 4648.798864 / 4648.798864 | 4266.923948 / 4266.923948 |
| Weighted one-ring distance | 0.182286 / 0.182286 | 0.178350 / 0.178350 |
| Total | 4648.981150 / 4648.981150 | 4267.102298 / 4267.102298 |

The [paired LH](standard_sphere_nonlinear_one_ring_lh.json) and
[paired RH](standard_sphere_nonlinear_one_ring_rh.json) reports reproduce
the same first nonlinear outputs independently from each native input
checkpoint. Their Python coordinate input and output SHA256 also equal those
in the continuous reports. All four reports contain input/reference and
implementation SHA256, ordered candidate scores, selected time step, and
whole-surface coordinate errors. The isolated native SSE log SHA256 values
are `696c0643647b89e9f2c0b868563e6d634a58ca49e813b8f38c5ec2677b8bc927`
(LH) and `f2fc28cb37f0ac643f7ff088ea47e7303593e43ccf73fde5b7b0b67487a034bb`
(RH). The native logs remain in the isolated `unfold_first_{lh,rh}/nlarea_sse_probe`
directories; the official subject was never modified.

The latest continuous CPU run took 17.03 s LH / 16.13 s RH to build the
full metric, 4.81 / 4.70 s to select one-ring rows, then 1.74 / 1.96 s for
the first nonlinear gradient, 0.235 / 0.279 s for its line search, and
0.004 / 0.004 s to apply the update. These measurements include Numba
compilation or cache loading and server-load variation. The separate native
`logSSE` commands took 0.0103 h LH and 0.0073 h RH end-to-end with one CPU
thread; they perform multiple stages and diagnostics, so they are not matched
first-update timing comparators. The current Python port runs on CPU with
Numba; this is numerical checkpoint validation, not a GPU speed claim.

## Full isolated first-pass coefficient sweep and final file

The next bounded probe continues the **same Python coordinates** from each
original `inflated` input through every update saved by an isolated native
`mris_sphere -threads 1 -seed 1234 -a 0 -n 1 -p 1` run. It never injects a
native coordinate snapshot. After the initial 7 LH / 2 RH updates, both
hemispheres execute five nonlinear-area updates with `l_nlarea=1` and ordered
`l_dist` values `1e-6, 1e-5, 1e-3, 1e-2, 0.1`. The native log omits `1e-4`
and has no subsequent `l_dist=1` update in this isolated first pass. Python
uses the same one-ring metric throughout these five updates.

| Isolated first pass from original input | LH | RH |
| --- | ---: | ---: |
| Saved updates checked | 12 / 12 | 7 / 7 |
| Nonlinear candidate selections matching native | 5 / 5 | 5 / 5 |
| Saved coordinate components exactly matching at every update | 319,866 / 319,866 | 316,623 / 316,623 |
| Final isolated output coordinate components exactly matching | 319,866 / 319,866 | 316,623 / 316,623 |
| Final maximum ordered-vertex error | 0 mm | 0 mm |
| Final ordered faces matching | 213,240 / 213,240 | 211,078 / 211,078 |
| Raw vertex / face / volume-info payloads matching byte-for-byte | yes / yes / yes | yes / yes / yes |
| Full output file SHA matching | no | no |
| Python metric build / one-ring setup / all gradients / all searches | 14.12 / 4.32 / 15.58 / 5.76 s | 16.39 / 5.12 / 9.65 / 2.61 s |
| Sum of timed Python substeps including projections, excluding file reads and write | 41.92 s | 34.62 s |

The isolated native output applies one last radial projection after the
final saved update. Applying that projection in Python produces identical
float32 coordinates in source order. The
[Python surface writer](../../../src/fnit/recon_all/sphere_standard_python.py)
keeps the original `inflated` volume-geometry text block byte-for-byte;
Nibabel's default writer rounds that text and caused up to
`2.15e-9` difference in `cras` on the LH trial. The generated and native
files have identical raw vertex arrays, face arrays, and volume-geometry
blocks. Their whole-file SHA values differ because the creation stamps and
FreeSurfer provenance tags differ; the Python file omits those tags. The
[raw LH](standard_sphere_file_audit_lh.json) and
[raw RH](standard_sphere_file_audit_rh.json) reports give payload and file
SHA256 values. The [continuous LH](standard_sphere_continuous_first_pass_lh.json)
and [continuous RH](standard_sphere_continuous_first_pass_rh.json) reports
give frozen input and implementation hashes, all selected candidate scores,
per-step timings, and ordered-vertex comparisons. The native candidate SSEs
were logged to two decimals; Python differs from those printed values by at
most 0.0051 LH / 0.0049 RH. Every selected candidate matches and every
Python time step rounds to the native three-decimal printout.

These timings are from CPU Numba on headcw and include cache/JIT loading.
The native log reports only the whole isolated command in coarse hundredths
of an hour and includes other work. A matched stage speed ratio is therefore
not established. The focused tests, including the writer's exact
volume-geometry check, pass (`6 passed`) in the remote Python environment.

**Open boundary:** this validates one deliberately shortened, isolated
`MRISunfold` pass and its written LH/RH surfaces. The standard recon-all
`mris_sphere` run uses further iterations/passes; its final conventional
`sphere` surfaces and downstream cortical vertex metrics have not been
matched. The Python path here runs on CPU with Numba, not GPU.

To repeat the isolated diagnostic, use the frozen `inflated` and `smoothwm`
files from one subject. The native diagnostic deliberately returns status 1
after writing `distance.log`; the capture script checks for that file and
never writes into the official subject:

```bash
export FS_LICENSE=/path/to/private/license.txt
bash -lc 'source validation/recon_all/python_gpu_port/experimental/capture_standard_sphere_native_metric.sh /path/to/lh.inflated /path/to/temporary/metric_native'
PYTHONPATH=src python validation/recon_all/python_gpu_port/experimental/benchmark_standard_sphere_metric.py \
  /path/to/lh.smoothwm \
  --native-log 0:/path/to/native/v0.log \
  --native-table /path/to/temporary/metric_native/distance.log
```

Use the corresponding right-hemisphere paths for RH.
