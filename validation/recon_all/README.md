# Provisional FreeSurfer 8.2 numerical comparison

## SynthSeg numerical boundary on the development T1

With cuDNN FP32 and TF32 disabled, the PyTorch 33-class SynthSeg output for
`examples/data/sub-01_T1w.nii.gz` differed from the official CPU result at one
of 16,777,216 native voxels. At that voxel, the two leading postprocessed
probabilities differed by only `6.2585e-7`. Recomputing its full receptive
field in FP64 still selected the PyTorch label. The pinned recon-all GPU
wrapper therefore treats probabilities within `2^-20` of the maximum as a
numerical tie and selects the first channel, consistent with ordinary argmax
for exact ties. On this T1 it changed only that voxel, yielding identical
native SynthSeg labels and affine. The rule does not use voxel coordinates or
anatomical label IDs. This is an empirical FS8.2 compatibility rule, not a
claim of improved mathematical precision or cross-subject equivalence.

The earlier one-voxel mismatch propagated to 2,174 white-matter voxels and
changed `white.preaparc` vertex counts by +71 (left) and -79 (right). Full
surface and regional metrics still require the checks below.

These are **development thresholds, not frozen acceptance limits or FreeSurfer
standards**. No full independent GPU workflow is validated by this configuration.
Calibrate on development subjects and repeated official runs, then freeze before
held-out evaluation. Do not relax thresholds after inspecting held-out failures.

The canonical configuration is `tests/recon_all/tolerances_numeric.json`;
`validation/recon_all/tolerances.provisional.json` is an identical distribution
copy. Unlisted numeric fields retain exact comparison. Integer stats counts,
ordered mesh connectivity, annotation color tables and cortex masks remain exact.
No required files are excluded by the profile.

Official 8.2 `cortex.label` files may contain repeated vertex IDs. The comparator
accepts these and compares both boolean vertex membership and entry multiplicity
per ID exactly, independently of line order. It reports duplicate counts and
rejects negative or out-of-range IDs. A duplicate-count mismatch fails even when
the boolean masks match.

## Run

From the repository root, using the same Python environment as `standalone.py`:

```bash
PYTHONPATH=src python -m freesurfer_torch.recon_all.compare_subject \
  /absolute/reference/subject /absolute/candidate/subject \
  --tolerances tests/recon_all/tolerances_numeric.json \
  --min-label-dice 0.995 --min-annotation-dice 0.995 \
  --max-outlier-ids 1000 --output work/comparison.numeric.json
```

Exit 0 means the comparator's mandatory checks pass; exit 1 means one or more
checks failed, were missing, could not be read, or were blocked by unavailable
native vertex correspondence. `outlier_ids` are zero-based native vertex IDs,
or voxel indices for volumes. Large lists explicitly report truncation while
retaining the full outlier count. Additional aggregate criteria below must also
pass; the comparator exit code alone does not implement those criteria.

For the gpucw1 reference, run this on gpucw1 from
`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth`, substituting the existing
runtime interpreter and the candidate subject directory (no new SSH/tmux needed):

```bash
PYTHONPATH=src "$FS_TORCH_PYTHON" -m freesurfer_torch.recon_all.compare_subject \
  work/reconall_reference_gpucw1/fs_sub01 "$CANDIDATE_SUBJECT_DIR" \
  --tolerances tests/recon_all/tolerances_numeric.json \
  --min-label-dice 0.995 --min-annotation-dice 0.995 \
  --max-outlier-ids 1000 --output work/comparison.numeric.json
```

`FS_TORCH_PYTHON` must name the already installed runtime interpreter;
`CANDIDATE_SUBJECT_DIR` must name the actual completed candidate. Comparing the
reference with itself is a comparator smoke check, not workflow validation.

## Criteria and units

Every scalar vertex value must satisfy
`abs(candidate - reference) <= atol + rtol * abs(reference)`; the permitted
outlier count is **zero**, including vertices outside the cortex mask. Coordinates
use Euclidean displacement and an absolute mm limit, without registration.

| Object | Enforced by the command | Additional provisional criterion |
|---|---|---|
| Native meshes | Same vertex count and ordered triangle indices; connected closed genus-zero topology; every displacement <=0.25 mm | For white/pial, mean displacement <=0.01 mm and P99 <=0.05 mm; inspect intersection/normal/degenerate-geometry QC separately |
| Thickness | 0.005 mm + 0.001 × absolute reference, at every vertex | MAE <=0.005 mm |
| White/pial/mid vertex area | 0.001 mm² + 0.001 × absolute reference | Check zero-mask rules and regional/hemisphere totals |
| TH3 vertex volume | 0.005 mm³ + 0.001 × absolute reference | Check the actual TH3 mask and zero-mask rules |
| Curv, curv.pial, white.H | 0.001 mm⁻¹ + 0.001 × absolute reference | Preserve the source surface, fitting neighbourhood and smoothing |
| White.K | 0.001 mm⁻² + 0.001 × absolute reference | Preserve principal-curvature estimation and sign conventions |
| Sulc, inflated.H/K | 0.001 + 0.001 × absolute reference, in original file values | Confirm final writeout/normalization and units before freezing; no replacement with another curvature/depth definition |
| Aseg and aparc+aseg | Exact grid/affine; every present label Dice >=0.995, no lost/new labels | Foreground macro Dice >=0.999; report small structures separately and inspect boundary errors |
| DK, DKT, Destrieux annotations | Every region Dice >=0.995; exact name/color tables and valid native indexing | Total vertex agreement >=0.999 for each hemisphere/atlas; inspect parcellation boundaries |
| ROI thickness mean/SD | Absolute difference <=0.01 mm | Each row must pass, never only the mean across ROIs |
| ROI printed area/GrayVol | One printed unit (1 mm² / 1 mm³) + 0.1% of reference | The absolute term explicitly covers integer text quantization and tiny regions |
| Aseg printed Volume_mm3 | 0.1 mm³ + 0.1% of reference | This is a stats-column check, not a replacement for label Dice |
| SynthSeg soft volumes, including eTIV | 10 mm³ + 0.1% of reference per CSV column | Reports every structure and eTIV separately; hard-label agreement cannot establish soft-volume agreement |
| Other stats | Explicit per-column/per-Measure bounds in JSON; unlisted fields exact | All rows, all columns and global Measures must remain present |

`normMean`/`normStdDev` use 0.01 native intensity units +0.1%; `normMin`,
`normMax`, `normRange` allow one native intensity unit. Printed `MeanCurv` and
`GausCurv` allow 0.001+0.1%; `FoldInd` allows 1+0.1%, and `CurvInd` allows
0.1+0.1%. Listed global volume/area Measures allow 1 native unit+0.1%, global
mean thickness allows 0.01 mm, and listed dimensionless ratios allow
1e-6+0.1%. Defect counts and vertex/voxel counts stay exact. These are deliberate
provisional engineering bounds, not empirically established agreement.

The SynthSeg soft-volume check reads `stats/synthseg.vol.csv`, requires the same
ordered columns, and reports signed, absolute, relative and allowed error in
mm³ for each column. Its 0.1% term matches the printed stats profile; the
10 mm³ floor covers small posterior channels. For the development T1, the
official no-crop route keeps posteriors in float32, and changing accumulation
between Torch/NumPy float32 and float64 moved eTIV by at most 0.276 mm³,
compared with a 464.8 mm³ GPU-versus-official difference. This is a posterior
inference difference, so the comparator retains a separate gate instead of
using hard segmentation Dice as a proxy. These values are from one subject
and do not establish performance across subjects.

The current label criterion applies to **all** present structures, including
small ones. It does not implement a two-voxel exception. Labels absent in both
outputs are omitted from Dice rather than counted as perfect matches. Use
foreground labels (exclude ID 0) for the additional macro Dice criterion.

The additional report criteria can be checked without rereading the images:

```python
import json
from statistics import mean

report = json.load(open("work/comparison.numeric.json"))
assert report["passed"], report["failed_checks"]
checks = report["checks"]
for hemi in ("lh", "rh"):
    for surface in ("white", "pial"):
        error = checks[f"surf/{hemi}.{surface}"]["displacement_mm"]
        assert error["mae"] <= 0.01 and error["p99_abs_error"] <= 0.05
    assert checks[f"surf/{hemi}.thickness"]["mae"] <= 0.005
    for atlas in ("aparc", "aparc.DKTatlas", "aparc.a2009s"):
        assert checks[f"label/{hemi}.{atlas}.annot"]["agreement"] >= 0.999
for name in ("aseg.mgz", "aparc+aseg.mgz"):
    rows = checks[f"mri/{name}"]["labels"]
    assert mean(row["dice"] for label, row in rows.items() if label != "0") >= 0.999
```

## Scientific definitions and completeness

The completed 8.2.0-1 reference uses `vertexvol --th3` for vertex `volume`, but
`mris_anatomical_stats -no-th3` for the ROI tables. Compare actual `GrayVol`
columns; do not substitute sums of TH3 vertex volume. The comparator records
the stats command and rejects conflicting explicit `-th3` / `-no-th3` modes.
`white.H/K` come from `white.preaparc`; `curv/curv.pial` come from their final
surfaces; `area.mid` is the average of white and pial area values.

The comparator currently requires **52 files**: white, pial, white.preaparc,
inflated; 12 vertex maps; all DK/DKT/Destrieux annotations; cortex.label;
DK white/pial, DKT and Destrieux stats for both hemispheres; aseg/aparc+aseg
volumes, aseg stats and SynthSeg soft-volume CSV. Missing files in **both** subjects still fail. No
nearest-neighbour or spherical resampling is used to rescue changed topology.

`standalone.py` has an additional mandatory output manifest. Its checks for
`orig.mgz`, ribbon, wmparc and sphere.reg remain required; those outputs are not
numerically compared by this comparator yet. Preserve both completeness gates.
Neither gate proves independent generation, conform equivalence, surface
self-intersection safety, repeatability, speedup, or absence of external
FreeSurfer installations; those remain separate workflow gates.

The available local Bert subject has all 51 comparator files and every output
in the current standalone manifest. It is an older reference subject, not an
8.2.0-1 end-to-end reference. Its self-comparison and the synthetic localized
perturbation tests establish comparator behaviour only.

## Matched end-to-end timing

Use [benchmark_pair.template.md](benchmark_pair.template.md) and
`tools/benchmark_recon_all_pair.py` for one serial official A / verified GPU C
pair. The script defaults to a read-only plan; `--execute` is required to run.
It refuses existing output paths and records the T1 hash, effective commands,
`/usr/bin/time -v`, CPU/GPU load and required output checks. Both runs request
`-all -parallel -openmp 4 -itkthreads 1`; C uses logical `cuda:1`. Compare the
completed pair only after the numerical gates above pass. A single pair on a
shared node is an observation, not an isolated speed estimate.
