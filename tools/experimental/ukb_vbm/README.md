# Experimental UKB v1.5 VBM workflow

These scripts reproduce the subject-level structure of UK Biobank v1.5 VBM
and test a CUDA PyTorch replacement. The PyTorch path is an experimental
alternative. Grey matter can come from SynthSeg-derived probabilities or
TorchFAST GM PVE, and a PyTorch registration replaces FNIRT. The registration
optimizes at scales 4, 2 and 1; at each scale its image loss is global
normalized correlation plus `0.2 * MSE`, rather than a windowed local
correlation. Its outputs use the same filenames and template grid as `bb_vbm`,
but the measurements are not
FNIRT-equivalent.

The packaged raw-T1 → SynthStrip → TorchFAST → GPU VBM interface is documented
under [`docs/fast_vbm/`](../../../docs/fast_vbm/README.md). The scripts here remain
the research harness for the FSL reference, SynthSeg arm, template comparisons,
and aggregate reports; they are not the stable package entry point.

The method, source correspondence and validation limits are documented in
[`docs/ukb_vbm/README.md`](../../../docs/ukb_vbm/README.md). Aggregate results
and cohort figures are in [`validation/ukb_vbm/`](../../../validation/ukb_vbm/).

## Files

| File | Purpose |
|---|---|
| `run_gpu_vbm.py` | One raw T1 to SynthSeg or TorchFAST GM, warped GM, nonlinear Jacobian and modulated GM |
| `gpu_gm.py` | SynthSeg-derived GM probability estimation with a persistent model |
| `gpu_register.py` | CUDA/CPU GM-to-template registration, Jacobian and modulation |
| `run_gpu_raw.py`, `run_gpu_batch.py` | Anonymous-manifest batch runner used in validation |
| `run_fsl_reference.py` | UKB v1.5 FSL reference subset from raw T1 through `bb_vbm` |
| `geometry_fallback.py` | Header-derived FSL transform used only when `xyztrans.sch` fails |
| `template_pair_eval.py` | Fixed-mask paired LOO evaluation and exact label permutation |
| `evaluate.py`, `qc_figures.py` | Method agreement, timing summaries and aggregate figures |
| `build_public_report.py` | Converts aggregate-only JSON into the public report |

## One T1 on one GPU

From the repository root, install the package. The default
`--gm-method synthseg` needs the official WMH-SynthSeg checkpoint. The checkpoint
is verified and its directory is saved, so `--weights` can be omitted afterward.

```bash
python -m pip install -e .
python tools/setup_weights.py --model wmh-synthseg

python tools/experimental/ukb_vbm/run_gpu_vbm.py \
  --input examples/data/sub-02_T1w.nii.gz \
  --template /path/to/ukb/template_GM.nii.gz \
  --output-dir work/ukb_vbm/sub-02 \
  --device cuda:0
```

For `--gm-method torch-fast`, prepare SynthStrip instead. TorchFAST itself has
no checkpoint. By default this path uses SynthStrip to obtain a brain image and
mask, then runs TorchFAST with bias-field correction enabled:

```bash
python tools/setup_weights.py --model synthstrip

python tools/experimental/ukb_vbm/run_gpu_vbm.py \
  --input examples/data/sub-02_T1w.nii.gz \
  --template /path/to/ukb/template_GM.nii.gz \
  --output-dir work/ukb_vbm/sub-02-fast \
  --device cuda:0 --gm-method torch-fast
```

Pass an input-grid mask with `--brain-mask` to skip SynthStrip. Bias correction
remains on in either case; `--fast-no-bias` is the explicit ablation switch.
`--synthstrip-weights` selects a checkpoint when the saved weight directory is
not used.

The command writes:

| Output | Meaning |
|---|---|
| `GM_prob.nii.gz` | SynthSeg-derived GM probability or TorchFAST GM PVE on the input T1 grid |
| `brain_mask.nii.gz` | Hard intracranial mask on the input T1 grid |
| `T1_GM_to_template_GM.nii.gz` | GM probability resampled to the template grid |
| `T1_GM_JAC_nl.nii.gz` | Nonlinear pull-map Jacobian on the template grid |
| `T1_GM_to_template_GM_mod.nii.gz` | Warped GM multiplied by the nonlinear Jacobian |
| `report.private.json` | Input and template paths, settings, GM method, deformation checks and cold-start wall times |

The TorchFAST branch additionally writes `T1_brain.nii.gz`,
`T1_brain_pve_0/1/2.nii.gz`, `T1_brain_seg.nii.gz`,
`T1_brain_pveseg.nii.gz`, `T1_brain_mixeltype.nii.gz`,
`T1_brain_bias.nii.gz` and `T1_brain_restore.nii.gz`. The report includes local
paths and is therefore a private intermediate.

The validated experiment used `--affine-steps 50 --deform-steps 40
--smoothness 10`. The smoothness value was selected on one tuning case and is
therefore an experiment setting rather than a generally established default.

## Several subjects and several GPUs

The validation manifest has this shape:

```json
{
  "cases": [
    {"case_id": "case01", "path": "/data/sub-001/T1w.nii.gz"},
    {"case_id": "case02", "path": "/data/sub-002/T1w.nii.gz"}
  ]
}
```

`run_gpu_raw.py --gm-method synthseg` keeps one SynthSeg estimator in memory
while it processes its assigned cases. With `--gm-method torch-fast`, it keeps
one SynthStrip instance and one TorchFAST instance instead and writes under
`T1_gpu_fast`; this batch path uses TorchFAST's default bias correction.
`run_gpu_batch.py` then registers those cases sequentially on the same device.
To use two GPUs, split the case IDs into disjoint groups and start one process
per GPU. The following example runs the SynthSeg arm:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/experimental/ukb_vbm/run_gpu_raw.py \
  --manifest cases.private.json --subjects-root work/vbm_subjects \
  --weights /path/to/WMH-SynthSeg_v10_231110.pth --device cuda:0 \
  --cases case01 case03 &
CUDA_VISIBLE_DEVICES=1 python tools/experimental/ukb_vbm/run_gpu_raw.py \
  --manifest cases.private.json --subjects-root work/vbm_subjects \
  --weights /path/to/WMH-SynthSeg_v10_231110.pth --device cuda:0 \
  --cases case02 case04 &
wait

CUDA_VISIBLE_DEVICES=0 python tools/experimental/ukb_vbm/run_gpu_batch.py \
  --manifest cases.private.json --subjects-root work/vbm_subjects \
  --ukb-template /path/to/ukb/template_GM.nii.gz --device cuda:0 \
  --arms gpu_raw_ukb --smoothness 10 --cases case01 case03 &
CUDA_VISIBLE_DEVICES=1 python tools/experimental/ukb_vbm/run_gpu_batch.py \
  --manifest cases.private.json --subjects-root work/vbm_subjects \
  --ukb-template /path/to/ukb/template_GM.nii.gz --device cuda:0 \
  --arms gpu_raw_ukb --smoothness 10 --cases case02 case04 &
wait
```

Each process writes only its assigned case directories. One worker per GPU
avoids loading duplicate 790 MB checkpoints on the same device. This workflow
was validated as sequential batching on one H100; the two-GPU example is the
supported scheduling pattern, not a reported two-GPU speed measurement.

For the TorchFAST arm, add `--gm-method torch-fast` and
`--synthstrip-weights /path/to/synthstrip.1.pt` to each `run_gpu_raw.py`
command, omit the WMH `--weights`, and register with `--arms gpu_fast_ukb`.
The case split and one process per GPU rule are unchanged.
`run_gpu_vbm.py --fast-no-bias` is the single-case bias ablation;
`run_gpu_raw.py` intentionally records the default bias-corrected batch path.

## Reference and evaluation

`run_fsl_reference.py` requires FSL and the UKB assets described in the method
document. `--arms ukb` runs only the official UKB template; adding `hcp` runs a
second user-supplied comparison template.

```bash
python tools/experimental/ukb_vbm/run_fsl_reference.py \
  --manifest cases.private.json --case case01 \
  --output-root work/fsl_subjects --assets /path/to/ukb/assets \
  --arms ukb
```

Formal template comparison uses a mask made only from the two templates. It
checks every shape and affine, recomputes leave-one-out references after every
within-subject arm-label assignment, and does not calculate a naive bootstrap
interval from dependent LOO scores.

```bash
python tools/experimental/ukb_vbm/template_pair_eval.py \
  --subjects-root work/vbm_subjects \
  --template-a /path/to/ukb/template_GM.nii.gz \
  --template-b /path/to/comparison/template_GM.nii.gz \
  --arm-a gpu_raw_ukb --arm-b gpu_raw_hcp \
  --permutation-device cuda:0 --out work/template_pair.private.json
```

`evaluate.py` writes anonymous case-level JSON and CSV. The formal template
evaluation JSON also contains local template paths. These files are private
intermediates; publish only aggregate output produced by `build_public_report.py`.
