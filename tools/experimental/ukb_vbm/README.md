# Experimental UKB v1.5 VBM workflow

These scripts reproduce the subject-level structure of UK Biobank v1.5 VBM
and test a CUDA PyTorch replacement. The PyTorch path is an experimental
alternative. It replaces FAST with SynthSeg-derived grey-matter probabilities
and replaces FNIRT with a multiscale PyTorch registration. Its outputs use the
same filenames and template grid as `bb_vbm`, but the measurements are not
FNIRT-equivalent.

The method, source correspondence and validation limits are documented in
[`docs/ukb_vbm/README.md`](../../../docs/ukb_vbm/README.md). Aggregate results
and cohort figures are in [`validation/ukb_vbm/`](../../../validation/ukb_vbm/).

## Files

| File | Purpose |
|---|---|
| `run_gpu_vbm.py` | One raw T1 to GM, warped GM, nonlinear Jacobian and modulated GM |
| `gpu_gm.py` | SynthSeg-derived GM probability estimation with a persistent model |
| `gpu_register.py` | CUDA/CPU GM-to-template registration, Jacobian and modulation |
| `run_gpu_raw.py`, `run_gpu_batch.py` | Anonymous-manifest batch runner used in validation |
| `run_fsl_reference.py` | UKB v1.5 FSL reference subset from raw T1 through `bb_vbm` |
| `geometry_fallback.py` | Header-derived FSL transform used only when `xyztrans.sch` fails |
| `template_pair_eval.py` | Fixed-mask paired LOO evaluation and exact label permutation |
| `evaluate.py`, `qc_figures.py` | Method agreement, timing summaries and aggregate figures |
| `build_public_report.py` | Converts aggregate-only JSON into the public report |

## One T1 on one GPU

From the repository root, install the package and download the official
WMH-SynthSeg checkpoint. The checkpoint is verified and its directory is saved,
so `--weights` can be omitted afterward.

```bash
python -m pip install -e .
python tools/setup_weights.py --model wmh-synthseg

python tools/experimental/ukb_vbm/run_gpu_vbm.py \
  --input examples/data/sub-02_T1w.nii.gz \
  --template /path/to/ukb/template_GM.nii.gz \
  --output-dir work/ukb_vbm/sub-02 \
  --device cuda:0
```

The command writes:

| Output | Meaning |
|---|---|
| `GM_prob.nii.gz` | SynthSeg-derived GM probability on the input T1 grid |
| `brain_mask.nii.gz` | Hard intracranial mask on the input T1 grid |
| `T1_GM_to_template_GM.nii.gz` | GM probability resampled to the template grid |
| `T1_GM_JAC_nl.nii.gz` | Nonlinear pull-map Jacobian on the template grid |
| `T1_GM_to_template_GM_mod.nii.gz` | Warped GM multiplied by the nonlinear Jacobian |
| `report.json` | Settings, deformation checks and cold-start wall times |

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

`run_gpu_raw.py` keeps one segmentation model in memory while it processes its
assigned cases. `run_gpu_batch.py` then registers those cases sequentially on
the same device. To use two GPUs, split the case IDs into disjoint groups and
start one process per GPU. For example:

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
