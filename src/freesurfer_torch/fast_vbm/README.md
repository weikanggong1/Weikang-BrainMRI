# FastVBM

`FastVBM` runs the package pipeline from one raw T1 image to modulated gray
matter on a supplied template grid:

1. SynthStrip brain extraction, unless an input-grid brain mask is supplied;
2. TorchFAST three-tissue PVE estimation and bias-field correction;
3. multiscale PyTorch GM registration;
4. nonlinear pull-Jacobian estimation and GM modulation.

```python
from freesurfer_torch.fast_vbm import FastVBM

pipeline = FastVBM(device="cuda:0", synthstrip_weights="synthstrip.1.pt")
result = pipeline("T1w.nii.gz", "template_GM.nii.gz")
result.save("subject_vbm")
```

`FastVBMResult.volumes()` returns the brain image and mask, three FAST PVE
maps, FAST label and bias-correction products, warped GM, nonlinear Jacobian,
and modulated GM. `fast_vbm_report.json` records the settings, timing boundary,
FAST summary, and registration QC and is written last as the completion marker.

The registration optimizes global normalized correlation plus 0.2 MSE with a
sparse displacement grid. It is experimental and is not an implementation of
FSL FNIRT. See [`docs/fast_vbm/README.md`](../../../docs/fast_vbm/README.md) for
the full API, output definitions, validation, and comparison with UKB/FSL VBM.
