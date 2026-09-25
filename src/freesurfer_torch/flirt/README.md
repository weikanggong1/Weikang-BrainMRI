# PyTorch FLIRT

This directory contains the public FLIRT implementation:

- `core.py`: source-derived 12-DOF correlation-ratio registration;
- `standalone.py`: FSL-style Python file API with atomic outputs;
- `cli.py` and `__main__.py`: `fs-torch-flirt` and module commands;
- `legacy.py`: the earlier NCC/Adam method, named `LegacyTorchFLIRT`.

```python
from freesurfer_torch.flirt import run_flirt

result = run_flirt(
    "subject_GM.nii.gz",
    "template_GM.nii.gz",
    output="subject_GM_to_template.nii.gz",
    omat="subject_GM_to_template.mat",
    device="cuda:0",
)
```

```bash
fs-torch-flirt \
  -in subject_GM.nii.gz \
  -ref template_GM.nii.gz \
  -out subject_GM_to_template.nii.gz \
  -omat subject_GM_to_template.mat \
  -dof 12 -cost corratio --device cuda:0
```

The input is the moving image. The reference defines the output grid. The
matrix maps input to reference in FSL scaled-mm coordinates; it is not a
world-RAS affine.

With CUDA TF32 enabled by default, the fixed 0.05 mm ten-case matrix gate
passed 0 of 10 cases; the median and maximum RMS differences were 0.185545 mm
and 0.456724 mm. The implementation therefore reports
`validated_fsl_equivalent=false`. See `docs/flirt/README.md` in the source
repository for the full input/output contract, argument-by-argument examples,
coordinate conversion, validation table, and timing context.

The modified port and its vendored upstream sources are covered by the
non-commercial FSL Software Licence, Release 6.0.
