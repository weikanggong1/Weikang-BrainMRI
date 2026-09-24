# PyTorch T1 tissue segmentation

`TorchFAST` performs single-channel, three-class T1 segmentation with an
HMRF-EM model, joint multiplicative bias-field correction and partial-volume
estimation. It runs entirely in PyTorch and accepts a brain-extracted image.

```python
from freesurfer_torch.fast import TorchFAST

result = TorchFAST(device="cuda:0")("T1_brain.nii.gz")
result.pve_gm.save("T1_brain_pve_1.nii.gz")
result.bias_field.save("T1_brain_bias.nii.gz")
result.restored.save("T1_brain_restore.nii.gz")
```

The three PVE maps are ordered CSF, GM and WM. Image outputs remain on the
input voxel grid. `bias_field` is the multiplicative field in the acquired
image, so `restored = input / bias_field` inside the brain mask.

The algorithm follows the main stages and single-channel T1 defaults of FAST4
2111.3. GPU spatial updates are synchronous and PyTorch convolution, percentile
and floating-point reductions differ from FSL. The outputs are therefore not
claimed to be bitwise FAST results. See the [full method and API
description](../../../docs/fast/README.md).

FAST and this adaptation are covered by the non-commercial FSL 6.0 license.
The complete unmodified FAST4 2111.3 source tree is included in
`upstream_fast4/` for source transmission and provenance; it is not compiled or
imported by the Python package.
