# SynthSeg 与皮层分区模块

本目录包含 FreeSurfer 8.2 SynthSeg 2.0 的 PyTorch 推理代码。

- `synthseg.py` 提供公开的 33 类 `SynthSeg`、`SynthSegResult` 和软体积 CSV；
- `segment.py`、`preprocess.py`、`postprocess.py` 实现网络推理、影像预处理和标签后处理；
- `pipeline.py` 与 `model.py` 提供 GPU recon-all 使用的皮层分区路径。

独立入口只覆盖单幅 T1 的非 robust、非 parcellated 33 类路径：

```python
from fnit import SynthSeg

model = SynthSeg(device="cuda:0")
result = model("subject_T1w.nii.gz")
result.segmentation.save("subject_synthseg.nii.gz")
result.write_volumes_csv("subject_T1w.nii.gz", "subject_synthseg.vol.csv")
```

单被试 CLI、原版 `mri_synthseg` 参数对应、权重和验证边界见
[功能说明](../../../docs/synthseg/README.md)。GPU recon-all 中的使用方式见
[recon-all 说明](../../../docs/recon_all/README.md)。

同输入原版对照见 [CUDA 验证记录](../../../validation/recon_all/python_gpu_port/CONNECTED_SYNTHSEG_GPU_20260926.md)。
