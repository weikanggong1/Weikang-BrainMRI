# SynthSR

这里实现单幅 3D MRI/CT 到 1 mm 合成 T1w 的推理。`model.py` 定义与官方 HDF5 权重对应的 PyTorch U-Net，`spatial.py` 处理重采样、方向和填充，`pipeline.py` 连接读图、推理、后处理和写盘；`__init__.py` 导出公开接口。推理不调用 FreeSurfer 或 TensorFlow。

```python
from freesurfer_torch.synthsr import SynthSR

sr = SynthSR(device="cuda:0")
result = sr("case_FLAIR.nii.gz")
result.image.save("case_synthsr.nii.gz")
```

`result.image.data` 是写盘前量化为 `uint8` 的 0–255 体素，`result.image.affine` 描述 1 mm 输出网格。MGZ 经 nibabel 保存后重新读入会报告 `float32` 存储类型，数值仍是这组量化值。构造时加载一次权重，可复用于后续单被试调用。接口参数、原版 `mri_synthsr` 对应关系、权重和输出格式见[完整说明](../../../docs/synthsr/README.md)。

12 例完整单例命令的耗时中位数（秒）：

| 原版 CPU | 本包 CPU | 原版 GPU | 本包 GPU |
|---:|---:|---:|---:|
| 103.60 | 42.32 | 53.27 | 13.80 |

基准条件、逐例范围及数值对照见[验证记录](../../../validation/synthsr/README.md)。
