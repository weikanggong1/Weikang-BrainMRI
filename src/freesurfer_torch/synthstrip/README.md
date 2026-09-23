# SynthStrip

脑提取功能的独立实现目录。`model.py` 定义官方 U-Net；`pipeline.py` 负责影像处理、`SynthStrip` 和 `StripResult`；`__init__.py` 导出公开接口；`__main__.py` 保留模块执行入口。

```python
from freesurfer_torch.synthstrip import SynthStrip

model = SynthStrip(weights="/path/to/weights", device="cuda:0")
result = model("subject_T1w.nii.gz")
result.mask.save("subject_mask.nii.gz")
```

返回 `image`、`mask`、`distance` 三个 Surfa Volume。输入为 3D 或逐帧处理的 4D，复用实例可避免重复加载权重。

[完整参数、CLI、源码分析与验证](../../../docs/synthstrip/README.md) · [权重](../../../docs/WEIGHTS.md) · [批量执行](../../../docs/ARCHITECTURE.md#批量执行)
