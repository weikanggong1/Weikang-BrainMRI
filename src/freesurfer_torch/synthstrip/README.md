# SynthStrip

这里实现脑提取。`model.py` 定义官方 U-Net，`pipeline.py` 实现影像处理及 `SynthStrip`、`StripResult`，`__init__.py` 导出接口。单被试命令使用统一的 `fs-torch synthstrip` 入口。

```python
from freesurfer_torch.synthstrip import SynthStrip

model = SynthStrip(weights="/path/to/weights", device="cuda:0")
result = model("subject_T1w.nii.gz")
result.mask.save("subject_mask.nii.gz")
```

调用返回三个 Surfa Volume：去颅骨图像 `image`、脑掩膜 `mask` 和符号距离图 `distance`。输入可为 3D 图像或逐帧处理的 4D 图像。

12 例完整单例命令的耗时中位数（秒；原版 GPU 为未修改官方脚本在 CUDA 环境运行）：

| 原版 CPU | 本包 CPU | 原版 GPU | 本包 GPU |
|---:|---:|---:|---:|
| 16.92 | 16.96 | 16.92 | 18.27 |

基准条件、四分位数及数值对照见[功能说明](../../../docs/synthstrip/README.md)。

[完整参数、CLI、源码分析与验证](../../../docs/synthstrip/README.md) · [权重](../../../docs/WEIGHTS.md)
