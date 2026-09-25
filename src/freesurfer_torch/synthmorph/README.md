# SynthMorph

这里实现图像配准。`models.py` 定义网络并读取官方 HDF5 权重，`spatial.py` 实现采样、积分和变换组合，`pipeline.py` 提供 `SynthMorph`、`RegistrationResult` 和 `apply_transform`；`__init__.py` 导出接口。

```python
from freesurfer_torch.synthmorph import SynthMorph, apply_transform

model = SynthMorph(weights="/path/to/weights", device="cuda:0", model="joint")
result = model("moving_T1w.nii.gz", "fixed_T1w.nii.gz")
result.moved.save("moving_in_fixed.nii.gz")
result.transform.save("moving_to_fixed.mgz")
```

可选择 joint、deform、affine 或 rigid 模型。调用返回双向配准图像和带几何信息的变换。配准接受单帧 3D 图像。`apply_transform` 可处理 3D 或 4D 图像，并使用 Surfa 在 CPU 上重采样。

默认 joint 模式的 12 例完整单例命令耗时中位数（秒）：

| 原版 CPU | 本包 CPU | 原版 GPU | 本包 GPU |
|---:|---:|---:|---:|
| 164.55 | 122.33 | 116.59 | 17.62 |

基准条件、四分位数及数值对照见[功能说明](../../../docs/synthmorph/README.md)。

[完整参数、CLI、源码分析与验证](../../../docs/synthmorph/README.md) · [权重](../../../docs/WEIGHTS.md)
