# SynthMorph

配准功能的独立实现目录。`models.py` 定义网络并读取官方 HDF5；`spatial.py` 实现采样、积分和变换组合；`pipeline.py` 负责 `SynthMorph`、`RegistrationResult` 和 `apply_transform`；`__init__.py` 导出公开接口。

```python
from freesurfer_torch.synthmorph import SynthMorph, apply_transform

model = SynthMorph(weights="/path/to/weights", device="cuda:0", model="joint")
result = model("moving_T1w.nii.gz", "fixed_T1w.nii.gz")
result.moved.save("moving_in_fixed.nii.gz")
result.transform.save("moving_to_fixed.mgz")
```

支持 joint、deform、affine、rigid。结果包含双向图像和带几何的变换。配准输入为单帧 3D；应用已有变换接受 3D/4D，并使用 Surfa CPU 重采样。

[完整参数、CLI、源码分析与验证](../../../docs/synthmorph/README.md) · [权重](../../../docs/WEIGHTS.md) · [批量执行](../../../docs/ARCHITECTURE.md#批量执行)
