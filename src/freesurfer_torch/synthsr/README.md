# SynthSR

这里实现单幅 3D MRI/CT 到 1 mm 合成 T1w 的推理。`model.py` 定义与官方 HDF5 权重对应的 PyTorch U-Net，`spatial.py` 处理重采样、方向和填充，`pipeline.py` 连接读图、推理、后处理和写盘；`__init__.py` 导出公开接口。推理不调用 FreeSurfer 或 TensorFlow。

```python
from freesurfer_torch.synthsr import SynthSR

sr = SynthSR(device="cuda:0")
result = sr("case_FLAIR.nii.gz")
result.image.save("case_synthsr.nii.gz")
```

`result.image.data` 是写盘前量化为 `uint8` 的 0–255 体素，`result.image.affine` 描述 1 mm 输出网格。MGZ 经 nibabel 保存后重新读入会报告 `float32` 存储类型，数值仍是这组量化值。构造时加载一次权重；多被试 Python 接口为 `sr.predict_batch(table)`，`output` 列指定不带扩展名的绝对输出前缀。接口参数、原版 `mri_synthsr` 对应关系、权重和输出格式见[完整说明](../../../docs/synthsr/README.md)。
