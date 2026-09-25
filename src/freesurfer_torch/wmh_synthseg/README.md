# WMH-SynthSeg

这里实现 WMH 与脑结构联合分割。`model.py` 定义与官方权重匹配的 3D U-Net，`spatial.py` 处理图像方向和 1 mm 重采样，`pipeline.py` 完成预处理、推理及标签和病灶概率输出。单例 Python 与命令行用法、对应的原版指令及验证结果见[功能说明](../../../docs/wmh_synthseg/README.md)。

12 例完整单例命令的耗时中位数（秒；原版 GPU 为未修改官方源码在 CUDA 环境运行）：

| 原版 CPU | 本包 CPU | 原版 GPU | 本包 GPU |
|---:|---:|---:|---:|
| 97.38 | 70.69 | 8.25 | 8.41 |

基准条件、逐例范围及数值对照见[验证记录](../../../validation/wmh/README.md)。
