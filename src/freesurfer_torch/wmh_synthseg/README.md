# WMH-SynthSeg

这里实现 WMH 与脑结构联合分割。`model.py` 定义与官方权重匹配的 3D U-Net，`spatial.py` 处理图像方向和 1 mm 重采样，`pipeline.py` 完成预处理、推理及标签和病灶概率输出。多被试 Python 接口 `model.predict_batch(table)` 从每行的绝对输出前缀保存分割图、病灶概率图和软体积 CSV。单例 Python 与命令行用法、对应的原版指令及验证结果见[功能说明](../../../docs/wmh_synthseg/README.md)。
