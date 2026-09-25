# GPU recon-all 模块

本目录封装固定的 FreeSurfer 8.2 单 T1 `recon-all -all` 流程。六项神经网络命令由
本包的 PyTorch/CUDA 实现执行；影像转换、表面、拓扑和统计仍由已验证原生运行包
中的 CPU 程序执行。

公开 Python 入口为：

```python
from fnit.recon_all.standalone import (
    run_recon_all,
    run_recon_all_batch,
)
```

`run_recon_all()` 运行一个被试；`run_recon_all_batch()` 在 Python 中把独立病例分配
到一张或多张 GPU，每张卡同时运行一例。单被试命令行是 `fnit-recon-all`；
不提供多被试命令行。

运行需要与固定配置匹配且通过清单校验的原生运行包、用户自己的 FreeSurfer license
以及模型目录。参数、输出、构建条件和验证范围见
[完整说明](../../../docs/recon_all/README.md)与
[验收记录](../../../validation/recon_all/README.md)。
