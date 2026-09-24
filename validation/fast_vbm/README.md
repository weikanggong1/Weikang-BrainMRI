# FastVBM package release validation

[返回 FastVBM 文档](../../docs/fast_vbm/README.md) · [既有科学对照](../fast/README.md#原始-t1-到-vbm)

本目录验证把既有 GPU FAST VBM 数值流程迁入可安装包后的 API、CLI、双 GPU
调度和发布边界。科学方法与 FSL/FNIRT 的 10 例比较仍以
[`validation/fast`](../fast/) 为准；这里不重复解释为新的算法准确性实验。

## 10 例真实 T1w、两张 GPU

`FastVBM` 通过 Python `BatchRunner` 在 gpucw1 的两张 H100 上运行 10 例真实 T1w，
每卡一个 worker，每个 worker 完成 5 例。每例保存 13 幅影像；130/130 个有限值检查
和 130/130 个输入/模板网格检查均通过。批量墙钟时间为 106.63 s，单例
`FastVBM.__call__` 中位数为 9.65 s；后者不含 BatchRunner 随后的 NIfTI 写出。

最终 nonlinear Jacobian 范围为 0.20016–4.77491，10/10 例均无非正值。
`deformation_scale` 范围为 0.67969–0.99609，说明部分原始位移场经过整体回退；因此
正 Jacobian 必须和该 QC 一起解释。

PVE 均位于 0–1。31,825,357 个 SynthStrip mask 体素中，有 6,943 个边缘体素
（0.0218%）不属于 TorchFAST 的有效强度支持，三类 PVE 都为 0；其余建模体素的
CSF+GM+WM 最大和误差为 2.98e-8。

## 重构与 CPU/CUDA 检查

迁移前 `gpu_register.py` 源码和包内 registration backend 在同一 CPU runtime、同一
phantom 和相同参数下运行，warped GM、Jacobian、displacement、4×4 pull affine 和
fit score 的最大绝对差均为 0。旧实验脚本现在导入包内 backend，避免两份算法继续
分叉。

另用一例真实 T1w、显式 mask 和缩短的 3/2 步优化进行 CPU/CUDA smoke。GM PVE 的
Pearson 为 0.9999997；warped GM、Jacobian 和 modulated GM 分别为 0.99650、
0.97276 和 0.99609，所有输出有限且两端 Jacobian 均为正。这是跨后端一致性 smoke，
不是默认 50/40 步科学结果。共享节点观测到 CPU 88.29 s、CUDA 5.05 s；由于优化步数
缩短且运行负载未隔离，该数值不作为默认 pipeline 加速倍数。

## 安装包验收

- 完整测试：120 passed、3 skipped；4 条 warning 来自既有 surfa 测试路径。
- wheel 在独立 venv 中安装，并从 `/tmp` 导入；`FastVBM`、`FastVBMResult`、
  `fs-torch --version` 和 `fs-torch fast-vbm --help` 均可用。
- 独立 wheel 环境的 CPU phantom 生成 13 幅有限影像和最后写入的
  `fast_vbm_report.json`。
- wheel 与 sdist 清单不含 checkpoint、模型权重、MRI、UKB archive 或 GM template；
  wheel 含五份许可/notice 文件和 FastVBM 源码及局部 README，sdist 还包含专属文档
  与流程图。最终产物的文件大小和 SHA-256 见 `report.public.json`。

聚合数据见 [`report.public.json`](report.public.json) 和
[`cpu_cuda.public.json`](cpu_cuda.public.json)。它们不含病例标识、源数据路径或
逐例数值。在本目录运行 `sha256sum -c SHA256SUMS` 可校验这三份公开记录。
