# Weikang-BrainMRI

独立的 PyTorch 脑 MRI 处理工具。当前提供 **SynthStrip 脑提取**和 **SynthMorph 配准**，支持 CPU、CUDA、模型复用和多 GPU 批量执行。推理无需安装 FreeSurfer、TensorFlow、VoxelMorph 或 Neurite。

仓库名为 `Weikang-BrainMRI`；为保持已有代码兼容，安装包名仍为 `freesurfer-torch`，Python 导入名为 `freesurfer_torch`，CLI 为 `fs-torch`。0.2.0 将实现按功能分目录，公开 API 不变。

| 功能 | 专属文档 | 实现目录 |
|---|---|---|
| 脑提取、脑掩膜、距离场 | [SynthStrip](docs/synthstrip/README.md) | [synthstrip/](src/freesurfer_torch/synthstrip/) |
| 刚性、仿射、非线性、联合配准及应用变换 | [SynthMorph](docs/synthmorph/README.md) | [synthmorph/](src/freesurfer_torch/synthmorph/) |
| 多 GPU / 同 GPU 多进程批量调度 | [批量使用与架构](docs/ARCHITECTURE.md#批量执行) | [batch.py](src/freesurfer_torch/batch.py) |

## 安装与权重

要求 Python ≥ 3.10。GPU 推理需要与驱动兼容的 CUDA 版 PyTorch。

```bash
git clone https://github.com/weikanggong1/Weikang-BrainMRI.git
cd Weikang-BrainMRI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Git 仓库和 wheel 均不包含模型权重。** 官方下载地址、版本、SHA-256 校验及许可证见 [权重说明](docs/WEIGHTS.md)。只需准备所用功能对应的文件，随后设置：

```bash
export FREESURFER_TORCH_WEIGHTS=/path/to/weights
```

推理时不会自动联网下载。也可在 Python 中显式传入 `weights=`，或在 CLI 中使用 `--weights`。

## 快速使用

```python
from freesurfer_torch import SynthStrip, SynthMorph

extract = SynthStrip(device="cuda:0")
brain = extract("subject_T1w.nii.gz")
brain.image.save("subject_brain.nii.gz")
brain.mask.save("subject_mask.nii.gz")

register = SynthMorph(device="cuda:0", model="joint")
registration = register("subject_T1w.nii.gz", "template_T1w.nii.gz")
registration.moved.save("subject_in_template.nii.gz")
registration.transform.save("subject_to_template.mgz")
```

将 `device` 改为 `"cpu"` 即可在 CPU 运行。同一模型实例可重复调用；多病例并行使用 [BatchRunner 或批量 CLI](docs/ARCHITECTURE.md#批量执行)。网络和部分空间变换可在 GPU 上执行，文件 I/O、Surfa 处理和最终影像重采样在 CPU 上执行。

```bash
fs-torch synthstrip -i subject_T1w.nii.gz -o subject_brain.nii.gz \
  -m subject_mask.nii.gz --device cuda:0
fs-torch synthmorph subject_T1w.nii.gz template_T1w.nii.gz \
  -o subject_in_template.nii.gz -t subject_to_template.mgz --device cuda:0
fs-torch --help
```

## 验证与维护

- [详细功能和数值对照](docs/COMPARISON.md)：0.1.0 参考实验包含 12 例真实 T1w、96 次单例运行及 24 个批量任务。数据只发布匿名统计，不包含原始影像。
- [0.2.0 结构重整回归](validation/refactor/report.public.json)：新布局与 0.1.0 的对照记录；历史计时不能当作 0.2.0 的重新计时。
- [架构、公共 API 与批量任务格式](docs/ARCHITECTURE.md)。
- [新增功能指南](docs/ADDING_FUNCTIONS.md)：每个功能的实现、文档和测试均有独立目录。
- [来源与模型哈希](docs/provenance.json)、[第三方许可与引用](THIRD_PARTY_NOTICES.md)。

```bash
python -m pip install pytest
python -m pytest tests
```

SynthStrip 官方模型本身即为 PyTorch；本项目将其整理为独立可复用接口。SynthMorph 将指定 FreeSurfer 8.2.0 构建中的 TensorFlow 网络与空间运算移植到 PyTorch，直接读取相应官方权重。各功能文档说明支持范围、差异和复现入口。
