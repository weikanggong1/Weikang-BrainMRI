# 代码结构与批量执行

[返回首页](../README.md) · [新增功能](ADDING_FUNCTIONS.md)

0.2.0 按功能组织源码、说明文档和测试。共享层只保留命令行、权重定位和批量调度；模型实现及特有的空间运算放在所属功能目录。

```text
src/freesurfer_torch/
├── __init__.py                 # 按需导出公开 API
├── cli.py                      # fs-torch 命令行
├── weights.py                  # 统一权重定位
├── batch.py                    # 常驻 worker、结果与输出检查
├── synthstrip/
│   ├── __init__.py              # 功能公开接口
│   ├── __main__.py              # python -m ... 的兼容入口
│   ├── model.py                # 官方 U-Net 结构
│   ├── pipeline.py             # 影像流程和结果对象
│   └── README.md               # 代码目录入口
├── synthmorph/
│   ├── __init__.py              # 功能公开接口
│   ├── models.py               # 网络与 HDF5 权重加载
│   ├── pipeline.py             # 配准、变换应用和影像几何
│   ├── spatial.py              # 网络采样、积分和变换组合
│   └── README.md               # 代码目录入口
├── spatial.py                  # 旧导入路径的转导出
└── synthmorph_models.py         # 旧导入路径的转导出
docs/
├── synthstrip/README.md         # 参数、用法、源码分析和验证
├── synthmorph/README.md
├── WEIGHTS.md                  # 官方权重获取与许可
├── COMPARISON.md               # 0.1.0 对照实验
├── ARCHITECTURE.md
└── ADDING_FUNCTIONS.md
tests/
├── synthstrip/
├── synthmorph/
├── batch/
└── test_public_api.py
```

`tools/` 是验证和报告脚本，普通推理不依赖该目录。`validation/` 和 `benchmark/` 保存可分享的实验记录。模型权重独立存放，不加入 Git、源码发行包或 wheel。

## 公开 API 与兼容性

以下导入在重整前后保持一致：

```python
from freesurfer_torch import (
    SynthStrip, StripResult,
    SynthMorph, RegistrationResult, apply_transform,
    BatchRunner, BatchResult, run_batch,
)
```

功能也可从其专属模块导入：

```python
from freesurfer_torch.synthstrip import SynthStrip
from freesurfer_torch.synthmorph import SynthMorph, apply_transform
```

顶层使用按需导入，单纯 `import freesurfer_torch` 不加载 Torch、Surfa 或权重。旧 `freesurfer_torch.spatial` 和 `freesurfer_torch.synthmorph_models` 仅转导出新目录中的同一个对象，避免维护两份实现。新增代码应使用 `freesurfer_torch.synthmorph.spatial` 和 `freesurfer_torch.synthmorph.models`。

构造函数加载模型并选择设备；调用实例处理输入，返回带影像几何的结果对象。调用者决定保存哪些输出。权重查找顺序为显式路径、`FREESURFER_TORCH_WEIGHTS`、用户缓存目录、已设置的 `FREESURFER_HOME/models/`。最后一项是兼容已有安装的回退，运行时不要求安装 FreeSurfer。

## 批量执行

批量调度以病例为单位使用独立进程，并非将不同形状的影像拼接成网络 tensor batch。每个 worker 绑定一个设备，按任务名称和模型参数缓存模型。一个 `BatchRunner` 内可连续提交多批，复用进程和已加载模型。

`jobs.json` 是任务字典的列表：

```json
[
  {
    "task": "synthstrip",
    "model": {"weights": "/path/to/weights", "no_csf": false},
    "kwargs": {"image": "/data/sub01_T1w.nii.gz", "border": 1},
    "outputs": {
      "image": "/results/sub01_brain.nii.gz",
      "mask": "/results/sub01_mask.nii.gz"
    }
  },
  {
    "task": "synthmorph",
    "model": {
      "weights": "/path/to/weights", "model": "joint",
      "extent": 256, "hyper": 0.5, "steps": 7
    },
    "kwargs": {
      "moving": "/data/sub02_T1w.nii.gz",
      "fixed": "/data/template_T1w.nii.gz"
    },
    "outputs": {
      "moved": "/results/sub02_in_template.nii.gz",
      "transform": "/results/sub02_to_template.mgz"
    }
  }
]
```

| 字段 | 规则 |
|---|---|
| `task` | `synthstrip` 或 `synthmorph` |
| `model` | 可省略；模型构造参数，不包含 `device`；SynthMorph 内层 `model` 指配准模式 |
| `kwargs` | 实例调用参数；SynthStrip 至少有 `image`，SynthMorph 至少有 `moving` 和 `fixed` |
| `outputs` | 至少一个输出，键为结果属性，值为文件路径 |

SynthStrip 输出键为 `image`、`mask`、`distance`；SynthMorph 为 `moved`、`fixed_moved`、`transform`、`inverse`。变换应用 `apply_transform` 是 CPU 后处理，不是当前 batch 的任务类型；可按需循环应用。

```bash
fs-torch batch jobs.json --devices cuda:0 cuda:1 \
  --workers-per-device 1 --threads-per-worker 4 \
  --report results/batch_report.json
```

Python 批处理应放在可导入的脚本中，并保留 `__main__` guard：

```python
import json
from pathlib import Path
from freesurfer_torch import BatchRunner

def main():
    jobs = json.loads(Path("jobs.json").read_text())
    with BatchRunner(
        devices=("cuda:0", "cuda:1"),
        workers_per_device=1,
        threads_per_worker=4,
    ) as runner:
        for result in runner.run(jobs):
            print(result.index, result.device, result.ok, result.outputs, result.error)
        # 后续可 runner.run(next_jobs)，复用 worker 与模型。

if __name__ == "__main__":
    main()
```

进程以 `spawn` 启动。Notebook 中使用 CLI 或运行上述脚本。`run_batch(jobs, devices=("cuda:0",))` 是单批便利接口，执行后关闭 worker；跨批复用使用 `BatchRunner`。CPU 批处理可指定 `devices=("cpu",)`。

## 设备、输出与错误

`cuda:N` 遵循 `CUDA_VISIBLE_DEVICES` 的编号映射。默认每设备一个 worker；同 GPU 可设 `workers_per_device=2`，每个进程独立保留模型、激活和卷积工作区。多个模型参数组合会增加缓存量，增加 worker 数不保证吞吐提升。

`BatchRunner` 默认每 worker 1 个 Torch 线程，批量 CLI 默认每 worker 4 个线程，可显式设定。`SynthStrip(threads=...)` 也会设置当前进程的 Torch 线程数；批量任务通常只在 runner 指定线程数，避免模型参数覆盖它。两模型构造时均关闭当前进程的 PyTorch TF32；SynthStrip 还设置其官方卷积后端选项。

批次先整体检查输出路径，再创建目录和分发。默认拒绝覆盖已有文件，`overwrite=True` / `--overwrite` 可允许覆盖。即使允许覆盖，同批任务之间也不能共享输出路径。SynthMorph 的调试目录三个输出同样参与冲突检查。

每个 `BatchResult` 保存输入索引、任务类型、设备、PID、起止时间、成功写出的路径、错误和 traceback；返回顺序保持输入顺序。单任务错误保留在结果中，其他任务仍可完成。CLI 写 JSON 报告，任一任务失败时返回非零退出码。如果一个任务后续写出失败，已经保存的文件仍保留。worker 崩溃后应关闭并重建 runner，再重试受影响任务。

## 验证记录

0.1.0 的[真实病例批量记录](../benchmark/real_batch/execution.public.json)和[输出比较](../benchmark/real_batch/comparison.public.json)覆盖 24 个任务、60 个输出，验证了进程复用、任务时间重叠和单例/批量逐元素一致。该历史实验的性能口径及共享 GPU 条件见 [COMPARISON.md](COMPARISON.md)。0.2.0 的布局与导入回归独立记录在 [refactor/report.public.json](../validation/refactor/report.public.json)。
