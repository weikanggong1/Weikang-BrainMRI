# 代码结构与批量执行

[返回首页](../README.md) · [新增功能](ADDING_FUNCTIONS.md)

源码、说明文档和测试按功能组织。0.3.0 加入 WMH-SynthSeg，0.4.0 加入 SynthSR，
0.5.0 加入 TorchFAST。命令行、权重定位和批量调度放在共享层，模型及其空间运算
放在各功能目录。

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
├── wmh_synthseg/               # 结构与白质高信号分割
│   ├── __init__.py              # 功能公开接口
│   ├── model.py                 # 官方权重对应的 3D U-Net
│   ├── pipeline.py              # 预处理、推理和结果对象
│   ├── spatial.py               # 方向调整与重采样
│   └── README.md               # 代码目录入口
├── synthsr/                    # 单幅影像合成 1 mm T1w
│   ├── __init__.py              # 功能公开接口
│   ├── model.py                 # 3D U-Net 与 HDF5 权重读取
│   ├── pipeline.py              # 预处理、推理和结果保存
│   ├── spatial.py               # 重采样、方向调整与填充
│   └── README.md               # 代码目录入口
├── fast/                       # 单通道 T1 三组织分割与偏置场校正
│   ├── __init__.py              # 功能公开接口
│   ├── algorithm.py             # HMRF-EM、bias field 与 PVE 张量算法
│   ├── pipeline.py              # 影像几何、TorchFAST 与 FASTResult
│   ├── upstream_fast4/          # 原样保留的 FAST4 2111.3 源码；不参与构建
│   └── README.md               # 代码目录入口
├── spatial.py                  # 旧导入路径的转导出
└── synthmorph_models.py         # 旧导入路径的转导出
docs/
├── synthstrip/README.md         # 参数、用法、源码分析和验证
├── synthmorph/README.md
├── wmh_synthseg/README.md
├── synthsr/README.md
├── fast/README.md
├── WEIGHTS.md                  # 官方权重获取与许可
├── COMPARISON.md               # 0.1.0 对照实验
├── ARCHITECTURE.md
└── ADDING_FUNCTIONS.md
tests/
├── synthstrip/
├── synthmorph/
├── wmh_synthseg/
├── synthsr/
├── fast/
├── batch/
└── test_public_api.py
```

`tools/` 包含验证和报告脚本；`validation/` 与 `benchmark/` 保存可分享的实验记录。权重单独存放，不加入 Git 或安装包。

## 公开 API 与兼容性

0.5.0 的顶层公开导入如下；此前版本已有的导入保持兼容：

```python
from freesurfer_torch import (
    SynthStrip, StripResult,
    SynthMorph, RegistrationResult, apply_transform,
    WMHSynthSeg, WMHResult,
    SynthSR, SynthSRResult, SynthSRImage,
    TorchFAST, FASTResult, FASTConfig, FASTTensorResult, segment_t1,
    BatchRunner, BatchResult, run_batch,
)
```

功能也可从其专属模块导入：

```python
from freesurfer_torch.synthstrip import SynthStrip
from freesurfer_torch.synthmorph import SynthMorph, apply_transform
from freesurfer_torch.wmh_synthseg import WMHSynthSeg
from freesurfer_torch.synthsr import SynthSR
from freesurfer_torch.fast import (
    TorchFAST, FASTResult, FASTConfig, FASTTensorResult, segment_t1,
)
```

顶层按需导入：`import freesurfer_torch` 本身不加载 Torch、Surfa 或权重。旧路径 `freesurfer_torch.spatial` 和 `freesurfer_torch.synthmorph_models` 转导出新目录中的对象；新增代码直接从 `freesurfer_torch.synthmorph.spatial` 和 `freesurfer_torch.synthmorph.models` 导入。`TorchFAST` 是数值算法，不读取 checkpoint；`FASTConfig`、`FASTTensorResult` 和 `segment_t1` 是无文件 I/O 的张量层接口。

学习模型的构造函数加载权重并选择设备；TorchFAST 构造函数只保存算法参数和设备。
调用实例处理输入，返回带影像几何的结果对象，由调用者决定保存哪些输出。学习模型
的权重查找顺序为显式路径、`FREESURFER_TORCH_WEIGHTS`、配置脚本保存的目录、用户
缓存目录、已设置的 `FREESURFER_HOME/models/`。最后一项兼容已有安装，运行时不要求
安装 FreeSurfer。WMH-SynthSeg 和 SynthSR 输出的空间网格通常与输入不同；分别见
[WMH-SynthSeg](wmh_synthseg/README.md) 和 [SynthSR](synthsr/README.md) 的说明。

## 批量执行

每例影像是一个任务，由独立进程处理；不同形状的影像无需拼成同一个 tensor。每个 worker 绑定一个设备，按任务名称和模型参数缓存模型。同一 `BatchRunner` 可以连续提交多批，复用进程和模型。

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
  },
  {
    "task": "wmh_synthseg",
    "model": {"weights": "/path/to/weights"},
    "kwargs": {"image": "/data/sub03_FLAIR.nii.gz", "crop": true},
    "outputs": {"segmentation": "/results/sub03_wmh_seg.nii.gz"}
  },
  {
    "task": "synthsr",
    "model": {"lowfield": false},
    "kwargs": {"image": "/data/sub04_FLAIR.nii.gz"},
    "outputs": {"image": "/results/sub04_synthsr.nii.gz"}
  },
  {
    "task": "fast",
    "kwargs": {"image": "/data/sub05_T1_brain.nii.gz"},
    "outputs": {
      "pve_gm": "/results/sub05_pve_1.nii.gz",
      "bias_field": "/results/sub05_bias.nii.gz",
      "restored": "/results/sub05_restore.nii.gz"
    }
  }
]
```

| 字段 | 规则 |
|---|---|
| `task` | `synthstrip`、`synthmorph`、`wmh_synthseg`、`synthsr` 或 `fast` |
| `model` | 可省略；功能构造参数，不包含 `device`；SynthMorph 内层 `model` 指配准模式，SynthSR 可在此选 `lowfield` 或 `v1`，FAST 可设置 `bias_fwhm_mm`、`pve_chunk_size` 等算法参数 |
| `kwargs` | 实例调用参数；SynthStrip、WMH-SynthSeg、SynthSR、FAST 至少有 `image`，SynthMorph 至少有 `moving` 和 `fixed`；FAST 还可传同网格 `mask` |
| `outputs` | 至少一个输出，键为结果属性，值为文件路径 |

SynthStrip 输出键为 `image`、`mask`、`distance`；SynthMorph 为 `moved`、`fixed_moved`、`transform`、`inverse`；WMH-SynthSeg 为 `segmentation`、`lesion_probability`；SynthSR 为 `image`；FAST 为 `pve_csf`、`pve_gm`、`pve_wm`、`hard_segmentation`、`pve_segmentation`、`mixel_type`、`bias_field`、`restored`。若请求 WMH 病灶概率输出，worker 会启用相应推理选项。`volumes_mm3`、FAST 的 `tissue_means` 和 `tissue_variances` 是 Python 返回的数值，不是可调用 `.save()` 的批量输出。变换应用 `apply_transform` 是 CPU 后处理，不是当前 batch 的任务类型；可按需循环应用。

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

`BatchRunner` 默认每 worker 1 个 Torch 线程，批量 CLI 默认每 worker 4 个线程，可显式设定。功能构造函数也可能设置当前进程的 Torch 线程数；批量任务通常只在 runner 指定线程数，避免模型参数覆盖它。SynthStrip、SynthMorph 和 SynthSR 在 CUDA 构造时关闭当前进程的 PyTorch TF32；SynthStrip 还设置其官方卷积后端选项。

批次先整体检查输出路径，再创建目录和分发。默认拒绝覆盖已有文件，`overwrite=True` / `--overwrite` 可允许覆盖。即使允许覆盖，同批任务之间也不能共享输出路径。SynthMorph 的调试目录三个输出同样参与冲突检查。

每个 `BatchResult` 保存输入索引、任务类型、设备、PID、起止时间、成功写出的路径、错误和 traceback；返回顺序保持输入顺序。单任务错误保留在结果中，其他任务仍可完成。CLI 写 JSON 报告，任一任务失败时返回非零退出码。如果一个任务后续写出失败，已经保存的文件仍保留。worker 崩溃后应关闭并重建 runner，再重试受影响任务。

## 验证记录

0.1.0 的[真实病例批量记录](../benchmark/real_batch/execution.public.json)和[输出比较](../benchmark/real_batch/comparison.public.json)覆盖 24 个任务、60 个输出：worker 跨批复用，任务执行时间重叠，批量与单例结果逐元素一致。共享 GPU 条件和计时口径见 [COMPARISON.md](COMPARISON.md)；0.2.0 的结构回归见 [refactor/report.public.json](../validation/refactor/report.public.json)。
