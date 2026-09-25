# 代码结构与批量执行

[返回首页](../README.md) · [新增功能](ADDING_FUNCTIONS.md)

SynthStrip、SynthMorph、WMH-SynthSeg、33 类 SynthSeg、SynthSR、TorchFAST 和 FastVBM 分别位于 `src/freesurfer_torch/` 的功能目录；每个目录包含实现及简短说明。共享的 `cli.py` 提供单例命令行，`weights.py` 定位官方权重，`_batch_table.py` 检查 SynthStrip、SynthMorph、WMH-SynthSeg 和 SynthSR 的多被试输入表，`batch.py` 调度 Python 多进程任务。各功能的参数、原版对应和验证见专属页面。

FastVBM 的活跃配准链位于 `fast_vbm/linear.py`、`fast_vbm/flirt.py`、`fast_vbm/registration.py`、`fast_vbm/synthmorph_backend.py` 和 `fast_vbm/fnirt_backend.py`。`flirt.py` 提供 FSL 文件角色与 scaled-mm matrix 契约；`registration.py` 在本包 PyTorch SynthMorph `deform` 与 PyTorch FNIRT-style cubic B-spline 后端之间调度。`fast_vbm/legacy_registration.py` 只保留早期实验工具的旧导入兼容。

| 功能 | 详细说明 | 多被试入口 |
|---|---|---|
| SynthStrip | [脑提取](synthstrip/README.md) | 两列表 `predict_batch()` |
| SynthMorph | [配准](synthmorph/README.md) | 两列表 `predict_batch()` |
| WMH-SynthSeg | [结构及 WMH 分割](wmh_synthseg/README.md) | 两列表 `predict_batch()` |
| 33 类 SynthSeg | [T1 结构分割](synthseg/README.md) | Python 复用 `SynthSeg` 逐例处理 |
| SynthSR | [合成 T1w](synthsr/README.md) | 两列表 `predict_batch()` |
| TorchFAST | [三组织分割及偏置校正](fast/README.md) | Python `BatchRunner` |
| FastVBM | [原始 T1w 到 modulated GM](fast_vbm/README.md) | Python `BatchRunner` |

## 公开 API 与兼容性

```python
from freesurfer_torch import (
    SynthStrip, SynthMorph, WMHSynthSeg, SynthSeg, SynthSR,
    TorchFAST, FastVBM, FastVBMResult, VBMRegistrationResult,
    TorchFLIRT, FLIRTResult, PyTorchFNIRTRegistration, FNIRTVBMResult,
    LinearRegistrationResult, register_affine, register_gm,
    BatchRunner, BatchResult, run_batch,
    apply_transform,
)
```

五个学习模型构造时加载权重并选择 `device="cpu"` 或 `device="cuda:0"`；TorchFAST 与 TorchFLIRT 不加载权重。单例调用返回带几何信息的结果对象，由调用者选择保存字段。FastVBM 组合 SynthStrip、TorchFAST、PyTorch 12-DOF affine 和一个可选非线性后端。`registration_backend="synthmorph"` 延迟加载官方 deform checkpoint；`registration_backend="fnirt"` 构造无 checkpoint 的 PyTorch B-spline 优化器。旧导入路径 `freesurfer_torch.spatial` 和 `freesurfer_torch.synthmorph_models` 继续转导出对应实现。权重查找顺序为显式路径、`FREESURFER_TORCH_WEIGHTS`、配置脚本保存的目录、用户缓存目录、已设置的 `FREESURFER_HOME/models/`；见[权重说明](WEIGHTS.md)。

## 批量执行

多被试仅通过 Python 调用。前四项功能的 `predict_batch()` 接受 pandas `DataFrame`，列名必须恰好为 `input`、`output`，一行对应一例。`input` 填入输入影像路径；`output` 必须是不带扩展名的绝对路径前缀，含被试的 base name，例如 `/results/sub-01`。它不是输出目录或完整文件名。模型按下表追加后缀并创建父目录；方法按表的行顺序返回 `list[dict[str, pathlib.Path]]`。同一表中不能重复使用输出前缀。

```python
from pathlib import Path
import pandas as pd
from freesurfer_torch import SynthStrip

inputs = Path("/data/t1w")
outputs = Path("/results/brain")
table = pd.DataFrame({
    "input": [str(inputs / "sub-01_T1w.nii.gz"), str(inputs / "sub-02_T1w.nii.gz")],
    "output": [str(outputs / "sub-01"), str(outputs / "sub-02")],
})
if __name__ == "__main__":
    model = SynthStrip(device="cuda:0")
    saved = model.predict_batch(table, workers=2)
    print(saved[0]["image"], saved[0]["mask"], saved[0]["distance"])
```

| 模型方法 | 输出前缀为 `/results/sub-01` 时生成的文件 | 路径字典键 |
|---|---|---|
| `SynthStrip.predict_batch(table, border=1, fill=None, workers=1, threads_per_worker=1)` | `_brain.nii.gz`、`_mask.nii.gz`、`_sdt.nii.gz` | `image`、`mask`、`distance` |
| `SynthMorph.predict_batch(table, fixed, workers=1, threads_per_worker=1)` | `_moved.nii.gz`、`_fixed_moved.nii.gz`、`_transform.mgz`/`.lta`、`_inverse.mgz`/`.lta` | `moved`、`fixed_moved`、`transform`、`inverse` |
| `WMHSynthSeg.predict_batch(table, crop=False, workers=1, threads_per_worker=1)` | `_seg.nii.gz`、`_lesion_probs.nii.gz`、`_volumes.csv` | `segmentation`、`lesion_probability`、`volumes_csv` |
| `SynthSR.predict_batch(table, ct=False, disable_flipping=False, disable_sharpening=False, workers=1, threads_per_worker=1)` | `_synthsr.nii.gz` | `image` |

SynthMorph 的 `fixed` 可以是全表共用的一幅目标图像，也可以是与表行顺序一一对应、长度相同的目标图像列表；`workers=2` 时目标图像须为文件路径或路径列表。`joint`/`deform` 保存 `.mgz` 位移变换，`affine`/`rigid` 保存 `.lta` 仿射变换。

默认 `workers=1` 在当前程序中逐例复用模型。表中至少有两行时，`workers=2` 用当前程序和一个 spawn 子进程在同一指定设备上各加载一份模型，按表行顺序返回结果。每次调用都会新建并关闭子进程，下次调用需重新加载子进程模型。多进程时 `threads_per_worker` 控制每个进程的 Torch CPU 线程数；脚本须以 `if __name__ == "__main__":` 保护调用。两种模式的每次网络推理均为 B=1；只有一行时实际只运行一个进程。各功能子页的 B2 对照使用未发布的合批实验路径，不代表单被试加速。

FastVBM 多病例使用 Python `BatchRunner`，每例一个 `fast_vbm` job。`model` 是传给 `FastVBM(...)` 的共享构造参数，其中 `registration_backend` 选择 `synthmorph` 或 `fnirt`；`kwargs` 至少包含 `image` 和 `template`，可加与输入同网格的 `brain_mask`；`outputs` 将 `pve_gm`、`warped_gm`、`jacobian`、`modulated_gm` 等结果属性映射到完整文件路径。下面在两张 GPU 上处理两例，完整输出字段见[FastVBM 多病例说明](fast_vbm/README.md#多被试python-batchrunner)：

```python
from freesurfer_torch import BatchRunner

def main():
    jobs = []
    for subject in ("sub-01", "sub-02"):
        jobs.append({
            "task": "fast_vbm",
            "model": {"registration_backend": "fnirt"},
            "kwargs": {
                "image": f"/data/{subject}_T1w.nii.gz",
                "template": "/data/template_GM.nii.gz",
            },
            "outputs": {
                "modulated_gm": f"/results/{subject}/T1_GM_to_template_GM_mod.nii.gz",
            },
        })
    with BatchRunner(devices=("cuda:0", "cuda:1"), workers_per_device=1) as runner:
        reports = runner.run(jobs)
    if any(not report.ok for report in reports):
        raise RuntimeError([report.error for report in reports if not report.ok])

if __name__ == "__main__":
    main()
```

`BatchRunner` 每个 worker 绑定一张设备，并按 `model` 字典缓存一套 pipeline，可连续提交多批。FNIRT worker 缓存 B-spline 注册器；SynthMorph worker 还缓存 deform 网络。默认拒绝覆盖已有输出。返回的 `BatchResult` 按任务顺序记录成功路径或错误，失败信息可能含本地路径，应作为私有运行记录。TorchFAST 也可通过 Python `BatchRunner` 调用，见[功能说明](fast/README.md#多病例并行)。单例 CLI 保留；命令行不接受多被试表或任务清单。

## 验证记录

历史 [0.1.0 批量记录](../benchmark/real_batch/execution.public.json)与[输出比较](../benchmark/real_batch/comparison.public.json)使用旧的多进程 `BatchRunner`，覆盖 24 个任务、60 个输出；它们不代表当前 `predict_batch()` 的提速结果。[B1/B2/P2 对照](../benchmark/batch_modes_2026-09-24.md)中的 P2 是跨 cold/warm 两轮保持常驻的两个独立程序，不能直接作为新 `workers=2` API 的计时。数值复现、运行环境和单例计时见[对照报告](COMPARISON.md)。
