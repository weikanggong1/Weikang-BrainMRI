# WMH-SynthSeg：脑结构与白质高信号分割

[返回首页](../../README.md) · [官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg) · [官方源码](https://github.com/freesurfer/freesurfer/tree/dev/mri_WMHsynthseg/WMHSynthSeg)

WMH-SynthSeg 同时分割脑结构和白质高信号（WMH，FreeSurfer 标签 **77**），支持 T1w、FLAIR 等对比度。**FreeSurfer 原版已使用 PyTorch**。本包基于已验证的源码与官方 `WMH-SynthSeg_v10_231110.pth`，提供独立安装、模型复用和多 GPU 批量调用；推理不调用 FreeSurfer 程序。

## 原版源码流程与输出几何

参考源为 gpucw1 安装的 FreeSurfer 8.2.0-1：`bin/mri_WMHsynthseg` 启动 `python/packages/WMHSynthSeg/inference.py`，模型定义来自同目录的 `unet3d`。输入是单幅 3D `.nii`、`.nii.gz` 或 `.mgz`。模型把影像重排到 RAS 方向，用最大值归一化，再重采样到 **1 mm 等方体素**。网络是五层 3D U-Net，输入单通道、输出 39 通道；前 33 通道用于标签后验概率。它还对第一空间轴翻转后的输入再预测一次，交换左右标签通道后平均两个概率图。

`crop=False` 时在完整重采样视野上推理，网络输入各轴填充到 32 的倍数。`crop=True` 时先在 192×224×192 的区域进行定位，再围绕估计的脑中心裁出不超过该大小的区域用于正式预测。原版官网指出 GPU 运行需要 `--crop`；裁剪可能移除视野边缘。**输出是模型处理后的 RAS/1 mm 网格，通常不等于输入网格**。分割体素值来自 33 个 FreeSurfer 标签；另可保存标签 77 的浮点概率图。CSV 体积来自各标签后验概率在 1 mm 网格上的求和，单位 mm³，因此通常不等于硬分割中各标签的体素计数。

原版 CLI 的单例指令：

```bash
mri_WMHsynthseg --i case_FLAIR.nii.gz --o case_seg.nii.gz \
  --device cpu --threads 8 --crop \
  --save_lesion_probabilities --csv_vols case_volumes.csv
```

`--i` 是唯一输入影像；`--o` 是硬分割图；`--crop` 启用上述两遍定位/裁剪；`--save_lesion_probabilities` 另写 `case_seg.lesion_probs.nii.gz`；`--csv_vols` 汇总各结构体积；`--device` 与 `--threads` 控制 PyTorch。输入、输出也可以分别是目录；原版遍历输入目录第一层的支持格式，为每个文件在输出目录写 `_seg` 后缀。原版 CSV 的第一列表头为 `Input-file`，实际写入的是**输出分割路径**。原版 3D 流程经过验证；安装源码中的 4D 均值分支存在参数错误，不在本包支持范围内。

## 本包 Python、命令行和批量接口

官方权重无需随 GitHub 仓库下载；用专属脚本从 FreeSurfer 官方地址获取并自动记录位置：

```bash
python tools/setup_weights.py --model wmh-synthseg
```

Python 模型构造一次即可复用；输入可为 3D `.nii`、`.nii.gz`、`.mgz` 路径或 `surfa.Volume`，返回 `WMHResult`。`crop` 和 `save_lesion_probabilities` 默认均为 `False`，下面显式开启二者以适应 GPU 并保存概率图。

```python
from freesurfer_torch import WMHSynthSeg

model = WMHSynthSeg(device="cuda:0")
result = model("case_FLAIR.nii.gz", crop=True, save_lesion_probabilities=True)
result.segmentation.save("case_seg.nii.gz")
result.lesion_probability.save("case_seg.lesion_probs.nii.gz")
print(result.volumes_mm3)
```

| 参数或字段 | 类型、含义 | 原版对应 |
|---|---|---|
| `WMHSynthSeg(weights=None, device="cpu", threads=None)` | 模型构造时加载一次官方 `.pth`；`weights` 可为文件或目录；`device` 选 CPU/GPU | `--device`、`--threads`；原版从 `$FREESURFER_HOME/models` 加载固定文件 |
| `model(image, crop=False, save_lesion_probabilities=False)` | 单幅 3D 影像；`crop=True` 两遍定位并限制推理区域 | `--i`、`--crop`、`--save_lesion_probabilities` |
| `result.segmentation` | `surfa.Volume`，33 类整数标签；WMH 为 77 | `--o` 指定的图像 |
| `result.lesion_probability` | 请求时为 `surfa.Volume`，否则为 `None`；体素值为 WMH 后验概率 | `--save_lesion_probabilities` 额外写出的 `.lesion_probs` 图 |
| `result.volumes_mm3` | `{标签编号: 软体积}` 字典，单位 mm³，包括背景 0 | `--csv_vols` 的各标签体积列；CSV 不输出背景列 |

两幅图像结果都使用原版同样的处理后网格，原图的坐标变换保留在输出仿射矩阵中；不保证与输入数组同形状。调用者通过 `.save(path)` 写出影像；Python 字典若需 CSV，可用下面的 CLI 直接生成与原版列名相同的表。原版 `Intracranial-volume` 列是非背景软体积之和，`Input-file` 列实际记录**输出分割路径**。

对应的新命令为：

```bash
fs-torch wmh-synthseg --i case_FLAIR.nii.gz --o case_seg.nii.gz \
  --device cuda:0 --threads 4 --crop \
  --save_lesion_probabilities --csv_vols case_volumes.csv
```

这条命令的 `--i` 读取单幅 FLAIR，`--o` 写标签图；`--crop` 将正式预测限制在脑周围的区域；`--save_lesion_probabilities` 额外写 `case_seg.lesion_probs.nii.gz`；`--csv_vols` 写软体积 CSV；`--device` 指定 GPU，`--threads` 指定 Torch 的 CPU 线程。与原版一样，`--i` 和 `--o` 也可同时指定为目录，此时逐个处理输入目录第一层的 `.nii`、`.nii.gz`、`.mgz` 文件，输出名增加 `_seg` 后缀。CLI 会创建输出父目录，并在遇到错误时以非零码结束；原版逐图捕获异常，可能返回零码但未产生某例输出，因此批量核对时应检查文件是否存在。

## 多病例、多 GPU

批量清单中每例一个独立任务，`task` 必须为 `wmh_synthseg`；`kwargs` 传给上表的模型调用，`outputs` 中的 `segmentation` 对应 `--o`，`lesion_probability` 对应额外的概率图。请求后者时 worker 会自动设置 `save_lesion_probabilities=True`。下面两例展示清单格式；仓库已提供包含三例的可运行 [examples/wmh_jobs.json](../../examples/wmh_jobs.json)：

```json
[
  {
    "task": "wmh_synthseg",
    "kwargs": {"image": "examples/wmh_data/sub-02_FLAIR.nii.gz", "crop": true},
    "outputs": {
      "segmentation": "examples/results/wmh/sub-02_seg.nii.gz",
      "lesion_probability": "examples/results/wmh/sub-02_seg.lesion_probs.nii.gz"
    }
  },
  {
    "task": "wmh_synthseg",
    "kwargs": {"image": "examples/wmh_data/sub-03_FLAIR.nii.gz", "crop": true},
    "outputs": {
      "segmentation": "examples/results/wmh/sub-03_seg.nii.gz",
      "lesion_probability": "examples/results/wmh/sub-03_seg.lesion_probs.nii.gz"
    }
  }
]
```

```bash
fs-torch batch examples/wmh_jobs.json --devices cuda:0 cuda:1 \
  --workers-per-device 1 --threads-per-worker 4 \
  --report examples/results/wmh/batch_report.json
```

`--devices` 指定两张可见 GPU；`--workers-per-device 1` 在每张 GPU 上建立一个进程。两例可以同时运行，更多病例由空闲 worker 领取。worker 首次处理 WMH 任务时加载 checkpoint，随后复用模型；每例仍独立推理。`--report` 按清单顺序记录设备、进程、时间、输出文件和异常。运行前会检查输出路径冲突；每张 GPU 的 worker 数决定同时加载的模型份数，应结合显存设置。批量接口保存影像；若需原版格式的 CSV，可使用单例/目录 CLI，或从 Python 的 `volumes_mm3` 自行生成。

Python 批量调用同一清单时，使用 `from freesurfer_torch import BatchRunner`，然后在 `if __name__ == "__main__":` 保护下执行 `with BatchRunner(devices=("cuda:0", "cuda:1"), workers_per_device=1, threads_per_worker=4) as runner: reports = runner.run(jobs)`；`jobs` 即上面的列表。完整多模型示例和输出冲突规则见[批量架构](../ARCHITECTURE.md#批量执行)。

## 验证边界

数值对照固定同一官方权重、相同输入、设备、`--crop` 和线程，检查输出网格、所有标签、WMH 标签 77、病灶概率图及软体积；分别记录原版 CPU、官方源码 CUDA、本包 CPU、本包 CUDA 的完整命令运行时间。原版随 FreeSurfer 安装的 `fspython` 在 gpucw1 上为 CPU 版 PyTorch，因此 GPU 参考以未改动的官方 `inference.py` 在 CUDA PyTorch 环境中运行，并单独标记。公开病例没有人工 WMH 标注时，原版/本包的一致性不能解释为病灶检测准确率。完整 12 例结果、运行环境与复现命令见[WMH 验证记录](../../validation/wmh/README.md)。

在 12 例公开 FLAIR 上，CPU 原版/本包和 CUDA 原版/本包两组的**逐例**标签、病灶概率、数值仿射和 CSV 软体积完全一致（WMH Dice=1，概率最大绝对差=0）。完整单例命令的中位时间依次为 **97.38、70.69、8.25、8.41 秒**。前两臂分别使用 FreeSurfer 的 Torch 2.1.2+cpu 与本包的 Torch 2.5.1，后两臂均用 Torch 2.5.1/CUDA 11.8；这不是控制 PyTorch 版本后的纯网络加速试验。原版写盘与 Surfa 写盘的 NIfTI qform/sform *code* 可能不同，体素值和数值仿射在本实验中相同。

### 原版与本包示意图

下图使用仓库公开的 `sub-04` FLAIR。左列为输入，中、右列分别叠加 FreeSurfer
原版和本包输出的标签 77（红色）；两次运行均使用官方权重、CUDA 和 `--crop`。

![公开 FLAIR、FreeSurfer WMH-SynthSeg 与本包 WMH-SynthSeg](../figures/wmh_synthseg_comparison.png)

该图只显示同一物理位置的二维切面。完整三维标签、病灶概率、软体积及仿射矩阵
用于数值比较；生成命令和公开数据来源见[图示记录](../figures/README.md)和
[FLAIR 示例](../../examples/WMH.md)。
