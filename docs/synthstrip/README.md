# SynthStrip：脑提取

[返回首页](../../README.md) · [源码目录](../../src/freesurfer_torch/synthstrip/) · [权重](../WEIGHTS.md) · [批量执行](../ARCHITECTURE.md#批量执行)

SynthStrip 从脑影像预测有符号距离场，生成脑掩膜和去除背景的影像。官方模型已使用 PyTorch。本模块沿用其网络、权重和影像处理流程，提供可重复调用的 Python API；模型未重新训练。

参考版本为 **FreeSurfer 8.2.0**，build `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`。分析对象是 `$FREESURFER_HOME/python/scripts/mri_synthstrip`，不是 `bin/` 下的包装脚本。脚本 SHA-256 为 `bbc2ff8f8779862039401b05d5cd6039fb4f3583e0032a793ac9adb3f4521590`，全部来源信息见 [provenance.json](../provenance.json)。

## Python API

```python
from pathlib import Path
from freesurfer_torch import SynthStrip

out = Path("results")
out.mkdir(exist_ok=True)
extract = SynthStrip(weights="/path/to/weights", device="cuda:0", threads=4)
result = extract("subject_T1w.nii.gz", border=1, fill=0)
result.image.save(out / "subject_brain.nii.gz")
result.mask.save(out / "subject_mask.nii.gz")
result.distance.save(out / "subject_sdt.nii.gz")
# 可继续 extract("another_T1w.nii.gz")，复用模型。
```

`from freesurfer_torch.synthstrip import SynthStrip, StripResult` 是等价的功能模块入口。

### 模型构造

`SynthStrip(weights=None, device="cpu", no_csf=False, threads=None)`：

| 参数 | 含义 |
|---|---|
| `weights` | 官方 PT 文件或所在目录；省略时使用[统一查找顺序](../ARCHITECTURE.md#公开-api-与兼容性) |
| `device` | `"cpu"` 或 `"cuda:N"`；CUDA 编号遵循 `CUDA_VISIBLE_DEVICES` |
| `no_csf` | 为 `True` 时使用排除 CSF 的官方权重 |
| `threads` | 当前进程的 Torch 线程数；`None` 保留当前值 |

模型进入 eval 模式，使用 FP32 推理，并保留官方卷积后端设置。构造会关闭当前进程的 PyTorch TF32。重复使用实例可避免重复加载权重。

### 单次调用与结果

`extract(image, border=1, fill=None) -> StripResult`：

| 参数或字段 | 含义 |
|---|---|
| `image` 输入 | 文件路径或 `surfa.Volume`；支持 3D 和逐帧处理的 4D |
| `border` | SDT 阈值，单位 mm，默认 1 |
| `fill` | 掩膜外的强度；省略时为 `min(image.min(), 0)` |
| `result.image` | 掩膜外已填充的影像，保留原网格和几何 |
| `result.mask` | 二值脑掩膜 |
| `result.distance` | 有符号距离场，单位 mm |

三个返回字段均为 `surfa.Volume`，可用 `.save(path)` 保存为 NIfTI、MGH、MGZ 等支持的格式。调用不会修改输入对象。直接使用 Python 保存时，由调用者准备输出父目录。

## 命令行

```bash
fs-torch synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz --weights /path/to/weights --device cuda:0
```

对应的 FreeSurfer 原版指令为：

```bash
mri_synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz
```

两条命令的 `-o`、`-m`、`-d` 分别保存脑图、掩膜和毫米单位的距离场；Python 返回字段依次为 `result.image`、`result.mask`、`result.distance`。原版 `-g` 选择可见 GPU，统一 CLI 用 `--device cuda:N` 指定设备；原版 `-t` 和 `--model` 分别对应统一 CLI 的 `-j` 与 `--weights`。

| CLI 参数 | 对应 API / 行为 |
|---|---|
| `-i`, `--image` | 输入影像，必需 |
| `-o`, `--out` | `result.image` |
| `-m`, `--mask` | `result.mask` |
| `-d`, `--sdt` | `result.distance` |
| `--weights` | PT 文件或目录 |
| `--device` | 默认 `cpu` |
| `--no-csf` | `no_csf=True` |
| `-b`, `--border` | 默认 1 mm |
| `-f`, `--fill` | 可选背景强度 |
| `-j`, `--threads` | Torch 线程数，统一 CLI 默认 4 |

至少指定一个输出。统一 CLI 会创建输出父目录。为兼容早期脚本，还保留 `python -m freesurfer_torch.synthstrip`，其参数接近官方命令，使用 `-g` 选择 CUDA、`-t` 设线程、`--model` 传权重文件；它与统一 `fs-torch` 的参数名不同，详情见各自 `--help`。

## 多被试 Python

`predict_batch(table, border=1, fill=None, workers=1, threads_per_worker=1)` 接受恰有 `input`、`output` 两列的 pandas 表。`input` 填入影像路径；`output` 是不带扩展名的绝对路径前缀，含被试 base name，不是目录。每行生成 `<output>_brain.nii.gz`、`<output>_mask.nii.gz` 和 `<output>_sdt.nii.gz`，按行顺序返回路径字典列表，键为 `image`、`mask`、`distance`。

```python
from pathlib import Path
import pandas as pd
from freesurfer_torch import SynthStrip

table = pd.DataFrame({
    "input": ["/data/sub-01_T1w.nii.gz", "/data/sub-02_T1w.nii.gz"],
    "output": ["/results/sub-01", "/results/sub-02"],
})
if __name__ == "__main__":
    extract = SynthStrip(device="cuda:0")
    saved: list[dict[str, Path]] = extract.predict_batch(table, workers=2)
    print(saved[0]["image"], saved[0]["mask"], saved[0]["distance"])
```

默认 `workers=1` 逐例复用模型；`workers=2` 在同一设备上启用两个 Python 进程，各加载一份模型，每个进程仍逐例推理。脚本中的多进程调用需放在 `if __name__ == "__main__":` 下。共享路径规则见[批量执行说明](../ARCHITECTURE.md#批量执行)。

### 实验性 B2 对照

在 gpucw1 的一张共享 H100 上，用相同的 12 例输入保存全部三项输出。B1 是单个常驻 Python 程序逐例运行，B2 使用未发布的实验代码在单个常驻程序中合批运行（12 例均实际进入 B=2 网络批），P2 是两个独立常驻 Python 程序各按 B=1 运行。正序和逆序各做一次 cold 与 warm 队列；下表是两轮 warm 队列总耗时的中位数。

| B1 | B2 | P2 |
|---:|---:|---:|
| 135.38 s | 145.61 s | 73.82 s |

本次 B2 慢于 B1，也慢于 P2。表中的 P2 由两个独立常驻脚本运行，并非当前 `workers=2` API 的实测；该对照衡量多被试队列吞吐。完整条件与逐轮结果见[批量性能报告](../../benchmark/batch_modes_2026-09-24.md)。

公开 Python 表格接口另在同一台 gpucw1 对 12 例做了四组配对调用：`workers=1/2` 的整队列耗时中位数为 **129.19/77.98 s**，观察到 **1.66 倍**吞吐差；四组的 144 个输出文件对逐字节相同。`workers=2` 的计时已包含每次新建子进程并重载其模型。逐轮数据及共享 GPU 负载见[Python 接口验证](../../benchmark/batch_modes_2026-09-24.md#python-table-api-with-two-processes)。

## 权重与 CPU/GPU 分工

| 文件 | 用途 |
|---|---|
| `synthstrip.1.pt` | 默认脑提取 |
| `synthstrip.nocsf.1.pt` | `no_csf=True` |

通过 `checkpoint["model_state_dict"]` 严格加载，无权重转换或精度压缩。下载、许可和 SHA-256 见 [WEIGHTS.md](../WEIGHTS.md)。推理使用本地权重，不调用 FreeSurfer 命令。

U-Net 在所选设备执行。影像读写、Surfa conform/crop、归一化、SDT 扩展、连通域和最终重采样在 CPU 执行。单例的 4D 输入逐帧处理；`predict_batch()` 按表行处理，每次网络推理 B=1，`workers=2` 时由两个进程并行处理。GPU 可加快网络部分，完整进程耗时还取决于预后处理和 I/O。

## 源码组织

| 文件 | 责任 |
|---|---|
| [model.py](../../src/freesurfer_torch/synthstrip/model.py) | `ConvBlock`、`StripModel`，保留官方网络和参数名 |
| [pipeline.py](../../src/freesurfer_torch/synthstrip/pipeline.py) | `SynthStrip`、`StripResult`、`extend_sdt` 及兼容 CLI |
| [__init__.py](../../src/freesurfer_torch/synthstrip/__init__.py) | 功能公开导出 |
| [__main__.py](../../src/freesurfer_torch/synthstrip/__main__.py) | 兼容模块执行入口 |

## 网络

- 输入为 `[N, 1, X, Y, Z]` float32，默认 API 每次推理一个 frame。
- 3D U-Net 共 7 个分辨率层级、6 次 `MaxPool3d(2)`。
- 编码器每级两次 `Conv3d(kernel=3, stride=1, padding=1)`；通道依次为
  16、32、64、64、64、64。
- 解码器每级两次卷积，然后最近邻 2 倍上采样，与相应编码器特征在通道维拼接。
  最后保留两层 16 通道卷积，再输出一个通道。
- 中间激活均为 `LeakyReLU(0.2)`；最后一层不加激活。没有 BatchNorm 或 Dropout。
- 默认网络共 **2,566,561 个可训练参数**、27 个卷积层；预测的是有符号距离，
  不直接输出 softmax 分割。保留原网络可选 `return_mask=True` 架构入口，官方 SDT
  权重和高级 API 使用默认 `False`。
- `encoder.*`、`decoder.*`、`remaining.*` 参数名称与官方权重一致。

## 每帧处理

1. 使用原影像几何将输入变换至 LIA 方向、1 mm 各向同性、float32，使用最近邻重采样。
2. 按非零包围盒裁剪；各轴尺寸向上取 64 的倍数，再限制在 192–320；居中裁剪或补零。
   此上限是官方行为，大视野超过 320 mm 的部分可能被裁掉。
3. 减去全图最小值，除以第 99 百分位数，限制到 `[0,1]`。这里的百分位数包含零值。
4. U-Net 预测窄带 signed distance transform (SDT)。`border` 大于窄带范围时，按官方
   `extend_sdt` 重算外部距离；负值内部预测保持不变。
5. SDT 线性重采样回原始体素空间，视野外填 100。以 `SDT < border` 得到 mask，
   保留最大连通域并填洞。默认 `border=1` mm；`--no-csf` 切换另一份官方权重。
6. 原影像 mask 外填 `min(image.min(), 0)`，也可显式提供 `fill`。4D 输入逐帧执行，
   再恢复 frame 维；原数据几何、体素格和有效区强度保留。

算法沿用官方输入范围：非有限强度、纯常数图像等没有增设修复规则。若归一化分母
为零，不能将所得输出视为有效提取结果。

## 功能差异与验证

官方脚本集成了参数解析和执行流程，本包允许导入并缓存模型；文件格式和几何处理仍使用独立的 Surfa 库。日志、版本和帮助格式由本包维护，未要求与 FreeSurfer 逐字一致。当前统一 CLI 名称为 `fs-torch synthstrip`，不会替换系统 `mri_synthstrip`。

以下数值来自 **0.1.0 参考实验**：12 例真实 T1w 在同设备对照中，脑图、掩膜和距离场全部逐元素一致；跨 CPU/GPU 时仅一个病例出现 1 个掩膜体素差异，最低 Dice 为 `0.999999858562`。0.2.0 是结构重整，单独的回归记录见 [refactor/report.public.json](../../validation/refactor/report.public.json)，历史运行时间未据此重新命名。

12 例完整单例命令的耗时中位数如下，单位为秒；计时包含启动、权重加载、推理和写盘。CPU 固定 8 线程，GPU 使用同一张 H100，四臂均关闭 TF32。原版 GPU 指未修改官方脚本在 CUDA Python 环境运行；已安装的 FreeSurfer 自带 PyTorch 仅支持 CPU。[完整四分位数与逐例条件](../COMPARISON.md#cpugpu-时间)。

| 原版 CPU | 本包 CPU | 原版 GPU | 本包 GPU |
|---:|---:|---:|---:|
| 16.92 | 16.96 | 16.92 | 18.27 |

以下使用同一份去面容的公开 `sub-02` T1w 输入和官方权重，展示原版与本包的脑图。两行分别为轴位和冠状位；三个面板使用同一切面及灰度范围。图片的制作步骤与完整影像比较见[图示记录](../figures/README.md)。

![公开 T1w 输入、FreeSurfer 脑图与本包脑图](../figures/synthstrip_comparison.png)

| 检查 | 记录 |
|---|---|
| 同设备官方类与本包类、默认/8 mm/no-CSF/4D 分支 | [模板验证](../../validation/synthstrip_fp32/synthstrip_validation.json) |
| 原版 CPU 与本包 CPU | [CPU 验证](../../validation/synthstrip_cpu/synthstrip_validation.json) |
| 12 例真实 T1w、四组 CPU/GPU 计时与数值比较 | [匿名数据](../../benchmark/summary.public.json)、[比较报告](../COMPARISON.md) |
| 0.1.0 旧 `BatchRunner` 输出与单例输出一致性 | [历史真实批量比较](../../benchmark/real_batch/comparison.public.json) |

测试源码在 [tests/synthstrip/](../../tests/synthstrip/)；原版对照工具在 [tools/validate_synthstrip.py](../../tools/validate_synthstrip.py)：

```bash
python -m pytest tests/synthstrip tests/test_public_api.py
# 只有与原版比较时才需要 FreeSurfer。
module load freesurfer
python tools/validate_synthstrip.py --image /path/to/test_T1w.nii.gz \
  --out-dir validation/synthstrip --device cuda --reference-device cpu
```

该工具从指定原脚本 AST 提取网络类，用相同权重比较参数名、参数量和随机 `64³` 输入的预测，再执行完整影像流程；正式流程仍使用官方最小 `192³` 网格。参考安装自带的 Torch 是 CPU build，因此历史 GPU 原版参考使用未修改官方脚本与 CUDA Python 环境，报告明确标为 `official_source_cuda`。

这些检查衡量与原版的数值一致性。真实病例没有人工脑掩膜真值，无法从中得出临床提取准确率。大视野裁切和常数输入的处理见上文。

原方法：Hoopes et al., *SynthStrip: Skull-Stripping for Any Brain Image*, NeuroImage (2022), [doi:10.1016/j.neuroimage.2022.119474](https://doi.org/10.1016/j.neuroimage.2022.119474)。
