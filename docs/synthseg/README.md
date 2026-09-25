# 独立 33 类 SynthSeg

[返回首页](../../README.md) · [源码目录](../../src/freesurfer_torch/synthseg_parc/) · [权重](../WEIGHTS.md) · [验证记录](../../validation/recon_all/)

`SynthSeg` 使用 FreeSurfer 8.2 的非 robust、非 parcellated SynthSeg 2.0 模型，从单幅 T1 生成 33 类结构标签和各结构软体积。它与 WMH-SynthSeg 是不同模型；本入口不输出 WMH 标签，也不生成皮层分区。推理使用 PyTorch，不需要安装 FreeSurfer、FSL 或 TensorFlow。

## 下载模型

在仓库根目录运行，或安装后用 `fs-torch-setup-weights` 代替 `python tools/setup_weights.py`。配置脚本下载固定的 `.h5` 和三个 `.npy` 文件，逐个核对大小与 SHA-256，然后记录权重目录。推理本身不联网。

```bash
python tools/setup_weights.py --model synthseg --dest /path/to/weights
```

这个组仅需四个文件，总计 53,087,016 字节，不需要下载 790 MB 的 WMH-SynthSeg 权重。已有模型时可用 `--verify-only` 检查。完整文件名、官方地址及哈希见[权重说明](../WEIGHTS.md)。

## Python API

```python
from freesurfer_torch import SynthSeg

model = SynthSeg(device="cuda:0", threads=4)
result = model("sub-01_T1w.nii.gz")
result.segmentation.save("sub-01_synthseg.nii.gz")
result.write_volumes_csv("sub-01_T1w.nii.gz", "sub-01_synthseg.vol.csv")
print(result.total_intracranial_mm3, result.volumes_mm3)
```

`SynthSeg(weights=None, device="cpu", threads=None)` 在构造时加载一次模型，适合顺序处理多幅图像。`weights` 可传包含四个文件的目录，或直接传 `synthseg_2.0.h5` 路径；三个 `.npy` 必须与这份 `.h5` 位于同一目录。省略 `weights` 时先查 `FREESURFER_TORCH_WEIGHTS`、配置脚本记录的目录，再查默认缓存。输入是单幅 3D `.nii`、`.nii.gz` 或 `.mgz` T1 路径。

`result.segmentation` 是 `surfa.Volume`，默认位于 SynthSeg 预处理后的 RAS 方向、约 1 mm 网格。`model(image, keep_geometry=True)` 会将标签以最近邻法重采样到输入网格。`color_lut="/path/to/FreeSurferColorLUT.txt"` 可选地附加色表；默认不读取 FreeSurfer 文件。

`result.volumes_mm3` 是 `{前景标签编号: 软体积}`，`result.total_intracranial_mm3` 是所有前景软体积之和，均在预处理后的网格计算并保留三位小数；`result.label_names` 对应 32 个前景结构名。CSV 列顺序、总量和近似并列标签规则与本仓库 GPU recon-all 的 `mri_synthseg` 入口相同。`result.near_tie_voxels` 记录近似并列规则相对于普通 `argmax` 更改的体素数。

## 原版命令与参数对应

本模块对应 FreeSurfer 8.2 `mri_synthseg` 的单幅、SynthSeg 2.0、非 robust、非
parcellated 路径。原版单例命令为：

```bash
mri_synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --vol sub-01_synthseg.vol.csv --threads 4
```

本包执行同一输入输出角色的命令为：

```bash
fs-torch synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

| 本包参数 | 原版参数 | 输入或输出 |
|---|---|---|
| `--i` | `--i` | 一幅 3D T1；本包接受 `.nii`、`.nii.gz` 或 `.mgz` 文件 |
| `--o` | `--o` | 33 类硬分割图；默认位于预处理后的 RAS、约 1 mm 网格 |
| `--csv-vols` | `--vol` | 前景结构软体积和 total intracranial volume，单位 mm³ |
| `--weights` | `--model` | 官方 `synthseg_2.0.h5`；本包还从同目录读取三份标签、名称和拓扑 `.npy` |
| `--device cpu` | `--cpu` | 强制 CPU；本包也可用 `--device cuda:N` 选择具体 GPU |
| `--threads` | `--threads` | PyTorch CPU 线程数；本包 CLI 默认 4，原版默认 1 |
| `--keep-geometry` | `--keepgeom` | 以最近邻法把硬标签重采样回输入网格 |
| `--color-lut` | `--addctab` / `--noaddctab` | 本包只在显式提供 FreeSurfer LUT 时附加色表；原版默认附加色表 |

原版还支持目录输入、robust、皮层分区、QC、posterior、CT、Photo-SynthSeg 和其他
模型路径；本接口没有实现这些选项，也不会把它们近似为 33 类单幅 T1 路径。

## 命令行

```bash
fs-torch synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

可选参数为 `--weights /path/to/weights`、`--keep-geometry` 和 `--color-lut /path/to/FreeSurferColorLUT.txt`。命令行每次处理一幅图像；Python 可复用同一个模型依次处理多幅。独立入口不依赖 recon-all 的原生运行包或个人 license。

## 多被试 Python

独立 `SynthSeg` 当前没有多被试命令行或 `predict_batch()`。在 Python 中构造一次
模型，再按病例循环，可避免重复加载权重：

```python
from pathlib import Path
from freesurfer_torch import SynthSeg

inputs = sorted(Path("/data/t1w").glob("*_T1w.nii.gz"))
output_dir = Path("/results/synthseg")
output_dir.mkdir(parents=True, exist_ok=True)

model = SynthSeg(device="cuda:0", threads=4)
for image in inputs:
    subject = image.name.removesuffix("_T1w.nii.gz")
    result = model(image)
    result.segmentation.save(output_dir / f"{subject}_synthseg.nii.gz")
    result.write_volumes_csv(
        image, output_dir / f"{subject}_synthseg.vol.csv"
    )
```

这个循环在一个 Python 进程和一张设备上顺序处理病例。它不启动隐藏的批量 CLI，
也不把多例合成一个网络 batch。

## 验证边界

公开 CLI、Python API 和 recon-all 内部启动器调用同一套 33 类推理与软体积代码，
相应接口测试见 [`tests/synthseg_parc/`](../../tests/synthseg_parc/)。
[外置模型检查](../../validation/recon_all/external_models_2026-09-25.md)记录：在
`sub-01` 的同一 `orig.mgz` 上，独立 `fs-torch synthseg` 与 v0.6 recon-all
SynthSeg 输出的硬标签和软体积 CSV 相同，硬标签变化体素为 0。

完整 recon-all 与 FreeSurfer 8.2 的 `sub-01` 比较中，硬分割及下游检查通过；
SynthSeg 软体积通过预设字段容差，但并非逐位相同。独立的 `sub-02` 检查中，硬分割
相同，CSF 软体积差为 389.03125 mm³，超过固定的 360.4035 mm³ 上限。现有证据
不支持多病例逐体素等价结论；范围和数值见
[单例记录](../../validation/recon_all/gpucw1_sub01_2026-09-24.md)及
[批量集成记录](../../validation/recon_all/gpucw1_batch_integration_2026-09-24.md)。
