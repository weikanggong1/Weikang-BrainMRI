# 独立 33 类 SynthSeg

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

## 命令行

```bash
fs-torch synthseg --i sub-01_T1w.nii.gz --o sub-01_synthseg.nii.gz \
  --csv-vols sub-01_synthseg.vol.csv --device cuda:0 --threads 4
```

可选参数为 `--weights /path/to/weights`、`--keep-geometry` 和 `--color-lut /path/to/FreeSurferColorLUT.txt`。命令行每次处理一幅图像；Python 可复用同一个模型依次处理多幅。独立入口不依赖 recon-all 的原生运行包或个人 license。
