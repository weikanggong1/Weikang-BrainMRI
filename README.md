# Weikang-BrainMRI

这个独立 Python 包提供 **SynthStrip 脑提取**、**SynthMorph 配准**、**WMH-SynthSeg 脑结构及白质高信号分割**和 **SynthSR 1 mm T1w 合成**。单例可在 CPU 或 CUDA 上运行；批量任务可分配到多张 GPU。推理无需安装 FreeSurfer、TensorFlow、VoxelMorph 或 Neurite。

仓库名为 `Weikang-BrainMRI`。安装包名 `freesurfer-torch`、Python 导入名 `freesurfer_torch` 和命令 `fs-torch` 保持已有接口不变。0.4.0 增加了 SynthSR；各功能分别存放源码、测试和说明。

| 功能 | 专属文档 | 实现目录 |
|---|---|---|
| 脑提取、脑掩膜、距离场 | [SynthStrip](docs/synthstrip/README.md) | [synthstrip/](src/freesurfer_torch/synthstrip/) |
| 刚性、仿射、非线性、联合配准及应用变换 | [SynthMorph](docs/synthmorph/README.md) | [synthmorph/](src/freesurfer_torch/synthmorph/) |
| 脑结构及白质高信号分割 | [WMH-SynthSeg](docs/wmh_synthseg/README.md) | [wmh_synthseg/](src/freesurfer_torch/wmh_synthseg/) |
| 单幅 MRI/CT 合成 1 mm T1w | [SynthSR](docs/synthsr/README.md) | [synthsr/](src/freesurfer_torch/synthsr/) |
| 多 GPU / 同 GPU 多进程批量调度 | [批量使用与架构](docs/ARCHITECTURE.md#批量执行) | [batch.py](src/freesurfer_torch/batch.py) |

仓库附有 [3 例 T1w](examples/README.md) 和 [3 例 FLAIR](examples/WMH.md) 供直接试运行。它们来自 [OpenNeuro ds000114](https://openneuro.org/datasets/ds000114) 和 [ds003592](https://openneuro.org/datasets/ds003592) 的 CC0 影像；发布前清除了远离脑组织的影像强度。原图地址、处理过程及校验值见 [T1w 清单](examples/data/SOURCES.json) 和 [FLAIR 清单](examples/wmh_data/SOURCES.json)。

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

**Git 仓库和 wheel 均不包含模型权重。** 下列脚本从 FreeSurfer 官方地址下载文件、校验哈希，并保存权重目录；后续 Python API 和 CLI 会自动使用该目录。下面分别准备默认 SynthStrip、joint SynthMorph、WMH-SynthSeg 和默认 SynthSR 所需的文件：

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint \
  --model wmh-synthseg --model synthsr
```

`python tools/setup_weights.py --all` 会配置全部官方权重。安装后也可运行 `fs-torch-setup-weights`。文件默认存入用户缓存目录；`--dest /path/to/weights` 可改下载位置，`--verify-only` 可检查已有文件。调用时可通过 Python 的 `weights=`、CLI 的 `--weights` 或环境变量 `FREESURFER_TORCH_WEIGHTS` 指定另一目录。推理过程不会联网下载。各权重的地址、版本、SHA-256 和许可见[权重说明](docs/WEIGHTS.md)。

## 单例 Python 调用：输入、输出和每步作用

先配置权重，并将下面的示例路径换成自己的文件。Python API 接受文件路径或 `surfa.Volume`，返回对象；只有调用 `.save()` 才写文件。SynthStrip 保留输入网格，SynthMorph 返回目标网格中的重采样图，WMH-SynthSeg 和 SynthSR 输出处理后的 1 mm 网格。前三个功能保存前需创建输出目录；SynthSR 的 `.save()` 会自动创建父目录。

```python
from pathlib import Path
from freesurfer_torch import SynthStrip, SynthMorph, apply_transform

out = Path("results")
out.mkdir(parents=True, exist_ok=True)          # Python API 不会自动创建输出目录。

strip = SynthStrip(device="cuda:0")              # 加载官方 SynthStrip 权重至 GPU 0；只加载一次。
brain = strip("subject_T1w.nii.gz")              # 输入一幅 T1w；预测距离场并生成脑掩膜和去颅骨图像。
brain.image.save(out / "subject_brain.nii.gz")  # 保存原网格上的脑图，对应原版 -o。
brain.mask.save(out / "subject_mask.nii.gz")    # 保存原网格上的二值掩膜，对应原版 -m。
brain.distance.save(out / "subject_sdt.nii.gz") # 保存毫米单位的有符号距离场，对应原版 -d。

morph = SynthMorph(device="cuda:0", model="joint")  # 加载默认联合仿射与非线性配准权重。
reg = morph("subject_T1w.nii.gz", "template_T1w.nii.gz")
# 上行第一个输入是 moving（待配准图像），第二个是 fixed（目标模板）；两者都须为单帧 3D。
reg.moved.save(out / "subject_in_template.nii.gz")   # moving 重采样到 fixed 网格，对应原版 -o。
reg.fixed_moved.save(out / "template_in_subject.nii.gz") # fixed 重采样到 moving 网格，对应 -O。
reg.transform.save(out / "subject_to_template.mgz")  # moving→fixed 变换，对应原版 -t。
reg.inverse.save(out / "template_to_subject.mgz")    # fixed→moving 变换，对应原版 -T。

labels = apply_transform("subject_labels.nii.gz", reg.transform, method="nearest", dtype="int16")
# 将 moving 空间的离散标签随正向变换送至 fixed 空间；最近邻插值避免产生新标签值。
labels.save(out / "labels_in_template.nii.gz")      # 对应原版 mri_synthmorph apply。
```

`brain.image`、`brain.mask`、`brain.distance` 都是 `surfa.Volume`，保留输入网格。脑图保留掩膜内的原始强度，掩膜为二值，距离场以 mm 计；默认以 `distance < 1` 确定脑边界。掩膜外的脑图强度默认填 `min(输入影像最小值, 0)`。SynthStrip 可处理单帧 3D 或逐帧处理 4D；`strip(image, border=1, fill=0)` 可指定边界和背景值。

`reg.moved` 与 `reg.fixed_moved` 也是 `surfa.Volume`，分别位于 fixed 与 moving 网格。`reg.transform` 和 `reg.inverse` 记录源/目标几何：`joint`/`deform` 返回 RAS 位移场，建议保存为 `.mgz`；`affine`/`rigid` 返回仿射变换，建议保存为 `.lta`。Python 调用会计算两个方向，即使只保存其中一个。`apply_transform` 用 Surfa 在 CPU 上重采样已有变换；对离散标签使用 `method="nearest"`。同一个 `strip` 或 `morph` 实例可继续处理其他图像，无需重新加载权重。将 `device` 设为 `"cpu"` 可使用 CPU；使用 CUDA 时，网络及部分空间运算在 GPU 上执行，读写和最终 Surfa 重采样仍在 CPU 上。

WMH-SynthSeg 接受单幅 3D T1w 或 FLAIR。下面使用 FLAIR；同一个模型实例可依次处理多例，避免重复加载权重：

```python
from pathlib import Path
from freesurfer_torch import WMHSynthSeg

out = Path("results")
out.mkdir(parents=True, exist_ok=True)
wmh = WMHSynthSeg(device="cuda:0", threads=4)  # 加载官方权重，指定 GPU 和 CPU 线程数。
result = wmh("subject_FLAIR.nii.gz", crop=True, save_lesion_probabilities=True)
# crop=True 对应原版 --crop：先定位脑，再在最多 192×224×192 的区域内预测。
result.segmentation.save(out / "subject_wmh_seg.nii.gz")
# 保存 33 类标签图；标签 77 是白质高信号，对应原版 --o。
result.lesion_probability.save(out / "subject_wmh_seg.lesion_probs.nii.gz")
# 保存每体素 WMH 后验概率，对应原版 --save_lesion_probabilities。
print(result.volumes_mm3[77])  # WMH 软体积，单位 mm³，对应原版 CSV 的 WMH(77) 列。
```

`result.segmentation` 是 `surfa.Volume`；显式请求概率图时，`result.lesion_probability` 也是 `surfa.Volume`，否则为 `None`。两者位于处理后的 RAS、1 mm 网格，与原版输出位置相同，可能不同于输入网格。`result.volumes_mm3` 是各标签的软体积字典，由后验概率求和得到，不等于整数标签体素数；上面的 Python 调用不会写 CSV。文件格式与 `crop` 的细节见 [WMH-SynthSeg 说明](docs/wmh_synthseg/README.md)。

SynthSR 从单幅 MRI 或 CT 合成 1 mm T1w。这里以 FLAIR 为输入；模型构造一次后可继续处理其他病例：

```python
from freesurfer_torch import SynthSR

sr = SynthSR(device="cuda:0")              # 加载官方通用 v2 权重到 GPU 0。
synthetic = sr("subject_FLAIR.nii.gz")       # 对应原版 mri_synthsr --i；输出 1 mm 合成 T1w。
synthetic.image.save("results/subject_synthsr.nii.gz")  # 对应原版 --o。
print(synthetic.image.data.shape, synthetic.image.affine)
```

`synthetic.image.data` 是 3D `uint8` 数组，`synthetic.image.affine` 是输出的 RAS 仿射矩阵；1 mm 输出通常与原图不同形状。`SynthSR(weights=None, device="cpu", lowfield=False, v1=False, threads=None)` 可选择权重、低场或 v1 模型、设备和 CPU 线程；`sr(image, ct=False, disable_flipping=False, disable_sharpening=False)` 对应原版的 CT 截断、翻转推理和锐化开关。输入支持 `.nii`、`.nii.gz`、`.mgz`、`.npz` 路径或 `surfa.Volume`；`.npz` 的原版输出规则与 NIfTI/MGZ 不同，见 [SynthSR 说明](docs/synthsr/README.md#python-输入与输出)。

| Python 输入与类型 | 返回字段与类型 | 原版文件指令 |
|---|---|---|
| `strip(image, border=1, fill=None)`：3D/4D NIfTI 等文件路径或 `surfa.Volume` | `image`、`mask`、`distance`：原输入网格的 `surfa.Volume`，距离单位 mm | `mri_synthstrip -i ... -o ... -m ... -d ...` |
| `morph(moving, fixed, init=None, mid_space=False, header_only=False)`：两幅单帧 3D 路径或 `surfa.Volume` | `moved`、`fixed_moved`：各自目标网格的 `surfa.Volume`；`transform`、`inverse`：带几何的仿射或 RAS 位移场 | `mri_synthmorph register moving fixed -o ... -O ... -t ... -T ...` |
| `apply_transform(image, transformation, method="linear", fill=0, dtype="float32")`：3D/4D 影像及已有变换 | 重采样后的 `surfa.Volume` | `mri_synthmorph apply transform image output` |
| `wmh(image, crop=False, save_lesion_probabilities=False)`：3D T1w/FLAIR 路径或 `surfa.Volume` | `segmentation`：33 类标签图；按需返回 `lesion_probability`；`volumes_mm3`：各类软体积 | `mri_WMHsynthseg --i ... --o ... [--crop] [--save_lesion_probabilities] [--csv_vols ...]` |
| `sr(image, ct=False, disable_flipping=False, disable_sharpening=False)`：单幅 3D MRI/CT 路径或 `surfa.Volume` | `image`：含 `uint8` 体素、1 mm 网格仿射矩阵和 `.save(path)` 的 `SynthSRImage` | `mri_synthsr --i ... --o ... [--ct] [--disable_flipping] [--disable_sharpening]` |

`morph` 的正向变换把 moving 的影像或标签送到 fixed 空间，反向变换用于相反方向。各功能的其余参数与几何约定见上表链接的专属文档。

## 单例命令行：与 FreeSurfer 原指令逐项对应

下列新旧指令均使用相同的输入影像和官方权重。`fs-torch` 使用本包独立的命令入口；原版指令只在安装并加载 FreeSurfer 的环境中可用。示例中 `subject_T1w.nii.gz` 是 moving，`template_T1w.nii.gz` 是 fixed。

```bash
fs-torch synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz --device cuda:0

fs-torch synthmorph subject_T1w.nii.gz template_T1w.nii.gz \
  -m joint -o results/subject_in_template.nii.gz \
  -O results/template_in_subject.nii.gz \
  -t results/subject_to_template.mgz -T results/template_to_subject.mgz \
  --device cuda:0

fs-torch apply results/subject_to_template.mgz subject_labels.nii.gz \
  results/labels_in_template.nii.gz --method nearest --dtype int16

fs-torch wmh-synthseg --i subject_FLAIR.nii.gz \
  --o results/subject_wmh_seg.nii.gz --device cuda:0 --threads 4 --crop \
  --save_lesion_probabilities --csv_vols results/subject_wmh_volumes.csv

fs-torch synthsr --i subject_FLAIR.nii.gz \
  --o results/subject_synthsr.nii.gz --device cuda:0 --threads 4
```

这五条命令依次完成：

1. `synthstrip` 读取 `-i` 指定的 T1w，分别用 `-o`、`-m`、`-d` 保存脑图、掩膜和距离场；至少指定一个输出。`--device cuda:0` 选择 GPU 0。
2. `synthmorph` 把第一个位置参数 moving 配准到第二个位置参数 fixed。`-m joint` 选择仿射加非线性模型，`-o/-O` 保存两个方向的重采样图，`-t/-T` 保存正反变换；`--device` 选择网络运行设备。
3. `apply` 用正向变换把 moving 空间的标签映射到 fixed 空间。`--method nearest` 保持标签值，`--dtype int16` 指定输出类型；已有变换的重采样在 CPU 上执行。
4. `wmh-synthseg` 从 `--i` 读取 3D FLAIR，向 `--o` 写解剖与 WMH 标签。`--crop` 先定位脑再裁出推理区域，`--save_lesion_probabilities` 另写 `subject_wmh_seg.lesion_probs.nii.gz`，`--csv_vols` 写软体积表；`--device` 与 `--threads` 选择设备和 PyTorch CPU 线程。
5. `synthsr` 从 `--i` 读取单幅 3D FLAIR，用通用 v2 模型生成 1 mm T1w，并由 `--o` 写出 `uint8` 图像；`--device` 选择 GPU，`--threads` 设置 PyTorch CPU 线程。`--lowfield`、`--v1`、`--ct`、`--disable_flipping`、`--disable_sharpening` 和 `--model` 对应原版同名选项。

CLI 会创建输出父目录；更多参数见 `fs-torch --help`。

原版 FreeSurfer 的对应指令为：

```bash
mri_synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz

mri_synthmorph register -m joint \
  -o results/subject_in_template.nii.gz -O results/template_in_subject.nii.gz \
  -t results/subject_to_template.mgz -T results/template_to_subject.mgz \
  subject_T1w.nii.gz template_T1w.nii.gz

mri_synthmorph apply -m nearest -t int16 \
  results/subject_to_template.mgz subject_labels.nii.gz \
  results/labels_in_template.nii.gz

mri_WMHsynthseg --i subject_FLAIR.nii.gz \
  --o results/subject_wmh_seg.nii.gz --device cpu --threads 4 --crop \
  --save_lesion_probabilities --csv_vols results/subject_wmh_volumes.csv

mri_synthsr --i subject_FLAIR.nii.gz \
  --o results/subject_synthsr.nii.gz --threads 4
```

原版 `mri_synthstrip` 的 `-i/-o/-m/-d` 分别对应新 CLI 的同名参数和 Python 的 `StripResult.image/mask/distance`。原版 `mri_synthmorph register` 的 `-o/-O/-t/-T` 对应 `RegistrationResult.moved/fixed_moved/transform/inverse`；`register` 一词在原版中可省略。原版 `apply -m nearest -t int16` 对应新 CLI 的 `apply --method nearest --dtype int16`。原版一条 `apply` 指令可处理多组影像，新 CLI 每次处理一组，Python 可循环。WMH 两版的 `--i/--o/--crop/--save_lesion_probabilities/--csv_vols` 对应相同文件。gpucw1 上安装的原生 `fspython` 只有 CPU 版 PyTorch；CUDA 对照使用相同权重和**未改动的官方 `inference.py`**，在 CUDA PyTorch 环境中运行。SynthSR 两版的 `--i/--o` 也相同；原版 TensorFlow 自动选择可用 GPU，本包用 `--device cuda:N` 指定设备，`--cpu` 可强制使用 CPU。

| 其余原版选项 | 本包对应 | 说明 |
|---|---|---|
| SynthStrip `-g`、`-t`、`--model` | `--device cuda:0`、`-j`、`--weights`；Python `device`、`threads`、`weights` | 原版 `-g` 只表示使用可见 GPU；本包显式选择设备。原版 `--model` 是单个 PT 文件。|
| SynthStrip `--no-csf`、`-b`、`-f` | 同名 CLI 选项；Python `no_csf`、`border`、`fill` | 默认边界 1 mm，默认背景填充值相同。|
| SynthMorph `-m`、`-r`、`-n`、`-e` | 同名短选项；Python `model`、`hyper`、`steps`、`extent` | 默认分别为 joint、0.5、7、256；`deform` 要求事先对齐或使用初始仿射。|
| SynthMorph `-i`、`-M`、`-H` | 同名短选项；Python `init`、`mid_space`、`header_only` | `-M` 需搭配 `-i`；`-H` 只适用于 affine/rigid。|
| SynthMorph `-g`、`-j`、可重复的 `-w` | `--device`、`-j`、`--weights`；Python `device`、`weights` | 新 CLI 从权重目录取所需 H5；原版 `-w` 可多次指定文件。原版 `-j` 管 TensorFlow 线程，新 CLI 管 Torch 线程。|
| SynthSR `--lowfield`、`--v1`、`--ct`、`--disable_flipping`、`--disable_sharpening`、`--model` | 同名选项；`--weights` 是本包 `--model` 的别名 | `--v1` 优先于 `--lowfield`；CT 输入须以 Hounsfield 单位保存。|

**与原版的输出比较。** SynthStrip 的三类输出保留原输入网格；12 例真实 T1w 的同设备脑图、掩膜和距离场均逐元素相同。SynthMorph 的 moving/fixed 顺序、输出方向和目标几何对应原版。TensorFlow 到 PyTorch 的浮点运算存在差异：12 例默认 joint 配准的变换最大差为 0.000790 mm，文件并非逐字节相同。原版普通配准只重采样请求保存的方向，Python API 会计算双向结果。原版 `-d` 调试目录生成 6 个文件，本包生成两幅网络输入和 `network_transforms.npz`；原版允许不指定保存输出，新 CLI 至少要求一个输出或调试目录。参考构建的原生 `-i` 初始化因 dtype 错误失败，该分支与两行修复后的参考源码比较。逐例结果及运行时间见[对照报告](docs/COMPARISON.md)。

WMH-SynthSeg 的 `--i`、`--o`、可选 `.lesion_probs` 概率图和 `--csv_vols` 软体积表与原版对应，输出位于处理后的 RAS/1 mm 网格。12 例公开 FLAIR 中，**CPU 对 CPU**、**CUDA 对 CUDA** 的标签、概率体素、数值仿射及 CSV 软体积均与原版相同。完整命令的时间中位数为：原版 CPU **97.38 s**、本包 CPU **70.69 s**；原版源码 CUDA **8.25 s**、本包 CUDA **8.41 s**。原生 CPU 与本包使用不同 PyTorch 版本，不能只凭时间差判断算法加速。两版 NIfTI 的 qform/sform code 可能不同，因此文件字节不一定相同。病例选择、环境、逐例结果和复现命令见[WMH 验证记录](validation/wmh/README.md)。

SynthSR 在 gpucw1 上以同一官方通用 v2 权重验证了 12 例真实 T1w。原版 CPU/GPU 与本包 CPU/GPU 四组输出的形状、仿射和 `uint8` 类型均一致；本包 GPU 对原版 CPU 的逐例体素完全一致比例不低于 **99.992%**，最大差值 1 灰度级。完整单例命令的时间中位数依次为原版 CPU **103.60 s**、原版 GPU **53.27 s**、本包 CPU **42.32 s**、本包 GPU **13.80 s**。不同框架的启动、模型加载及共享节点负载都进入该计时；详细方法、匿名统计和复现命令见[SynthSR 验证记录](validation/synthsr/README.md)。

## 公开样例与原版对照图

从仓库根目录运行 `python examples/check_data.py` 可核对三个 T1w 文件。按[权重文档](docs/WEIGHTS.md)准备官方权重后，[示例说明](examples/README.md)给出 CPU 单例、双 GPU 批量命令及各输出文件的位置。下面的图使用同一份去面容的 `sub-02` 输入和相同官方权重，分别运行 FreeSurfer 8.2.0 原版与本包 0.2.0；`sub-01` 是配准的 fixed 图像。图片是实际程序输出，制作步骤与逐文件比较见 [图示记录](docs/figures/README.md)。

![同一 T1w 的原图、FreeSurfer 脑提取结果和 PyTorch 脑提取结果](docs/figures/synthstrip_comparison.png)

上图从左到右为去面容输入、FreeSurfer 提取的 brain、本包提取的 brain；两行展示轴位与冠状位。各列使用相同体素切面和灰度范围，定量掩膜比较见[图示记录](docs/figures/README.md)。

![FreeSurfer 与 PyTorch 非线性配准后的脑图并排比较](docs/figures/synthmorph_comparison.png)

上图展示 moving、fixed，以及两种实现的 joint 配准结果。两列配准结果都位于 fixed 网格；图中仅为展示使用同一 fixed 脑掩膜，数值误差在完整保存的影像和形变场上计算。两次推理使用相同输入与权重，原版在 CPU、本包在 GPU；该图不用于比较运行速度。

WMH-SynthSeg 的公开 FLAIR 从仓库根目录运行 `python examples/check_wmh_data.py` 校验；[专属示例步骤](examples/WMH.md)给出单例和双 GPU 批量命令。下图的 `sub-04` 是同一份发布的脑外清零 FLAIR，左列为输入，中、右列分别为原版 FreeSurfer 与本包的标签 77（红色）叠加结果。两者都在 CUDA 上使用官方权重及 `--crop`；完整三维标签、概率图、软体积和仿射矩阵一致。图示与复现命令见[图示记录](docs/figures/README.md)。

![同一 FLAIR 上 FreeSurfer 与本包 WMH-SynthSeg 的病灶标签对照](docs/figures/wmh_synthseg_comparison.png)

SynthSR 也可直接使用这三例公开 FLAIR。下图在相同 RAS 切面显示 `sub-04` 输入、FreeSurfer 原版合成 T1w 与本包 PyTorch 合成 T1w；后两列使用相同灰度范围。完整三维比较和图像生成命令见[SynthSR 专属说明](docs/synthsr/README.md#对照验证)。

![公开 FLAIR 与 FreeSurfer、PyTorch SynthSR 合成 T1w 对照](docs/synthsr/figures/synthsr_flair_comparison.png)

## 多病例批量并行

批量调度以**一例影像调用一次功能**为单位，不把多例拼成一个网络 tensor。每个 worker 是绑定一张 GPU 的独立进程，处理下一例时可复用已加载的模型。下面提交五项互不依赖的任务：三例 T1w 各做一次脑提取，`sub-02/03` 另配准到 `sub-01`。配置权重后，将代码保存为仓库根目录的 `run_many.py` 并运行 `python run_many.py`；仓库也提供等价的 [examples/run_batch.py](examples/run_batch.py)：

```python
from freesurfer_torch import BatchRunner


def main():
    subjects = ("sub-01", "sub-02", "sub-03")
    template = "examples/data/sub-01_T1w.nii.gz"
    jobs = []
    for subject in subjects:
        moving = f"examples/data/{subject}_T1w.nii.gz"
        jobs.append({
            "task": "synthstrip",
            "kwargs": {"image": moving},
            "outputs": {
                "image": f"examples/results/readme_python/{subject}_brain.nii.gz",
                "mask": f"examples/results/readme_python/{subject}_mask.nii.gz",
            },
        })
        if subject != "sub-01":
            jobs.append({
                "task": "synthmorph",
                "model": {"model": "joint"},
                "kwargs": {"moving": moving, "fixed": template},
                "outputs": {
                    "moved": f"examples/results/readme_python/{subject}_in_sub-01.nii.gz",
                    "transform": f"examples/results/readme_python/{subject}_to_sub-01.mgz",
                },
            })

    with BatchRunner(devices=("cuda:0", "cuda:1"),
                     workers_per_device=1, threads_per_worker=4) as runner:
        results = runner.run(jobs)
    for result in results:
        print(result.index, result.task, result.device, result.outputs, result.error)
    if any(not result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

每项 job 的 `task` 指定功能，`kwargs` 是单例调用参数，`outputs` 把结果属性映射到文件路径。`model` 是构造模型的参数；这里的 `"model": "joint"` 表示 SynthMorph 在估计的仿射中间空间进行非线性配准，并输出组合变换。`sub-01` 保存脑图和掩膜，`sub-02/03` 还保存配准影像和前向变换，合计十个文件。`BatchRunner` 启动两个带模型缓存的 worker；`runner.run(jobs)` 分发任务，按提交顺序返回 `BatchResult`。每项结果记录设备、进程号、起止时间、文件及错误，`result.ok` 表示成功。离开 `with` 时关闭 worker。多进程使用 `spawn`，脚本必须保留 `if __name__ == "__main__":`。

命令行直接读取仓库的 [examples/jobs.json](examples/jobs.json)：清单含三个 SynthStrip 任务和两个 joint SynthMorph 任务；后两项分别把 `sub-02/03` 的原始 T1w 配准到 fixed `sub-01`。它与上面的 Python 示例处理相同病例，结果保存到独立的 `examples/results/cli/`。从仓库根目录运行：

```bash
fs-torch batch examples/jobs.json --devices cuda:0 cuda:1 \
  --workers-per-device 1 --threads-per-worker 4 \
  --report examples/results/cli/batch_report.json
```

`examples/jobs.json` 提供全部任务和各自输出；`--devices` 指定可用 GPU，编号遵循 `CUDA_VISIBLE_DEVICES`；`--workers-per-device 1` 表示每张 GPU 一个进程；`--threads-per-worker 4` 限制每个进程的 PyTorch CPU 线程；`--report` 写逐任务 JSON 结果。任一任务失败时命令退出码非零。完成后可检查五项任务及十个输出：

```bash
python - <<'PY'
import json
from pathlib import Path
jobs = json.loads(Path("examples/jobs.json").read_text())
results = json.loads(Path("examples/results/cli/batch_report.json").read_text())
assert len(jobs) == len(results) == 5
for index, (job, result) in enumerate(zip(jobs, results)):
    assert result["index"] == index and result["task"] == job["task"]
    assert result["error"] is None, result["error"]
    assert set(result["outputs"]) == set(job["outputs"])
    assert all(Path(path).is_file() for path in result["outputs"].values())
print("5 jobs completed; 10 output files present")
PY
```

用 CPU 并行时将设备选项改为 `--devices cpu --workers-per-device 2`；Python 中对应 `devices=("cpu",), workers_per_device=2`。若先运行了 GPU 示例，应更换输出目录或显式允许覆盖。增加同 GPU worker 会各自加载模型并占用更多显存，不保证更快。

任务在同一批中没有先后依赖。如果要把**脑提取后的影像**用于配准，先完成所有 SynthStrip 任务，再以保存的 `_brain.nii.gz` 作为 `moving` 提交第二批 SynthMorph 任务；同一个 `BatchRunner` 可连续调用 `run`，复用 worker。默认拒绝覆盖已有文件，即使指定 `--overwrite` / `overwrite=True`，同一批也不能让两个任务写入同一路径。更多输出字段和错误处理见[批量使用与架构](docs/ARCHITECTURE.md#批量执行)。

WMH-SynthSeg 使用相同的 job 格式。下面读取 `inputs/` 中的每幅 3D FLAIR，在两张 GPU 上各启动一个 worker；每个 worker 加载一次 WMH 网络，再逐例处理分配给自己的影像：

```python
from pathlib import Path
from freesurfer_torch import BatchRunner


def main():
    jobs = []
    for image in sorted(Path("inputs").glob("*_FLAIR.nii.gz")):
        name = image.name.removesuffix(".nii.gz")
        jobs.append({
            "task": "wmh_synthseg",
            "kwargs": {"image": str(image), "crop": True},
            "outputs": {
                "segmentation": f"results/{name}_seg.nii.gz",
                "lesion_probability": f"results/{name}_seg.lesion_probs.nii.gz",
            },
        })
    with BatchRunner(devices=("cuda:0", "cuda:1"),
                     workers_per_device=1, threads_per_worker=4) as runner:
        reports = runner.run(jobs)
    if any(not report.ok for report in reports):
        raise RuntimeError([report.error for report in reports if not report.ok])


if __name__ == "__main__":
    main()
```

`kwargs` 中的 `crop=True` 对应原版 `--crop`；`outputs.segmentation` 对应 `--o`，`outputs.lesion_probability` 对应可选的 `.lesion_probs` 文件。`runner.run` 按提交顺序返回结果，空闲 worker 动态领取下一例。此批量调用只保存分割图与概率图；要保存 CSV 软体积，可读取上述 `WMHSynthSeg` Python 返回值，或逐例/目录运行 `fs-torch wmh-synthseg --csv_vols ...`。每个 GPU worker 都加载一份约 791 MB 的权重及网络中间激活，同一张卡上增加 worker 会增加显存占用。

可直接运行的三例 FLAIR 清单是 [examples/wmh_jobs.json](examples/wmh_jobs.json)：`fs-torch batch examples/wmh_jobs.json --devices cuda:0 cuda:1 --workers-per-device 1 --threads-per-worker 4 --report examples/results/wmh_batch/report.json`。在 gpucw1 上验证时，两张 H100 各执行至少一例，三例共六个输出影像均与相同输入的原版 CUDA 结果逐体素一致；清单字段、运行前检查和结果核验详见 [FLAIR 示例说明](examples/WMH.md)。

SynthSR 使用相同的调度器，任务名为 `synthsr`。以下脚本把已公开的三例 FLAIR 分别合成 T1w；每例的 `kwargs.image` 对应 `mri_synthsr --i`，`outputs.image` 对应 `--o`。两张 GPU 各启动一个进程，空闲进程领取下一例；每个进程只加载一次通用 v2 权重。

```python
from pathlib import Path
from freesurfer_torch import BatchRunner


def main():
    jobs = []
    for image in sorted(Path("examples/wmh_data").glob("*_FLAIR.nii.gz")):
        name = image.name.removesuffix(".nii.gz")
        jobs.append({
            "task": "synthsr",
            "kwargs": {"image": str(image)},
            "outputs": {"image": f"examples/results/synthsr/{name}_synthsr.nii.gz"},
        })
    with BatchRunner(devices=("cuda:0", "cuda:1"),
                     workers_per_device=1, threads_per_worker=4) as runner:
        reports = runner.run(jobs)
    if any(not report.ok for report in reports):
        raise RuntimeError([report.error for report in reports if not report.ok])


if __name__ == "__main__":
    main()
```

也可将相同任务保存为 JSON，使用 `fs-torch batch jobs.json --devices cuda:0 cuda:1 --workers-per-device 1 --threads-per-worker 4 --report results/report.json`。低场任务在该例的 `model` 中写 `{"lowfield": true}`；显卡由 `--devices` 分配，不写在单例任务里。任务字段及原版目录和 `.txt` 输入方式见 [SynthSR 专属说明](docs/synthsr/README.md#命令行与多病例)。

## 验证与维护

- [详细功能和数值对照](docs/COMPARISON.md)：0.1.0 参考实验包含 12 例真实 T1w、96 次单例运行及 24 个批量任务。该临床数据只发布匿名统计，不包含原始影像；仓库另附三例公开 OpenNeuro 衍生样例。
- [0.2.0 结构重整回归](validation/refactor/report.public.json)：新布局与 0.1.0 的对照记录；历史计时不能当作 0.2.0 的重新计时。
- [WMH-SynthSeg 0.3.0 对照](validation/wmh/README.md)：12 例公开 FLAIR 的原版 CPU/官方源码 CUDA 与本包 CPU/CUDA 逐例输出、时间和三例双 GPU 示例。
- [SynthSR 0.4.0 说明](docs/synthsr/README.md)与[验证记录](validation/synthsr/README.md)：原版指令、模型变体、输出格式、多 GPU 调用及 12 例四组计时和数值对照。
- [架构、公共 API 与批量任务格式](docs/ARCHITECTURE.md)。
- [新增功能指南](docs/ADDING_FUNCTIONS.md)：每个功能的实现、文档和测试均有独立目录。
- [来源与模型哈希](docs/provenance.json)、[第三方许可与引用](THIRD_PARTY_NOTICES.md)。

```bash
python -m pip install pytest
python -m pytest tests
```
