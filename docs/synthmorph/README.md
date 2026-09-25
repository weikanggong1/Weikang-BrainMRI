# SynthMorph：配准与变换应用

[返回首页](../../README.md) · [源码目录](../../src/freesurfer_torch/synthmorph/) · [权重](../WEIGHTS.md) · [批量执行](../ARCHITECTURE.md#批量执行)

本模块将指定 FreeSurfer 8.2.0 构建中的 TensorFlow/Keras SynthMorph 移植为 PyTorch，支持刚性、仿射、非线性和联合配准，直接读取官方 HDF5 权重。推理不导入 TensorFlow、VoxelMorph、Neurite，也不调用 FreeSurfer。

参考 build 为 `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`，并非随时变化的开发分支。准确源文件和权重哈希见 [provenance.json](../provenance.json)。

## Python API

```python
from pathlib import Path
from freesurfer_torch import SynthMorph, apply_transform

out = Path("results")
out.mkdir(exist_ok=True)
register = SynthMorph(
    weights="/path/to/weights", device="cuda:0", model="joint",
    extent=256, hyper=0.5, steps=7,
)
result = register("moving_T1w.nii.gz", "fixed_T1w.nii.gz")
result.moved.save(out / "moving_in_fixed.nii.gz")
result.fixed_moved.save(out / "fixed_in_moving.nii.gz")
result.transform.save(out / "moving_to_fixed.mgz")
result.inverse.save(out / "fixed_to_moving.mgz")

# 离散标签与原 moving 图像应具有相同几何，使用最近邻。
labels = apply_transform(
    "moving_labels.nii.gz", result.transform,
    method="nearest", dtype="int16",
)
labels.save(out / "labels_in_fixed.nii.gz")
```

`from freesurfer_torch.synthmorph import SynthMorph, RegistrationResult, apply_transform` 是等价的功能模块入口。模型实例可重复用于后续影像对。

### 模型构造

`SynthMorph(weights=None, device="cpu", model="joint", extent=256, hyper=0.5, steps=7)`：

| 参数 | 含义 |
|---|---|
| `weights` | 权重目录，或将 `affine`、`rigid`、`deform` 映射到权重文件的字典；省略时使用统一查找顺序 |
| `device` | `"cpu"` 或 `"cuda:N"` |
| `model="joint"` | 默认，联合仿射与非线性配准 |
| `model="deform"` | 非线性阶段；输入应已具有适当的仿射对齐 |
| `model="affine"` / `"rigid"` | 仿射 / 刚性配准 |
| `extent` | 192 或 256，每轴网络网格大小，分辨率 1 mm，默认 256 |
| `hyper` | 非线性正则化参数，`0 < hyper < 1`，默认 0.5；实例构造时固定 |
| `steps` | scaling-and-squaring 次数，至少 5，默认 7 |

模型使用 float32 张量；CUDA 构造默认允许 TF32 matmul 和 cuDNN 内核，不使用 float16 或 bfloat16。不同 `hyper` 需构造另一实例；底层 `DeformNetwork.set_hyper()` 是显式重新计算特化权重的入口。改变实例的普通属性不会自动更新这些权重。Python 的 CPU 线程数可用 `torch.set_num_threads()` 设置；统一 CLI 提供 `-j` 参数。

### 配准调用与结果

`register(moving, fixed, init=None, mid_space=False, header_only=False, output_dir=None)`：

| 参数 | 含义 |
|---|---|
| `moving`, `fixed` | 文件路径或 `surfa.Volume`，必须为单帧 3D 图像；网格和方向可不同 |
| `init` | 可选初始仿射：`.lta` 路径或带源/目标几何的 Surfa Affine，几何须匹配输入 |
| `mid_space` | 使用初始仿射的中间空间；为 `True` 时必须提供 `init` |
| `header_only` | 仅改变影像头信息，限 affine / rigid |
| `output_dir` | 调试输出目录：`inp_1.nii.gz`、`inp_2.nii.gz` 和 `network_transforms.npz` |

返回 `RegistrationResult`：

| 字段 | 含义 |
|---|---|
| `moved` | moving 在 fixed 空间的 `surfa.Volume` |
| `fixed_moved` | fixed 在 moving 空间的 `surfa.Volume` |
| `transform` | moving → fixed 的带几何变换 |
| `inverse` | fixed → moving 的带几何变换 |

重采样时图像采用目标网格；`header_only=True` 保留数据并更新其几何。每次调用都会计算双向结果。对于 affine / rigid，变换为 world-space `surfa.Affine`，建议保存为 `.lta`；joint / deform 返回 RAS 位移 `surfa.Warp`，建议保存为 `.mgz`。普通三通道数组不携带足够的源/目标几何，不能直接替代这些变换文件。直接在 Python 中保存时，由调用者准备输出父目录。

### 应用已有变换

`apply_transform(image, transformation, method="linear", fill=0, dtype="float32", header_only=False)`：

| 参数 | 含义 |
|---|---|
| `image` | 路径或 `surfa.Volume`；接受 3D 和带 frame 维的 4D |
| `transformation` | `.lta` 路径、warp 文件路径或 Surfa Affine/Warp |
| `method` | `linear` 或 `nearest`，标签使用 `nearest` |
| `fill` | 视野外强度，默认 0 |
| `dtype` | 输出类型，默认 `float32` |
| `header_only` | 只更新头信息，限 affine |

返回 `surfa.Volume`。warp 的源几何必须与输入图像一致；配准输入仍仅支持 3D，这一限制不适用于应用已有变换。

## 命令行

```bash
fs-torch synthmorph moving_T1w.nii.gz fixed_T1w.nii.gz \
  --model joint --device cuda:0 --weights /path/to/weights \
  -o results/moving_in_fixed.nii.gz -O results/fixed_in_moving.nii.gz \
  -t results/moving_to_fixed.mgz -T results/fixed_to_moving.mgz

fs-torch apply results/moving_to_fixed.mgz moving_labels.nii.gz \
  results/labels_in_fixed.nii.gz --method nearest --dtype int16
```

对应的 FreeSurfer 原版指令为：

```bash
mri_synthmorph register -m joint \
  -o results/moving_in_fixed.nii.gz -O results/fixed_in_moving.nii.gz \
  -t results/moving_to_fixed.mgz -T results/fixed_to_moving.mgz \
  moving_T1w.nii.gz fixed_T1w.nii.gz

mri_synthmorph apply -m nearest -t int16 \
  results/moving_to_fixed.mgz moving_labels.nii.gz \
  results/labels_in_fixed.nii.gz
```

两版的第一个输入均为 moving、第二个为 fixed；`-o/-O` 保存两个方向的影像，`-t/-T` 保存对应变换。`apply` 的原版 `-m/-t` 分别对应本包的 `--method/--dtype`。离散标签使用最近邻插值，以免引入新标签值。

| 配准参数 | 含义 |
|---|---|
| `moving fixed` | 两个必需位置参数 |
| `-m`, `--model` | joint / deform / affine / rigid，默认 joint |
| `--weights`, `--device` | 权重目录、设备 |
| `-o`, `--out-moving` / `-O`, `--out-fixed` | 正向 / 反向图像 |
| `-t`, `--trans` / `-T`, `--inverse` | 正向 / 反向变换 |
| `-i`, `--init` / `-M`, `--mid-space` | 初始仿射 / 中间空间初始化 |
| `-H`, `--header-only` | 仅更新头信息 |
| `-e`, `--extent` / `-r`, `--hyper` / `-n`, `--steps` | 网络网格、正则化和积分步数，与 API 默认值一致 |
| `-j`, `--threads` | Torch 线程数，CLI 默认 4 |
| `-d`, `--output-dir` | 调试目录 |

配准至少请求一个影像、变换或调试输出。统一 CLI 会创建输出父目录。`fs-torch apply` 的位置参数依次是变换、影像、输出；支持 `--method`、`--fill`、`--dtype`、`--header-only`。CLI dtype 选择为 `uint8`、`uint16`、`int16`、`int32`、`float32`，默认 `float32`。apply 使用 Surfa CPU 重采样，不接受设备参数。当前 apply CLI 每次处理一对 image/output；多个图像可在 Python 中循环调用。

## 多被试 Python

`predict_batch(table, fixed, workers=1, threads_per_worker=1)` 接受恰有 `input`、`output` 两列的 pandas 表。`input` 是每例 moving 图像路径；`output` 是不带扩展名的绝对路径前缀，含被试 base name。`fixed` 可为全表共用的目标图像，或与表中行顺序一一对应、长度相同的目标图像列表。每行保存双向配准图像及变换，返回键为 `moved`、`fixed_moved`、`transform`、`inverse` 的路径字典。

```python
from pathlib import Path
import pandas as pd
from freesurfer_torch import SynthMorph

table = pd.DataFrame({
    "input": ["/data/sub-01_T1w.nii.gz", "/data/sub-02_T1w.nii.gz"],
    "output": ["/results/sub-01", "/results/sub-02"],
})
if __name__ == "__main__":
    register = SynthMorph(device="cuda:0", model="joint")
    saved: list[dict[str, Path]] = register.predict_batch(
        table,
        fixed=["/data/template_A.nii.gz", "/data/template_B.nii.gz"],
        workers=2,
    )
    print(saved[0]["moved"], saved[0]["transform"])
```

示例中两幅 `fixed` 依次对应表中的两行；也可传入一个共享的目标图像路径。`workers=2` 时 `fixed` 须为路径或路径列表，单进程仍可使用 `surfa.Volume`。`output="/results/sub-01"` 生成 `_moved.nii.gz`、`_fixed_moved.nii.gz`、`_transform.mgz` 和 `_inverse.mgz`；`affine`/`rigid` 模式的后两个文件改为 `.lta`。默认 `workers=1` 逐例复用模型；`workers=2` 在同一设备上使用两个 Python 进程、各加载一份模型，输出影像保留各自的输入及目标几何。多进程脚本须保护主入口；共享路径规则见[批量执行说明](../ARCHITECTURE.md#批量执行)。

### 未发布 B2 原型对照

在 gpucw1 的一张共享 H100 上，以相同的 4 例输入运行 joint 配准，每例保存双向图像和变换，共 16 个文件。B1 为单个常驻 Python 程序逐例运行；B2 为未发布原型在单个常驻程序中合批运行，4 例均实际进入 B=2 网络批；P2 为两个独立常驻程序各按 B=1 处理 2 例。正序和逆序各运行一次 cold 与 warm 队列；下表为 warm 队列总耗时。

| 模式 | 正序 | 逆序 | 两轮中位数 |
|---|---:|---:|---:|
| B1 | 293.89 s | 217.96 s | 255.93 s |
| B2 | 261.63 s | 239.26 s | 250.45 s |
| P2 | 128.13 s | 126.18 s | 127.16 s |

B2 与 B1 的快慢随运行顺序翻转，不能据此认定 B2 稳定提速；P2 在两轮中均约快 2 倍。表中的 P2 由两个独立常驻脚本运行，并非当前 `workers=2` API 的实测；该对照衡量多被试队列吞吐。完整条件与逐轮结果见[批量性能报告](../../benchmark/batch_modes_2026-09-24.md)。

公开 Python 表格接口在 gpucw1 的 4 例 `joint` 配准中，两组 `workers=1/2` 调用耗时中位数为 **231.99/136.10 s**，观察到 **1.70 倍**吞吐差。双向图像逐字节相同，正反 MGZ 变换解码后的位移差为 **0 mm**；另以两例测试了逐行对应的 `fixed` 路径列表。逐轮数据见[Python 接口验证](../../benchmark/batch_modes_2026-09-24.md#python-table-api-with-two-processes)。

## 权重和执行位置

| 权重 | 使用模式 |
|---|---|
| `synthmorph.affine.2.h5` | affine；joint 的仿射阶段 |
| `synthmorph.rigid.1.h5` | rigid |
| `synthmorph.deform.3.h5` | deform；joint 的非线性阶段 |

官方地址、文件大小、SHA-256 和许可证见 [WEIGHTS.md](../WEIGHTS.md)。独立推理只需本地权重；HDF5 加载器只支持这里记录的架构，不是任意 Keras 网络转换器。

网络空间采样、网络、速度场积分和原始坐标组合在所选设备执行。HDF5 读取和超网络权重特化、初始仿射的矩阵平方根、影像 I/O、最终 Surfa 重采样在 CPU 执行。对于固定 `hyper`，构造时把大型超网络特化为普通卷积权重，随后可复用。这一初始化与重复调用的时间分配不同于原实现，比较性能时需区分完整 CLI 进程和已加载 API。

## 源码组织

| 文件 | 责任 |
|---|---|
| [models.py](../../src/freesurfer_torch/synthmorph/models.py) | affine/rigid 特征网络、HyperVxmJoint、HDF5 读取与权重特化 |
| [pipeline.py](../../src/freesurfer_torch/synthmorph/pipeline.py) | 图像几何、预后处理、双向结果与 apply |
| [spatial.py](../../src/freesurfer_torch/synthmorph/spatial.py) | pull 采样、仿射/位移组合和积分 |
| [__init__.py](../../src/freesurfer_torch/synthmorph/__init__.py) | 功能公开导出 |

## Source map

Source paths below are relative to the reference `$FREESURFER_HOME`; Python dependency paths refer to `python/lib/python3.8/site-packages/`. Line numbers refer to the recorded 8.2.0 build.

| Source | Lines | Role |
|---|---:|---|
| `python/scripts/mri_synthmorph` | CLI parser near end | `register`/`apply`; model choices and parameter validation; explicitly selects TensorFlow backends |
| `python/packages/synthmorph/registration.py` | 10–15 | Official model weights: affine.2, rigid.1, deform.3 |
| same | 18–54 | Build image-specific 1 mm isotropic LIA network geometry |
| same | 57–100 | Resample into network coordinates, zero outside volume, min–max normalize |
| same | 103–156 | Flexible nested Keras H5 loading |
| same | 159–242 | Input geometry validation, affine initialization, network selection |
| same | 244–310 | Native-coordinate composition, Surfa transforms, outputs and debug network-space exports |
| `voxelmorph/tf/networks.py` | 1238–1459 | `VxmAffineFeatureDetector` |
| same | 1462–1685 | `HyperVxmJoint` |
| `neurite/tf/layers.py` | 2668–2803 | `HyperConvFromDense`: predict complete convolution kernel/bias from a dense hypernetwork |
| `neurite/tf/utils/utils.py` | 73 onward | Multilinear interpolation with whole-sample fill or border clamping |
| same | 512–578 | Centered normalized feature barycenters, divide-no-nan |
| `voxelmorph/tf/utils/utils.py` | 253–348 | Mixed affine/dense pull-transform composition |
| same | 350 onward | Scaling-and-squaring integration |
| same | 794–982 | Affine parameters and intrinsic XYZ Euler rotations |
| same | 983–1046 | Cholesky decomposition stripping scale/shear |
| same | 1049–1098 | Weighted least-squares affine fit |

The bundled `voxelmorph/torch/networks.py` contains ordinary `VxmDense`; it does **not** implement the affine feature detector or HyperVxmJoint. Changing `VXM_BACKEND` alone is insufficient. The command additionally forces the TensorFlow backend.

## Model and weight conversion

**Affine/rigid.** One shared single-image detector has four 3×3×3 convolutions with 256 output channels, each followed by LeakyReLU(0.2) and factor-two max pooling; four additional 256-channel convolutions at the coarsest scale; and a 64-channel ReLU output. Standalone registration downsamples the normalized network images by selecting index coordinates `(2i,2j,2k)`. Its feature maps yield 64 centered barycenters. Coordinates are divided by feature-map extent and multiplied by the detector input extent, rather than taking a generic resized coordinate grid.

The product of the two images' normalized feature masses weights a normal-equation affine fit in both directions. The first result is averaged with the inverse of the second, and the reverse result is the inverse of that average. Rigid mode uses a separately trained checkpoint and strips scale and shear by Cholesky/Euler decomposition; an SVD nearest-rotation approximation would change the algorithm. The return matrices operate on zero-based voxel indices, so the centered estimate is conjugated by translations `(shape−1)/2`.

**Deformable.** A scalar user-selected regularization input passes through four Dense(32, ReLU) layers. Each of the 13 convolutions has its entire kernel and bias predicted by separate linear maps from this 32-vector. The main network has four 256-channel encoder convolutions/pools; four 256-channel decoder convolutions followed by nearest-neighbor ×2 upsampling and skip concatenation; four 256-channel extra convolutions; and a three-channel linear SVF output. Decoder convolutions 5–8 (zero-based indexes) consume 512 channels. Features are always ordered as moving then fixed.

Keras Conv3D kernels are `(Ki,Kj,Kk,Cin,Cout)`. PyTorch Conv3D kernels are `(Cout,Cin,Ki,Kj,Kk)`: transpose `(4,3,0,1,2)` with **no spatial reversal**. Dense kernels are `(input,output)`; evaluating `h @ W + b` preserves that layout. Hyperkernel flat outputs must first be reshaped to the Keras five-dimensional kernel shape and only then transposed. The source predicts kernel and bias linearly; neither has an extra activation.

The published deformable H5 stores several GB of hypernetwork coefficients. The port streams one hyperkernel matrix at a time and evaluates its 32-dimensional projection with torch, retaining only the resulting ordinary convolution weights. This is an algebraic specialization for a fixed regularization value, not a reduced architecture. Calling `DeformNetwork.set_hyper(r)` recomputes these weights for any `0 < r < 1`; repeated calls at the same value are cached. Inference does not import TensorFlow. Alternative weights must follow this exact 8.2 architecture; arbitrary future Keras model graphs are not automatically translated.

The installed H5 dataset shapes imply 12,837,696 affine parameters (51,350,784 FP32 bytes) and 877,133,827 deformable hypernetwork parameters (3,508,535,308 FP32 bytes). Specialization retains 26,579,715 deformable convolution parameters (106,318,860 FP32 bytes). These figures count parameter storage only, not activation tensors, convolution workspaces, HDF5 metadata or process memory.

## Symmetry and coordinate transforms

For both directions, the same deformable network runs on swapped inputs and the SVF is antisymmetrized: `v = (D(m,f) − D(f,m))/2`; the reverse SVF is `−v`. Two scaling-and-squaring integrations compute `exp(v)` and `exp(−v)`, at half resolution. With seven steps, initialize `u=v/128` and repeat `u ← u + u(Id+u)` seven times. Sampling the vector field uses border extension, not zero fill.

Joint registration estimates a symmetric affine transform in the half-resolution image space and computes separate principal square roots for the forward and reverse matrices. It resamples both full-resolution images directly into this affine mid-space. The nonlinear step therefore differs from running standalone affine followed by ordinary deformable registration, unless the latter uses the documented mid-space initialization. The port computes the 4×4 real principal roots with double-precision Denman–Beavers iterations, then converts back to FP32. This differs numerically from TensorFlow's Schur-based matrix root, so the final transform error must be measured.

At half resolution the composed pull map is `A_half_to_full ∘ exp(v) ∘ S_0.5 ∘ A_half_to_full`. Deform-only omits the final two factors and uses `A_half_to_full=S_2`. To produce a full-resolution displacement field, the source composes with a *dense* `S_0.5` pull map defined on the full grid. Generic `F.interpolate(..., align_corners=...)` is not an equivalent replacement at the last boundary.

All network transforms are **pull** maps: the first maps fixed coordinates to moving coordinates for sampling the moving image. The registration wrapper conjugates by network/native coordinate maps (`net_to_mov ∘ fw ∘ fix_to_net`). Matrix transforms are exchanged before constructing Surfa Affines because LTA follows the opposite convention; nonlinear fields are constructed as `disp_crs`, then exported as `disp_ras`. A plain three-channel NIfTI array without correct source/target geometry and RAS conversion would not reproduce the command.

## Pre/postprocessing and option compatibility

The wrapper preserves single-frame NIfTI/MGZ image geometry. Joint/affine/rigid network grids are centered on each input's field of view. Deform-only centers the moving network grid on the fixed image, reflecting assumed prior affine alignment. Images are resampled to LIA 1 mm with extent 192 or 256 and globally min–max normalized after resampling, including the zero-filled exterior. It is not percentile normalization or per-slice normalization.

The original CLI also supports bidirectional image and transform output, affine initialization, mid-space initialization, header-only affine application, interpolation/dtype/fill control for apply, alternative checkpoint paths, and debug network-space outputs. These functions belong to the wrapper and must be compared separately from neural forward equivalence.

Neural inference, network-space resampling, SVF integration and native-coordinate composition use PyTorch on the selected device. Final moved-image resampling and applying a saved transform use the standalone Surfa library on CPU, exactly as the original CLI does. These operations do not invoke FreeSurfer executables or TensorFlow.

Two interpolation conventions must remain separate. Neurite's network sampler accepts coordinates in `[0,n−1]`, filling a sample outside that closed interval. Surfa 0.6.3's final linear sampler checks the floored coordinate, accepting `[0,n)` and extending the final voxel over `[n−1,n)`. Surfa nearest-neighbor sampling also accepts `[0,n)` and rounds positive half-integers upward, whereas TensorFlow/PyTorch round ties to even. Reusing the network sampler for final outputs therefore changes edge voxels even when the estimated transforms agree. The port resamples using the original native voxel/CRS transforms before converting the returned transforms to world/RAS format, preserving the original operation order and avoiding unnecessary coordinate round trips.

The installed original command has an initialization-path dtype failure: with `-i` (also `-i -M`), `net_to_mov` and sometimes `net_to_fix` become floating TensorFlow tensors at double precision. VoxelMorph composition preserves floating tensor dtypes but casts NumPy inputs to single precision, leading to a mixed double/float matrix multiplication with the FP32 network result. The option validator preserves the unmodified command's failed exit code and logs. For a separate functional comparison it copies the official `synthmorph` Python package into the task's `work/reference_dtype_fix/` and inserts only `net_to_mov = np.asarray(net_to_mov)` and `net_to_fix = np.asarray(net_to_fix)` immediately before native-space composition. It runs the official CLI through `fspython` with `FS_LOCAL_PYTHONPATH` selecting that copy, verifies the imported source path, and records original/copied hashes. The installed FreeSurfer files remain untouched. These initialized-registration comparisons are explicitly labeled **patched reference**, and must not be described as successful runs of the unmodified original command.

Saved-transform application accepts multi-frame (4D) input images; neural registration still requires single-frame (3D) images. The option validator exercises four distinct frames with both affine and nonlinear saved transforms, linear and nearest-neighbor interpolation, and checks image data bitwise against the original `apply` command. It separately checks output dtype, nonzero fill, and header-only affine application.

## 验证、差异与限制

数值检查覆盖网络层、空间运算层和完整影像流程。以下结果来自 **0.1.0 参考实验**；0.2.0 的结构回归另见 [refactor/report.public.json](../../validation/refactor/report.public.json)。

- 130 项空间/插值差分检查通过，涵盖边界、半整数取整、积分和变换组合，见 [spatial_validation.json](../../validation/spatial/spatial_validation.json)。
- 192³ 模板对涵盖 affine、rigid、deform、joint 四模式和双向输出，见 [full192/report.json](../../validation/full192/report.json)。
- 默认 joint、256³ 的 12 例真实 T1w，同设备最大形变向量差为 `0.000790 mm`，最大正向 moved NRMSE 为 `4.05e-5`，见 [匿名逐例结果](../../benchmark/summary.public.json)。
- 15 项选项检查的记录见 [options/report.json](../../validation/options/report.json)；初始化两项使用上文明确区分的 patched reference，其余 saved-transform apply 与未修改原版比较。

下表是 0.8 及更早版本在 TF32 关闭条件下的历史默认 joint 12 例计时，不代表 0.9 的 TF32 默认性能。CPU 固定 8 线程，GPU 使用同一张 H100，计时包含启动、权重加载、推理和写盘。原版 CPU/GPU 均运行 FreeSurfer 原生命令。[四分位数和环境差异](../COMPARISON.md#cpugpu-时间)保留在完整对照报告中。

| 原版 CPU | 本包 CPU | 原版 GPU | 本包 GPU |
|---:|---:|---:|---:|
| 164.55 | 122.33 | 116.59 | 17.62 |

下图使用公开 T1w 样例，展示 moving、fixed 以及原版和本包的 joint 配准图像。两列配准结果均在 fixed 网格；图中为展示使用相同的 fixed 脑掩膜。原版在 CPU、本包在 GPU 上推理，因此这张图用于查看结果，不用于比较运行速度。图像制作与完整体数据比较见[图示记录](../figures/README.md)。

![公开 T1w 输入及 FreeSurfer 与本包的 joint 配准结果](../figures/synthmorph_comparison.png)

模板反向图像存在少量采样有效域边界跳变：deform 为 1 个、joint 为 2 个体素，其强度误差超过输入最大强度的 0.1%。微小坐标误差使域外填零变为域内采样，因此接近的位移并不保证全部输出逐元素一致。完整误差、几何和定位见 [COMPARISON.md](../COMPARISON.md) 及[异常体素报告](../../validation/full192/reverse_output_diagnosis/report.json)。本包保留原边界规则，没有通过放宽规则掩盖这些差异。

参考版本的 `-i` / `-i -M` 原命令因 float64/float32 混合而失败。记录同时保留原失败和两行类型转换副本的比较结果；调试输出布局、日志及线程默认值也与原 CLI 不同。

测试入口：

```bash
python -m pytest tests/synthmorph tests/test_public_api.py
# 原版差分比较才需要 FreeSurfer。
module load freesurfer
python tools/validate_spatial.py --out-dir validation/spatial --device cuda
python tools/compare_synthmorph_networks.py --help
python tools/validate_synthmorph.py --help
python tools/validate_registration_options.py --help
```

[validate_synthmorph.py](../../tools/validate_synthmorph.py) 执行完整流程；[compare_synthmorph_networks.py](../../tools/compare_synthmorph_networks.py) 分开运行 TensorFlow 和 PyTorch，比较特征、矩阵、SVF 和位移。全部实际通过范围、计时口径和未覆盖边界以 [COMPARISON.md](../COMPARISON.md) 为准。真实病例没有配准地标真值，这些数值对照验证参考实现的复现，不是独立解剖学准确率评估。

## 官方来源与引用

- [FreeSurfer SynthMorph 源码](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthmorph)
- [VoxelMorph TensorFlow 分支](https://github.com/voxelmorph/voxelmorph/tree/dev-tensorflow)
- [联合 SynthMorph 方法论文](https://doi.org/10.1162/imag_a_00197)

上述链接用于追溯项目来源；具体移植依据为已记录哈希的 FreeSurfer 8.2.0 安装版本。
