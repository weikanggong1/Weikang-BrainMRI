# TorchFNIRT

`TorchFNIRT` 是 FSL 6.0.7.4 中 FNIRT 灰质配准路径的 PyTorch 实现。本页面的
独立接口只实现 `GM_2_MNI152GM_2mm.cnf`：输入灰质概率图、灰质模板、可选 FLIRT
仿射矩阵和 binary reference mask，输出 cubic B-spline 系数、配准后的输入图以及
非线性 Jacobian。运行时不调用 FSL。

## 命令行调用

```bash
python -m freesurfer_torch.fnirt \
  --in subject_GM.nii.gz \
  --ref template_GM.nii.gz \
  --aff subject_GM_to_template_GM.mat \
  --cout subject_GM_to_template_GM_warp.nii.gz \
  --iout subject_GM_to_template_GM.nii.gz \
  --jout subject_GM_JAC_nl.nii.gz \
  --refmask MNI152_T1_2mm_brain_mask_dil.nii.gz \
  --config GM_2_MNI152GM_2mm.cnf \
  --device cuda:0
```

每一行的作用如下：

| 参数 | 作用 |
|---|---|
| `python -m freesurfer_torch.fnirt` | 启动本包的单被试 PyTorch FNIRT，不调用外部 `fnirt` executable。 |
| `--in` | moving/input 灰质概率图；配准要把它变换到 `--ref`。 |
| `--ref` | fixed/reference 灰质模板；决定 `--iout` 和 `--jout` 的 shape、voxel size 和空间网格。 |
| `--aff` | 可选 FLIRT `.mat`；方向是 input → reference，坐标是 FSL scaled-mm。代码先转换成带 source/target geometry 的 world-RAS affine，再传给 `TorchFNIRT`。省略时使用 FSL scaled-mm identity。 |
| `--cout` | cubic B-spline residual coefficient 文件。它是 FSL intent `2007`，不是 dense warp。省略时与 FSL 一样，从 input 文件名生成 `<input>_warpcoef`。 |
| `--iout` | 原始 `--in` 经仿射和非线性变换后，在 reference 网格上的图像。优化阶段的平滑图不会写到这里。 |
| `--jout` | 非线性位移的 Jacobian determinant；与 FSL `fnirt::SaveJacobian` 一致，不包含 FLIRT affine determinant。 |
| `--refmask` | reference 网格上的 binary `0/1` mask。GM 配置只在最后一级按 `applyrefmask=0,0,0,1` 使用它。省略时仅在 `FSLDIR/data/standard/` 可找到官方 mask 的情况下自动使用。 |
| `--config` | 只接受官方、未修改的 `GM_2_MNI152GM_2mm.cnf` 或该名称。其他配置会直接报错。 |
| `--device` | `cpu`、`cuda` 或 `cuda:N`；CLI 和 Python API 均默认优先使用可用 CUDA。 |
| `--overwrite` | 允许替换已有输出。未给出时，任何一个输出已存在都会在模型运行前报错。 |

`cout` 始终生成；省略 `--cout` 时，`subject_GM.nii.gz` 对应默认文件
`subject_GM_warpcoef.nii.gz`。输出 root 可以不带扩展名：`FSLOUTPUTTYPE=NIFTI`
补 `.nii`，`NIFTI_GZ` 或未设置时补 `.nii.gz`；其他 `FSLOUTPUTTYPE` 会明确报错。
三个输出必须使用不同路径，也不能覆盖 input、reference、affine 或 mask。输出先
写入同目录临时文件。未指定 `--overwrite` 时以原子 hard-link 创建最终文件，若
并行任务发生路径冲突，本次已创建的输出会回滚，已有文件不会被覆盖。

对应的 FSL 命令是：

```bash
fnirt \
  --in=subject_GM.nii.gz \
  --ref=template_GM.nii.gz \
  --aff=subject_GM_to_template_GM.mat \
  --cout=subject_GM_to_template_GM_warp.nii.gz \
  --iout=subject_GM_to_template_GM.nii.gz \
  --jout=subject_GM_JAC_nl.nii.gz \
  --refmask=MNI152_T1_2mm_brain_mask_dil.nii.gz \
  --config=GM_2_MNI152GM_2mm.cnf
```

两条命令的参数角色相同。额外的 `--device` 和 `--overwrite` 是本包选项。

## Python 调用

```python
from freesurfer_torch.fnirt.standalone import run_fnirt

result = run_fnirt(
    "subject_GM.nii.gz",                 # input/moving GM
    "template_GM.nii.gz",                # reference/fixed GM
    "subject_GM_to_template_GM.mat",     # FSL scaled-mm input -> reference
    cout="subject_GM_to_template_GM_warp.nii.gz",
    iout="subject_GM_to_template_GM.nii.gz",
    jout="subject_GM_JAC_nl.nii.gz",
    refmask="MNI152_T1_2mm_brain_mask_dil.nii.gz",
    config="GM_2_MNI152GM_2mm.cnf",
    device="cuda:0",
    overwrite=False,
)
```

`affine` 可以传 `None`，含义与 FSL 省略 `--aff` 相同，即使用 scaled-mm identity。
`device=None` 时与 CLI 一样自动选择 CUDA，否则使用 CPU。若 `input` 是文件路径，
省略 `cout` 会自动生成同目录的 `<input>_warpcoef`；若 `input` 是内存中的
`surfa.Volume`，必须显式给出 `cout`，因为没有文件名可用于派生输出路径。

返回值是 `TorchFNIRTResult`。主要内存结果为：

| 属性 | 含义 |
|---|---|
| `result.coefficient_image` | 与 `--cout` 相同的 intent-2007 NIfTI image。 |
| `result.coefficients` | `[Cx, Cy, Cz, 3]` coefficient array。 |
| `result.moved` | 与 `--iout` 相同的 reference-grid `surfa.Volume`。 |
| `result.nonlinear_jacobian` | 与 `--jout` 相同的非线性 Jacobian。 |
| `result.full_pull_jacobian` | 包含 affine 的完整 pull Jacobian；FSL `fnirt --jout` 不写这一项。 |
| `result.pull_transform` | reference → input 的 fixed-grid world-RAS pull warp。它不能作为 FSL scaled-mm warp array 直接使用。 |
| `result.qc` | 优化层级、拓扑约束和当前数值验证状态。 |

## 坐标和 coefficient 文件

FLIRT `.mat` 不是 NIfTI world affine。设 `Vin`、`Vref` 是 input/reference voxel
到 FSL scaled-mm 的矩阵，`Win`、`Wref` 是对应 voxel-to-world affine，`A` 是
`--aff`，则传给 `TorchFNIRT` 的 input → reference world-RAS affine 为：

```text
Wref · inverse(Vref) · A · Vin · inverse(Win)
```

`--cout` 使用 FSL cubic coefficient intent `2007`。它的 sform 保存 input →
reference FLIRT affine；qform offset 保存 dense reference field size；pixdim 保存
knot spacing；intent parameters 保存 dense field voxel size。设 `A` 为 sform 中的
forward FLIRT affine，`x_ref`、`x_in` 为 FSL scaled-mm 坐标，展开系数得到的 residual
field 为 `d`，则 coefficient 文件定义的 pull 关系是：

```text
x_in = inverse(A) · x_ref + d(x_ref)
```

因此 residual 的定义域是 reference grid，分量位于 input scaled-mm；不能把
coefficient array 当作 `[X,Y,Z,3]` dense displacement。
本包的 `TorchApplyWarp` 可以直接读取该文件。

`--jout` 是
`det(I + ∂d_nonlinear / ∂x_reference)`。它排除 affine determinant，正是 FSL
FNIRT 的 `SaveJacobian` 输出和 UKB VBM 非线性 modulation 使用的量。

## 固定配置

内置参数对应 FSL 6.0.7.4 的官方配置：

| 项目 | 值 |
|---|---|
| subsampling | `4,2,1,1` |
| maximum iterations | `5,5,10,5` |
| input/reference FWHM, mm | `6,4,2,2` / `4,2,0,0` |
| bending regularization | `150,75,50,30`，按当前 SSD 加权 |
| intensity model | `global_linear`，前三层估计，最后一层固定 |
| warp resolution | `10,10,10 mm` |
| reference mask schedule | `0,0,0,1` |
| implicit input/reference masking | 关闭 |
| prescribed Jacobian range | `0.2,5` |

若 `--config` 指向一个实际文件，文件名和 SHA-256 都必须与官方
`GM_2_MNI152GM_2mm.cnf` 一致。这样不会把修改过的配置静默当成已实现配置。

## 实现边界

当前接口只接收单个 3D input/reference、一个可选 4×4 FLIRT affine 和一个
binary reference mask。不支持 `T1_2_MNI152_2mm.cnf`、FA 配置、自定义 schedule、
`--inwarp`、`--intin`、`--inmask`、`--fout`、`--refout`、局部 intensity model、
DCT basis 或 quadratic spline。CLI 遇到这些参数会报告未识别参数；Python API
遇到其他 config 会抛出 `NotImplementedError`，不会退化为近似实现。

## 许可和验证门

本实现根据 FSL FNIRT 2203.0、basisfield 2203.1、miscmaths 2203.2、newimage
2203.11 和 warpfns 2203.0 修改，受
[`FSL Software Licence, Release 6.0`](../../../licenses/FSL-6.0.txt) 约束，仅限
该许可允许的非商业用途。本项目不是官方 FSL 发布。未修改的上游源码和版本、
commit、文件哈希保存在
[`_vendor_fsl`](../_vendor_fsl/README.md)。

intent-2007 header、B-spline expansion、ZoomField、bending energy、拓扑约束和
Jacobian 都有独立 FSL oracle 测试。只有在真实 GM 数据上通过以下外部门后，
才能把端到端结果标记为 FSL 数值等价：使用同一个 input、reference、FLIRT
matrix 和官方 reference mask；逐例比较 coefficient-expanded residual、`iout`、
`jout` 和 modulated GM；报告最大误差、MAE、RMSE、Pearson 及运行时间。目前
`TorchFNIRT.qc["fsl_fnirt_numerically_equivalent"]` 仍为 `False`，文档不把组件
级 oracle 通过写成端到端等价。

### FSL 6.0.7.4 实例诊断

同一幅真实 FAST GM、同一模板、官方 FLIRT matrix、官方 reference mask 和
`GM_2_MNI152GM_2mm.cnf` 的严格 case01 对照结果如下。数值比较覆盖完整 reference
grid；运行时间在 H100 上同步 CUDA 后测量。

| 输出 | Pearson r | MAE | RMSE |
|---|---:|---:|---:|
| cubic coefficients | 0.999416 | 0.028560 | 0.063963 |
| warped GM | 0.999089 | 0.002453 | 0.011100 |
| nonlinear Jacobian | 0.999669 | 0.002840 | 0.005656 |
| modulated GM | 0.999206 | 0.002818 | 0.012855 |

PyTorch CUDA 总时间为 69.32 秒，其中保持 FSL Jacobian 范围的 topology projection
占 62.57 秒。首两次 accepted coefficient update 对 FSL 的 MAE 为
`5.45e-7` 和 `2.05e-6`；第三次开始，FSL `SpMat` 的固定稀疏列累加顺序与本包
matrix-free Hessian 的 reduction 顺序使 1e-3 截断 PCG 走向不同 Krylov 轨迹。
因此标量输出高度接近，但不满足“仅浮点误差”或逐体素数值等价。完整无私有路径
记录见
[`fnirt_fsl_6074_real_case_01.public.json`](../../../validation/fast_vbm/fnirt_fsl_6074_real_case_01.public.json)。
