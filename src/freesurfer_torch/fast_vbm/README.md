# FastVBM 模块

`FastVBM` 将 raw T1w 处理为模板空间 modulated GM：

```text
raw T1w
  → SynthStrip
  → TorchFAST 三组织 PVE + bias-field correction
  → source-derived TorchFLIRT（FSL default path）
  → PyTorch SynthMorph deform 或 source-derived TorchFNIRT GM config
  → common FSL warp conversion + GPU TorchApplyWarp
  → common nonlinear-only Jacobian
  → warped GM × Jacobian
```

线性阶段固定使用 FSL default correlation-ratio/Brent FLIRT 实现。
早期 NCC + Adam 仿射和 FNIRT-style/Adam 实现已从安装包删除。

非线性阶段由 `registration_backend` 选择：

- `"synthmorph"` 调用本包 `SynthMorph(model="deform")`，读取官方 `synthmorph.deform.3.h5`；
- `"fnirt"` 使用 `freesurfer_torch.fnirt.TorchFNIRT` 的 GM config，不读取非线性 checkpoint。

两分支共用同一 FAST GM、source-derived `TorchFLIRT`、FSL 坐标契约和 template
grid。两分支的 full RAS pull 都转为
`u=source_fsl-inv(FLIRT)@target_fsl`，然后由同一 GPU `TorchApplyWarp`、
`det(I + ∂u/∂q)` 和 modulation 代码处理。FNIRT 与 FSL 配对比较时必须向
`FastVBM.run(..., reference_mask=...)` 传入官方 FNIRT 使用的同一
reference mask；默认 `template > 0` 会在 QC 中标为非 FSL-exact。
该 mask 记录在两个分支的共同上下文中，但 SynthMorph 网络没有 mask 输入；
mask 使用属于两个 nonlinear estimator 之间的算法差异。因而，两个公开后端
唯一影响输出的分支是 nonlinear estimator 本身，包括各自的目标函数、正则化和
mask 使用；其余配准、重采样、Jacobian 和 modulation 步骤相同。
FNIRT 的 spline analytic Jacobian 只作为 estimator QC 与 common dense Jacobian
对照，不参与 warped GM、Jacobian 或 modulated GM 的生成。

## 单被试 Python

```python
from freesurfer_torch import FastVBM

pipeline = FastVBM(
    device="cuda:0",
    threads=4,
    registration_backend="fnirt",  # 或 synthmorph
)
result = pipeline.run(
    "subject_T1w.nii.gz",
    "template_GM.nii.gz",
    "results/sub-01",
    reference_mask="MNI152_T1_2mm_brain_mask_dil.nii.gz",
    overwrite=False,
)
```

构造一次后可继续调用，worker 会复用已加载组件。只需内存结果时使用 `result = pipeline(image, template)`；已有同网格 mask 时增加 `brain_mask="mask.nii.gz"`，可跳过 SynthStrip。

## 单被试命令行

```bash
fs-torch fast-vbm \
  -i subject_T1w.nii.gz \
  --template template_GM.nii.gz \
  -o results/sub-01 \
  --registration-backend fnirt \
  --reference-mask MNI152_T1_2mm_brain_mask_dil.nii.gz \
  --device cuda:0 \
  --threads 4
```

`-i` 是 raw T1w，`--template` 是 fixed GM template，`-o` 是完整输出目录。
`--registration-backend` 选择 `synthmorph` 或 `fnirt`；`--reference-mask` 进入
共同配准上下文，但只有 `TorchFNIRT` 使用。多被试只用 Python `BatchRunner`；
`model` 字典中的 `registration_backend` 控制所有 job 的后端，每个 worker 固定到
一张 GPU 并缓存一套 pipeline。

## 独立 PyTorch FLIRT

```python
from freesurfer_torch import TorchFLIRT

result = TorchFLIRT(device="cuda:0").run(
    input="moving.nii.gz",
    reference="fixed.nii.gz",
    output="moved.nii.gz",
    omat="moving_to_fixed.mat",
)
```

```bash
fs-torch flirt -in moving.nii.gz -ref fixed.nii.gz \
  -out moved.nii.gz -omat moving_to_fixed.mat \
  -dof 12 -cost corratio --device cuda:0
```

`moved` 与 reference 的 shape/geometry 一致；`.mat` 是 input → reference 的
FSL scaled-mm matrix。当前 `TorchFLIRT` 来自 FSL 2111.2 默认
correlation-ratio/Brent 路径源码，不是旧
NCC/Adam 实现。当前十例验证使用 FSL reference image intensity-weighted COG
和 80 mm radius 计算 `rmsdiff`；汇总完成前仍标记
`validated_fsl_equivalent=false`。详见
[`docs/flirt/README.md`](../../../docs/flirt/README.md)。

## 输出

两个后端使用同一组 13 个结果键与文件名：

| 结果键 | 文件名 |
|---|---|
| `brain` | `T1_brain.nii.gz` |
| `brain_mask` | `brain_mask.nii.gz` |
| `pve_csf` | `T1_brain_pve_0.nii.gz` |
| `pve_gm` | `T1_brain_pve_1.nii.gz` |
| `pve_wm` | `T1_brain_pve_2.nii.gz` |
| `hard_segmentation` | `T1_brain_seg.nii.gz` |
| `pve_segmentation` | `T1_brain_pveseg.nii.gz` |
| `mixel_type` | `T1_brain_mixeltype.nii.gz` |
| `bias_field` | `T1_brain_bias.nii.gz` |
| `restored` | `T1_brain_restore.nii.gz` |
| `warped_gm` | `T1_GM_to_template_GM.nii.gz` |
| `jacobian` | `T1_GM_JAC_nl.nii.gz` |
| `modulated_gm` | `T1_GM_to_template_GM_mod.nii.gz` |

后三项在 GM template 网格。文件名、网格角色和 `modulated_gm = warped_gm × jacobian` 对应 UKB/FSL VBM；算法和体素值不保证相同。

## 权重

```bash
python tools/setup_weights.py --model fast-vbm
```

该别名配置 SynthStrip 与 SynthMorph deform，是两个后端的权重超集。只运行
`TorchFNIRT` 分支可改用 `--model synthstrip`；若还提供显式脑 mask，则该分支
不需要 checkpoint。TorchFAST、`TorchFLIRT`、`TorchFNIRT`、`TorchApplyWarp`、
Jacobian 和 modulation 都没有模型权重。

当前 0.9 共享链路的正式 10 例双后端结果和 FNIRT 10 例汇总均待完成。
验证目录中的 0.8
FNIRT-style/NCC-Adam 数值是历史记录，只用于追溯旧版本，不能作为当前
实现的准确度或计时结果。

完整参数、坐标与 Jacobian 约定、Python BatchRunner 多 GPU 示例、FSL/UKB 对应和验证入口见 [`docs/fast_vbm/README.md`](../../../docs/fast_vbm/README.md)。公开统计只以 [`validation/fast_vbm`](../../../validation/fast_vbm/README.md) 为准。
