# FastVBM 模块

`FastVBM` 将 raw T1w 处理为模板空间 modulated GM：

```text
raw T1w
  → SynthStrip
  → TorchFAST 三组织 PVE + bias-field correction
  → PyTorch 12-DOF affine
  → PyTorch SynthMorph deform 或 PyTorch FNIRT-style cubic B-spline
  → nonlinear-only Jacobian
  → warped GM × Jacobian
```

线性阶段使用 FLIRT 的 input/reference 角色和 12 参数类别，但优化器是 NCC + Adam。独立 `TorchFLIRT` 接口输出 reference-grid image 和 input → reference 的 FSL scaled-mm 4×4 matrix。它不复现 FSL FLIRT 的 cost、搜索、优化器或插值细节。

非线性阶段由 `registration_backend` 选择：

- `"synthmorph"` 调用本包 `SynthMorph(model="deform")`，读取官方 `synthmorph.deform.3.h5`；
- `"fnirt"` 使用 PyTorch cubic B-spline、SSD、bending energy 和 Jacobian 约束，不读取非线性 checkpoint。

FNIRT-style 分支在内部使用 FSL scaled-mm residual displacement，并按 FSL nonlinear-only 约定计算 `det(I + ∂d_nl/∂x_fixed_fsl)`。它使用 PyTorch Adam，不读写 FSL coefficient/dense-warp 文件，也不与 FSL FNIRT 数值等价。两个后端对外都返回 fixed-grid `surfa.Warp(format=disp_ras)` 和相同的 FastVBM 文件名。

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
  --device cuda:0 \
  --threads 4
```

`-i` 是 raw T1w，`--template` 是 fixed GM template，`-o` 是完整输出目录。`--registration-backend` 选择 `synthmorph` 或 `fnirt`。多被试只用 Python `BatchRunner`；`model` 字典中的 `registration_backend` 控制所有 job 的后端，每个 worker 固定到一张 GPU 并缓存一套 pipeline。

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
  -dof 12 -cost normcorr --device cuda:0
```

`moved` 与 reference 的 shape/geometry 一致；`.mat` 是 input → reference 的 FSL scaled-mm matrix。该契约不表示输出数值与 FSL FLIRT 相同。

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

该别名配置 SynthStrip 与 SynthMorph deform，是两个后端的权重超集。只运行 FNIRT-style 分支可改用 `--model synthstrip`；若还提供显式脑 mask，则该分支不需要 checkpoint。TorchFAST、affine、FNIRT-style 优化、Jacobian 和 modulation 都没有模型权重。

完整参数、坐标与 Jacobian 约定、Python BatchRunner 多 GPU 示例、FSL/UKB 对应和验证入口见 [`docs/fast_vbm/README.md`](../../../docs/fast_vbm/README.md)。公开统计只以 [`validation/fast_vbm`](../../../validation/fast_vbm/README.md) 为准。
