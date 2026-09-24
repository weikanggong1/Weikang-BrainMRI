# FastVBM 模块

`FastVBM` 将 raw T1w 处理为模板空间 modulated GM：

```text
raw T1w
  → SynthStrip
  → TorchFAST 三组织 PVE + bias-field correction
  → 独立 PyTorch FLIRT-compatible 12-DOF affine（NCC + Adam）
  → 本包 PyTorch SynthMorph deform（官方权重，init affine）
  → nonlinear-only pull Jacobian
  → warped GM × Jacobian
```

线性阶段只在 12-DOF 参数含义和用途上与 FLIRT 相容。FSL FLIRT 默认使用
correlation ratio 和 Brent/direction-set 优化；本模块使用 normalized correlation
和 Adam，因此不属于 FSL FLIRT 的数值等价实现。非线性阶段直接调用本包
`freesurfer_torch.synthmorph.SynthMorph(model="deform")`，读取官方
`synthmorph.deform.3.h5`，并计算 nonlinear-only pull Jacobian。

FSL FLIRT `.mat` 是 input → reference 的 scaled-mm 变换，不能直接当作 world-RAS
矩阵。本模块的线性结果是 moving → fixed world-RAS 仿射及其 fixed → moving 逆矩阵；
SynthMorph/Surfa warp 定义在 fixed 网格，`disp_ras` 保存 fixed → moving 的
target-to-source pull displacement。转换公式和 Jacobian 约定见完整文档。

## 单被试 Python 调用

```python
from freesurfer_torch import FastVBM

pipeline = FastVBM(device="cuda:0", threads=4)
result = pipeline.run(
    "subject_T1w.nii.gz",      # 单帧 3D raw T1w
    "template_GM.nii.gz",      # fixed GM；决定模板空间输出网格
    "results/sub-01",          # 13 幅影像和 JSON 报告的目录
    overwrite=False,            # 已有同名文件时停止
)
```

构造一次 `pipeline` 后可继续调用，已加载的 SynthStrip 和 SynthMorph 模型会复用。
只需内存结果时使用
`result = pipeline("subject_T1w.nii.gz", "template_GM.nii.gz")`；已有同网格 mask
时增加 `brain_mask="mask.nii.gz"`，可跳过 SynthStrip。

## 单被试命令行

```bash
fs-torch fast-vbm \
  -i subject_T1w.nii.gz \
  --template template_GM.nii.gz \
  -o results/sub-01 \
  --device cuda:0 \
  --threads 4
```

`-i` 是 raw T1w，`--template` 是 fixed GM 模板，`-o` 是完整输出目录，
`--device` 选择计算设备，`--threads` 设置当前进程的 PyTorch CPU 线程数。多被试只用
Python `BatchRunner`；每个 worker 固定到一张 GPU，并按相同 model 配置缓存一套
FastVBM 模型。

## 输出

13 个稳定结果键及文件名为：

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

后三项 VBM 文件在名称、模板网格用途和 modulation 公式上对应 UKB v1.5：

```bash
fsl_reg T1_brain_pve_1.nii.gz template_GM.nii.gz \
  T1_GM_to_template_GM -fnirt \
  "--config=GM_2_MNI152GM_2mm.cnf --jout=T1_GM_JAC_nl"
fslmaths T1_GM_to_template_GM -mul T1_GM_JAC_nl \
  T1_GM_to_template_GM_mod -odt float
```

FastVBM 用 PyTorch affine + SynthMorph deform 替代 FLIRT + FNIRT，不能把结果解释为
FSL 数值等价。脑提取对应本包对官方 `mri_synthstrip` 的实现；非线性调用对应
`mri_synthmorph register -m deform -i INIT ...` 且不启用 `-M`。

## 权重与完整说明

```bash
python tools/setup_weights.py --model fast-vbm
```

该模型名配置 `synthstrip.1.pt` 和 `synthmorph.deform.3.h5`。TorchFAST、线性配准、
Jacobian 和 modulation 不需要其他 checkpoint。

完整构造参数、返回字段、13 项输出含义、Python BatchRunner 多 GPU 示例、FSL/UKB
逐阶段对照和验证入口见
[`docs/fast_vbm/README.md`](../../../docs/fast_vbm/README.md)。当前统计和图示只以
[`validation/fast_vbm`](../../../validation/fast_vbm/README.md) 为准。
