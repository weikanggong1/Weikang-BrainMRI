# FastVBM 验证记录

[返回 FastVBM 文档](../../docs/fast_vbm/README.md) · [FSL/UKB 参考方法](../../docs/ukb_vbm/README.md)

本目录只保留当前 0.9 FastVBM 的两个后端共用 FAST GM、
source-derived `TorchFLIRT`、FSL 坐标契约、GPU `TorchApplyWarp`、dense
nonlinear-only Jacobian 和 modulation。唯一影响输出的分支是 nonlinear estimator：
PyTorch SynthMorph deform 或 source-derived `TorchFNIRT` GM config。reference mask
写入两分支的共同上下文和 `pre_nonlinear_signature`；只有 `TorchFNIRT` estimator
使用它，SynthMorph 网络没有 mask 输入。

## 当前 0.9 验证状态

| 记录 | 状态 | 文件 |
|---|---|---|
| source-derived `TorchFLIRT`，默认 TF32，10 例真实 GM | 10/10 通过预设 matrix functional gate（`rmsdiff ≤ 0.05 mm`） | [`report.v0.9.public.json`](report.v0.9.public.json) |
| source-derived `TorchFNIRT`，10 例独立 matched-input 对照 | 已完成；相关很高但不满足数值等价 | [`fnirt_fsl_10case.v0.9.public.json`](fnirt_fsl_10case.v0.9.public.json) |
| FNIRT 最终源码继承核验 | 已完成；这是源码继承证明，不是新的数值运行 | [`fnirt_source_equivalence.v0.9.public.json`](fnirt_source_equivalence.v0.9.public.json) |
| `TorchApplyWarp` 已支持子集 | 已完成 | [`../applywarp/report.json`](../applywarp/report.json) |
| 当前共享链路的 10 例双后端 FastVBM 对照 | 已完成；正式报告只运行 `end_to_end` | [`report.v0.9.public.json`](report.v0.9.public.json) |

正式 FastVBM 报告只包含 raw T1w 起点的 `end_to_end` 层。下面的 direct FNIRT
结果来自另一项独立 matched-input 实验；它固定 FSL FAST GM、模板、官方 FSL
FLIRT affine 和 reference mask，不能写成这次正式报告又运行了 matched-GM 或
matched-affine 两层。

### FLIRT reference-suite matrix gate

| 指标 | median [Q1–Q3] | maximum | 判据 |
| --- | --- | --- | --- |
| matrix RMS difference | 0.008544 [0.006265–0.019436] mm | 0.028984 mm | 10/10 ≤ 0.05 mm |
| CUDA-synchronized compute | 24.015 [23.100–30.327] s | 85.818 s | 描述性计时 |

这是指定 10 例 reference suite 的 tolerance-based matrix functional gate。
运行时 QC 仍固定报告 `validated_fsl_equivalent=false`、
`current_input_compared_with_fsl=false` 和
`complete_numerical_equivalence_claimed=false`。
`reference_validation_matrix_gate_passed=true` 只记录上述套件事实，不表示当前输入已与
FSL 比较，也不表示 bitwise 或完整数值等价。

### End-to-end：与 UKB/FSL reference 的输出差异

下表为 10 例 median [Q1–Q3]。Pearson 和 Dice 越高、MAE 和 RMSE 越低，表示更接近
本次 FSL reference；FSL reference 不是人工解剖真值。

| nonlinear backend | 输出 | Pearson | MAE | RMSE | Dice |
| --- | --- | --- | --- | --- | --- |
| TorchFNIRT | warped GM | 0.898466 [0.888843–0.923235] | 0.085994 [0.072976–0.091053] | 0.174951 [0.152526–0.182325] | 0.914901 [0.910153–0.923871] |
| TorchFNIRT | nonlinear-only Jacobian | 0.894475 [0.876889–0.915720] | 0.102445 [0.090271–0.107419] | 0.167041 [0.141769–0.172346] | 0.862602 [0.848450–0.882966] |
| TorchFNIRT | modulated GM | 0.886469 [0.877303–0.907874] | 0.105144 [0.088433–0.110944] | 0.219846 [0.192876–0.236301] | 0.911365 [0.906487–0.920568] |
| SynthMorph | warped GM | 0.721780 [0.702484–0.730865] | 0.170587 [0.163596–0.178022] | 0.291724 [0.283062–0.302674] | 0.828445 [0.822360–0.836998] |
| SynthMorph | nonlinear-only Jacobian | 0.310599 [0.286666–0.325480] | 0.226411 [0.214676–0.230775] | 0.335042 [0.329786–0.351709] | 0.640193 [0.634200–0.656967] |
| SynthMorph | modulated GM | 0.636723 [0.621506–0.648671] | 0.217052 [0.207119–0.226813] | 0.388714 [0.369035–0.395304] | 0.824325 [0.816793–0.831633] |

### End-to-end 运行时间

| nonlinear backend | CUDA-synchronized compute (s) | 3 个 VBM 输出写盘 (s) | compute + save (s) |
| --- | --- | --- | --- |
| TorchFNIRT | 179.841 [147.700–210.411] | 0.252 [0.249–0.258] | 180.096 [147.947–210.670] |
| SynthMorph | 34.974 [33.661–37.394] | 0.240 [0.238–0.246] | 35.216 [33.902–37.633] |

### Release gates

| 范围 | 通过 | 失败 | 未评估 | 状态 |
| --- | --- | --- | --- | --- |
| source-derived FLIRT component | 1 | 0 | 0 | 通过 |
| shared FastVBM chain | 3 | 0 | 0 | 通过 |
| FNIRT numerical-equivalence claim | 0 | 1 | 0 | 失败 |
| FNIRT functional scalar-output gate | 0 | 0 | 24 | 未评估 |
| 全部声明 gate | 4 | 1 | 24 | 有失败项 |

`FNIRT numerical-equivalence claim` 的失败项是实现明确报告
`fsl_fnirt_numerically_equivalent=false`。`FNIRT functional scalar-output gate` 属于
matched-GM/matched-affine 设计；正式报告只运行 `end_to_end`，所以这些 gate 为
`not_evaluated`，不能由其他实验结果补成通过。

### 独立 direct FNIRT matched-input 对照

| 输出 | Pearson | MAE | RMSE | maximum absolute error | NIfTI contract |
| --- | --- | --- | --- | --- | --- |
| coefficient | 0.999089 [0.998947–0.999240] | 0.033959 [0.032730–0.036433] | 0.074783 [0.071679–0.082324] | 2.106445 [2.005102–3.042672] | 10/10 |
| nonlinear residual | 0.999508 [0.999457–0.999613] | 0.028455 [0.026858–0.030319] | 0.051817 [0.047223–0.054161] | 0.980345 [0.839263–1.141533] | 10/10 |
| warped GM (`iout`) | 0.998695 [0.998625–0.998868] | 0.003101 [0.003057–0.003363] | 0.013682 [0.012832–0.014601] | 0.642930 [0.596209–0.775278] | 10/10 |
| nonlinear-only Jacobian (`jout`) | 0.999267 [0.999025–0.999502] | 0.003395 [0.003065–0.003450] | 0.007417 [0.006340–0.008993] | 0.240333 [0.145420–0.300674] | 10/10 |
| modulated GM | 0.998507 [0.998036–0.998864] | 0.003759 [0.003488–0.004088] | 0.017655 [0.014843–0.019880] | 1.186854 [0.876111–1.486011] | 10/10 |

FSL 6.0.7.4 CPU FNIRT 为 855.583 [822.899–910.276] s，TorchFNIRT H100 为 1244.896 [1054.397–1337.331] s，FSL/Torch 比为 0.686 [0.652–0.750]。两端都来自共享节点；本次 TorchFNIRT 更慢，不能称为 GPU 加速。
warped GM、Jacobian 和 modulated GM 的中位 Pearson 分别为
0.998695、0.999267 和 0.998507，但绝对
误差明显大于舍入误差，因此保持 `fsl_fnirt_numerically_equivalent=false`。

### 对端到端相关偏低的解释边界

FLIRT 已通过这套 10 例 matrix gate，因此不能再把端到端相关偏低简单归为 affine
未达到该 gate。独立 direct FNIRT 固定上游输入后与 FSL 很接近，但仍不是数值等价；
同一 brain-only T1 输入上的 TorchFAST/FSL FAST GM Pearson 中位数为
0.98488。端到端还同时包含 SynthStrip 与 UKB 脑提取/前处理的差异和
TorchFAST/FSL FAST 的 GM 估计差异。SynthMorph 分支本来就采用不同的非线性算法，
而 modulated GM 是 warped GM 与 Jacobian 的乘积，两个量的偏差会共同进入结果。
当前正式报告没有新的 matched-GM 或 matched-affine 消融，不能用这些观察量化各步骤的
因果占比。

### 公开文件

当前 0.9 的正式报告、表格、测试摘要和构建清单为
[`report.v0.9.public.json`](report.v0.9.public.json)、[`backend_comparison.v0.9.public.csv`](backend_comparison.v0.9.public.csv)、[`test_summary.v0.9.public.json`](test_summary.v0.9.public.json) 和 [`release.v0.9.public.json`](release.v0.9.public.json)。Direct FNIRT 数值在
[`fnirt_fsl_10case.v0.9.public.json`](fnirt_fsl_10case.v0.9.public.json)，最终源码继承
边界在 [`fnirt_source_equivalence.v0.9.public.json`](fnirt_source_equivalence.v0.9.public.json)。
