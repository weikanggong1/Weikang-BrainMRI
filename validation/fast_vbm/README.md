# FastVBM 验证记录

[返回 FastVBM 文档](../../docs/fast_vbm/README.md) · [FSL/UKB 参考方法](../../docs/ukb_vbm/README.md)

本目录按实现版本保存验证结果。当前 0.9 FastVBM 的两个后端共用 FAST GM、
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

### 历史 FSL wall time

| 历史 FSL 阶段 | n | wall time median [Q1–Q3] (s) |
| --- | --- | --- |
| preprocessing through FAST | 9 | 2736.085 [2370.179–3041.439] |
| GM registration + modulation | 10 | 897.294 [833.112–937.614] |
| raw T1w through modulated GM | 9 | 3637.203 [3195.138–3840.994] |

| nonlinear backend | paired n | FSL wall / candidate compute+save |
| --- | --- | --- |
| TorchFNIRT | 9 | 21.381 [17.021–25.807] |
| SynthMorph | 9 | 92.367 [86.865–115.321] |

FSL 数值来自已有日志；原始硬件和线程数没有记录，部分 raw-T1 timing 还因前处理阶段
缺失而被排除。候选运行也在共享节点上完成，而且两端算法、设备与中间文件 I/O 边界
不同。上述比值只描述这批病例的观测 wall time，不能解释为受控 CPU/GPU speedup。

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
`not_evaluated`，不能沿用旧版本结果补成通过。

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

另保留一次关闭 TF32 的早期 FLIRT 运行：10 例中 9 例通过，matrix RMS difference
中位数为 0.00875803 mm、最大值为 0.0541455 mm。它只用于历史追溯，不与当前默认
TF32 结果合并；见
[`flirt_exact_target_10case.public.json`](flirt_exact_target_10case.public.json)。

## 历史版本记录

下列 0.8、0.7 和 0.6 结果均用于版本追溯，不是当前 0.9 共享链路的验证结果。
0.8 使用旧 NCC/Adam affine 和旧 FNIRT-style nonlinear backend；0.7 只含当时的
SynthMorph 分支；0.6 使用更早的独立非线性优化器。

历史 0.8 双后端报告使用同一批 raw T1w、同一 GM template、同一 FSL/UKB reference 输出和
固定 template mask。warped GM、nonlinear-only Jacobian 与 modulated GM 分别报告
Pearson、MAE、RMSE 和 Dice@0.2；计时区分 CUDA-synchronized API 计算、NIfTI 写出、
模型构造和整组 invocation wall time。FSL reference 是 UKB v1.5 脚本步骤在 FSL
6.0.7.4 上的方法级复现；没有使用 UKB 冻结的旧 FSL 生产构建，因此不能标为逐字节
或原生产环境复现。

| 记录 | 状态 | 文件 |
| --- | --- | --- |
| 0.8 源码测试、构建和 wheel 导入验收 | 历史版本已完成 | [`release.v0.8.public.json`](release.v0.8.public.json) |
| 0.8 的 10 例 SynthMorph/FNIRT-style 三层配对对照 | 历史版本已完成 | [`report.v0.8.public.json`](report.v0.8.public.json)、[`backend_comparison.v0.8.public.csv`](backend_comparison.v0.8.public.csv) |
| 0.8 旧 TorchFLIRT 输入、输出网格和 FSL matrix 消费验证 | 历史版本已完成 | [`flirt_io.v0.8.public.json`](flirt_io.v0.8.public.json) |
| 0.7 线性配准后端对照 | 历史版本已完成 | [`linear_backends.v0.7.public.json`](linear_backends.v0.7.public.json) |
| 0.7 的 10 例真实 T1w 批量验证 | 历史版本已完成 | [`report.v0.7.public.json`](report.v0.7.public.json) |
| 0.7 PyTorch/FreeSurfer SynthMorph 一致性 | 历史版本已完成 | [`synthmorph_parity.v0.7.public.json`](synthmorph_parity.v0.7.public.json) |
| 0.7 FSL scaled-mm 与 world-RAS 坐标转换 | 历史版本已完成 | [`warp_coordinates.v0.7.public.json`](warp_coordinates.v0.7.public.json) |
| 0.6 的 10 例、CPU/CUDA 与安装包验收 | 历史结果 | [`report.v0.6.public.json`](report.v0.6.public.json)、[`cpu_cuda.v0.6.public.json`](cpu_cuda.v0.6.public.json) |

## 历史记录：0.8 的 10 例双后端对照

验证使用 10 例真实 T1w 和 UKB GM template。评估 mask 在处理前固定为
`template_GM > 0.01`，共 207,268 个体素；所有 scalar output 均先检查 shape 和
voxel-to-world affine 与模板一致。三层实验逐步固定上游输入：

| 层级 | 输入和可变步骤 |
|---|---|
| end-to-end | raw T1w；本包 SynthStrip、TorchFAST、PyTorch FLIRT 和所选非线性后端 |
| matched GM | 两端使用同一 FSL FAST GM；本包 PyTorch FLIRT 和所选非线性后端 |
| matched affine | 两端使用同一 FSL FAST GM 和转换后的同一 FSL FLIRT affine；只替换非线性后端 |

下表均为 10 例中位数，参考端为 FSL 6.0.7.4 按 UKB v1.5 方法生成的结果。Pearson 和
Dice 越高表示越接近 FSL reference，MAE 和 RMSE 越低表示越接近。

### Warped GM

| 层级 | 后端 | Pearson | MAE | RMSE | Dice@0.2 |
|---|---|---:|---:|---:|---:|
| end-to-end | SynthMorph | 0.622648 | 0.226509 | 0.336405 | 0.838938 |
| end-to-end | FNIRT-style | 0.715598 | 0.189795 | 0.289572 | 0.868437 |
| matched GM | SynthMorph | 0.642618 | 0.218398 | 0.330894 | 0.850312 |
| matched GM | FNIRT-style | 0.762635 | 0.168323 | 0.269371 | 0.888726 |
| matched affine | SynthMorph | 0.643430 | 0.218148 | 0.330396 | 0.851953 |
| matched affine | FNIRT-style | 0.768303 | 0.166854 | 0.267016 | 0.888884 |

### Modulated GM

| 层级 | 后端 | Pearson | MAE | RMSE | Dice@0.2 |
|---|---|---:|---:|---:|---:|
| end-to-end | SynthMorph | 0.535652 | 0.287252 | 0.443202 | 0.835073 |
| end-to-end | FNIRT-style | 0.677062 | 0.231171 | 0.357691 | 0.866982 |
| matched GM | SynthMorph | 0.563404 | 0.277751 | 0.432140 | 0.847820 |
| matched GM | FNIRT-style | 0.732124 | 0.209591 | 0.331914 | 0.885593 |
| matched affine | SynthMorph | 0.562853 | 0.279279 | 0.434065 | 0.849825 |
| matched affine | FNIRT-style | 0.733626 | 0.210787 | 0.333675 | 0.886592 |

同病例配对差定义为 `FNIRT-style − SynthMorph`。因此 Pearson/Dice 的正值和 MAE/RMSE
的负值都表示 FNIRT-style 更接近 FSL reference：

| 层级 | Warped GM Δ：Pearson / MAE / RMSE / Dice | Modulated GM Δ：Pearson / MAE / RMSE / Dice | Jacobian Pearson Δ |
|---|---|---|---:|
| end-to-end | +0.098899 / −0.038713 / −0.046917 / +0.030486 | +0.146343 / −0.053569 / −0.075454 / +0.031554 | +0.408424 |
| matched GM | +0.114449 / −0.047580 / −0.060559 / +0.038174 | +0.171570 / −0.067733 / −0.097645 / +0.037651 | +0.447229 |
| matched affine | +0.117349 / −0.048244 / −0.061676 / +0.037447 | +0.172227 / −0.068725 / −0.102479 / +0.037143 | +0.456698 |

### Jacobian 与运行时间

| 层级 | 后端 | Jacobian Pearson | Jacobian MAE | Jacobian RMSE | 非正 Jacobian fraction | CUDA-synchronized compute，中位数 [IQR] |
|---|---|---:|---:|---:|---:|---:|
| end-to-end | SynthMorph | 0.308498 | 0.247797 | 0.365173 | 0 | — |
| end-to-end | FNIRT-style | 0.723328 | 0.173258 | 0.258433 | 0 | 13.321 [11.360–15.326] s |
| matched GM | SynthMorph | 0.318663 | 0.245196 | 0.363990 | 0 | 7.215 [6.802–7.477] s |
| matched GM | FNIRT-style | 0.766695 | 0.160076 | 0.247398 | 0 | 6.146 [5.171–6.955] s |
| matched affine | SynthMorph | 0.309328 | 0.249027 | 0.367283 | 0 | 5.958 [5.655–6.178] s |
| matched affine | FNIRT-style | 0.769513 | 0.160431 | 0.245940 | 0 | 2.605 [2.428–3.146] s |

端到端 SynthMorph 的 10 个 accuracy output 来自同一病例的既有 0.7 运行，本次只重算
固定 mask 指标；`executed_outputs=0`、`reused_outputs=10`、`executed_timings=0`，所以
上表没有为它补写计时。其他五个组合各实际执行 10 例并记录 CUDA-synchronized API
时间。FNIRT-style 的 end-to-end 保存时间中位数为 0.308 s，compute + save 为
13.664 s。SynthMorph matched 注册器的 eager checkpoint setup 为 16.363 s，独立于
逐例 API compute。FNIRT 三层 30 次调用总 wall time 为 253.838 s；SynthMorph 两个
matched 层 20 次调用总 wall time 为 159.475 s。

FSL reference 的计时口径如下：

| FSL 阶段 | 中位数 [IQR] |
|---|---:|
| preprocessing through FAST | 2675.898 [2372.092–2999.863] s |
| GM registration + Jacobian modulation | 897.294 [833.112–937.614] s |
| raw T1w through modulated GM | 3646.025 [3226.343–3814.595] s |

同病例配对运行时间按可对齐的阶段计算：

| 配对边界 | reference / candidate compute，中位数 | reference − candidate compute，中位数 | reference / candidate compute+save，中位数 |
|---|---:|---:|---:|
| FSL raw T1→modulated GM / FNIRT-style end-to-end | 280.881× | 3634.188 s | 273.663× |
| FSL GM registration+modulation / SynthMorph matched GM | 127.049× | 890.349 s | 122.137× |
| FSL GM registration+modulation / FNIRT-style matched GM | 148.221× | 890.580 s | 142.099× |

matched affine 没有对应的 FSL nonlinear-only 分阶段时间，因此不计算时间比。上表是同病例
配对后的描述性比值；它同时包含 CPU/GPU、算法和实现差异。

候选后端在 H100 PCIe、PyTorch 2.5.1、CUDA 11.8、4 CPU threads 上运行；FSL 时间来自
既有 CPU reference。节点负载没有隔离，且算法、前处理和运行设备不同，所以这些数字
描述实际运行时间，不能把 FSL/候选时间比解释为纯 GPU 加速倍数。

在这个固定 cohort、template mask 和 FSL reference 下，FNIRT-style 在三个层级的
warped GM、modulated GM 与 Jacobian 指标均比 SynthMorph 更接近 FSL。这个结论衡量
FSL 相似性；FSL 不是人工解剖真值，因此不能据此推断 FNIRT-style 的解剖准确性或其他
数据集上的优劣。完整分布、计时边界、执行/复用计数和隐私边界见 0.8 JSON；CSV 提供
同一聚合的表格形式。0.7 章节使用全模板网格计算旧指标，不能与本节固定前景 mask 的
数值直接比较。

## 历史记录：0.8 TorchFLIRT 输入输出合同

本节检查的是 0.8 的旧 NCC/Adam `TorchFLIRT`，不是 0.9 source-derived
`TorchFLIRT`。一例真实 T1w-derived GM 用于检查其文件合同。package output 与 FSL
`applywarp --premat` 都使用本包写出的同一个 `.mat`，因此这个实验隔离的是 matrix
方向、FSL scaled-mm handedness 转换、reference grid 和重采样坐标，不比较两个优化器
各自估计出的 matrix。

| 检查 | 结果 |
|---|---:|
| output shape 与 reference | 相同 |
| output affine 最大绝对差 | 0 |
| package/FSL dtype | float32 / float32 |
| package/FSL qform、sform code | 2、2 / 2、2 |
| 同 matrix、whole-grid scalar Pearson | 0.999999999961 |
| 同 matrix、whole-grid scalar MAE / RMSE | 6.58e-7 / 2.38e-6 |
| 同 matrix、template>0.01 scalar Pearson | 0.999999999922 |
| 同 matrix、template>0.01 scalar MAE / RMSE | 2.74e-6 / 4.87e-6 |
| coordinate-ramp 最大误差，x / y / z | 2.58e-5 / 3.35e-5 / 2.56e-5 voxel |

FSL `flirt -applyxfm` 与 `applywarp --premat` 在降采样时有不同的最终插值行为。package
output 对齐 UKB `fsl_reg` 最终使用的 `applywarp` trilinear 路径；与
`flirt -applyxfm -interp trilinear` 的 whole-grid Pearson/MAE/RMSE 为
0.955340/0.031970/0.081995。因此验证结论是 input/reference 角色、reference-grid
header、scaled-mm matrix 和 `applywarp` 采样坐标一致；`TorchFLIRT` 仍使用独立 NCC +
Adam，不能据此声称与 FSL FLIRT 优化器或所有 `-applyxfm` 插值模式数值等价。

## 历史记录：0.7 线性配准后端对照

匹配实验使用同一幅常驻内存的真实 GM 图像、同一幅 group GM template 和零初始化。
LibTorch C++/ATen CUDA 与 Python/PyTorch 都执行 12 DOF、NCC 加同一正则项、
`4/2/1` 三层分辨率、每层 `80/60/50` 个 Adam step；两端都缓存采样网格，并在计时
边界调用 CUDA synchronize。五次热运行交替执行，避免固定先后顺序。

| 实现 | 五次热运行中位数 | 独立进程冷启动计算 |
| --- | ---: | ---: |
| LibTorch C++ 控制层 + ATen CUDA | 1.2978 s | 1.942 s |
| Python + PyTorch CUDA | 1.9351 s | 4.993 s |

C++ GPU 路径的热运行是 Python 路径的 1.49 倍速度，每例节省 0.6373 s。两种实现的
输出 Pearson 为 0.997258、MAE 为 0.005691、RMSE 为 0.020348，阈值 0.5 的 Dice
为 0.974168。

这里的 C++ GPU 路径是独立的 LibTorch 控制程序，计算由 ATen CUDA 完成；它不是
FSL FLIRT 源码的 CUDA 移植。正式包继续使用 Python/PyTorch，因为 0.64 s 左右的热
运行收益不足以抵消 C++ 扩展的编译、PyTorch ABI 和多平台 wheel 维护成本。

### FSL FLIRT 参考运行

FSL FLIRT 是 CPU C++ 程序。一次 GPU 主机上的 `normcorr` 运行用时 61.70 s；另一个
CPU 主机的三次 `normcorr` 为 42.0819、44.5421、46.1724 s，中位数 44.5421 s；三次
`corratio` 为 43.0572、41.8134、42.5524 s，中位数 42.5524 s。两台共享节点在测试
期间都有严重争用，这些时间只能描述当次运行。

FSL 与两个 GPU 实现没有匹配优化器和搜索策略，不能把时间或输出差异归因于语言、
CPU/GPU 后端。统一重采样到 template 网格后，FSL 与 C++ GPU 的 Pearson/Dice 为
0.934779/0.853187，与 Python/PyTorch 的 Pearson/Dice 为 0.934871/0.853429；输出与
template 的相关分别为 0.808364、0.717773 和 0.717664。这些数值用于界定实现差异，
不构成后端速度的因果比较。

## 历史记录：0.8 坐标和 warp 约定

FSL 与 FreeSurfer/SynthMorph 的变换不能按数组元素直接比较：

- FSL FLIRT `.mat` 表示 input 到 reference 的 **scaled-mm** 变换，并不是 RAS 世界坐标矩阵。
- FastVBM 的线性结果保存 moving 到 fixed 的 world-RAS 仿射；重采样使用它的逆，即
  fixed 到 moving 的 world-RAS pull。
- PyTorch SynthMorph 在 fixed 网格上使用 fixed 到 moving 的 target-to-source pull；
  对外保存前转换为物理 RAS displacement。
- PyTorch FNIRT-style 后端内部把 affine 和 nonlinear residual 表示在 FSL scaled-mm；
  对外同样转换为 fixed-grid target-to-source world-RAS displacement。

因此，FSL 对照先比较同一 template 网格上的 warped volume。若后续比较位移场，必须
先把两端转换到同一 fixed 网格、同一 fixed-to-moving 方向和同一物理 RAS 基底，并
明确是否已经合入线性项。本目录不会把 FSL `.mat` 当作 RAS 仿射，也不会直接逐元素
比较原始 FSL warp 与 SynthMorph warp。

坐标转换另用一例真实 GM 做了三项检查。`img2imgcoord` 的 3 个点与转换后结果最大相差
0.00003743 voxel；3 个坐标 ramp 在 5 个选定点的最大差为 0.000237 voxel。最近邻
体积重采样的 Pearson 为 0.9999798、MAE 为 6.82e-6，99.9980% 的体素完全相同。
这些结果验证的是 scaled-mm 基底和变换方向处理后的点坐标映射。独立 `TorchFLIRT`
因此可以按 FSL input → reference scaled-mm 契约写出 `.mat`；它没有复现 FSL 的搜索、
优化器、插值和边界实现，不能据此声称估计矩阵或任意图像具有 voxelwise parity。

FSL FNIRT coefficient file、dense field 和本包 Surfa `disp_ras` warp 不是同一文件格式。
FNIRT-style 分支按 FSL residual 定义输出 nonlinear-only Jacobian，但不读写 `--cout` 或
`--fout`。双后端报告比较同一 template 网格上的最终影像，不直接逐元素比较这些 raw
warp 表示。

## 历史记录：0.7 的 10 例真实 T1w

10 例均通过 Python `BatchRunner` 完成。运行时另一张 GPU 被占用，因此两个 worker
同驻 `cuda:1`，每个 worker 处理 5 例。130/130 个有限值检查和 130/130 个输入或
template 几何检查全部通过。批量墙钟时间为 396.41 s；单例 `FastVBM.__call__`
中位数为 24.03 s，后者不含批量 NIfTI 写出。墙钟时间还包括 worker 与模型启动和
输出写盘，测试节点存在严重共享争用，所以这些时间只描述本次运行。

PVE 位于 0–1，建模 mask 内三类 PVE 和的最大误差为 0。SynthStrip mask 中有 6,943
个不属于 TorchFAST 有效强度支持的边缘体素，占 0.0218%。线性配准 NCC 中位数为
0.74468，范围为 0.71771–0.76294；SynthMorph 后最终 NCC 中位数为 0.82487，范围为
0.79060–0.83591。非线性 Jacobian 范围为 0.23179–3.67057，10 例中没有非正
Jacobian；最大非线性位移的中位数为 7.184 mm，范围为 6.506–7.890 mm。

与 FSL/FNIRT 参考结果的比较只在同一 template 网格上进行。warped GM 的 Pearson、
MAE、RMSE 和 Dice@0.2 中位数分别为 0.81540、0.05523、0.16659 和 0.82551；
modulated GM 分别为 0.74673、0.07043、0.22203 和 0.82138。FastVBM 的仿射优化器、
SynthMorph 非线性模型与 FSL/FNIRT 不同，这些数值描述两条完整 pipeline 的输出差异，
不表示 FNIRT warp 与 SynthMorph warp 可以直接互换。

## 历史记录：0.7 SynthMorph 一致性

FastVBM 在运行时直接调用本包
`freesurfer_torch.synthmorph.SynthMorph(model="deform")`。FreeSurfer
`mri_synthmorph register -m deform` 只作为外部参考，不是 FastVBM 的运行时依赖。
一致性测试固定同一幅 moving GM、fixed template、官方
`synthmorph.deform.3.h5`、同一 moving-to-fixed world-RAS 初始仿射，并都使用
`mid_space=False`。

两端 warped GM 位于同一 fixed 网格。Pearson 为 0.999999999929，MAE 为 9.17e-7，
RMSE 为 3.23e-6，99% 绝对差为 1.59e-5，最大绝对差为 7.53e-5，Dice@0.2 为 1.0。
由于 FreeSurfer 与包内场的存储方向和表示需要先统一，本测试没有直接逐元素比较 raw
warp。

本包单独加载模型用时 8.54 s，同一初始变换下推理为 10.97 s，冷启动总计 19.51 s；
FreeSurfer 外部参考冷启动为 475.07 s，本次观测比值为 24.35。两个命令都是共享、
严重争用 GPU 节点上的单次冷进程，这一比值不能作为隔离硬件加速倍数。

## 历史记录：0.8 与 0.7 测试和安装包验收

下列模块名、类名和安装包内容是 **0.8 当时归档**。旧模块与类已从当前安装包删除，
不是当前 API。

0.8.0 在 gpucw1 完成 70 项 FastVBM/批量/API 定向测试，并完成全量源码测试：
210 passed、9 skipped。16 条 warning 来自 Surfa 对弃用 NumPy binary `fromstring` 用法的
提示。wheel 和 sdist 均包含新的 `flirt.py` 与 `fnirt_backend.py`，且不含模型权重、
checkpoint 或 MRI 文件。

0.8.0 wheel 从临时 target 目录安装，并在 `/tmp` 以已有运行时依赖导入；版本、
`FastVBM`、`TorchFLIRT`、`PyTorchFNIRTRegistration`、两个 FLIRT 坐标转换函数以及
单被试 CLI 的 FLIRT/FastVBM 参数均通过检查。wheel 含 72 个文件，大小 171,499 bytes，
SHA-256 为 `3c46c13c23237daaab4d0178f6e41fe8cb87b69103cc90f871105d44a4559863`；
sdist 含 214 个文件，大小 1,617,100 bytes，SHA-256 为
`4c8f5ebd5683e4c36a00c240a852ede78c28384002ef26098cd0eb5feb0cd70a`。完整字段见
[`release.v0.8.public.json`](release.v0.8.public.json)。

以下是 0.7.0 的历史验收记录。

最终源码测试为 148 passed、3 skipped；4 条 warning 来自既有 Surfa 测试路径。
0.7.0 wheel 在独立环境中从 `/tmp` 导入，版本、FastVBM API、FLIRT 坐标转换函数、
单被试 CLI 帮助和两个 FastVBM 官方权重的离线校验均通过。解包后的
`synthmorph_backend.py` 确认调用本包
`freesurfer_torch.synthmorph.SynthMorph(model="deform")`，没有外部
`mri_synthmorph` 调用。

wheel 和 sdist 分别包含 70 和 210 个文件，均未包含 checkpoint、模型权重、MRI、
GM template 或私有 `work` 文件。wheel 为 157,084 字节，SHA-256 是
`3ac9cf93692ed0f4906271d59684b83423faf8eacc02f48c93f966c4a1f64104`；sdist 为
1,538,779 字节，SHA-256 是
`2572cd511434014ef1869b2a1d1b0d1884e2dbb4fa3818f2016ce1a3cf05e9e3`。
完整字段见 [`report.v0.7.public.json`](report.v0.7.public.json)。

## 0.6 历史结果

[`report.v0.6.public.json`](report.v0.6.public.json) 保存 0.6 包的 10 例双 GPU、旧
registration backend 回归和安装包验收；[`cpu_cuda.v0.6.public.json`](cpu_cuda.v0.6.public.json)
保存缩短优化步数的一例 CPU/CUDA smoke。它们对应 0.6.0 的独立非线性优化器和
`deformation_scale` 回退机制，不能作为 0.7 SynthMorph pipeline 的验证结果。

公开记录不含源病例标识、私有数据路径或 PID。新组件报告中的 `case01` 至
`case10` 是与源身份断开的顺序别名；含逐例数值的报告会明确标注该范围。运行
`sha256sum -c SHA256SUMS` 可校验本目录纳入清单的公开记录。0.9 的
[detached release manifest](https://github.com/weikanggong1/Weikang-BrainMRI/blob/main/validation/fast_vbm/release.v0.9.public.json)
在 wheel 和 sdist 构建完成后生成，用于记录两个归档本身的 SHA-256；它不写入
sdist，也不纳入 `SHA256SUMS`，从而避免归档哈希自引用。
