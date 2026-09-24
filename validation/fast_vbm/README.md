# FastVBM 验证记录

[返回 FastVBM 文档](../../docs/fast_vbm/README.md) · [FSL/UKB 方法对照](../fast/README.md#原始-t1-到-vbm)

本目录按实现版本保存验证结果。FastVBM 0.7 的配准链为：PyTorch 12 DOF
仿射配准、PyTorch SynthMorph deform 非线性配准、非线性 Jacobian 调制。0.6 使用的
独立非线性优化器已经退出当前 pipeline，其结果只作历史记录。

| 记录 | 状态 | 文件 |
| --- | --- | --- |
| 0.7 线性配准后端对照 | 已完成 | [`linear_backends.v0.7.public.json`](linear_backends.v0.7.public.json) |
| 0.7 的 10 例真实 T1w 批量验证 | 已完成 | [`report.v0.7.public.json`](report.v0.7.public.json) |
| 0.7 PyTorch/FreeSurfer SynthMorph 一致性 | 已完成 | [`synthmorph_parity.v0.7.public.json`](synthmorph_parity.v0.7.public.json) |
| 0.7 FSL scaled-mm 与 world-RAS 坐标转换 | 已完成 | [`warp_coordinates.v0.7.public.json`](warp_coordinates.v0.7.public.json) |
| 0.6 的 10 例、CPU/CUDA 与安装包验收 | 历史结果 | [`report.v0.6.public.json`](report.v0.6.public.json)、[`cpu_cuda.v0.6.public.json`](cpu_cuda.v0.6.public.json) |

## 0.7 线性配准后端对照

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

## 坐标和 warp 约定

FSL 与 FreeSurfer/SynthMorph 的变换不能按数组元素直接比较：

- FSL FLIRT `.mat` 表示 input 到 reference 的 **scaled-mm** 变换，并不是 RAS 世界坐标矩阵。
- FastVBM 的线性结果保存 moving 到 fixed 的 world-RAS 仿射；重采样使用它的逆，即
  fixed 到 moving 的 world-RAS pull。
- PyTorch SynthMorph 在 fixed 网格上使用 fixed 到 moving 的 target-to-source pull；
  对外保存前转换为物理 RAS displacement。

因此，FSL 对照先比较同一 template 网格上的 warped volume。若后续比较位移场，必须
先把两端转换到同一 fixed 网格、同一 fixed-to-moving 方向和同一物理 RAS 基底，并
明确是否已经合入线性项。本目录不会把 FSL `.mat` 当作 RAS 仿射，也不会直接逐元素
比较原始 FSL warp 与 SynthMorph warp。

坐标转换另用一例真实 GM 做了三项检查。`img2imgcoord` 的 3 个点与转换后结果最大相差
0.00003743 voxel；3 个坐标 ramp 在 5 个选定点的最大差为 0.000237 voxel。最近邻
体积重采样的 Pearson 为 0.9999798、MAE 为 6.82e-6，99.9980% 的体素完全相同。
这些结果验证的是 scaled-mm 基底和变换方向处理后的点坐标映射。FSL `applyxfm` 与
PyTorch `grid_sample` 的插值、边界和坐标归一化实现不同，不能据此声称任意图像及
任意插值设置下均有 voxelwise parity。

## 0.7 的 10 例真实 T1w

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

## 0.7 SynthMorph 一致性

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

## 测试与安装包验收

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

所有公开 JSON 都不含病例标识、源数据路径、PID 或逐例结果。在本目录运行
`sha256sum -c SHA256SUMS` 可校验公开记录。
