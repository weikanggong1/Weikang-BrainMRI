# 官方实现与 PyTorch 实现的影像对照

前两张图使用仓库的[去面容 T1w 样例](../../examples/README.md)。原图来自 [OpenNeuro ds000114](https://openneuro.org/datasets/ds000114)，许可为 CC0；发布的三个文件保留脑掩膜外扩 12 mm 内的体素，来源和 SHA-256 见 [SOURCES.json](../../examples/data/SOURCES.json)。数据没有人工标注，以下数值只比较两种实现。

![原始 T1w 与两种脑提取结果](synthstrip_comparison.png)

SynthStrip 图使用 `sub-02`：从左到右是去面容输入、FreeSurfer 8.2.0 CPU 输出、本包 PyTorch GPU 输出。上、下分别是相同原网格体素的轴位和冠状位切面；三列用同一灰度范围。两种输出的掩膜 Dice 为 **1.000000**，不一致体素 **0**，脑图全体素最大绝对差 **0**，输出几何相同。这里的脑图强度已经由两种程序各自的掩膜处理。

![移动与固定影像及两种 joint 配准结果](synthmorph_comparison.png)

SynthMorph 图以 `sub-02` 为 moving、`sub-01` 为 fixed，使用默认 `joint` 模型。从左到右为 moving 脑图、fixed 脑图、FreeSurfer CPU 配准结果、本包 PyTorch GPU 配准结果；上、下为轴位与冠状位。配准后两列都在 **fixed 的原网格**；为便于看脑组织，图中仅对显示的配准切片使用同一个 fixed 脑掩膜。数值比较针对**完整保存影像和位移场**，未使用显示掩膜：配准影像归一化 RMSE 为 **1.7880 × 10⁻⁵**，RAS 位移向量最大误差 **0.000622 mm**，输出几何相同。强度范围对 moving 与两列配准影像一致；fixed 图像单独按其正强度分布设定亮度，因此 fixed 一列不能用于视觉判断强度误差。

机器可读指标见 [metrics.json](metrics.json)。12 例 T1w 的成对数值与时间比较见 [对照报告](../COMPARISON.md)。本页图示使用的原版单进程和本包双 GPU 批量运行条件不同，不据此计算加速比。

![同一公开 FLAIR 的 FreeSurfer 与本包 WMH-SynthSeg 输出](wmh_synthseg_comparison.png)

WMH-SynthSeg 图使用仓库的 [公开 FLAIR `sub-04`](../../examples/wmh_data/sub-04_FLAIR.nii.gz)。该文件来自 [OpenNeuro ds003592](https://openneuro.org/datasets/ds003592) CC0 原图，经原版 SynthStrip 脑掩膜外扩 6 mm 后清零其余强度；三例 FLAIR 的来源、处理步骤和 SHA-256 见 [FLAIR 清单](../../examples/wmh_data/SOURCES.json)。从左到右是输入、未改动的 FreeSurfer WMH-SynthSeg CUDA 源码、本包 PyTorch CUDA；上、下为相同输出网格的轴位与冠状位，红色显示标签 77。两次推理使用同一官方 checkpoint、`--crop`、GPU 和线程。**完整三维输出的标签不一致体素为 0，WMH Dice 为 1，概率图最大绝对差为 0，仿射矩阵相同**；图示指标见 [WMH 图示数据](wmh_metrics.json)。此结果是对官方实现的复现，不是对病灶真值的测量。

## 图示来源与重跑边界

前两张图和 [metrics.json](metrics.json) 由本包 **0.2.0**、FreeSurfer 8.2.0 和同一官方权重在 gpucw1 上生成。当时的 `examples/run_batch.py` 使用两张 GPU 的常驻 worker，写出脑图、掩膜、配准图像和变换；原版由 `examples/run_reference.py` 在 CPU 上运行。原版 CPU 脑提取两例分别耗时 50.28、52.42 秒，joint 配准耗时 212.65 秒；两侧执行条件不同，不能从图示运行时间计算加速比。

当前 `examples/run_batch.py` 已改为 pandas 表格接口的 SynthStrip 入门示例，会保存脑图、掩膜和距离场，但不运行 SynthMorph，也不采用历史图的输出布局。`examples/render_comparison.py` 仍按旧的文件布局和 0.2.0 标签写入指标；直接运行会覆盖历史 PNG 和 `metrics.json`。要做当前版本的新图，需另行生成配准结果并更新绘图脚本的输入路径和版本标签。当前多被试使用方法见[批量执行说明](../ARCHITECTURE.md#批量执行)，严格的历史计时比较见[详细对照报告](../COMPARISON.md)。

第三张图由本包 0.3.0 在 gpucw1 上生成。以下命令在仓库根目录运行；需另行安装 FreeSurfer 8.2.0-1，并在 CUDA Python 环境中安装本包及绘图依赖。原版 GPU 参考直接执行**未修改的** `inference.py`，脚本自动建立指向所选 FreeSurfer 安装及官方权重的临时目录：

```bash
python -m pip install . matplotlib
python examples/check_wmh_data.py
python tools/setup_weights.py --model wmh-synthseg
export WMH_WEIGHTS="$HOME/.cache/freesurfer_torch"
export FREESURFER_HOME=/path/to/freesurfer-8.2.0-1

python tools/benchmark_wmh.py --input-dir examples/wmh_data \
  --output-dir examples/results/wmh_reference_cuda --weights "$WMH_WEIGHTS" \
  --arm official-cuda --freesurfer-home "$FREESURFER_HOME" --threads 8
python tools/benchmark_wmh.py --input-dir examples/wmh_data \
  --output-dir examples/results/wmh_port_cuda --weights "$WMH_WEIGHTS" \
  --arm torch-cuda --threads 8
python examples/render_wmh_comparison.py \
  --image examples/wmh_data/sub-04_FLAIR.nii.gz \
  --reference examples/results/wmh_reference_cuda/sub-04_seg.nii.gz \
  --candidate examples/results/wmh_port_cuda/sub-04_seg.nii.gz \
  --output docs/figures/wmh_synthseg_comparison.png
```

`benchmark_wmh.py` 对三例逐例启动新进程、保存分割图、概率图与软体积 CSV；如只想运行本包，不需要 FreeSurfer，可单独执行第二条推理命令或使用 `fs-torch wmh-synthseg`。重跑前请清空对应 `examples/results/` 目录，脚本不会覆盖已有文件。
