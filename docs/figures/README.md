# 官方实现与 PyTorch 实现的影像对照

两张图使用仓库中同一份 [公开去面容样例](../../examples/README.md)，展示实际保存的输出。原始数据为 [OpenNeuro ds000114](https://openneuro.org/datasets/ds000114) 的 CC0 T1w；仓库内三个文件是经脑掩膜外扩 12 mm 后去除其余体素的衍生示例，来源和 SHA-256 见 [SOURCES.json](../../examples/data/SOURCES.json)。它们没有人工标注，图中的一致性衡量的是两种**实现之间**的差异，不是脑提取或配准的临床准确性。

![原始 T1w 与两种脑提取结果](synthstrip_comparison.png)

SynthStrip 图使用 `sub-02`：从左到右是去面容输入、FreeSurfer 8.2.0 CPU 输出、本包 PyTorch GPU 输出。上、下分别是相同原网格体素的轴位和冠状位切面；三列用同一灰度范围。两种输出的掩膜 Dice 为 **1.000000**，不一致体素 **0**，脑图全体素最大绝对差 **0**，输出几何相同。这里的脑图强度已经由两种程序各自的掩膜处理。

![移动与固定影像及两种 joint 配准结果](synthmorph_comparison.png)

SynthMorph 图以 `sub-02` 为 moving、`sub-01` 为 fixed，使用默认 `joint` 模型。从左到右为 moving 脑图、fixed 脑图、FreeSurfer CPU 配准结果、本包 PyTorch GPU 配准结果；上、下为轴位与冠状位。配准后两列都在 **fixed 的原网格**；为便于看脑组织，图中仅对显示的配准切片使用同一个 fixed 脑掩膜。数值比较针对**完整保存影像和位移场**，未使用显示掩膜：配准影像归一化 RMSE 为 **1.7880 × 10⁻⁵**，RAS 位移向量最大误差 **0.000622 mm**，输出几何相同。强度范围对 moving 与两列配准影像一致；fixed 图像单独按其正强度分布设定亮度，因此 fixed 一列不能用于视觉判断强度误差。

完整机器可读指标见 [metrics.json](metrics.json)。原版与 PyTorch 的 CPU/GPU 成对耗时、12 例真实 T1w 的数值对照见 [详细对照报告](../COMPARISON.md)；此处原版单进程与本包双 GPU 批量运行条件不同，不用它们计算加速比。

## 重现这些图

从仓库根目录开始，先安装本包和绘图依赖，再校验三份随仓库发布的影像。以下 Python 脚本均假设当前目录是仓库根目录。

```bash
python -m pip install . nibabel matplotlib
python examples/check_data.py
python tools/setup_weights.py --model synthstrip --model synthmorph-joint
python examples/run_batch.py
```

`run_batch.py` 在 `cuda:0`、`cuda:1` 上各启动一个常驻 worker，运行三项 SynthStrip 和两项 joint SynthMorph 任务；十个输出及逐任务 JSON 报告保存在 `examples/results/python/`。本次执行五项均成功，两个 GPU 都承担了任务。若机器只有 CPU，可按 [batch 文档](../../examples/README.md#双-gpu-批量命令行或-python)把设备改为 `cpu`，并相应修改脚本或调用 `fs-torch batch`；CPU 与 GPU 的浮点结果可能不同。

原版对照需要 FreeSurfer 8.2.0 和其 TensorFlow 依赖。以下脚本显式指定官方模型文件路径，禁用 CUDA，使原版在 CPU 执行；它对 `sub-01/02` 做脑提取，再将 `sub-02` 配准到 `sub-01`，结果写入 `examples/results/reference/`：

```bash
module load freesurfer
export FREESURFER_TORCH_WEIGHTS="$HOME/.cache/freesurfer_torch"
python examples/run_reference.py
python examples/render_comparison.py
```

如果权重脚本用了 `--dest`，将上述环境变量改为该目录。`module load freesurfer` 是 gpucw1 上的环境配置示例，其他机器按自己的 FreeSurfer 安装方式设置 `FREESURFER_HOME`。`run_reference.py` 使用原版 `mri_synthstrip -i/-o/-m` 和 `mri_synthmorph register -m joint -o/-t`；`render_comparison.py` 核对影像几何并计算指标，生成本目录的两个 PNG 和 `metrics.json`。它要求原版和本包的结果均已存在，且覆盖本目录图示。

这些图在 gpucw1 上使用 FreeSurfer 8.2.0、官方固定版本权重和本包 0.2.0 生成。原版 CPU 脑提取两例分别耗时 50.28、52.42 秒，joint 配准耗时 212.65 秒；本包示例采用两张 GPU 并行且复用模型，单任务报告中的时间包含 worker 执行、读写和可能的同时运行争用。严格的时间比较使用 [详细对照报告](../COMPARISON.md) 中同任务的成对测量。
