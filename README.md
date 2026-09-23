# Weikang-BrainMRI

独立的 PyTorch 脑 MRI 处理工具。当前提供 **SynthStrip 脑提取**和 **SynthMorph 配准**，支持 CPU、CUDA、模型复用和多 GPU 批量执行。推理无需安装 FreeSurfer、TensorFlow、VoxelMorph 或 Neurite。

仓库名为 `Weikang-BrainMRI`；为保持已有代码兼容，安装包名仍为 `freesurfer-torch`，Python 导入名为 `freesurfer_torch`，CLI 为 `fs-torch`。0.2.0 已将实现按功能分目录，公开 API 不变。

| 功能 | 专属文档 | 实现目录 |
|---|---|---|
| 脑提取、脑掩膜、距离场 | [SynthStrip](docs/synthstrip/README.md) | [synthstrip/](src/freesurfer_torch/synthstrip/) |
| 刚性、仿射、非线性、联合配准及应用变换 | [SynthMorph](docs/synthmorph/README.md) | [synthmorph/](src/freesurfer_torch/synthmorph/) |
| 多 GPU / 同 GPU 多进程批量调度 | [批量使用与架构](docs/ARCHITECTURE.md#批量执行) | [batch.py](src/freesurfer_torch/batch.py) |

仓库附有 [3 例可直接运行的公开 T1w](examples/README.md)；数据来自 [OpenNeuro ds000114](https://openneuro.org/datasets/ds000114) 的 CC0 影像，经去面容处理。每例的来源、处理方法和校验值见 [SOURCES.json](examples/data/SOURCES.json)。

## 安装与权重

要求 Python ≥ 3.10。GPU 推理需要与驱动兼容的 CUDA 版 PyTorch。

```bash
git clone https://github.com/weikanggong1/Weikang-BrainMRI.git
cd Weikang-BrainMRI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Git 仓库和 wheel 均不包含模型权重。** 运行专属脚本，从 FreeSurfer 官方下载并校验所需文件；脚本保存权重目录，后续 Python API 和 CLI 自动找到它。默认脑提取与 joint 配准需三个文件：

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint
```

要配置所有五个权重，运行 `python tools/setup_weights.py --all`；安装后的独立命令是 `fs-torch-setup-weights`。下载目标默认是用户缓存目录，也可用 `--dest /path/to/weights` 指定。已有权重可用 `--verify-only` 校验；显式 `weights=` / `--weights` 或环境变量 `FREESURFER_TORCH_WEIGHTS` 可覆盖保存的位置。推理时不会自动联网下载。官方下载地址、版本、SHA-256、许可和完整用法见[权重说明](docs/WEIGHTS.md)。

## 单例 Python 调用：输入、输出和每步作用

以下路径是示例；先按上节运行权重配置脚本。Python API 接受路径或 `surfa.Volume`，返回带原始影像几何的对象，不会自动保存文件。保存到嵌套目录前请先创建目录。

```python
from pathlib import Path                         # 用 Path 管理输出目录和文件名。
from freesurfer_torch import SynthStrip, SynthMorph, apply_transform

out = Path("results")                           # 所有输出放在同一目录。
out.mkdir(parents=True, exist_ok=True)          # Python API 保存前由调用者建目录。

strip = SynthStrip(device="cuda:0")              # 加载官方 SynthStrip 权重至 GPU 0；只加载一次。
brain = strip("subject_T1w.nii.gz")              # 输入一幅 T1w；预测距离场并生成脑掩膜和去颅骨图像。
brain.image.save(out / "subject_brain.nii.gz")  # 保存原网格上的脑图，对应原版 -o。
brain.mask.save(out / "subject_mask.nii.gz")    # 保存原网格上的二值掩膜，对应原版 -m。
brain.distance.save(out / "subject_sdt.nii.gz") # 保存毫米单位的有符号距离场，对应原版 -d。

morph = SynthMorph(device="cuda:0", model="joint")  # 加载默认联合仿射与非线性配准权重。
reg = morph("subject_T1w.nii.gz", "template_T1w.nii.gz")
# 上行第一个输入是 moving（待配准图像），第二个是 fixed（目标模板）；两者都须为单帧 3D。
reg.moved.save(out / "subject_in_template.nii.gz")   # moving 重采样到 fixed 网格，对应原版 -o。
reg.fixed_moved.save(out / "template_in_subject.nii.gz") # fixed 重采样到 moving 网格，对应 -O。
reg.transform.save(out / "subject_to_template.mgz")  # moving→fixed 变换，对应原版 -t。
reg.inverse.save(out / "template_to_subject.mgz")    # fixed→moving 变换，对应原版 -T。

labels = apply_transform("subject_labels.nii.gz", reg.transform, method="nearest", dtype="int16")
# 将 moving 空间的离散标签随正向变换送至 fixed 空间；最近邻插值避免产生新标签值。
labels.save(out / "labels_in_template.nii.gz")      # 对应原版 mri_synthmorph apply。
```

`brain.image`、`brain.mask`、`brain.distance` 都是 `surfa.Volume`：脑图保留输入网格及掩膜内强度；掩膜为二值；距离场以 mm 计，默认用 `distance < 1` 决定脑边界。脑图掩膜外默认填 `min(输入影像最小值, 0)`。SynthStrip 接受单帧 3D，也可逐帧处理 4D；可用 `strip(image, border=1, fill=0)` 控制边界和背景。

`reg.moved` 与 `reg.fixed_moved` 也是 `surfa.Volume`，分别在 fixed 与 moving 网格。`reg.transform` 和 `reg.inverse` 含源/目标几何：`joint`/`deform` 为 RAS 位移场，建议保存为 `.mgz`；`affine`/`rigid` 为仿射变换，建议保存为 `.lta`。两个方向在 Python 调用中都会计算，即使只保存一个文件。`apply_transform` 对已得变换做 Surfa CPU 重采样；标签使用 `method="nearest"`。相同 `strip`/`morph` 实例可处理后续图像，省去重复加载权重。将 `device` 改为 `"cpu"` 可在 CPU 上运行；网络和部分空间变换用 GPU，影像读写及最终 Surfa 重采样仍用 CPU。

| Python 输入与类型 | 返回字段与类型 | 原版文件指令 |
|---|---|---|
| `strip(image, border=1, fill=None)`：3D/4D NIfTI 等文件路径或 `surfa.Volume` | `image`、`mask`、`distance`：原输入网格的 `surfa.Volume`，距离单位 mm | `mri_synthstrip -i ... -o ... -m ... -d ...` |
| `morph(moving, fixed, init=None, mid_space=False, header_only=False)`：两幅单帧 3D 路径或 `surfa.Volume` | `moved`、`fixed_moved`：各自目标网格的 `surfa.Volume`；`transform`、`inverse`：带几何的仿射或 RAS 位移场 | `mri_synthmorph register moving fixed -o ... -O ... -t ... -T ...` |
| `apply_transform(image, transformation, method="linear", fill=0, dtype="float32")`：3D/4D 影像及已有变换 | 重采样后的 `surfa.Volume` | `mri_synthmorph apply transform image output` |

Python 直接调用返回对象；`save(...)` 才写文件。`morph` 的正向变换用于把 moving 的影像或标签送到 fixed 空间，反向变换用于相反方向。具体参数、限制和几何约定分别见 [SynthStrip](docs/synthstrip/README.md) 与 [SynthMorph](docs/synthmorph/README.md)。

## 单例命令行：与 FreeSurfer 原指令逐项对应

下列新旧指令均使用相同的输入影像和官方权重。`fs-torch` 使用本包独立的命令入口；原版指令只在安装并加载 FreeSurfer 的环境中可用。示例中 `subject_T1w.nii.gz` 是 moving，`template_T1w.nii.gz` 是 fixed。

```bash
fs-torch synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz --device cuda:0

fs-torch synthmorph subject_T1w.nii.gz template_T1w.nii.gz \
  -m joint -o results/subject_in_template.nii.gz \
  -O results/template_in_subject.nii.gz \
  -t results/subject_to_template.mgz -T results/template_to_subject.mgz \
  --device cuda:0

fs-torch apply results/subject_to_template.mgz subject_labels.nii.gz \
  results/labels_in_template.nii.gz --method nearest --dtype int16
```

第一条调用脑提取：`-i` 读取原始 T1w；`-o` 写脑图；`-m` 写脑掩膜；`-d` 写距离场；`--device cuda:0` 选 GPU 0。至少选一个输出。第二条将 moving 配准到 fixed：`-m joint` 选择默认的仿射加非线性模型；`-o/-O` 写两个方向的重采样脑图；`-t/-T` 写正反变换；`--device` 选执行网络的设备。第三条把 moving 空间的标签用正向变换映射到 fixed 空间，`--method nearest` 保留离散标签，`--dtype int16` 指定输出数据类型；应用已有变换在 CPU 执行。三条命令的输出父目录由 CLI 创建；`fs-torch --help` 可查看全部参数。

原版 FreeSurfer 的对应指令为：

```bash
mri_synthstrip -i subject_T1w.nii.gz \
  -o results/subject_brain.nii.gz -m results/subject_mask.nii.gz \
  -d results/subject_sdt.nii.gz

mri_synthmorph register -m joint \
  -o results/subject_in_template.nii.gz -O results/template_in_subject.nii.gz \
  -t results/subject_to_template.mgz -T results/template_to_subject.mgz \
  subject_T1w.nii.gz template_T1w.nii.gz

mri_synthmorph apply -m nearest -t int16 \
  results/subject_to_template.mgz subject_labels.nii.gz \
  results/labels_in_template.nii.gz
```

原版第一条的 `-i/-o/-m/-d` 与新 CLI 和 `StripResult` 的输入输出逐项相同。第二条的 `register` 是原版配准子命令（原版也允许省略该词）；其 `-o/-O/-t/-T` 分别对应 `RegistrationResult.moved/fixed_moved/transform/inverse`。第三条的 `apply` 对应本包 `fs-torch apply` 或 Python 的 `apply_transform`，其中原版 `-m nearest`、`-t int16` 对应新 CLI 的 `--method nearest`、`--dtype int16`。原版 `apply` 可在一条指令中传多组影像/输出，新 CLI 每次一组，Python 可循环。

| 其余原版选项 | 本包对应 | 说明 |
|---|---|---|
| SynthStrip `-g`、`-t`、`--model` | `--device cuda:0`、`-j`、`--weights`；Python `device`、`threads`、`weights` | 原版 `-g` 只表示使用可见 GPU；本包显式选择设备。原版 `--model` 是单个 PT 文件。|
| SynthStrip `--no-csf`、`-b`、`-f` | 同名 CLI 选项；Python `no_csf`、`border`、`fill` | 默认边界 1 mm，默认背景填充值相同。|
| SynthMorph `-m`、`-r`、`-n`、`-e` | 同名短选项；Python `model`、`hyper`、`steps`、`extent` | 默认分别为 joint、0.5、7、256；`deform` 要求事先对齐或使用初始仿射。|
| SynthMorph `-i`、`-M`、`-H` | 同名短选项；Python `init`、`mid_space`、`header_only` | `-M` 需搭配 `-i`；`-H` 只适用于 affine/rigid。|
| SynthMorph `-g`、`-j`、可重复的 `-w` | `--device`、`-j`、`--weights`；Python `device`、`weights` | 新 CLI 从权重目录取所需 H5；原版 `-w` 可多次指定文件。原版 `-j` 管 TensorFlow 线程，新 CLI 管 Torch 线程。|

**一致性的范围。** 默认脑提取的输入、三类输出及原网格语义与原版一致；12 例真实 T1w 的同设备脑图、掩膜、距离场均逐元素相同。配准的 moving/fixed 顺序、正反输出、变换方向和目标几何也对应原版；但 TensorFlow→PyTorch 移植有浮点差异，12 例默认 joint 配准的变换最大差为 0.000790 mm，不能宣称文件逐字节相同。原版普通配准只重采样请求保存的方向，本包 Python 调用会计算双向结果；原版 `-d` 调试目录生成 6 个文件，本包仅生成两幅网络输入和 `network_transforms.npz`。原版配准允许不指定保存输出，新 CLI 至少要求一个输出或调试目录。参考构建的原生 `-i` 初始化因 dtype 错误失败，该分支仅与两行修复后的参考源码比较。详细证据与运行时间见[对照报告](docs/COMPARISON.md)。

## 公开样例与原版对照图

从仓库根目录运行 `python examples/check_data.py` 可核对三个 T1w 文件。按[权重文档](docs/WEIGHTS.md)准备官方权重后，[示例说明](examples/README.md)给出 CPU 单例、双 GPU 批量命令及各输出文件的位置。下面的图使用同一份去面容的 `sub-02` 输入和相同官方权重，分别运行 FreeSurfer 8.2.0 原版与本包 0.2.0；`sub-01` 是配准的 fixed 图像。图片是实际程序输出，制作步骤与逐文件比较见 [图示记录](docs/figures/README.md)。

![同一 T1w 的原图、FreeSurfer 脑提取结果和 PyTorch 脑提取结果](docs/figures/synthstrip_comparison.png)

上图从左到右为去面容输入、FreeSurfer 提取的 brain、本包提取的 brain；两行展示轴位与冠状位。各列使用相同体素切面和灰度范围，定量掩膜比较见[图示记录](docs/figures/README.md)。

![FreeSurfer 与 PyTorch 非线性配准后的脑图并排比较](docs/figures/synthmorph_comparison.png)

上图展示 moving、fixed，以及两种实现的 joint 配准结果。两列配准结果都位于 fixed 网格；图中仅为展示使用同一 fixed 脑掩膜，数值误差在完整保存的影像和形变场上计算。两次推理使用相同输入与权重，原版在 CPU、本包在 GPU；该图不用于比较运行速度。

## 多病例批量并行

批量任务以**一例影像、一次功能调用**为单位；不会把不同病例拼成一个网络 tensor。每个 worker 是独立进程，绑定一张 GPU，并在进程内缓存已加载的模型。下例对仓库中的三例公开 T1w 运行三次脑提取，并把 `sub-02/03` 配准到 `sub-01`，共五个互不依赖的任务。先运行上文的权重配置脚本，再把以下代码保存为仓库根目录的 `run_many.py`，执行 `python run_many.py`；仓库中也提供等价的 [examples/run_batch.py](examples/run_batch.py)：

```python
from freesurfer_torch import BatchRunner


def main():
    subjects = ("sub-01", "sub-02", "sub-03")
    template = "examples/data/sub-01_T1w.nii.gz"
    jobs = []
    for subject in subjects:
        moving = f"examples/data/{subject}_T1w.nii.gz"
        jobs.append({
            "task": "synthstrip",
            "kwargs": {"image": moving},
            "outputs": {
                "image": f"examples/results/readme_python/{subject}_brain.nii.gz",
                "mask": f"examples/results/readme_python/{subject}_mask.nii.gz",
            },
        })
        if subject != "sub-01":
            jobs.append({
                "task": "synthmorph",
                "model": {"model": "joint"},
                "kwargs": {"moving": moving, "fixed": template},
                "outputs": {
                    "moved": f"examples/results/readme_python/{subject}_in_sub-01.nii.gz",
                    "transform": f"examples/results/readme_python/{subject}_to_sub-01.mgz",
                },
            })

    with BatchRunner(devices=("cuda:0", "cuda:1"),
                     workers_per_device=1, threads_per_worker=4) as runner:
        results = runner.run(jobs)
    for result in results:
        print(result.index, result.task, result.device, result.outputs, result.error)
    if any(not result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

`task` 选择功能；`model` 是模型构造参数，其中 SynthMorph 的 `"model": "joint"` 在估计的仿射中间空间进行非线性配准，输出组合后的变换；`kwargs` 传给该功能的单例调用；`outputs` 指定要保存的结果属性和路径。`sub-01` 只得到脑提取影像和掩膜；`sub-02/03` 各得到这两项以及配准后影像、前向变换，共十个输出。`BatchRunner` 启动两个 worker，各持有自己的模型缓存；`runner.run(jobs)` 分发任务并按输入顺序返回 `BatchResult`。每项结果包含设备、进程号、起止时间、已保存文件和错误；`result.ok` 表示该任务没有报错。`with` 退出时关闭 worker。由于使用 `spawn`，Python 脚本必须保留 `if __name__ == "__main__":`。

命令行直接读取仓库的 [examples/jobs.json](examples/jobs.json)：清单含三个 SynthStrip 任务和两个 joint SynthMorph 任务；后两项分别把 `sub-02/03` 的原始 T1w 配准到 fixed `sub-01`。它与上面的 Python 示例处理相同病例，结果保存到独立的 `examples/results/cli/`。从仓库根目录运行：

```bash
fs-torch batch examples/jobs.json --devices cuda:0 cuda:1 \
  --workers-per-device 1 --threads-per-worker 4 \
  --report examples/results/cli/batch_report.json
```

`examples/jobs.json` 提供全部任务和各自输出；`--devices` 指定可用 GPU，编号遵循 `CUDA_VISIBLE_DEVICES`；`--workers-per-device 1` 表示每张 GPU 一个进程；`--threads-per-worker 4` 限制每个进程的 PyTorch CPU 线程；`--report` 写逐任务 JSON 结果。任一任务失败时命令退出码非零。完成后可检查五项任务及十个输出：

```bash
python - <<'PY'
import json
from pathlib import Path
jobs = json.loads(Path("examples/jobs.json").read_text())
results = json.loads(Path("examples/results/cli/batch_report.json").read_text())
assert len(jobs) == len(results) == 5
for index, (job, result) in enumerate(zip(jobs, results)):
    assert result["index"] == index and result["task"] == job["task"]
    assert result["error"] is None, result["error"]
    assert set(result["outputs"]) == set(job["outputs"])
    assert all(Path(path).is_file() for path in result["outputs"].values())
print("5 jobs completed; 10 output files present")
PY
```

用 CPU 并行时将设备选项改为 `--devices cpu --workers-per-device 2`；Python 中对应 `devices=("cpu",), workers_per_device=2`。若先运行了 GPU 示例，应更换输出目录或显式允许覆盖。增加同 GPU worker 会各自加载模型并占用更多显存，不保证更快。

任务在同一批中没有先后依赖。如果要把**脑提取后的影像**用于配准，先完成所有 SynthStrip 任务，再以保存的 `_brain.nii.gz` 作为 `moving` 提交第二批 SynthMorph 任务；同一个 `BatchRunner` 可连续调用 `run`，复用 worker。默认拒绝覆盖已有文件，即使指定 `--overwrite` / `overwrite=True`，同一批也不能让两个任务写入同一路径。更多输出字段和错误处理见[批量使用与架构](docs/ARCHITECTURE.md#批量执行)。

## 验证与维护

- [详细功能和数值对照](docs/COMPARISON.md)：0.1.0 参考实验包含 12 例真实 T1w、96 次单例运行及 24 个批量任务。该临床数据只发布匿名统计，不包含原始影像；仓库另附三例公开 OpenNeuro 衍生样例。
- [0.2.0 结构重整回归](validation/refactor/report.public.json)：新布局与 0.1.0 的对照记录；历史计时不能当作 0.2.0 的重新计时。
- [架构、公共 API 与批量任务格式](docs/ARCHITECTURE.md)。
- [新增功能指南](docs/ADDING_FUNCTIONS.md)：每个功能的实现、文档和测试均有独立目录。
- [来源与模型哈希](docs/provenance.json)、[第三方许可与引用](THIRD_PARTY_NOTICES.md)。

```bash
python -m pip install pytest
python -m pytest tests
```

SynthStrip 官方模型本身即为 PyTorch；本项目将其整理为独立可复用接口。SynthMorph 将指定 FreeSurfer 8.2.0 构建中的 TensorFlow 网络与空间运算移植到 PyTorch，直接读取相应官方权重。各功能文档说明支持范围、差异和复现入口。
