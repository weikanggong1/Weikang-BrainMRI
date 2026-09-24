# 三例公开 T1w 的运行示例

这组三例 T1w 来自 OpenNeuro ds000114。三例都运行 SynthStrip；`sub-02`、`sub-03` 还用 joint SynthMorph 配准到固定影像 `sub-01`。配准直接读取原始 T1w。文件来源和校验值见 [data/SOURCES.json](data/SOURCES.json)。

仓库文件已从公开原图去除面部，保留脑组织及邻近颅骨。这些衍生文件没有人工标注真值，适合检查程序运行和输出格式。下载后先运行 `python examples/check_data.py` 校验文件及 NIfTI 头；处理方法和 SHA-256 也记录在上述清单。

## 准备环境与权重

在仓库根目录安装；已安装本包可跳过：

```bash
git clone https://github.com/weikanggong1/Weikang-BrainMRI.git
cd Weikang-BrainMRI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

下载并校验 SynthStrip 与 joint SynthMorph 所需的三个官方权重。脚本会记录存放目录，供 `fs-torch` 后续调用：

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint
```

权重不随仓库发布。已有官方权重的配置方法见[权重说明](../docs/WEIGHTS.md)中的 `--dest` 和 `--verify-only`。GPU 运行需安装与驱动兼容的 CUDA 版 PyTorch；用 `python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'` 检查。`cuda:0` 按 `CUDA_VISIBLE_DEVICES` 编号。

## CPU 单例：检查每个函数的基本输出

```bash
fs-torch synthstrip -i examples/data/sub-01_T1w.nii.gz \
  -o examples/results/cpu/sub-01_brain.nii.gz \
  -m examples/results/cpu/sub-01_mask.nii.gz --device cpu

fs-torch synthmorph examples/data/sub-02_T1w.nii.gz \
  examples/data/sub-01_T1w.nii.gz --model joint \
  -o examples/results/cpu/sub-02_in_sub-01.nii.gz \
  -t examples/results/cpu/sub-02_to_sub-01.mgz --device cpu
```

第一条对 `sub-01` 做脑提取，写出脑图和二值掩膜。第二条把 moving `sub-02` 配准到 fixed `sub-01`，写出 fixed 网格中的影像和前向变换；`--model joint` 在对称仿射中间空间估计非线性形变。这两条命令均可在没有 GPU 的机器上运行。

## GPU 多被试：Python 表格接口

每个功能用一张 pandas 表指定输入影像和输出前缀。`output` 是含被试 base name、不带扩展名的绝对路径，不是目录或完整输出文件名；`input` 可以是相对或绝对影像路径。下面用两个同设备 Python 进程处理三例脑提取，每个进程加载一次 SynthStrip 权重。

```python
from pathlib import Path
import pandas as pd
from freesurfer_torch import SynthStrip

data = Path("examples/data").resolve()
out = Path("examples/results/python").resolve()
subjects = ("sub-01", "sub-02", "sub-03")

table = pd.DataFrame({
    "input": [str(data / f"{subject}_T1w.nii.gz") for subject in subjects],
    "output": [str(out / subject) for subject in subjects],
})
if __name__ == "__main__":
    saved = SynthStrip(device="cuda:0").predict_batch(table, workers=2)
    print(saved)
```

方法按表的行顺序返回每例的路径字典，生成三例脑图、掩膜和距离场，共 9 个文件。SynthMorph 的多被试用法见[功能说明](../docs/synthmorph/README.md#多被试-python)。`examples/results/` 不纳入 Git。完整命名规则见[批量执行说明](../docs/ARCHITECTURE.md#批量执行)。

只运行三例 SynthStrip 的短脚本见 [run_batch.py](run_batch.py)：从仓库根目录执行 `python examples/run_batch.py`。
