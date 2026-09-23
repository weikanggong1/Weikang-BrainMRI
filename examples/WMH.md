# 三例公开 FLAIR：运行 WMH-SynthSeg

[返回主页](../README.md) · [功能说明](../docs/wmh_synthseg/README.md) · [数据来源与 SHA-256](wmh_data/SOURCES.json)

`wmh_data/` 中的 `sub-02`、`sub-03`、`sub-04` 源于 [OpenNeuro ds003592](https://openneuro.org/datasets/ds003592) 的 CC0 FLAIR。发布文件保留输入网格，清除了原版 SynthStrip 脑掩膜外扩 6 mm 以外的强度以及 NIfTI 头中的标识文字。仓库不包含原始影像或人工病灶标注。制作脚本、原图地址及发布文件哈希见 [`tools/prepare_wmh_examples.py`](../tools/prepare_wmh_examples.py) 和 [`SOURCES.json`](wmh_data/SOURCES.json)。

从仓库根目录执行：

```bash
python -m pip install .
python examples/check_wmh_data.py
python tools/setup_weights.py --model wmh-synthseg
```

依次安装本包、校验三份 FLAIR 的大小与 SHA-256 及 NIfTI 头，再从 FreeSurfer 官方下载并校验约 791 MB 的 WMH checkpoint。脚本会记录权重目录，推理时无需安装 FreeSurfer。GPU 推理需要与驱动兼容的 CUDA 版 PyTorch；用 `python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'` 检查。

先运行一例。下面的参数对应原版 `mri_WMHsynthseg --i ... --o ... --crop --save_lesion_probabilities --csv_vols ...`：

```bash
fs-torch wmh-synthseg \
  --i examples/wmh_data/sub-04_FLAIR.nii.gz \
  --o examples/results/wmh_single/sub-04_seg.nii.gz \
  --device cuda:0 --threads 4 --crop \
  --save_lesion_probabilities \
  --csv_vols examples/results/wmh_single/sub-04_volumes.csv
```

`--i` 读取 3D FLAIR；`--o` 写 33 类整数值标签图，文件存储类型与原版一样为 float32。`--crop` 先定位脑再限制推理区域；`--save_lesion_probabilities` 另写 `sub-04_seg.lesion_probs.nii.gz`；`--csv_vols` 写各类后验概率的软体积，单位 mm³。`--device` 与 `--threads` 设置设备和 PyTorch CPU 线程。输出位于处理后的 RAS/1 mm 网格，可能不同于输入网格。没有 GPU 时改用 `--device cpu`。

Python 调用返回影像对象和体积字典，不自动写文件：

```python
from pathlib import Path
from freesurfer_torch import WMHSynthSeg

Path("examples/results/wmh_single").mkdir(parents=True, exist_ok=True)
model = WMHSynthSeg(device="cuda:0", threads=4)
result = model("examples/wmh_data/sub-04_FLAIR.nii.gz",
               crop=True, save_lesion_probabilities=True)
result.segmentation.save("examples/results/wmh_single/sub-04_python_seg.nii.gz")
result.lesion_probability.save("examples/results/wmh_single/sub-04_python_lesion_probs.nii.gz")
print(result.volumes_mm3[77])  # WMH 软体积；不是硬标签 77 的体素数。
```

CLI 会自动创建输出父目录。同一模型实例可继续处理 `sub-02/03`，无需再次加载权重。

[批量清单](wmh_jobs.json) 可将三例分配到两张 GPU：

```bash
fs-torch batch examples/wmh_jobs.json --devices cuda:0 cuda:1 \
  --workers-per-device 1 --threads-per-worker 4 \
  --report examples/results/wmh_batch/report.json
```

清单为每例指定 `task="wmh_synthseg"`、FLAIR 路径、`crop=True` 和分割图、概率图的输出路径。`--devices` 指定两张 GPU，`--workers-per-device 1` 让每张卡启动一个进程并加载一份模型。两个进程可同时处理病例；第三例交给先空闲的进程。`--threads-per-worker 4` 限制每个进程的 PyTorch CPU 线程。结果 JSON 按清单顺序排列，`device`、`pid`、`error`、`outputs` 分别记录执行 GPU、进程、错误及保存的文件。已有旧输出时，改用新目录或加 `--overwrite`。

运行后可检查每例两幅图、标签 77 和概率范围：

```bash
python - <<'PY'
import json
from pathlib import Path
import nibabel as nib
import numpy as np

report = json.loads(Path("examples/results/wmh_batch/report.json").read_text())
assert len(report) == 3 and {row["device"] for row in report} == {"cuda:0", "cuda:1"}
for row in report:
    assert row["error"] is None and len(row["outputs"]) == 2, row
    seg = nib.load(row["outputs"]["segmentation"])
    prob = nib.load(row["outputs"]["lesion_probability"])
    assert seg.shape == prob.shape
    labels = np.asarray(seg.dataobj)
    values = np.asarray(prob.dataobj)
    assert np.any(labels == 77) and np.all((0 <= values) & (values <= 1))
print("3 subjects, 2 GPUs, 6 valid images")
PY
```

两版程序使用同一份发布输入的比较结果及并排图制作步骤见 [WMH 图示](../docs/figures/README.md) 和 [12 例对照](../validation/wmh/README.md)。这些公开图像没有人工 WMH 真值；这里的检查用于比较接口与输出，不能评价临床检测精度。
