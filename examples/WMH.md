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

三例 FLAIR 的多被试处理使用 Python 表格接口。`output` 是不带扩展名、含被试 base name 的绝对路径前缀；`workers=2` 在同一设备启动两个 Python 进程，各加载一次模型并逐例保存输出。

```python
from pathlib import Path
import pandas as pd
from freesurfer_torch import WMHSynthSeg

data = Path("examples/wmh_data").resolve()
out = Path("examples/results/wmh_batch").resolve()
subjects = ("sub-02", "sub-03", "sub-04")
table = pd.DataFrame({
    "input": [str(data / f"{subject}_FLAIR.nii.gz") for subject in subjects],
    "output": [str(out / subject) for subject in subjects],
})
if __name__ == "__main__":
    model = WMHSynthSeg(device="cuda:0", threads=1)
    saved = model.predict_batch(table, crop=True, workers=2)
    assert len(saved) == 3 and all(path.is_file() for files in saved for path in files.values())
    print(saved)
```

方法为每例保存分割图、病灶概率图和软体积 CSV，并按表行顺序返回含三项路径的字典。两幅影像输出位于处理后的 RAS/1 mm 网格。接口字段和路径规则见[批量执行说明](../docs/ARCHITECTURE.md#批量执行)。

两版程序使用同一份发布输入的比较结果及并排图制作步骤见 [WMH 图示](../docs/figures/README.md) 和 [12 例对照](../validation/wmh/README.md)。这些公开图像没有人工 WMH 真值；这里的检查用于比较接口与输出，不能评价临床检测精度。
