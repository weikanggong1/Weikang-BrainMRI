# 三例公开 T1w 的运行示例

本示例使用 OpenNeuro ds000114 的三例公开 T1w。文件来源和校验值见 [data/SOURCES.json](data/SOURCES.json)。`sub-01` 作为配准固定影像；三个 subject 都执行 SynthStrip 脑提取，`sub-02` 和 `sub-03` 另执行 joint SynthMorph 配准到 `sub-01`。配准读取原始 T1w，与脑提取任务互不依赖，因此五个任务可在同一批中并行调度。

仓库中的三份文件已从公开原图去除面部，仍保留脑组织及邻近颅骨。它们只用于功能演示；不是原始 OpenNeuro 数据，也没有人工标注真值。下载仓库后先运行 `python examples/check_data.py` 校验文件和 NIfTI 头；原始来源、处理方法及 SHA-256 见上述清单。

## 准备环境与权重

从仓库根目录运行以下命令；若已安装本包，可跳过安装步骤：

```bash
git clone https://github.com/weikanggong1/Weikang-BrainMRI.git
cd Weikang-BrainMRI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

用专属脚本下载并校验这两个功能需要的三个官方权重；脚本保存目录，此后 `fs-torch` 会自动找到权重：

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint
```

这三个权重不随仓库或示例影像发布。已有官方权重可按[权重说明](../docs/WEIGHTS.md)使用 `--dest` 和 `--verify-only` 检查并配置。GPU 运行需安装与本机驱动兼容的 CUDA 版 PyTorch，并先检查 `python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'`。以下双 GPU 命令要求至少两张可见 GPU；`cuda:0`、`cuda:1` 按 `CUDA_VISIBLE_DEVICES` 重新编号。

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

第一条读取 `sub-01` 的 T1w，保存脑提取影像及二值脑掩膜。第二条以 `sub-02` 为 moving、`sub-01` 为 fixed，保存重采样到 fixed 网格的影像及前向非线性变换；`--model joint` 在对称仿射中间空间联合估计非线性形变。两个命令均可在没有 GPU 的机器上运行。

## 双 GPU 批量：命令行或 Python

[jobs.json](jobs.json) 是五项任务的清单，每项都指定输入与输出。要在两张 GPU 上各启一个常驻 worker，从仓库根目录运行：

```bash
fs-torch batch examples/jobs.json --devices cuda:0 cuda:1 \
  --workers-per-device 1 --threads-per-worker 4 \
  --report examples/results/cli/batch_report.json
```

`--devices` 指定设备；`--workers-per-device 1` 为每张 GPU 启一个进程；`--threads-per-worker 4` 设置每个进程的 PyTorch CPU 线程；`--report` 保存逐任务结果。worker 按任务类型和模型参数缓存权重，并动态领取任务。报告中的 `index` 对应 `jobs.json` 的顺序，`device` 和 `pid` 显示实际执行进程，`outputs` 列出成功保存的文件，`error` 为 `null` 表示该任务成功。任一任务失败时，命令退出码非零。输出为三个 `_brain.nii.gz`、三个 `_mask.nii.gz`、两个 `_in_sub-01.nii.gz` 和两个 `_to_sub-01.mgz`，均保存在 `examples/results/cli/`。

也可用脚本动态生成同一组病例任务，结果保存到单独的 `examples/results/python/`：

```bash
python examples/run_batch.py
```

脚本使用 `BatchRunner(devices=("cuda:0", "cuda:1"))`，逐项检查 `BatchResult.ok`，并写入 `examples/results/python/batch_report.json`。Python 多进程使用 `spawn`，因此脚本保留 `if __name__ == "__main__":`。`BatchRunner.run()` 返回顺序与提交顺序一致，任务实际完成顺序可以不同。这里的五个任务输入独立；若改为用脑提取影像做 moving，须先运行完全部脑提取任务，再提交配准任务。

`examples/results/` 不纳入 Git。默认拒绝覆盖已有输出，重跑前可删除对应结果目录，或在 CLI 加 `--overwrite`、在 Python 中向 `runner.run` 传 `overwrite=True`。同一批任务即使允许覆盖，也不能共享一个输出路径。
