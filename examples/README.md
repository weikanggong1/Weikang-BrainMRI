# 三例公开 T1w 的运行示例

这组三例 T1w 来自 OpenNeuro ds000114。三例都运行 SynthStrip；`sub-02`、`sub-03` 还用 joint SynthMorph 配准到固定影像 `sub-01`。配准直接读取原始 T1w，和脑提取没有先后依赖，因此五个任务可在同一批中运行。文件来源和校验值见 [data/SOURCES.json](data/SOURCES.json)。

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

权重不随仓库发布。已有官方权重的配置方法见[权重说明](../docs/WEIGHTS.md)中的 `--dest` 和 `--verify-only`。GPU 运行需安装与驱动兼容的 CUDA 版 PyTorch；用 `python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'` 检查。下文的双 GPU 命令需要两张可见 GPU，`cuda:0`、`cuda:1` 按 `CUDA_VISIBLE_DEVICES` 编号。

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

## 双 GPU 批量：命令行或 Python

[jobs.json](jobs.json) 列出五项任务及各自的输入输出。在两张 GPU 上各启动一个常驻 worker：

```bash
fs-torch batch examples/jobs.json --devices cuda:0 cuda:1 \
  --workers-per-device 1 --threads-per-worker 4 \
  --report examples/results/cli/batch_report.json
```

`--devices` 指定 GPU；`--workers-per-device 1` 为每张卡启动一个进程；`--threads-per-worker 4` 限制每个进程的 PyTorch CPU 线程。worker 缓存用过的模型，空闲时领取下一项任务。`--report` 写出按清单顺序排列的逐任务结果：`device` 和 `pid` 标明执行进程，`outputs` 列出文件，`error` 为 `null` 表示成功；有任务失败时，命令退出码非零。成功后，`examples/results/cli/` 中应有三个脑图、三个掩膜、两个配准影像和两个前向变换，共十个文件。

也可用 Python 脚本生成相同任务，结果写入 `examples/results/python/`：

```bash
python examples/run_batch.py
```

脚本使用 `BatchRunner(devices=("cuda:0", "cuda:1"))`，检查每项 `BatchResult.ok`，并保存 `examples/results/python/batch_report.json`。多进程启动方式为 `spawn`，因此需要 `if __name__ == "__main__":`。`BatchRunner.run()` 按提交顺序返回结果，任务完成顺序可以不同。若改用脑提取后的影像做 moving，须等第一批脑提取全部完成，再提交配准任务。

`examples/results/` 不纳入 Git。默认拒绝覆盖已有输出，重跑前可删除对应结果目录，或在 CLI 加 `--overwrite`、在 Python 中向 `runner.run` 传 `overwrite=True`。同一批任务即使允许覆盖，也不能共享一个输出路径。
