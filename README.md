# Weikang-BrainMRI

面向脑 MRI 的独立 PyTorch 包，支持 CPU、CUDA 和多 GPU 批量推理。学习模型使用
FreeSurfer 官方权重；单项推理无需安装 FreeSurfer、FSL、TensorFlow、VoxelMorph
或 Neurite。`recon-all` 另需预先准备的 FreeSurfer 8.2 原生运行包和个人 license，
运行时无需安装系统 FreeSurfer。安装包名为 `freesurfer-torch`，Python 导入名为
`freesurfer_torch`，单项功能的统一命令为 `fs-torch`。

## 功能与文档

每个功能的 Python、命令行、批量调用、输入输出、原版指令对应、数值比较和示意图
均放在其专属页面。

| 功能 | 专属说明 | 实现目录 | 验证 |
|---|---|---|---|
| SynthStrip 脑提取、脑掩膜和距离场 | [SynthStrip](docs/synthstrip/README.md) | [synthstrip/](src/freesurfer_torch/synthstrip/) | [对照总表](docs/COMPARISON.md) |
| SynthMorph 刚性、仿射、非线性、联合配准及变换应用 | [SynthMorph](docs/synthmorph/README.md) | [synthmorph/](src/freesurfer_torch/synthmorph/) | [对照总表](docs/COMPARISON.md) |
| WMH-SynthSeg 脑结构及白质高信号分割 | [WMH-SynthSeg](docs/wmh_synthseg/README.md) | [wmh_synthseg/](src/freesurfer_torch/wmh_synthseg/) | [12 例验证](validation/wmh/README.md) |
| SynthSR 单幅 MRI/CT 合成 1 mm T1w | [SynthSR](docs/synthsr/README.md) | [synthsr/](src/freesurfer_torch/synthsr/) | [12 例验证](validation/synthsr/README.md) |
| TorchFAST 三组织 PVE、bias field 和校正图 | [TorchFAST](docs/fast/README.md) | [fast/](src/freesurfer_torch/fast/) | [10 例验证](validation/fast/README.md) |
| GPU FAST VBM：raw T1w 到 warped GM、Jacobian 和 modulated GM | [GPU FAST VBM](docs/fast_vbm/README.md) | [fast_vbm/](src/freesurfer_torch/fast_vbm/) | [包级与科学验证](validation/fast_vbm/README.md) |
| 单 T1 recon-all 与 Python 多 GPU 调度 | [GPU recon-all](docs/recon_all/README.md) | [recon_all/](src/freesurfer_torch/recon_all/) | [FreeSurfer 8.2 数值验收](validation/recon_all/README.md) |
| 多病例与多 GPU 调度 | [批量执行与架构](docs/ARCHITECTURE.md#批量执行) | [batch.py](src/freesurfer_torch/batch.py) | [批量记录](benchmark/real_batch/execution.public.json) |

TorchFAST 覆盖 FSL FAST 的单通道 T1、三组织、无 prior 路径。GPU FAST VBM 将
SynthStrip、TorchFAST 偏置场校正、GM 配准、Jacobian 和 modulation 串成完整流程；
它使用新的 PyTorch 配准方法，输出不能视为 FNIRT 的逐位或算法等价结果。

GPU recon-all 的 SynthStrip、33 类 SynthSeg、SynthMorph 及三项辅助神经分割使用
PyTorch/CUDA；影像转换、强度校正、皮层拓扑、表面生成和统计仍运行于打包的原生
CPU 程序。单被试有 CLI 和 Python 入口，多被试完整流程只提供 Python 入口。

## 安装

要求 Python 3.10 或更高版本。GPU 推理需要与本机驱动匹配的 CUDA 版 PyTorch。

```bash
git clone https://github.com/weikanggong1/Weikang-BrainMRI.git
cd Weikang-BrainMRI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

开发安装和测试：

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest tests
```

## 下载并部署官方权重

Git 仓库和 wheel 不包含模型权重。配置脚本从 FreeSurfer 官方地址下载文件，校验
大小和 SHA-256，并保存权重目录；后续 Python API 和 CLI 会自动查找该目录。

```bash
# 全部官方权重
python tools/setup_weights.py --all

# 按功能下载
python tools/setup_weights.py --model synthstrip
python tools/setup_weights.py --model synthmorph-joint
python tools/setup_weights.py --model wmh-synthseg
python tools/setup_weights.py --model synthsr
python tools/setup_weights.py --model fast-vbm
```

安装后也可运行 `fs-torch-setup-weights`。默认目录为
`~/.cache/freesurfer_torch/`；`--dest /path/to/models` 指定共享目录，
`--verify-only` 只校验已有文件。权重查找顺序为：调用时显式路径、
`FREESURFER_TORCH_WEIGHTS`、配置脚本保存的目录、默认缓存、已有
`FREESURFER_HOME/models/`。

TorchFAST 不使用权重。`fast-vbm` 只下载 GPU FAST VBM 所需的默认 SynthStrip
checkpoint；GM template 是独立输入，不由配置脚本下载。全部文件名、官方 URL、
SHA-256、版本、许可和离线部署方法见[权重文档](docs/WEIGHTS.md)。

`recon-all` 需额外提供与固定 FreeSurfer 8.2 流程匹配的本地原生运行包；上面的
下载命令不提供它。运行包及个人 license 均不随本仓库或 wheel 发布。构建、
单例与批量调用、输出和验收说明见[GPU recon-all 专页](docs/recon_all/README.md)。

## 其他入口

- [公开 T1w 示例](examples/README.md)与[公开 FLAIR 示例](examples/WMH.md)
- [公开 API、目录结构和批量规则](docs/ARCHITECTURE.md)
- [来源、参考版本和哈希](docs/provenance.json)
- [新增功能目录规范](docs/ADDING_FUNCTIONS.md)
- [较早的 UKB/FSL 注册与模板研究](docs/ukb_vbm/README.md)
- [第三方许可、修改说明和论文](THIRD_PARTY_NOTICES.md)

## 许可

各改写部分沿用上游许可。TorchFAST 及随包保存的 FAST4 源码受
[FSL 6.0 非商业许可](licenses/FSL-6.0.txt)约束；WMH-SynthSeg、SynthSR 和部分
FreeSurfer 改写受 [FreeSurfer Software License](licenses/FreeSurfer.txt)约束。
完整归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
