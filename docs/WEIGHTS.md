# 预训练权重：下载、校验与公开发布

Git 仓库和 wheel 均不包含权重。SynthStrip、SynthMorph、WMH-SynthSeg 和 SynthSR
使用 FreeSurfer 官方发布的模型文件；配置脚本下载文件、核对大小与 SHA-256，并
保存权重目录。此后 Python API 和 `fs-torch` 命令会自动查找它，下载过程无需安装
FreeSurfer。0.5.0 新增的 TorchFAST 是数值算法，不使用模型权重。0.6.0 的 FastVBM
仅在原始 T1 脑提取阶段复用默认 SynthStrip checkpoint。

## 一次配置，后续自动使用

在仓库根目录运行。默认下载下表九个权重到 `~/.cache/freesurfer_torch/`；SynthMorph 最大的文件约 3.51 GB，可通过 HTTP Range 续传。脚本先写 `.part`，完整校验后才更名为正式权重文件。

```bash
python tools/setup_weights.py --all
```

也可只下载所需模型。`joint` 需要 affine 和 deform 两份权重；下例再加默认 SynthStrip 权重，共三个文件：

```bash
python tools/setup_weights.py --model synthstrip --model synthmorph-joint
```

只运行 WMH-SynthSeg 时下载其单个 checkpoint：

```bash
python tools/setup_weights.py --model wmh-synthseg
```

SynthSR 默认、低场和 v1 是三份不同权重。只需通用 v2 时下载一份；需要全部变体时把三个模型名同时传给脚本：

```bash
python tools/setup_weights.py --model synthsr
python tools/setup_weights.py --model synthsr --model synthsr-lowfield --model synthsr-v1
```

有独立模型目录时，用 `--dest` 指定一次即可。脚本成功后把绝对路径保存在用户缓存目录的 `weights.json`，之后 API 和 CLI 可以省略 `weights=` / `--weights`：

```bash
python tools/setup_weights.py --all --dest /path/to/models
python tools/setup_weights.py --all --verify-only
```

`--verify-only` 只检查当前权重目录，不下载或修改配置。已从联网机器复制了权重时，运行 `python tools/setup_weights.py --all --dest /path/to/copied/models`：现有文件校验成功后直接保存目录，无需重新下载。安装 wheel 后也可使用相同选项的 `fs-torch-setup-weights` 命令。

可选模型名：`synthstrip`、`synthstrip-nocsf`、`synthmorph-rigid`、`synthmorph-affine`、`synthmorph-deform`、`synthmorph-joint`、`wmh-synthseg`、`synthsr`、`synthsr-lowfield`、`synthsr-v1` 和 `fast-vbm`。`fast-vbm` 是 SynthStrip 默认 checkpoint 的依赖别名，不增加第十个权重。`--model` 可重复；不写 `--model` 时等同 `--all`。显式 API/CLI 权重路径优先，其次是 `FREESURFER_TORCH_WEIGHTS` 环境变量，再次是脚本保存的目录，然后是默认缓存和现有 FreeSurfer 模型目录。`XDG_CACHE_HOME` 可改变缓存根目录。模型推理不会联网，只有运行配置脚本才会下载。

以下链接、HTTP 状态和文件大小于 **2026-09-23** 核验；九个端点均返回 HTTP 200。SynthStrip/SynthMorph 的 SHA-256 来自本包已完成数值验证的权重，并与 FreeSurfer 官方仓库的 git-annex 指针一致；WMH-SynthSeg 和 SynthSR v1 的 SHA-256 来自官方文件的完整下载校验。SynthSR v2 两份文件的大小和 SHA-256 与 FreeSurfer git-annex 对象名一致；配置脚本下载后还会逐字节校验。此处的版本号固定，不会自动跟随上游替换为新模型。

## 官方文件

| 功能 | 文件与官方下载链接 | 字节数 | 使用场景 |
|---|---|---:|---|
| SynthStrip | [synthstrip.1.pt](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt) | 30,851,709 | 默认脑提取 |
| SynthStrip | [synthstrip.nocsf.1.pt](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.nocsf.1.pt) | 30,851,709 | `no_csf=True` / `--no-csf` |
| SynthMorph | [synthmorph.affine.2.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.affine.2.h5) | 51,455,312 | affine；joint 的仿射阶段 |
| SynthMorph | [synthmorph.deform.3.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.deform.3.h5) | 3,508,630,424 | deform；joint 的非线性阶段 |
| SynthMorph | [synthmorph.rigid.1.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.rigid.1.h5) | 51,656,152 | rigid |
| WMH-SynthSeg | [WMH-SynthSeg_v10_231110.pth](https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth) | 790,531,383 | `wmh-synthseg`；解剖结构与 WMH 的联合分割 |
| SynthSR | [synthsr_v20_230130.h5](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/f08/bc9/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5/SHA256E-s106163752--a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b.h5) | 106,163,752 | `synthsr`；默认通用 v2 |
| SynthSR | [synthsr_lowfield_v20_230130.h5](https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/de0/799/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5/SHA256E-s106163752--a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84.h5) | 106,163,752 | `synthsr-lowfield`；低场单输入 v2 |
| SynthSR | [synthsr_v10_210712.h5](https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthsr/synthsr_v10_210712.h5) | 53,075,984 | `synthsr-v1`；2021 年通用模型 |

合计 **4,729,380,177 字节**，约 4.73 GB（4.40 GiB）。只使用默认 SynthStrip 时需要第一个文件；默认 joint 配准需要 affine 和 deform 两个文件；WMH-SynthSeg 只需其单独的 `.pth`；默认 SynthSR 只需通用 v2 的 `.h5`。[WMH 官方目录](https://github.com/freesurfer/freesurfer/tree/dev/mri_WMHsynthseg) · [SynthSR 官方目录](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthsr)

SHA-256：

```text
37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33  synthstrip.1.pt
62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28  synthstrip.nocsf.1.pt
1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6  synthmorph.affine.2.h5
95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536  synthmorph.deform.3.h5
284c145fce47e98ecf3fdeda2163f646ac3ebb0240e87dd50d71d879f4d5b3af  synthmorph.rigid.1.h5
0ece39dd651357aa95222fc4d45fa32d00f11e763d2583cae3f869989ce35988  WMH-SynthSeg_v10_231110.pth
a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b  synthsr_v20_230130.h5
a7c5ea91c94fe31f3c716252caae0d181629201bd884dc59af88ddfd75ed4b84  synthsr_lowfield_v20_230130.h5
2fd59e96196388360eba95254fb6dfc9eb9eb8638018b590575e47e0a387f255  synthsr_v10_210712.h5
```

也可在 [provenance.json](provenance.json) 查看 SynthStrip/SynthMorph 权重与参考实现的来源记录。SynthMorph 和 SynthSR v2 由 FreeSurfer 的 git-annex 管理；直接下载 GitHub 同名 `raw` 路径可能只得到链接文本。本表的 SynthSR v2 链接指向实际 annex 对象，v1 链接经完整下载校验。[官方 SynthMorph 目录](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthmorph)

## 手动下载示例

下面下载默认 SynthStrip 权重并检查 SHA-256。其他模型替换为上表的完整 URL、文件名和对应 SHA-256 即可。

```bash
mkdir -p weights
curl --fail --location --retry 3 \
  'https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt' \
  --output weights/synthstrip.1.pt.part
printf '%s  %s\n' \
  '37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33' \
  'weights/synthstrip.1.pt.part' | sha256sum --check - && \
  mv weights/synthstrip.1.pt.part weights/synthstrip.1.pt
export FREESURFER_TORCH_WEIGHTS="$PWD/weights"
```

`pip install`、导入模块和推理不下载权重。离线计算节点可从联网机器复制已校验的权重目录。

## TorchFAST 不需要权重

`TorchFAST`、`fs-torch fast` 和 batch 中的 `"task": "fast"` 直接运行 HMRF-EM、
bias field 和 PVE 数值计算，不读取 checkpoint，也不需要执行
`tools/setup_weights.py`。只有从原始、未去颅骨 T1 开始并先调用 SynthStrip 时，
才需要配置 `synthstrip.1.pt`。`setup_weights.py --all` 的九个文件均属于上表四个
学习模型，不含 TorchFAST 文件。

`FastVBM` / `fs-torch fast-vbm` 从原始 T1w 开始，默认调用 SynthStrip，因此需要
`synthstrip.1.pt`；其 TorchFAST、GPU registration、Jacobian 和 modulation 阶段不读取
其他权重。只运行该 pipeline 时执行
`python tools/setup_weights.py --model fast-vbm` 即可；该名称等同于只配置默认
SynthStrip checkpoint。GM template 是独立输入，不是
模型权重，也不由本仓库或配置脚本下载。

## 权重许可与归属

**九个文件中，SynthStrip 与 SynthMorph 的五个权重可选择 MIT 或 CC BY 4.0 许可。** 两个功能的官网 “Code and Weights” 均明确提供这一选择。[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)，[SynthMorph](https://synthmorph.io/#code)

这五个权重的公开镜像可按所选许可发布，保留原作者、原始模型名称、官方来源和相应许可文本；如果转换或修改文件，注明具体变更。权重归原作者所有，本项目提供独立的 PyTorch 实现及验证，不将这些模型声称为本项目训练所得。模型卡应链接原论文，并记录文件 SHA-256。[MIT 条款](https://choosealicense.com/licenses/mit/)，[CC BY 4.0 条款](https://creativecommons.org/licenses/by/4.0/)

**WMH-SynthSeg 和 FreeSurfer 发布的 SynthSR 权重遵循 [FreeSurfer Software License](https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense)。** 官方没有为这些文件宣布上述 MIT 或 CC BY 4.0 双许可。该许可对下载、使用和再分发要求保留条款与归属信息；原文说明软件为研究用途设计，临床应用未获审查或批准。[WMH-SynthSeg 官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg) · [SynthSR 官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSR)

上述许可针对权重。改编代码及依赖继续遵守 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) 中的 FreeSurfer、Apache 等条款。

## 如果需要自行提供公开镜像

目前使用上述官方 URL。本项目没有上传权重或建立模型镜像。如需自行托管，下表列出可选平台；WMH-SynthSeg 与 SynthSR 镜像还须遵守 FreeSurfer 许可。

| 方式 | 当前官方限制 | 对本项目的适用性 |
|---|---|---|
| Hugging Face 模型仓库 | 免费公开存储为 best-effort；单文件硬上限 500 GB | 文件大小允许原样保存；适合持续维护模型与版本。附模型卡、各自许可和 SHA-256 清单。 |
| Zenodo | 免费服务；默认每条记录总计 50 GB、最多 100 个文件，适用公平使用政策 | 可原样保存全套，适合带 DOI 的固定研究版本。 |
| GitHub Release | 专属 Release 文档规定每个附件小于 2 GiB；最多 1,000 个附件，无总大小或下载带宽上限 | deform 权重需拆成小于 2 GiB 的分卷；下载后拼接并核对完整 SHA-256。 |
| GitHub LFS | Free/Pro 单文件 2 GB；Team 4 GB；Enterprise Cloud 5 GB | Free/Pro 无法原样上传 deform 权重；LFS 下载消耗仓库所有者的流量额度。 |
| 普通 Git 提交 | 超过 100 MiB 的单文件被拒绝 | 不适合存放整套模型，保持代码仓库轻量。 |

限制来源：[Hugging Face 存储](https://huggingface.co/docs/hub/storage-limits)、[Zenodo 文件限制](https://help.zenodo.org/docs/deposit/manage-files/)、[Zenodo 免费与公平使用](https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo)、[GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)、[Git LFS 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)、[普通 Git 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

GitHub 的通用大文件页面与专属 Release 页面对附件上限表述不同：前者关联账号 LFS 计划，后者明确为 2 GiB。因此若使用 Release，采用每个附件小于 2 GiB 的分卷方案。LFS 的免费额度当前为 Free/Pro 每月 10 GiB 下载流量及 10 GiB 存储；Team/Enterprise 为各 250 GiB，超额处理依预算配置。[LFS 计费说明](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)

若后续建立 Hugging Face 镜像，可用官方 `huggingface_hub` 上传文件夹，并以完整 commit ID 固定下载版本；仍核对本页 SHA-256。公开存储的 best-effort 政策不等同于无限免费额度。[上传文档](https://huggingface.co/docs/huggingface_hub/guides/upload)、[下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)
