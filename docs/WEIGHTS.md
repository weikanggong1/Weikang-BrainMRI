# 预训练权重：下载、校验与公开发布

本仓库和 Python wheel 不包含权重。五个所需模型均已由原作者公开，无需安装完整 FreeSurfer 即可从官网获取；下载后放入同一目录，并用 `FREESURFER_TORCH_WEIGHTS` 或 API/CLI 的权重路径指定该目录。

以下链接、HTTP 状态和文件大小于 **2026-09-23** 核验；五个端点均返回 HTTP 200。SHA-256 来自本包已完成数值验证的权重，并与 FreeSurfer 官方仓库的 git-annex 指针一致。此处的版本号固定，不会自动跟随上游替换为新模型。

## 官方文件

| 功能 | 文件与官方下载链接 | 字节数 | 使用场景 |
|---|---|---:|---|
| SynthStrip | [synthstrip.1.pt](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt) | 30,851,709 | 默认脑提取 |
| SynthStrip | [synthstrip.nocsf.1.pt](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.nocsf.1.pt) | 30,851,709 | `no_csf=True` / `--no-csf` |
| SynthMorph | [synthmorph.affine.2.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.affine.2.h5) | 51,455,312 | affine；joint 的仿射阶段 |
| SynthMorph | [synthmorph.deform.3.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.deform.3.h5) | 3,508,630,424 | deform；joint 的非线性阶段 |
| SynthMorph | [synthmorph.rigid.1.h5](https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.rigid.1.h5) | 51,656,152 | rigid |

合计 **3,673,445,306 字节**，约 3.67 GB（3.42 GiB）。只使用默认 SynthStrip 时需要第一个文件；默认 joint 配准需要 affine 和 deform 两个文件。

SHA-256：

```text
37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33  synthstrip.1.pt
62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28  synthstrip.nocsf.1.pt
1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6  synthmorph.affine.2.h5
95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536  synthmorph.deform.3.h5
284c145fce47e98ecf3fdeda2163f646ac3ebb0240e87dd50d71d879f4d5b3af  synthmorph.rigid.1.h5
```

也可在 [provenance.json](provenance.json) 查看权重与参考实现的完整来源记录。不要直接下载 GitHub `raw` 页上的同名文件：FreeSurfer 用 git-annex 管理大文件，其 `raw` 内容可能只是几十到几百字节的链接文本。[官方 SynthMorph 仓库说明](https://github.com/freesurfer/freesurfer/tree/dev/mri_synthmorph)

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

下载是显式操作；`pip install`、导入模块和推理都不隐式下载权重。离线计算节点可从联网机器下载、校验后复制整个权重目录。

## 权重许可与归属

**五个权重均可选择 MIT 或 CC BY 4.0 许可。** SynthStrip 官网的 “Code and Weights” 和 SynthMorph 官网的 “Code and weights” 均明确提供这一选择；上表链接来自这两个官方页面。[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/)，[SynthMorph](https://synthmorph.io/#code)

公开镜像可以按所选许可发布，保留原作者、原始模型名称、官方来源和相应许可文本；如果转换或修改文件，注明具体变更。权重归原作者所有，本项目提供独立的 PyTorch 实现及验证，不将这些模型声称为本项目训练所得。模型卡应链接原论文，并记录文件 SHA-256。[MIT 条款](https://choosealicense.com/licenses/mit/)，[CC BY 4.0 条款](https://creativecommons.org/licenses/by/4.0/)

上述许可针对权重。改编代码及依赖继续遵守 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) 中的 FreeSurfer、Apache 等条款。

## 如果需要自行提供公开镜像

当前直接使用官方 URL 最简单。本次只发布代码和文档，没有另行上传权重或创建模型托管仓库。将来希望在自己账号下提供下载，可选择：

| 方式 | 当前官方限制 | 对本项目的适用性 |
|---|---|---|
| Hugging Face 模型仓库 | 免费公开存储为 best-effort；单文件硬上限 500 GB | 五个文件均可原样保存；适合持续维护模型与版本。附模型卡、许可和 SHA-256 清单。 |
| Zenodo | 免费服务；默认每条记录总计 50 GB、最多 100 个文件，适用公平使用政策 | 可原样保存全套，适合带 DOI 的固定研究版本。 |
| GitHub Release | 专属 Release 文档规定每个附件小于 2 GiB；最多 1,000 个附件，无总大小或下载带宽上限 | deform 权重需拆成小于 2 GiB 的分卷；下载后拼接并核对完整 SHA-256。 |
| GitHub LFS | Free/Pro 单文件 2 GB；Team 4 GB；Enterprise Cloud 5 GB | Free/Pro 无法原样上传 deform 权重；LFS 下载消耗仓库所有者的流量额度。 |
| 普通 Git 提交 | 超过 100 MiB 的单文件被拒绝 | 不适合存放整套模型，保持代码仓库轻量。 |

限制来源：[Hugging Face 存储](https://huggingface.co/docs/hub/storage-limits)、[Zenodo 文件限制](https://help.zenodo.org/docs/deposit/manage-files/)、[Zenodo 免费与公平使用](https://support.zenodo.org/help/en-gb/1-upload-deposit/80-what-are-the-size-limitations-of-zenodo)、[GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)、[Git LFS 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)、[普通 Git 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

GitHub 的通用大文件页面与专属 Release 页面对附件上限表述不同：前者关联账号 LFS 计划，后者明确为 2 GiB。因此若使用 Release，采用每个附件小于 2 GiB 的分卷方案。LFS 的免费额度当前为 Free/Pro 每月 10 GiB 下载流量及 10 GiB 存储；Team/Enterprise 为各 250 GiB，超额处理依预算配置。[LFS 计费说明](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)

若后续建立 Hugging Face 镜像，可用官方 `huggingface_hub` 上传文件夹，并以完整 commit ID 固定下载版本；仍核对本页 SHA-256。公开存储的 best-effort 政策不等同于无限免费额度。[上传文档](https://huggingface.co/docs/huggingface_hub/guides/upload)、[下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)
