# 验证记录

`../benchmark/` 中的早期数值结果使用 0.1.0 的固定权重和源码。0.2.0 的目录重组验证见 `refactor/report.public.json`。0.3.0 的 WMH-SynthSeg 验证包括 12 例 FLAIR 的四臂对照与双 GPU 批量运行，见 [wmh/README.md](wmh/README.md) 和 [wmh/report.public.json](wmh/report.public.json)。

FastVBM 0.9 的正式 10 例报告只包含 `end_to_end` 两后端对照；独立 direct FNIRT
matched-input 实验另行记录。入口见 [FastVBM 验证页](fast_vbm/README.md)，正式文件为
[`report.v0.9.public.json`](fast_vbm/report.v0.9.public.json)、[`backend_comparison.v0.9.public.csv`](fast_vbm/backend_comparison.v0.9.public.csv)、[`test_summary.v0.9.public.json`](fast_vbm/test_summary.v0.9.public.json) 和 [`release.v0.9.public.json`](fast_vbm/release.v0.9.public.json)。0.8 及更早结果在子页面中明确标为历史记录，不能作为
0.9 当前实现的准确度或计时。

公开记录不含账号、绝对私有路径或源病例 ID；私有路径用 `<PROJECT>`、
`<TEMPLATE_DIR>`、`<FREESURFER_HOME>` 和 `<PYTHON_BASE>` 等占位符替换。
为解释运行条件，记录可保留 GPU/CPU 型号等硬件标签，以及软件版本、测试结果、源码
和权重哈希。0.1.0 的哈希仅对应当时源码。临床病例原图、私有清单、权重及运行输出
没有上传。仓库提供三例公开去面容 T1w 和三例公开脑外清零 FLAIR 作为测试输入，并记录
来源与校验值。

模板测试发现的反向边界数值差异见 `../docs/COMPARISON.md`。公开统计没有人工真值，只检验对参考实现的复现程度。
