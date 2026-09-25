# 验证记录

每个功能只保留当前使用的说明与最新公开 benchmark：

- SynthStrip / SynthMorph：[`benchmark/public_report/summary.md`](../benchmark/public_report/summary.md) 与机器可读 CSV。
- WMH-SynthSeg：[`wmh/README.md`](wmh/README.md) 与 [`wmh/report.public.json`](wmh/report.public.json)。
- SynthSR：[`synthsr/README.md`](synthsr/README.md)。
- TorchFAST：[`fast/README.md`](fast/README.md) 与 [`fast/report.public.json`](fast/report.public.json)。
- TorchApplyWarp：[`applywarp/report.json`](applywarp/report.json)。
- FLIRT、FNIRT 与 FastVBM：[`fast_vbm/README.md`](fast_vbm/README.md) 及 0.9 正式文件。

公开记录不含账号、私有绝对路径、源病例 ID、权重或临床原图。公开样例及其来源校验见 [T1w 示例](../examples/README.md)和 [FLAIR 示例](../examples/WMH.md)。没有人工真值的报告只衡量与参考实现的一致性。
