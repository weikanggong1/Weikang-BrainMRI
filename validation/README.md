# 验证记录

`../benchmark/` 中的早期数值结果使用 0.1.0 的固定权重和源码。0.2.0 的目录重组验证见 `refactor/report.public.json`。0.3.0 的 WMH-SynthSeg 验证包括 12 例 FLAIR 的四臂对照与双 GPU 批量运行，见 [wmh/README.md](wmh/README.md) 和 [wmh/report.public.json](wmh/report.public.json)。

公开记录中的服务器绝对路径已替换为 `<PROJECT>`、`<TEMPLATE_DIR>`、`<FREESURFER_HOME>`、`<PYTHON_BASE>`，账号和主机标签已删除；数值、版本、测试结果及原始源码和权重哈希均保留。0.1.0 的哈希仅对应当时的源码。临床病例原图、私有清单、权重及运行输出没有上传。仓库提供三例公开去面容 T1w 和三例公开脑外清零 FLAIR 作为测试输入，并记录来源与校验值。

模板测试发现的反向边界数值差异见 `../docs/COMPARISON.md`。公开统计没有人工真值，只检验对参考实现的复现程度。
