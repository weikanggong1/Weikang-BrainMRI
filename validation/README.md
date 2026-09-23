# 验证记录

除 `refactor/` 外，本目录和 `../benchmark/` 的数值结果来自 0.1.0 的固定权重与源码。0.2.0 的功能目录重组由 `refactor/report.public.json` 单独验证。

发布时仅将服务器绝对路径替换为 `<PROJECT>`、`<TEMPLATE_DIR>`、`<FREESURFER_HOME>`、`<PYTHON_BASE>`，删除账号和主机标签；保留数值、版本、测试结果与原始源码/权重哈希。它们是历史来源记录，不等于当前重组源码的哈希。病例数据、私有清单、权重和运行输出不上传。

模板测试包含反向边界数值差异；见 `../docs/COMPARISON.md`。公开统计没有人工真值，检验的是对参考实现的复现。
