# 添加一个新功能

[返回首页](../README.md) · [现有架构](ARCHITECTURE.md)

新增功能使用独立目录，沿用明确的 Python 导出和 CLI 分支。当前无需插件发现、动态注册框架或通用模型基类。先实现实际需要的文件，再根据功能复杂度拆分。

## 1. 建立实现、文档和测试目录

以 `new_function` 为占位名称：

```text
src/freesurfer_torch/new_function/
├── __init__.py
├── pipeline.py
└── README.md
docs/new_function/README.md
tests/new_function/
```

只有在模型或空间运算足够独立时才增加 `model.py`、`models.py` 或 `spatial.py`。将功能特有逻辑留在该目录，不因名字相似就移入共享层。使用相对导入访问共享 `weights.py`。

可复用模型在构造时接收 `weights`、`device`，加载一次网络；`__call__` 处理单例输入，返回有明确字段的结果对象。若接入现有 batch，每个需要保存的结果字段应提供 `.save(path)`，模型参数必须能稳定序列化为缓存键。不要让调用流程解析 CLI、自动联网或启动另一个外部程序。

## 2. 明确公开接口

在功能的 `__init__.py` 导出模型、结果类型和确有需要的辅助函数，并声明 `__all__`。若希望提供顶层入口，在根 `__init__.py` 的 `__getattr__` 中增加对应分支，保留按需导入。

更新 `tests/test_public_api.py`，检查顶层对象与功能模块对象一致。调整旧功能时保留已公开的导入路径；可用很短的转导出模块兼容旧路径，不复制实现。

## 3. 接入共享入口

在 `cli.py` 添加子命令、参数和执行分支，明确输入、至少一个输出以及错误退出行为。参数默认值应与 Python API 一致；如果线程等运行参数不同，文档应写明。

接入 `batch.py` 的两个功能注册点：

1. 在 `_OUTPUTS` 中添加任务名和合法结果字段。
2. 在 `_model_class(task)` 中添加显式的任务分支，返回对应模型类。

同时更新 `_prepare_jobs` 的任务名错误提示。如果新功能在 `outputs` 之外写文件，将这些文件加入预执行路径冲突检查；不要绕过已有覆盖保护。其他调度、模型缓存、设备绑定、结果排序和错误记录通常无需修改。

## 4. 记录权重和依赖

权重通过共享解析器定位，推理默认只读本地文件。新增模型在 [WEIGHTS.md](WEIGHTS.md) 记录精确文件名、官方来源、大小、SHA-256、适用架构、许可证和引用要求。需要额外许可或来源说明时同步 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。不要把权重提交到 Git。

只有实现确实需要的新依赖才加入 `pyproject.toml`。当前 wheel 的 package-data 显式包含两个功能的 Markdown；新增功能后将其 README 路径加入列表，并检查 wheel 中实际包含该文件。

## 5. 写专属文档

`docs/new_function/README.md` 应包括：

- 任务和可接受的输入、返回类型及坐标/单位约定。
- 可运行的 Python、CLI 示例，模型构造和单次调用参数表。
- 权重需求、CPU/GPU 分工和模型复用行为。
- 与参考实现的对应关系、已知限制和实际验证入口。
- 如有性能或准确性数字，写清版本、设备、数据和计时范围。

源码目录的 `README.md` 用短示例说明核心 API，并链接完整文档。更新根 README 的功能表、[ARCHITECTURE.md](ARCHITECTURE.md) 的目录树；已有文档搬家时保留简短跳转页。

## 6. 验证后发布

为输入/输出几何、数值或行为的关键要求编写有意义的检查。GPU 功能应覆盖 CPU/GPU、模型复用，以及适用时的批量输出一致性。与原始工具比较时固定权重和输入，分别报告参考实现、运行环境和误差，保留失败记录。

```bash
python -m pip install -e .
python -m pip install pytest build
python -m pytest tests
python -m build
```

在独立环境安装生成的 wheel，检查新功能导入、CLI 帮助和至少一个代表性输入。检查源码发行包和 wheel 的文件列表、许可和版本；确认没有权重、原始影像、私人路径、凭据或临时文件。发布记录应区分新的实测结果和沿用的历史基准。
