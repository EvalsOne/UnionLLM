# 项目治理与改进计划

本页记录对源代码、测试、打包配置和文档的审查结论，以及每一项的当前状态。2026-07 做过第一次审查；2026-10 准备发布 0.3.0 时更新了一次。

## 已完成

| 项目 | 完成于 |
| --- | --- |
| 文档改为“README + 入门指南 + Provider 矩阵 + 架构说明 + 贡献指南”的结构；按 provider 拆分的旧页面保留为跳转页。 | 2026-07 |
| 从当前版本移除被跟踪的 `.DS_Store` 与 `dump.rdb`（后者是含任务元数据的 Redis 运行文件）。旧的 Git 历史中仍可能保留它们。 | 2026-07 |
| Python 版本要求从 `*` 改为 `>=3.10,<3.14`。 | 0.2.14 |
| LiteLLM 改为可选依赖，Azure OpenAI 与 OpenAI 兼容服务商改用 openai SDK；支持 openai 3.x 与 anthropic 1.x。 | 0.3.0 |
| 新增 CHANGELOG、0.3.0 迁移指南与可直接运行的示例；README 同时作为 PyPI 项目说明（此前 PyPI 页面没有项目说明）。 | 0.3.0 |
| 新增 `requirements-dev.txt`，测试所需的 `python-dotenv` 不再缺失；在 Python 3.10–3.13、openai 2.x/3.x、anthropic 0.x/1.x/2.x 下跑通测试。 | 0.3.0 |
| `provider="azure"` 改为按每次调用的模型选择分支，并去掉 `azure/` 前缀后判断 Claude，`unionchat` 与带前缀的模型名都能调用 Azure Claude；Azure Claude 固定 `anthropic-version: 2023-06-01`，API 版本为 `2026-10-21` 或更新时不再发送采样参数。补了模拟测试，并在 anthropic 0.x/1.x/2.x 下用真实请求验证。 | 0.3.0 |

## 待处理：代码问题

| 问题 | 影响 | 建议 |
| --- | --- | --- |
| Coze、Dify、FastGPT 适配器会构造 `context`，但 `ModelResponse` 只保留 `conversation_id`。 | 调用方拿不到知识库上下文，也没有回归测试。 | 决定是否公开 `context`：要支持就扩展响应模型并补测试，不支持就删掉适配器里的无用字段。 |
| `acompletion` 只是把同步调用放进线程池，`unionchat` 没有异步版本。 | 异步语义不清楚，流式时返回的仍是同步迭代器。 | 明确支持范围；要么实现原生异步，要么在下一个不兼容版本中标为实验性。 |
| 参数过滤和能力支持分散在各 provider 中。 | 调用方不知道某个参数会被忽略、改写还是报错。 | 建立声明式的 provider 能力与参数表，让代码和文档从同一来源生成。 |
| 测试集中在 OpenAI、Anthropic、Azure、Gemini、Moonshot、DeepSeek、智谱与 OpenAI 兼容路由，其他老 provider 没有单元测试。 | 修改老 provider 时没有回归保护。 | 为每个路由至少补一个无密钥的模拟测试，覆盖响应与错误的统一形状。 |

## 待处理：发布与维护

| 问题 | 建议 |
| --- | --- |
| 没有 CI、发布流水线、`SECURITY.md`、issue/PR 模板和行为准则。 | 加上 GitHub Actions：按 Python 3.10–3.13 与新旧 SDK 版本跑测试，并检查 README 能否在 PyPI 上渲染；配置 PyPI trusted publishing；补齐社区文件。 |
| 仓库没有版本标签，PyPI 上部分版本（如 0.2.2、0.2.3、0.2.10）找不到对应的版本提交。 | 从 0.3.0 起，每次发布都给发布提交打 `vX.Y.Z` 标签。 |
| 核心依赖没有下限，也没有锁文件。 | 在 `pyproject.toml` 中写上已验证的最低版本，CI 中同时测试最低版本与最新版本。 |
| `pytest.ini` 会递归收集被忽略的 `tests/my/`，其中的测试调用真实外部服务。 | 把真实服务测试移到单独的 integration 目录，用 marker 或环境变量开关隔离；目前文档统一使用 `--ignore=tests/my`。 |
| 旧 Git 历史中可能仍有 `dump.rdb`。 | 如果其中有敏感数据，协调后重写公开历史并清理缓存。 |
| `cookbook/`、`tests/my/` 等被忽略的目录只存在于本地。 | 区分私有材料与应当公开的内容，后者移入受版本控制的目录。 |

## 建议顺序

```text
发布之后        CI 与发布流程 → 补齐老 provider 的测试 → provider 能力表
长期            异步策略 → Agent 平台 context → 社区文件
```

每一步单独提交 PR，先建立可重复的验证基线，再做可能影响路由行为的重构。
