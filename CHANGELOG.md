# 变更日志

本文件记录 UnionLLM 每个版本中使用者能感知到的变化，格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)（0.x 期间，次版本号的提升可能包含不兼容变更）。

PyPI 上有些版本没有对应的版本提交，下面按发布日期合并记录，内容以仓库提交历史为准。

## [0.3.0] - 未发布

这一版把 LiteLLM 从必装依赖改为可选依赖。从 0.2.x 升级前，请先阅读[迁移指南](docs/migration-0.3.md)。

### 不兼容变更

- **LiteLLM 改为可选依赖。** `pip install unionllm` 不再安装 LiteLLM。仍需要 LiteLLM 路由的用户改用 `pip install "unionllm[litellm]"`。未安装时，选中 LiteLLM 路由会抛出 `LiteLLMUnavailableError`（`ProviderError` 的子类），错误信息里带有安装命令。
- **Azure 上的非 Claude 模型不再经过 LiteLLM。** `provider="azure"` 下的 GPT、o 系列等模型改由 openai SDK 的 `AzureOpenAI` 直接调用（`AzureOpenAIProvider`）。请求参数按 LiteLLM `drop_params` 的规则改写，但返回的 `ModelResponse` 由 UnionLLM 自己构造，细节可能与 LiteLLM 不同。
- **`xai`、`openrouter`、`groq` 不再经过 LiteLLM。** 它们改由 openai SDK 调用各自的 OpenAI 兼容接口（`OpenAICompatibleProvider`），只读取 `XAI_API_KEY` / `OPENROUTER_API_KEY` / `GROQ_API_KEY`，不会读取 `OPENAI_API_KEY`。
- **自定义 OpenAI 兼容端点不再经过 LiteLLM。** 不传 `provider`、传 `custom_llm_provider="openai"` 和 `api_base` 的调用改走 `OpenAICompatibleProvider`，地址原样使用，不再补 `/v1`。
- **`qwen` 的保留模型 `qwen-special-model` 不再经过 LiteLLM，** 改为直接调用 DashScope 的 OpenAI 兼容接口。

### 新增

- 支持 openai 3.x（基于 httpx2）与 anthropic 1.x。`unionllm.exceptions` 不再在运行时导入 `httpx`。
- Azure OpenAI：接受 `api_base` / `base_url` / `azure_endpoint` / `endpoint` 与 `api_version`，未指定版本时使用 `2025-04-01-preview`；地址以 `/openai/v1` 结尾时改用 Azure 的 v1 API；除 `AZURE_OPENAI_*` 外也接受 LiteLLM 使用的 `AZURE_API_KEY`、`AZURE_API_BASE`、`AZURE_API_VERSION`；模型名可带 LiteLLM 风格的 `azure/` 前缀。流式响应会跳过 Azure 开头那个只含 `prompt_filter_results` 的空块。
- 新增 `LiteLLMUnavailableError`，位于 `unionllm.providers.litellm`。

### 修复

- Anthropic 与 Azure Claude：`temperature`、`top_p`、`top_k` 改为放进 `extra_body` 发送。anthropic 1.x 的 `messages.create` 删除了这三个关键字参数，旧写法会在请求发出前就抛出 `TypeError`。开启推理时，以及在只接受默认采样的模型上，这三个参数会被丢弃；开启推理时也不再强制设置 `temperature=1`。
- 只支持 Adaptive Thinking 的模型列表补上了 Claude Haiku 5.5 与 Claude Opus 4.7 / 4.8。这些模型会拒绝 `budget_tokens` 形式的推理配置。

### 已知问题

- 已在 anthropic 2.x 下跑过测试，但 Azure Claude 分支未固定 `anthropic-version`，SDK 2.x 默认使用 API 版本 `2026-10-21`，该版本会拒绝任何采样参数。在这个组合下给 Azure Claude 传 `temperature` 可能得到 400，该组合尚未用真实请求验证。原生 `anthropic` 分支固定发送 `2023-06-01`，不受影响。
- 通过 `unionchat(provider="azure", model="claude-...")` 调用时，路由看不到 `model`，请求不会进入 Azure Claude 分支。需要 Azure Claude 时，请使用 `UnionLLM(provider="azure", model="claude-...")` 构造客户端，且模型名不要带 `azure/` 前缀。0.2.x 中同样存在这个问题。

## [0.2.14] - 2026-10-15

### 变更

- Python 版本要求从不限制改为 `>=3.10,<3.14`。

### 修复

- Azure Claude：流式请求不再丢失系统提示语，并会传递 `cache_control`。
- DeepSeek：遵循调用方传入的 `api_base`，并把 `timeout` 传给上游请求。

## [0.2.13] - 2026-07-27

### 修复

- OpenAI 原生 provider 调用 Responses API 时的问题。

## [0.2.12] - 2026-07-27

### 新增

- OpenAI 原生 provider：`provider="openai"` 直接使用官方 SDK，不再经过 LiteLLM；默认走 Chat Completions，可通过 `api_mode="responses"` 改走 Responses API。
- 改进 Anthropic 原生 provider。
- OpenAI 的缓存命中 token 会映射到统一的 `usage` 中。

### 修复

- Anthropic 流式迭代中的错误统一为 provider 错误。
- Gemini 上下文缓存的问题。

### 其他

- 从仓库中移除了被跟踪的运行产物，重写了 README 与 `docs/`。

## [0.2.10] – [0.2.11] - 2026-04-25

- Anthropic 原生 provider：`provider="anthropic"` 默认直接使用官方 SDK，不再经过 LiteLLM；可通过 `use_litellm_anthropic=True` 或 `UNIONLLM_USE_LITELLM_ANTHROPIC=1` 回到 LiteLLM。
- 改进 DeepSeek V4 的支持。
- 智谱 GLM 改为通过 OpenAI 兼容接口调用，不再依赖 `zhipuai` SDK。

## [0.2.9] - 2026-01-30

- 支持 Kimi K2.5（`moonshot`），包括图片与视频输入。

## [0.2.4] – [0.2.8] - 2025-11-23 至 2025-11-30

- 新增 Gemini 原生 provider（基于 `google-genai`）与 Azure Anthropic 原生 provider。
- 若干修复，包括 Azure Anthropic 的问题。

## [0.2.0] – [0.2.3] - 2025-08-14 至 2025-09-30

- 更新多家 provider 的连接方式。

## [0.1.0] – [0.1.32] - 2024-05-08 至 2025-07-01

0.1.x 期间的主要变化：

- 首批 provider：智谱、Moonshot、MiniMax、百川、通义千问、天工、文心，以及 Dify、FastGPT、Coze 等 Agent 平台；海外模型经 LiteLLM 调用。
- 新增零一万物（`lingyi`）、阶跃星辰（`stepfun`）、豆包（`doubao`）、DeepSeek、讯飞 HTTP（`xunfei_http`）。
- 支持图片、对象内容、`video_url` 与文件输入；流式模式返回 token 用量；返回缓存 token 用量。
- 改进 OpenAI 兼容模型的支持；DeepSeek 支持推理过程与工具调用；Gemini 支持多模态输入与图片生成。

[0.3.0]: https://pypi.org/project/unionllm/0.3.0/
[0.2.14]: https://pypi.org/project/unionllm/0.2.14/
[0.2.13]: https://pypi.org/project/unionllm/0.2.13/
[0.2.12]: https://pypi.org/project/unionllm/0.2.12/
[0.2.11]: https://pypi.org/project/unionllm/0.2.11/
[0.2.10]: https://pypi.org/project/unionllm/0.2.10/
[0.2.9]: https://pypi.org/project/unionllm/0.2.9/
[0.2.8]: https://pypi.org/project/unionllm/0.2.8/
[0.2.4]: https://pypi.org/project/unionllm/0.2.4/
[0.2.3]: https://pypi.org/project/unionllm/0.2.3/
[0.2.0]: https://pypi.org/project/unionllm/0.2.0/
[0.1.32]: https://pypi.org/project/unionllm/0.1.32/
[0.1.0]: https://pypi.org/project/unionllm/0.1.0/
