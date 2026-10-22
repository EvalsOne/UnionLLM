# 从 0.2.x 迁移到 0.3.0

0.3.0 把 LiteLLM 从必装依赖改为可选依赖，并把 Azure OpenAI、xAI、OpenRouter、Groq 和自定义 OpenAI 兼容端点改为直接用 openai SDK 调用。公开 API（`unionchat`、`UnionLLM`）和 `provider` 取值都没有变化。完整变更见[变更日志](../CHANGELOG.md)。

## 先判断是否受影响

| 你的用法 | 是否受影响 | 需要做什么 |
| --- | --- | --- |
| 只用国内 provider、Agent 平台，或 `openai`、`anthropic`、`gemini`、`deepseek` | 基本不受影响 | 升级即可。注意下文的 [SDK 版本](#sdk-版本)。 |
| `provider="azure"` 调用 GPT、o 系列等非 Claude 模型 | 受影响 | 不再经过 LiteLLM，见 [Azure OpenAI](#azure-openai)。 |
| `provider="azure"` 调用 Claude | 行为有变化 | 路由与 API 版本见 [Azure Claude](#azure-claude)，采样参数见 [Claude 的采样参数](#claude-的采样参数)。 |
| `provider` 为 `xai`、`openrouter`、`groq` | 受影响 | 不再经过 LiteLLM，见 [OpenAI 兼容服务商](#openai-兼容服务商)。 |
| 不传 `provider`，传 `custom_llm_provider="openai"` 和 `api_base` | 受影响 | 同上。 |
| `bedrock`、`vertex_ai`、`mistral`、`ollama` 等 LiteLLM 路由，或不传 `provider`、直接用 LiteLLM 模型名 | 受影响 | 安装 `unionllm[litellm]`，见 [仍需 LiteLLM 的路由](#仍需-litellm-的路由)。 |
| `use_litellm_anthropic=True` 或设置了 `UNIONLLM_USE_LITELLM_ANTHROPIC=1` | 受影响 | 安装 `unionllm[litellm]`。 |

## 安装

```bash
python -m pip install -U unionllm
```

需要 LiteLLM 路由时：

```bash
python -m pip install -U "unionllm[litellm]"
```

## SDK 版本

0.3.0 不再锁定 openai 与 anthropic 的大版本。下列组合已在 Python 3.10–3.13 下跑通仓库测试（测试使用模拟的上游响应）；Azure Claude 另外在这三组版本下用真实请求验证过：

| openai | anthropic | 说明 |
| --- | --- | --- |
| 2.54 | 0.120 | 0.2.x 时期的组合 |
| 3.31 | 1.15 | openai 3 / anthropic 1 |
| 3.36 | 2.1 | 撰写本文时的最新版本 |

openai 3 与 anthropic 1 起，SDK 的 HTTP 层从 `httpx` 换成了 `httpx2`。只给 UnionLLM 传普通参数（`api_key`、`api_base`、`timeout` 等）时无需改动；如果你在测试里模拟 `httpx` 或用 `httpx` 做链路追踪，需要改为 `httpx2` 才能拦截到 SDK 的请求。

anthropic 2.x 默认使用 API 版本 `2026-10-21`，该版本会拒绝 `temperature`、`top_p`、`top_k`。UnionLLM 的原生 `anthropic` 分支与 Azure Claude 分支都固定发送 `anthropic-version: 2023-06-01`，不受影响。另外，anthropic 2.x 把 API key 放在 `Authorization: Bearer` 请求头里发送，如果 `ANTHROPIC_BASE_URL` 指向只读取 `x-api-key` 的代理，需要相应调整代理。

## Azure OpenAI

非 Claude 模型改由 openai SDK 的 `AzureOpenAI` 客户端调用（`AzureOpenAIProvider`）。

**配置**

| 配置 | 参数 | 环境变量（按优先级） |
| --- | --- | --- |
| API key | `api_key` | `AZURE_OPENAI_API_KEY`、`AZURE_API_KEY` |
| 地址 | `api_base`、`base_url`、`azure_endpoint`、`endpoint` | `AZURE_OPENAI_ENDPOINT`、`AZURE_API_BASE` |
| API 版本 | `api_version` | `AZURE_OPENAI_API_VERSION`、`AZURE_API_VERSION`，都没有时用 `2025-04-01-preview` |

LiteLLM 使用的 `AZURE_API_*` 环境变量仍然有效，原来的配置一般不用改。地址以 `/openai/v1` 结尾时，会改用 Azure 的 v1 API（普通 OpenAI 客户端，不带 `api-version`）。

**模型名与部署名**

模型名去掉 LiteLLM 风格的 `azure/` 前缀后，作为 Azure 部署名使用：`azure/gpt-5.4` 与 `gpt-5.4` 等价。参数改写规则按部署名判断模型家族，所以**部署名应以模型名开头**。名为 `my-deployment` 的部署会被当作普通对话模型，`reasoning_effort` 会被丢弃。名字像旧模型、实际部署的是推理模型时问题更明显：名为 `gpt-4o`、实际是 gpt-5.1 的部署，传 `max_tokens` 会直接得到 400（该模型只接受 `max_completion_tokens`）。LiteLLM 同样按名字判断，0.2.x 中也是如此。

**参数改写**

与 LiteLLM 的 `drop_params` 一样，模型不接受的参数会被丢弃，而不是让请求失败：

| 模型家族（按部署名判断） | 改写规则 |
| --- | --- |
| o 系列（`o1`、`o3`、`o4-mini` 等） | `max_tokens` 改为 `max_completion_tokens`；丢弃 `presence_penalty`、`frequency_penalty`、`logit_bias`、`top_p`、`logprobs`、`top_logprobs`，以及不等于 1 的 `temperature` |
| GPT-5 及以后的推理模型（`gpt-5-chat` 除外） | 同上，另外丢弃 `stop`。对 gpt-5.1、gpt-5.4 及以后、gpt-6 及以后（`-chat`、`-codex`、`-pro` 变体除外），如果 `reasoning_effort` 为 `none` 或未指定，会保留采样参数 |
| 其他模型（gpt-4o、gpt-4.1、gpt-5-chat 等） | 丢弃 `reasoning_effort` |

`provider="openai"` 原有的规则在 Azure 上同样适用：GPT-5.6 在 Chat Completions 中搭配 function tools 时，`reasoning_effort` 会被设为 `none`。

**响应**

返回值由 UnionLLM 自己构造，不再是 LiteLLM 的对象。字段形状与其他 provider 一致（`choices`、`usage` 等），但不要依赖 LiteLLM 特有的字段或隐藏属性。流式响应会跳过 Azure 开头那个只含 `prompt_filter_results` 的空块。

旧的类名 `AzureAIProvider` 仍保留，现在是 `AzureOpenAIProvider` 的别名。

## Azure Claude

**分支选择。** `provider="azure"` 改为按每次调用的模型选择分支：模型名去掉 `azure/` 前缀后以 `claude` 开头的走 Azure Claude（Anthropic Foundry），其余走 Azure OpenAI。0.2.x 只看构造 `UnionLLM` 时的 `model`，所以 `unionchat` 和带 `azure/` 前缀的模型名都进不了 Claude 分支；现在下面几种写法都能调用 Azure Claude：

```python
unionchat(provider="azure", model="azure/claude-sonnet-4-5", messages=messages, api_key=..., api_base=...)

client = UnionLLM(provider="azure", api_key=..., api_base=...)
client.completion(model="claude-sonnet-4-5", messages=messages)
```

构造时传了 `model`，会立即创建对应分支的客户端，与 0.2.x 相同；没传时在第一次 `completion()` 时创建，缺少 API key 或地址的错误也在那时抛出。两个分支的客户端各创建一次并复用，都使用构造时的配置。

**API 版本。** Azure Claude 固定发送 `anthropic-version: 2023-06-01`，可用参数 `anthropic_version` 或环境变量 `AZURE_ANTHROPIC_VERSION` 改。0.2.x 使用 anthropic SDK 的默认版本，在 anthropic 2.x 下就是 `2026-10-21`；该版本对任何采样参数都返回 400，还会拒绝 assistant 消息里的 document 块和 user 消息里带 citations 的 text 块。如果你自己把版本设为 `2026-10-21` 或更新，UnionLLM 不再发送采样参数。

## OpenAI 兼容服务商

`xai`、`openrouter`、`groq` 与自定义端点改由 `OpenAICompatibleProvider` 调用：

| provider | API key 环境变量 | 默认地址 |
| --- | --- | --- |
| `xai` | `XAI_API_KEY` | `https://api.x.ai/v1` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `groq` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| 不传 `provider`，`custom_llm_provider="openai"` | 无，传 `api_key` | 必须传 `api_base` 或 `base_url` |

与 LiteLLM 路由的区别：

- **永远不读取 `OPENAI_API_KEY`。** 这是有意的，避免把 OpenAI 的 key 发给第三方地址。原来靠 `OPENAI_API_KEY` 访问这些服务商的调用，需要改设上表中的环境变量或传 `api_key`。
- **地址原样使用**，不会补 `/v1`。自定义端点请传完整的 API 根地址，通常以 `/v1` 结尾。
- 模型名带 `xai/`、`openrouter/`、`groq/` 前缀时会去掉前缀。
- 自定义端点可以不传 `api_key`（适合不鉴权的本地服务），此时发送占位值 `unused`。

## Claude 的采样参数

这部分与 LiteLLM 无关，但同样在 0.3.0 中改变，影响 `provider="anthropic"` 与 Azure Claude：

- `temperature`、`top_p`、`top_k` 改为放进 `extra_body` 发送（anthropic 1.x 的 `messages.create` 已删除这三个关键字参数）。
- 开启推理（`thinking` 或 `reasoning_effort`）时，这三个参数会被丢弃；开启推理时也不再强制设置 `temperature=1`。
- 只接受默认采样的模型（Claude 5 系列，包括 Haiku 5.5，以及 Claude Opus 4.7 / 4.8）上，这三个参数总是被丢弃。这些模型的 `reasoning_effort` 会映射为 Adaptive Thinking 与 `output_config.effort`。
- `anthropic-version` 设为 `2026-10-21` 或更新时，这三个参数也会被丢弃（该版本拒绝它们）。两个 Claude 分支默认都发送 `2023-06-01`。

## 仍需 LiteLLM 的路由

下列调用仍经过 LiteLLM，需要安装 `unionllm[litellm]`：

- `UnionLLM.check_litellm_providers()` 中尚无专用实现的 provider，如 `bedrock`、`vertex_ai`、`mistral`、`ollama`、`together_ai`、`perplexity`、`cohere` 等，完整列表见 [Provider 矩阵](providers.md#litellm-路由可选)；
- 不传 `provider`、直接使用 LiteLLM 模型名的调用（例如 `model="ollama/llama3.1"`）；
- `provider="anthropic"` 且 `use_litellm_anthropic=True` 或 `UNIONLLM_USE_LITELLM_ANTHROPIC=1`。

未安装 LiteLLM 时，构造 `UnionLLM` 就会抛出 `LiteLLMUnavailableError`：

```python
from unionllm import UnionLLM
from unionllm.providers.litellm import LiteLLMUnavailableError

try:
    client = UnionLLM(provider="bedrock")
except LiteLLMUnavailableError as exc:
    print(exc.message)  # 含安装命令：pip install 'unionllm[litellm]'
```

它是 `ProviderError` 的子类，`status_code` 为 501。

**LiteLLM 与 openai 3 的冲突。** LiteLLM 1.84 起声明 `openai<3`，1.83.1–1.83.8 把 openai 锁定在某个 2.x 版本。在已装 openai 3 的环境里安装 `unionllm[litellm]`，pip 可能把 openai 降回 2.x，也可能退回 1.83.0 或更早的 LiteLLM（这些版本没有声明 openai 上限，但未在 openai 3 下验证过）。需要 LiteLLM 路由时，建议使用 openai 2.x。

## 升级检查清单

1. 搜索代码中的 `provider="azure"`、`"xai"`、`"openrouter"`、`"groq"`、`custom_llm_provider`、`use_litellm_anthropic`，对照上文逐项确认。
2. 用到 LiteLLM 路由的，安装 `unionllm[litellm]`，并确认 openai 停在 2.x。
3. 依赖 `OPENAI_API_KEY` 访问第三方 OpenAI 兼容服务的，改用对应服务商的环境变量或 `api_key`。
4. Azure 部署名与模型名不一致的，确认参数改写是否符合预期。
5. 捕获异常的代码，按需处理 `LiteLLMUnavailableError`。
6. 测试中模拟 `httpx` 的，在 openai 3 / anthropic 1 及以上版本中改为模拟 `httpx2`。
