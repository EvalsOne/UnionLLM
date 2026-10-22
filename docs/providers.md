# Provider 矩阵与配置

本页列出 `unionllm/main.py` 当前实际能路由到的 provider，内容以代码为准，不是厂商的产品目录。可用模型、价格与配额请看各厂商的官方文档。

阅读约定：

- `provider` 不区分大小写。
- 除特别说明外，`api_key` 参数可以代替表中的环境变量；参数优先于环境变量。
- `api_base` 是多数适配器接受的自定义地址参数，但不是每个 provider 都支持，以表中说明为准。
- “支持某个 provider”不代表所有能力相同。多模态、工具调用、推理参数、缓存、流式 usage 都取决于具体适配器。

## 速查

| provider | 实现 | 需要 LiteLLM |
| --- | --- | --- |
| `zhipuai`、`moonshot`、`minimax`、`qwen`、`tiangong`、`baichuan`、`wenxin`、`xunfei`、`xunfei_http`、`lingyi`、`stepfun`、`doubao`、`deepseek` | 各自的适配器 | 否 |
| `openai`、`anthropic`、`gemini` | 官方 SDK | 否（`anthropic` 可选择改走 LiteLLM） |
| `azure` | Claude：`AnthropicFoundry`；其他模型：openai SDK 的 `AzureOpenAI` | 否 |
| `xai`、`openrouter`、`groq`，以及自定义 OpenAI 兼容端点 | openai SDK | 否 |
| `coze`、`dify`、`fastgpt` | 各自的适配器 | 否 |
| `bedrock`、`vertex_ai`、`mistral`、`ollama` 等 | LiteLLM | 是，见 [LiteLLM 路由（可选）](#litellm-路由可选) |

## 国内模型

| provider | 环境变量 | 地址与特有参数 | 说明 |
| --- | --- | --- | --- |
| `zhipuai` | `ZHIPU_API_KEY` | 默认 `https://open.bigmodel.cn/api/paas/v4`，可用 `api_base` 覆盖 | 经 openai SDK 调用智谱的 OpenAI 兼容接口，支持流式。 |
| `moonshot` | `MOONSHOT_API_KEY` | 默认 `https://api.moonshot.cn/v1` | `kimi-k2.5` 支持图片与视频输入，HTTP(S) URL 或本地文件会在调用前转为 data URI。 |
| `minimax` | `MINIMAX_API_KEY` | `api_base` 默认 `https://api.minimax.chat/v1`；也可直接传完整的 `endpoint_url` | 默认请求 `<api_base>/text/chatcompletion_v2`。 |
| `qwen` | `DASHSCOPE_API_KEY` | 默认使用 DashScope Python SDK | 保留模型名 `qwen-special-model` 改走 DashScope 的 OpenAI 兼容接口。 |
| `tiangong` | `TIANGONG_APP_KEY`、`TIANGONG_APP_SECRET` | 参数 `app_key`、`app_secret` | 需要两个凭证。 |
| `baichuan` | `BAICHUAN_API_KEY` | `api_base` | OpenAI 兼容适配器。 |
| `wenxin` | `ERNIE_API_KEY`，或 `ERNIE_CLIENT_ID` + `ERNIE_CLIENT_SECRET` | 参数 `api_key`，或 `client_id` + `client_secret` | 传 `api_key` 时调用千帆 v2 接口；传 `client_id` + `client_secret` 时调用旧的文心工作台接口。 |
| `xunfei` | `XUNFEI_APP_ID`、`XUNFEI_API_KEY`、`XUNFEI_API_SECRET` | 参数 `app_id`、`api_key`、`api_secret` | WebSocket 适配器。 |
| `xunfei_http` | `XUNFEI_HTTP_API_KEY` | 默认 `https://spark-api-open.xf-yun.com/v1`，可用 `api_base` 覆盖 | 讯飞星火的 HTTP 接口。 |
| `lingyi` | `LINGYI_API_KEY` | 默认 `https://api.lingyiwanwu.com/v1` | 零一万物，OpenAI 兼容适配器。 |
| `stepfun` | `STEPFUN_API_KEY` | 默认 `https://api.stepfun.com/v1` | 阶跃星辰，OpenAI 兼容适配器。 |
| `doubao` | `ARK_API_KEY` | `api_base` | 火山方舟；`model` 通常填推理接入点 ID。 |
| `deepseek` | `DEEPSEEK_API_KEY` | 默认 `https://api.deepseek.com/v1`；`beta=True` 时用 `https://api.deepseek.com/beta`；`api_base` 优先 | 支持工具调用与 `timeout`；`thinking` / `reasoning_effort` 会整理进 `extra_body`。 |

## 海外模型（官方 SDK）

### `openai`

| 环境变量 | 地址 | 特有参数 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 默认 `https://api.openai.com`；`OPENAI_BASE_URL` 或 `OPENAI_API_BASE` 覆盖；调用时可传 `base_url` / `api_base` | `api_mode` |

默认走 Chat Completions。GPT-5.6 在 Chat Completions 中搭配 function tools 时，`reasoning_effort` 会被设为 `none`。需要同时保留推理和工具时，显式传 `api_mode="responses"` 改走 Responses API：

```python
from unionllm import UnionLLM

client = UnionLLM(provider="openai")
response = client.completion(
    model="gpt-5.6-luna",
    messages=messages,
    tools=tools,
    reasoning_effort="high",
    api_mode="responses",
)
```

`api_mode` 可以在构造客户端时设默认值，也可以在单次调用时覆盖。省略时使用 Chat Completions，不会根据模型名自动切换。调用时传入的 `api_key`、`api_base` / `base_url` 会覆盖构造时的配置。

### `anthropic`

| 环境变量 | 特有参数 |
| --- | --- |
| `ANTHROPIC_API_KEY`；可选 `ANTHROPIC_BASE_URL`、`ANTHROPIC_VERSION`、`ANTHROPIC_BETA` | `anthropic_version`、`anthropic_beta`、`use_litellm_anthropic` |

- 使用 Anthropic Messages API。默认发送 `anthropic-version: 2023-06-01`，可用参数或 `ANTHROPIC_VERSION` 改。anthropic 2.x 自己的默认值是 `2026-10-21`，该版本拒绝采样参数；版本设为 `2026-10-21` 或更新时，采样参数会被丢弃。
- 未传 `max_tokens` 时默认 16384。
- 推理：Claude 5 系列（包括 Haiku 5.5）与 Claude Opus 4.7 / 4.8 只支持 Adaptive Thinking，`reasoning_effort` 会映射为 `thinking.type=adaptive` 与 `output_config.effort`；更早的模型使用 `budget_tokens`。
- 采样参数：`temperature`、`top_p`、`top_k` 放在 `extra_body` 中发送（anthropic 1.x 的 `messages.create` 删除了这三个关键字参数）。开启推理时，以及在上述只接受默认采样的模型上，这三个参数会被丢弃。
- `use_litellm_anthropic=True` 或 `UNIONLLM_USE_LITELLM_ANTHROPIC=1` 可改走 LiteLLM，需要安装 `unionllm[litellm]`。
- anthropic 2.x 把 API key 放在 `Authorization: Bearer` 请求头里发送。`ANTHROPIC_BASE_URL` 指向的代理如果只读取 `x-api-key`，需要相应调整。

### `gemini`

| 环境变量 | 特有参数 |
| --- | --- |
| `GEMINI_API_KEY`；可选 `GEMINI_CONTEXT_CACHE_MIN_TOKENS`、`GEMINI_CONTEXT_CACHE_TTL_SECONDS` | `enable_context_cache`、`context_cache_min_tokens`、`context_cache_ttl_seconds` |

使用 `google-genai`，支持多模态输入、工具调用与工具结果。`enable_context_cache=True` 时，会为足够长的稳定历史前缀创建显式上下文缓存。

## Azure

`provider="azure"` 按模型名分成两个分支：模型名去掉 `azure/` 前缀后以 `claude` 开头（不区分大小写）的走 Claude 分支，其余走 Azure OpenAI 分支。

判断依据是每次调用的模型，`unionchat` 与在 `completion()` 时才给出模型的 `UnionLLM(provider="azure")` 都能进入 Claude 分支。构造时传了 `model` 会立即创建对应分支的客户端；没传时在第一次调用时创建。两个分支的客户端各创建一次并复用，都使用构造时的配置。

### Azure OpenAI（GPT、o 系列等）

| 配置 | 参数 | 环境变量（按优先级） |
| --- | --- | --- |
| API key | `api_key` | `AZURE_OPENAI_API_KEY`、`AZURE_API_KEY` |
| 地址 | `api_base`、`base_url`、`azure_endpoint`、`endpoint` | `AZURE_OPENAI_ENDPOINT`、`AZURE_API_BASE` |
| API 版本 | `api_version` | `AZURE_OPENAI_API_VERSION`、`AZURE_API_VERSION`，都没有时用 `2025-04-01-preview` |

- 经 openai SDK 的 `AzureOpenAI` 调用；地址以 `/openai/v1` 结尾时改用 Azure 的 v1 API。
- 模型名去掉 `azure/` 前缀后作为部署名。参数改写按部署名判断模型家族，所以部署名应以模型名开头；名字与模型不符的部署（例如名为 `gpt-4o`、实际是 gpt-5.1）可能因参数不被接受而得到 400。
- 模型不接受的参数会被丢弃：o 系列与 GPT-5 及以后的推理模型把 `max_tokens` 改为 `max_completion_tokens`，并丢弃不支持的采样与惩罚参数；其他模型丢弃 `reasoning_effort`。完整规则见[迁移指南](migration-0.3.md#azure-openai)。
- 流式响应会跳过 Azure 开头那个只含 `prompt_filter_results` 的空块。

### Azure Claude

| 配置 | 参数 | 环境变量 |
| --- | --- | --- |
| 地址 | `api_base`、`endpoint` | `AZURE_ANTHROPIC_API_BASE`、`AZURE_ANTHROPIC_ENDPOINT` |
| API key | `api_key` | 不传时由 anthropic SDK 的 `AnthropicFoundry` 读取 `ANTHROPIC_FOUNDRY_API_KEY` |
| API 版本 | `anthropic_version` | `AZURE_ANTHROPIC_VERSION`，都没有时用 `2023-06-01` |

- 依赖 `anthropic` 与 `azure-identity`。
- 模型名去掉 `azure/` 前缀后作为部署名，`azure/claude-sonnet-4-5` 与 `claude-sonnet-4-5` 等价。
- 采样参数与推理的规则与 `anthropic` 相同。
- 未传 `max_tokens` 时默认 30000。这个值下 SDK 会拒绝非流式请求、要求改用流式，所以非流式调用请显式传 `max_tokens`。
- 固定发送 `anthropic-version: 2023-06-01`，不随 anthropic SDK 的默认值变化（2.x 默认 `2026-10-21`，该版本拒绝采样参数）。Azure Foundry 两个版本都接受；设为 `2026-10-21` 或更新时，采样参数会被丢弃。

## OpenAI 兼容服务商

经 openai SDK 调用各自的 Chat Completions 接口。地址原样使用，不会补 `/v1`；永远不读取 `OPENAI_API_KEY`，避免把 OpenAI 的 key 发到第三方地址。模型名带 `<provider>/` 前缀时会去掉前缀。

| provider | 环境变量 | 默认地址 |
| --- | --- | --- |
| `xai` | `XAI_API_KEY` | `https://api.x.ai/v1` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `groq` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| 不传 `provider`，传 `custom_llm_provider="openai"` | 无，传 `api_key`；可省略 | 必须传 `api_base` 或 `base_url` |

自定义端点不传 `api_key` 时发送占位值 `unused`，适合不鉴权的本地服务：

```python
from unionllm import unionchat

response = unionchat(
    custom_llm_provider="openai",
    api_base="http://localhost:8000/v1",
    model="my-local-model",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Agent 平台

这些适配器把消息交给已配置好的应用或 Bot，`model` 只是接口形状需要的占位值，实际调用什么由平台上的应用配置决定。

| provider | 环境变量 | 特有参数 | 默认地址 |
| --- | --- | --- | --- |
| `coze` | `COZE_API_KEY`、`COZE_BOT_ID` | `bot_id`、可选 `conversation_id` | `https://api.coze.com` |
| `dify` | `DIFY_API_KEY`、可选 `DIFY_API_BASE` | `conversation_id`、`user`（默认 `guest-user`） | `https://api.dify.ai/v1` |
| `fastgpt` | `FASTGPT_API_KEY` | `api_base` | `https://api.fastgpt.in/api/v1` |

这些平台的会话与知识库字段来自各自的 API，格式可能变化。Coze、Dify、FastGPT 适配器会构造 `context`，但统一的 `ModelResponse` 只保留 `conversation_id`，调用方不要依赖 `response.context`。

## LiteLLM 路由（可选）

需要安装 `unionllm[litellm]`。未安装时，构造 `UnionLLM` 就会抛出 `LiteLLMUnavailableError`（`ProviderError` 的子类，位于 `unionllm.providers.litellm`）。LiteLLM 与 openai 3 的版本冲突见[迁移指南](migration-0.3.md#仍需-litellm-的路由)。

下列 `provider` 在没有更早的专用分支时交给 LiteLLM，由 `UnionLLM.check_litellm_providers()` 判断：

- 模型名自动加上 `<provider>/` 前缀：`azure_ai`、`sagemaker`、`bedrock`、`vertex_ai`、`vertex_ai_beta`、`palm`、`mistral`、`cloudflare`、`huggingface`、`replicate`、`together_ai`、`baseten`、`nlp_cloud`、`petals`、`ollama`、`perplexity`、`anyscale`、`watsonx`、`voyage`、`xinference`。
- 模型名原样交给 LiteLLM：`cohere`、`ai21`、`deepinfra`、`alpha_alpha`。

此外，以下调用也走 LiteLLM：

- 不传 `provider`、也不传 `custom_llm_provider="openai"`，此时 `model` 按 LiteLLM 的格式书写，例如 `ollama/llama3.1`；
- `provider="anthropic"` 且开启了 `use_litellm_anthropic`。

`check_litellm_providers()` 的列表里还有 `openai`、`anthropic`、`deepseek`、`gemini`、`azure`、`openrouter`、`groq`，但它们都有更早的专用分支，实际不会进入 LiteLLM。这份列表是当前代码的分支，不代表 LiteLLM 支持的全部 provider。认证方式、模型名和高级参数以 LiteLLM 官方文档为准。

## 旧页面

`docs/` 下按 provider 拆分的旧页面保留为跳转页，避免外部链接失效，内容统一指向本页。本仓库不再复制模型列表、价格和厂商链接。
