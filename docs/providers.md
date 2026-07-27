# Provider 矩阵与配置

本页描述代码当前实际可路由的 provider，而不是厂商产品目录。`provider` 不区分大小写。除非特别说明，`api_key` 可以代替对应环境变量；`api_base` 是适配器普遍接受的自定义端点参数，但并非所有 provider 都支持它。

## 直接适配 provider

| Provider | 环境变量 | 默认/特殊配置 | 说明 |
| --- | --- | --- | --- |
| `zhipuai` | `ZHIPU_API_KEY` | 默认 OpenAI-compatible BigModel v4 地址 | 支持 OpenAI 风格 chat 参数与流式。 |
| `moonshot` | `MOONSHOT_API_KEY` | 默认 `https://api.moonshot.cn/v1` | `kimi-k2.5` 可在调用前将 HTTP(S) 图片/视频 URL 或本地文件转为 data URI。 |
| `minimax` | `MINIMAX_API_KEY` | `api_base`；也可传 `endpoint_url` | 默认 endpoint 为 `<api_base>/text/chatcompletion_v2`。 |
| `qwen` | `DASHSCOPE_API_KEY` | 默认使用 DashScope Python SDK | 特殊模型 `qwen-special-model` 走 LiteLLM OpenAI-compatible 分支。 |
| `tiangong` | `TIANGONG_APP_KEY`、`TIANGONG_APP_SECRET` | `app_key`、`app_secret` | 双凭证 provider。 |
| `baichuan` | `BAICHUAN_API_KEY` | `api_base` | OpenAI-compatible 适配器。 |
| `wenxin` | `ERNIE_API_KEY`，或 `ERNIE_CLIENT_ID` + `ERNIE_CLIENT_SECRET` | `api_key` 或 `client_id` + `client_secret` | 两套认证路径由实现决定。 |
| `xunfei` | `XUNFEI_APP_ID`、`XUNFEI_API_KEY`、`XUNFEI_API_SECRET` | `app_id`、`api_key`、`api_secret` | WebSocket 适配器。 |
| `xunfei_http` | `XUNFEI_HTTP_API_KEY` | `api_base` | 独立 HTTP 适配器，未在旧 README 中列出。 |
| `lingyi` | `LINGYI_API_KEY` | `api_base` | OpenAI-compatible 适配器。 |
| `stepfun` | `STEPFUN_API_KEY` | `api_base` | OpenAI-compatible 适配器。 |
| `doubao` | `ARK_API_KEY` | `api_base` | 模型值通常是方舟推理接入点 ID。 |
| `deepseek` | `DEEPSEEK_API_KEY` | `api_base`；`thinking` / `reasoning_effort` | 默认会在 `extra_body` 中处理 thinking 配置。 |
| `gemini` | `GEMINI_API_KEY` | `enable_context_cache`、`context_cache_min_tokens`、`context_cache_ttl_seconds` | 使用 `google-genai`；支持工具结果与可选显式上下文缓存。 |
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` 或 `OPENAI_API_BASE`；调用级 `base_url` / `api_base` | 使用原生 OpenAI SDK，不再经 LiteLLM。默认使用 Chat Completions；GPT-5.6 搭配 function tools 时会将 Chat 的 `reasoning_effort` 设为 `none`。显式传 `api_mode="responses"` 可改走 Responses API，并保留 reasoning。 |
| `anthropic` | `ANTHROPIC_API_KEY`、可选 `ANTHROPIC_BASE_URL`、`ANTHROPIC_VERSION`、`ANTHROPIC_BETA` | `use_litellm_anthropic=True` 或 `UNIONLLM_USE_LITELLM_ANTHROPIC=1` 可改走 LiteLLM | 默认使用 Anthropic Messages API；缺少 `max_tokens` 时默认 16384。Claude 5 会将 `reasoning_effort` 映射为 Adaptive Thinking 的 `thinking.type=adaptive` 与 `output_config.effort`。 |
| `azure` | Claude 原生分支使用 `AZURE_ANTHROPIC_API_BASE` 或 `AZURE_ANTHROPIC_ENDPOINT` | 也可传 `api_base` / `endpoint`；仅当模型名以 `claude` 开头走原生分支 | 非 Claude Azure 模型交给 LiteLLM；Claude 原生分支依赖 `anthropic` 与 `azure-identity`。当前实现只提供非流式 Claude 调用。 |

### Agent 应用 provider

这些适配器把聊天消息交给已配置的应用/Bot，而不是通过 `model` 选择一个基础模型。因此 `model` 是 API 形状所需的占位字段，选择由目标应用配置决定。

| Provider | 环境变量 | 特有参数 | 默认地址/说明 |
| --- | --- | --- | --- |
| `coze` | `COZE_API_KEY`、`COZE_BOT_ID` | `bot_id`、可选 `conversation_id` | Coze API 适配器。 |
| `dify` | `DIFY_API_KEY`、可选 `DIFY_API_BASE` | `conversation_id`、`user`（默认 `guest-user`） | 默认 `https://api.dify.ai/v1`。 |
| `fastgpt` | `FASTGPT_API_KEY` | `api_base` | 默认 `https://api.fastgpt.in/api/v1`。 |

这些平台的上下文、会话和知识库字段来自各自 API，格式可能变动。当前 Coze、Dify、FastGPT 适配器会构造 `context`，但统一 `ModelResponse` 构造器只保留 `conversation_id`；因此调用方目前不应依赖 `response.context`。这是已记录在[开源改进计划](open-source-review.md)中的契约缺口。

## 最小示例

```python
from unionllm import unionchat

response = unionchat(
    provider="gemini",
    model="gemini-2.0-flash",
    api_key="...",  # 或设置 GEMINI_API_KEY
    messages=[{"role": "user", "content": "Hello"}],
)
```

使用特定 provider 前，先确认上游账户已启用对应模型/应用，并用厂商官方模型名替换示例值。

### OpenAI GPT-5.6 reasoning 与 function tools

`api_mode` 是 OpenAI 原生路由的切换开关，可作为实例默认值，也可在单次调用中覆盖：

```python
from unionllm import UnionLLM

# 默认值：保留 Chat Completions。
# GPT-5.6 同时使用 function tools 时，reasoning_effort 会被设为 "none"。
client = UnionLLM(provider="openai", api_mode="chat")

# 需要同时保留 reasoning 与 function tools 时，显式切换到 Responses API。
response = client.completion(
    model="gpt-5.6-luna",
    messages=messages,
    tools=tools,
    reasoning_effort="high",
    api_mode="responses",
)
```

省略 `api_mode` 时仍默认使用 Chat Completions，不会因为模型名称自动切换 API。

## LiteLLM allow-list 和行为

下列 `provider` 由 `UnionLLM.check_litellm_providers()` 处理：

- 自动为模型名添加 provider 前缀：`azure`、`azure_ai`、`anthropic`、`deepseek`、`sagemaker`、`bedrock`、`vertex_ai`、`vertex_ai_beta`、`palm`、`gemini`、`mistral`、`cloudflare`、`huggingface`、`replicate`、`together_ai`、`openrouter`、`baseten`、`nlp_cloud`、`petals`、`ollama`、`perplexity`、`groq`、`anyscale`、`watsonx`、`voyage`、`xinference`。
- 原样把模型名交给 LiteLLM：`openai`、`cohere`、`ai21`、`deepinfra`、`alpha_alpha`。
- 以 OpenAI-compatible 方式调用：`xai`、`qwen`。

其中 `openai`、`anthropic`、`deepseek`、`gemini`、`qwen` 和 `azure` 已有前置的专用路由，实际是否进入 LiteLLM 取决于[请求路由](architecture.md)。此列表是当前代码分支而非“LiteLLM 全部 provider”承诺。

## 迁移旧页面

旧的单 provider 页面保留在 `docs/` 下供外部链接使用，并统一跳转到本页。新文档不再复制模型列表、价格和厂商链接，避免把随时变动的商业信息当作库的静态事实。
