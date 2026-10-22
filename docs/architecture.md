# 架构说明

## 请求路径

```text
unionchat(model, messages, **kwargs)
  -> UnionLLM(provider=kwargs["provider"], **kwargs)
  -> provider 选择（unionllm/main.py）
  -> 专用 provider / OpenAI-compatible provider / LiteLLM provider（可选依赖）
  -> 上游 SDK 或 HTTP/WebSocket API
  -> 统一 ModelResponse 或流式 ModelResponse chunk
```

`unionllm/main.py` 是唯一的路由权威。它先处理直接适配 provider 和 OpenAI-compatible provider，然后对固定 allow-list 使用 LiteLLM，其他未知 provider 会抛出 `ProviderError`。传入空 provider 时，带 `custom_llm_provider="openai"` 的自定义端点走 OpenAI-compatible provider，其余使用 LiteLLM。

LiteLLM 是可选依赖（`unionllm[litellm]`），只在选中 LiteLLM 分支时才导入；未安装时构造 `UnionLLM` 即抛出 `LiteLLMUnavailableError`，并给出安装提示。

## 目录责任

| 路径 | 责任 |
| --- | --- |
| `unionllm/main.py` | 公共入口、provider 路由、LiteLLM 分支规则。 |
| `unionllm/providers/` | 厂商协议、认证、请求参数过滤/映射、流式转换。 |
| `unionllm/utils.py` | 统一响应对象、消息内容检查和跨 provider 的输入能力规则。 |
| `unionllm/exceptions.py` | 顶层异常。 |
| `tests/` | 当前已覆盖的回归测试，主要集中在近期原生适配器。 |
| `docs/` | 面向使用者与贡献者的文档。 |
| `examples/` | 只依赖环境变量的最小示例。 |
| `CHANGELOG.md` | 每个版本的使用者可见变化。 |
| `requirements-dev.txt` | 运行测试所需的额外依赖。 |

`cookbook/`、`tests/my/`、构建产物与缓存均被忽略，不属于发布源代码或稳定文档。

## 统一层的边界

1. 统一的是 Python 调用形状和大部分响应外形，不是所有厂商能力的交集。
2. 上游 provider 会各自过滤或改写关键字参数。参数透传必须以对应 adapter 的代码和测试为依据。
3. `ModelResponse` 模仿 chat-completions 对象，并保留 `conversation_id`；它不是任意上游响应的无损包装。特别是 Agent 适配器目前构造的 `context` 不会被该对象保留。
4. 多模态支持取决于 provider 与模型。基础检查可能拒绝不支持的输入，或将某些对象内容重写为文本。
5. LiteLLM 是可选依赖，不是镜像：本库的 allow-list、模型名前缀和版本约束都可能与 LiteLLM 的最新能力不同。

## 选择适配器时的特殊分支

- `openai` 总是选择 `OpenAINativeProvider`。
- `anthropic` 默认选择 `AnthropicAIProvider`，但 `use_litellm_anthropic` 或环境变量可切至 LiteLLM。
- `azure` 的模型名以 `claude` 开头时选择 `AzureAnthropicProvider`；其他模型选择 `AzureOpenAIProvider`（openai SDK 的 `AzureOpenAI`）。判断用的是构造 `UnionLLM` 时传入的 `model`；`unionchat` 不把 `model` 交给路由，所以经 `unionchat` 调用时总会选到 `AzureOpenAIProvider`。
- `qwen` 默认使用 DashScope；只有代码中的保留模型名 `qwen-special-model` 走 DashScope 的 OpenAI-compatible 接口。
- `xai`、`openrouter`、`groq` 选择 `OpenAICompatibleProvider`，使用各自的默认地址与环境变量密钥。
- 不传 `provider` 且 `custom_llm_provider="openai"` 时选择 `OpenAICompatibleProvider`，原样使用调用方的 `api_base` / `base_url`。
- 其他命中 LiteLLM allow-list 的 provider 使用 LiteLLM；未知值报错。

这些分支是文档、测试和未来重构时必须共同维护的契约。
