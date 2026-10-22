# UnionLLM

用一套 OpenAI 风格的 `model` / `messages` 调用方式，访问国内外的大模型和 Agent 平台的 Python 库。

UnionLLM 负责选择 provider、适配请求与响应格式，并尽量把结果统一成 OpenAI 风格的 `ModelResponse`。它是一个 Python 库，不是 HTTP 代理服务。

> **0.3.0 起，LiteLLM 改为可选依赖。** Azure OpenAI、xAI、OpenRouter、Groq 和自定义 OpenAI 兼容端点改为直接用 openai SDK 调用，支持 openai 3.x 与 anthropic 1.x。从 0.2.x 升级前请阅读[迁移指南](https://github.com/EvalsOne/UnionLLM/blob/main/docs/migration-0.3.md)。

## 特点

- **一套调用方式**：`unionchat(provider=..., model=..., messages=...)`，流式与非流式都返回 OpenAI 风格的对象。
- **国内模型直接适配**：智谱、Moonshot（Kimi）、通义千问、DeepSeek、豆包、文心、讯飞星火、MiniMax、百川、零一万物、阶跃星辰、天工。
- **海外模型用官方 SDK**：OpenAI、Anthropic、Gemini、Azure OpenAI 与 Azure 上的 Claude。
- **Agent 平台**：Coze、Dify、FastGPT。
- **其他 provider 可选 LiteLLM**：Bedrock、Vertex AI、Mistral、Ollama 等，按需安装。

## 安装

```bash
python -m pip install unionllm
```

需要 LiteLLM 路由时：

```bash
python -m pip install "unionllm[litellm]"
```

Python 要求 3.10–3.13。LiteLLM 要求 openai 2.x，详见[迁移指南](https://github.com/EvalsOne/UnionLLM/blob/main/docs/migration-0.3.md#仍需-litellm-的路由)。

## 快速开始

```bash
export ZHIPU_API_KEY="..."
```

```python
from unionllm import unionchat

response = unionchat(
    provider="zhipuai",
    model="glm-4",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

流式调用返回可迭代的 chunk：

```python
for chunk in unionchat(
    provider="openai",
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "用一句话介绍 UnionLLM"}],
    stream=True,
):
    if chunk.choices and getattr(chunk.choices[0].delta, "content", None):
        print(chunk.choices[0].delta.content, end="")
```

需要复用配置时，使用 `UnionLLM` 客户端：

```python
from unionllm import UnionLLM

client = UnionLLM(provider="deepseek")
response = client.completion(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
)
```

调用约定、异步、流式与错误处理见[入门指南](https://github.com/EvalsOne/UnionLLM/blob/main/docs/getting-started.md)；更多示例见 [examples/](https://github.com/EvalsOne/UnionLLM/tree/main/examples)。

## 支持范围

| 类别 | `provider` 取值 | 实现 |
| --- | --- | --- |
| 国内模型 | `zhipuai`、`moonshot`、`qwen`、`deepseek`、`doubao`、`wenxin`、`xunfei`、`xunfei_http`、`minimax`、`baichuan`、`lingyi`、`stepfun`、`tiangong` | 各自的适配器 |
| 海外模型 | `openai`、`anthropic`、`gemini` | 官方 SDK |
| Azure | `azure` | Claude 走 `AnthropicFoundry`，其他模型走 openai SDK 的 `AzureOpenAI` |
| OpenAI 兼容 | `xai`、`openrouter`、`groq`，以及 `custom_llm_provider="openai"` 加 `api_base` 的自定义端点 | openai SDK |
| Agent 平台 | `coze`、`dify`、`fastgpt` | 各自的适配器 |
| LiteLLM（可选） | `bedrock`、`vertex_ai`、`mistral`、`ollama`、`together_ai`、`perplexity`、`cohere` 等 | 需安装 `unionllm[litellm]` |

“支持”不代表能力相同。多模态、工具调用、推理参数、缓存和流式 usage 都取决于具体适配器。每个 provider 的环境变量、默认地址和特有参数见 [Provider 矩阵](https://github.com/EvalsOne/UnionLLM/blob/main/docs/providers.md)。

## 兼容性

| 依赖 | 范围 |
| --- | --- |
| Python | 3.10–3.13 |
| openai | 2.x、3.x |
| anthropic | 0.x、1.x、2.x（Azure Claude 在 2.x 下的采样参数尚未用真实请求验证，见[已知问题](https://github.com/EvalsOne/UnionLLM/blob/main/docs/migration-0.3.md#已知问题)） |
| litellm（可选） | `>=1.75,<1.95`，要求 openai 2.x |

以上组合在仓库测试中验证，测试使用模拟的上游响应。

## 文档

- [入门指南](https://github.com/EvalsOne/UnionLLM/blob/main/docs/getting-started.md)：安装、`unionchat` 与 `UnionLLM`、流式、异步、响应与错误。
- [Provider 矩阵](https://github.com/EvalsOne/UnionLLM/blob/main/docs/providers.md)：所有路由、环境变量、特有参数。
- [从 0.2.x 迁移到 0.3.0](https://github.com/EvalsOne/UnionLLM/blob/main/docs/migration-0.3.md)：LiteLLM 改为可选后的变化与检查清单。
- [架构说明](https://github.com/EvalsOne/UnionLLM/blob/main/docs/architecture.md)：请求路径、目录分工与统一层的边界。
- [贡献指南](https://github.com/EvalsOne/UnionLLM/blob/main/docs/contributing.md)：本地测试与提交要求。
- [变更日志](https://github.com/EvalsOne/UnionLLM/blob/main/CHANGELOG.md)
- [项目治理与改进计划](https://github.com/EvalsOne/UnionLLM/blob/main/docs/open-source-review.md)

## 边界

- 公开 API 只有 `unionchat` 与 `UnionLLM`，入口在 `unionllm/main.py`。
- 适配器不支持的参数可能被过滤、改写或忽略，不要假设任意 OpenAI 参数在所有 provider 上都生效。
- Agent 平台的 `model` 只是占位值，实际调用什么由平台上的应用配置决定。
- 本仓库只包含 Python 库，不包含服务端、命令行工具或文档站点。

## 贡献

欢迎提交 issue 和 PR。提交前请阅读[贡献指南](https://github.com/EvalsOne/UnionLLM/blob/main/docs/contributing.md)，并至少运行：

```bash
python -m pip install -e . -r requirements-dev.txt
python -m pytest --ignore=tests/my
```

许可证：[MIT](https://github.com/EvalsOne/UnionLLM/blob/main/LICENSE)。
