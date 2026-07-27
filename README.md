# UnionLLM

一个以统一 `model` / `messages` 调用面连接多家 LLM、Agent 应用和 LiteLLM provider 的 Python 库。它的职责是路由请求、适配部分请求/响应格式，并尽可能返回统一的 OpenAI 风格 `ModelResponse`；它不是一个 OpenAI HTTP 代理服务。

本文档以仓库中的运行代码和测试为事实来源；模型是否仍可用、配额和厂商参数以相应厂商的官方文档为准。

## 快速开始

```bash
python -m pip install unionllm
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

流式调用返回可迭代的统一 chunk：

```python
for chunk in unionchat(
    provider="openai",
    model="gpt-5.6-luna",
    messages=[{"role": "user", "content": "用一句话介绍 UnionLLM"}],
    stream=True,
):
    delta = chunk.choices[0].delta
    if getattr(delta, "content", None):
        print(delta.content, end="")
```

详细的安装、调用约定、流式响应与异常处理见[入门与调用约定](docs/getting-started.md)。

## 当前支持范围

| 路由类别 | provider 值 |
| --- | --- |
| 原生/直接适配 | `zhipuai`、`moonshot`、`minimax`、`qwen`、`tiangong`、`baichuan`、`wenxin`、`xunfei`、`xunfei_http`、`dify`、`fastgpt`、`coze`、`lingyi`、`stepfun`、`doubao`、`deepseek`、`gemini`、`anthropic`、`openai` |
| Azure | `azure`：Claude 模型走 Azure Anthropic 原生实现；其他模型经 LiteLLM |
| LiteLLM 路由 | `bedrock`、`ollama`、`mistral`、`groq`、`openrouter` 等固定 allow-list；也可省略 `provider`，直接提供 LiteLLM 格式的 `model` |

不要把“支持”理解为所有 provider 具有相同能力。多模态、工具调用、推理参数、缓存、流式和上下文信息均取决于具体适配器。请先查看完整的[Provider 矩阵与配置](docs/providers.md)。

## 文档导航

- [入门与调用约定](docs/getting-started.md)：安装、`unionchat`、`UnionLLM`、流式、响应和异常。
- [Provider 矩阵与配置](docs/providers.md)：所有当前路由、环境变量、特殊参数和迁移说明。
- [架构说明](docs/architecture.md)：真实请求路径、边界与目录责任。
- [贡献指南](docs/contributing.md)：本地校验、provider 改动的最小交付面。
- [项目治理与开源改进计划](docs/open-source-review.md)：本次整理的发现、优先级和阶段计划。

历史 provider 页面仍保留为跳转页，避免外部链接失效；其中不再维护重复示例。

## 项目状态与边界

- 公开 API 只有 `unionchat` 与 `UnionLLM`；请求入口位于 `unionllm/main.py`。
- `unionchat` 需要 `model` 和 OpenAI 风格的 `messages` 列表，其他关键字参数传给选中的 provider。
- 若适配器不支持某个参数，可能过滤、映射或忽略它；不要假定任意 OpenAI 参数在所有 provider 上生效。
- Agent 平台的 `model` 在目前实现中常是占位值；真正调用目标由应用/Bot 配置决定。
- 本仓库不包含服务端、CLI、环境变量模板、发布流水线或文档站点。它是一个 Python library 仓库。

## 贡献

欢迎 issue 和 PR。提交前请阅读[贡献指南](docs/contributing.md)，并至少运行：

```bash
python -m pytest --ignore=tests/my
```

许可证为 [MIT](LICENSE)。
