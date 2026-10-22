# 入门与调用约定

## 安装

```bash
python -m pip install unionllm
```

需要 LiteLLM 路由（`bedrock`、`vertex_ai`、`mistral`、`ollama` 等）时，安装可选依赖：

```bash
python -m pip install "unionllm[litellm]"
```

- Python 要求 `>=3.10,<3.14`。仓库测试在 3.10、3.11、3.12、3.13 下运行。
- 不锁定 openai 与 anthropic 的大版本，已验证的组合见[迁移指南](migration-0.3.md#sdk-版本)。
- LiteLLM 要求 openai 2.x，见[迁移指南](migration-0.3.md#仍需-litellm-的路由)。

## 最小调用

```python
from unionllm import unionchat

response = unionchat(
    provider="deepseek",
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.2,
)
print(response.choices[0].message.content)
```

`unionchat(model, messages, **kwargs)` 先用 `kwargs` 创建 `UnionLLM(**kwargs)`，再调用 `completion(model, messages, **kwargs)`。`provider`、认证信息和 provider 专属参数都以关键字参数传入；`model` 与 `messages` 必填。

消息采用常见的 Chat Completions 结构：

```python
messages = [
    {"role": "system", "content": "回答应简洁。"},
    {"role": "user", "content": "解释这个库的作用。"},
]
```

基础输入校验要求 `system` 消息位于首位。`content` 为对象列表时（如 `image_url`、`video_url`），各 provider 会检查自己能否处理：可能原样保留、转成文本，也可能报错。多模态支持取决于 provider 与模型，不是统一承诺。

## 使用客户端实例

需要复用配置时，直接使用 `UnionLLM`：

```python
from unionllm import UnionLLM

client = UnionLLM(provider="openai", api_key="...")
response = client.completion(
    model="gpt-5.4-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

调用时的参数会与构造时的参数一起交给适配器，但各适配器的覆盖规则不完全一致。`openai`、`azure`（非 Claude）与 OpenAI 兼容服务商允许在调用时用 `api_key`、`api_base` / `base_url` 覆盖构造时的配置。

部分路由在构造时就要看 `model`，例如 `azure` 按模型名选择 Claude 或 Azure OpenAI 分支。这种情况下请把 `model` 也传给 `UnionLLM(...)`，见 [Provider 矩阵](providers.md#azure)。

## 流式响应

```python
stream = unionchat(
    provider="zhipuai",
    model="glm-4",
    messages=[{"role": "user", "content": "写一行诗"}],
    stream=True,
)

for chunk in stream:
    if not chunk.choices:
        continue
    text = getattr(chunk.choices[0].delta, "content", None)
    if text:
        print(text, end="")
```

适配器尽量返回 `ModelResponse(object="chat.completion.chunk")`。文本、工具调用增量、推理内容和最终 usage 可能出现在不同的块里，最后一个只带 usage 的块可能没有 `choices`。读取时请用 `getattr` 并检查 `choices`，不要假设每个块都有 `content` 或 `usage`。

## 异步调用

`UnionLLM.acompletion()` 把同步的 `completion()` 放进线程池执行，返回值与 `completion()` 相同；流式时得到的仍是同步迭代器。它不是原生异步实现，`unionchat` 也没有异步版本。

```python
import asyncio
from unionllm import UnionLLM

async def main():
    client = UnionLLM(provider="deepseek")
    response = await client.acompletion(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "你好"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## 响应

非流式结果通常包含 `id`、`choices`、`created`、`model`、`object` 与 `usage`。各家响应会被转换成 OpenAI 风格的对象，但不保证每家都返回完整的 usage，也不保证保留上游的所有字段。

## 错误处理

| 异常 | 何时抛出 |
| --- | --- |
| `unionllm.exceptions.ProviderError` | `provider` 不受支持，或客户端未初始化。基类是 `UnionLLMError`。 |
| `unionllm.providers.litellm.LiteLLMUnavailableError` | 选中了 LiteLLM 路由但没有安装 LiteLLM。是 `ProviderError` 的子类，`status_code` 为 501。 |
| 各 provider 自己的 `*Error` | 认证缺失、参数不合法、上游返回错误等。例如 `OpenAINativeError`、`AnthropicProviderError`、`AzureProviderError`、`DeepSeekError`。 |

各 provider 的错误类没有共同的基类，但通常带有 `status_code` 与 `message` 属性；缺少 API key 时一般是 `status_code=422`。建议这样处理：

```python
try:
    response = unionchat(provider="deepseek", model="deepseek-chat", messages=messages)
except Exception as exc:
    status = getattr(exc, "status_code", None)
    message = getattr(exc, "message", str(exc))
    # 记录 status 与 message，不要记录 API key 或完整的提示词
    raise
```

## 认证与配置

优先使用环境变量，不要把密钥写进源码：

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

也可以在调用时传 `api_key`。每个 provider 的环境变量、默认地址和特有参数见 [Provider 矩阵](providers.md)。

## LiteLLM 路由（可选）

安装 `unionllm[litellm]` 后，`UnionLLM.check_litellm_providers()` 列表中的 provider 会交给 LiteLLM，模型名按规则加上 provider 前缀：

```python
unionchat(
    provider="bedrock",
    model="anthropic.claude-instant-v1",
    messages=[{"role": "user", "content": "Hello"}],
)
```

也可以不传 `provider`，直接使用 LiteLLM 格式的模型名：

```python
unionchat(
    model="ollama/llama3.1",
    messages=[{"role": "user", "content": "Hello"}],
)
```

这部分行为取决于安装时解析到的 LiteLLM 版本。认证、模型名和高级参数以 LiteLLM 官方文档为准。完整列表见 [Provider 矩阵](providers.md#litellm-路由可选)。
