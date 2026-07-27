# 入门与调用约定

## 安装

项目使用 Poetry 构建；发布包可通过 pip 安装：

```bash
python -m pip install unionllm
```

从源码运行测试时，使用仓库依赖：

```bash
python -m pip install -r requirements.txt
python -m pytest --ignore=tests/my
```

`pyproject.toml` 的 Python 约束目前是 `*`，这不等于项目已经在所有 Python 版本上验证过。贡献者应在自己目标运行时执行测试。当前工作区中的 `tests/my/` 是被忽略的本地/真实服务测试，可能依赖外部应用配置；稳定的受版本控制测试应使用上面的 `--ignore=tests/my` 命令。

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

`unionchat(model, messages, **kwargs)` 会创建 `UnionLLM(**kwargs)`，再调用 `completion`。因此 `provider`、认证信息和 provider 专属参数都作为关键字参数传入；`model` 和 `messages` 是必填位置/关键字参数。

消息遵循常见的 chat-completions 结构：

```python
messages = [
    {"role": "system", "content": "回答应简洁。"},
    {"role": "user", "content": "解释这个库的作用。"},
]
```

基础输入校验要求 `system` 消息位于首位。带对象的 `content`（如 `image_url`）会被 provider 能力检查；结果可能是保留、转换成文本或报错，不是全 provider 统一的多模态承诺。

## 使用客户端实例

需要复用 provider 配置时可以直接使用 `UnionLLM`：

```python
from unionllm import UnionLLM

client = UnionLLM(provider="openai", api_key="...")
response = client.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

调用级别的参数会与初始化参数一起交给适配器。不同适配器对此的覆盖规则并不完全一致；对于 `openai`，调用级 `api_key`、`api_base` / `base_url` 可覆盖初始化配置。

## 流式响应

```python
stream = unionchat(
    provider="zhipuai",
    model="glm-4",
    messages=[{"role": "user", "content": "写一行诗"}],
    stream=True,
)

for chunk in stream:
    choice = chunk.choices[0]
    text = getattr(choice.delta, "content", None)
    if text:
        print(text, end="")
```

适配器尽量返回 `ModelResponse(object="chat.completion.chunk")`，但 chunk 里的字段会随上游事件而不同：文本、工具调用增量、推理内容和最终 usage 都可能出现在不同 chunk。消费端必须使用 `getattr`，不要假定每个 chunk 都有 `content` 或 `usage`。

## 响应与错误

非流式结果通常包含 `id`、`choices`、`created`、`model`、`object` 和 `usage`。统一层会把若干厂商响应转换为 OpenAI 风格对象，但并不保证所有厂商都返回完整 usage，也不保证所有上游字段被保留。

各 provider 抛出自己的 `*Error`，通常具有 `status_code` 与 `message` 属性。应用代码应捕获 `Exception` 后按该属性做可观测记录，且不要记录 API key 或完整敏感提示词。

## 认证与配置

优先使用环境变量，避免将密钥写入源码：

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

也可以将 `api_key` 传入调用。每个 provider 的环境变量、默认端点和必要的额外参数都列在[Provider 矩阵](providers.md)。

## LiteLLM 路由

对于 `UnionLLM.check_litellm_providers()` 中的 provider，库会把 provider 前缀按当前规则加入模型名，交给 LiteLLM。例如：

```python
unionchat(
    provider="bedrock",
    model="anthropic.claude-instant-v1",
    messages=[{"role": "user", "content": "Hello"}],
)
```

也可省略 `provider` 并直接采用 LiteLLM 模型格式：

```python
unionchat(
    model="ollama/llama3.1",
    messages=[{"role": "user", "content": "Hello"}],
)
```

此行为依赖安装时解析到的 LiteLLM 版本。对认证、模型名和高级参数，请以 LiteLLM 官方文档为准。
