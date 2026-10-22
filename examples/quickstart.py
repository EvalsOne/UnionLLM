"""最小调用：发一条消息，打印回复和 token 用量。

运行前设置对应 provider 的 API key，例如：

    export DEEPSEEK_API_KEY="..."
    python examples/quickstart.py

换成其他 provider 时设置 UNIONLLM_PROVIDER 和 UNIONLLM_MODEL，
需要自定义地址时设置 UNIONLLM_API_BASE。
"""
import os

from unionllm import unionchat

provider = os.getenv("UNIONLLM_PROVIDER", "deepseek")
model = os.getenv("UNIONLLM_MODEL", "deepseek-chat")
extra = {"api_base": os.environ["UNIONLLM_API_BASE"]} if os.getenv("UNIONLLM_API_BASE") else {}

response = unionchat(
    provider=provider,
    model=model,
    messages=[
        {"role": "system", "content": "回答应简洁。"},
        {"role": "user", "content": "用一句话介绍你自己。"},
    ],
    **extra,
)

print(response.choices[0].message.content)
usage = getattr(response, "usage", None)
if usage:
    print(f"\n[usage] prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
