"""流式调用：边生成边打印。

    export DEEPSEEK_API_KEY="..."
    python examples/streaming.py

与 quickstart.py 一样，可用 UNIONLLM_PROVIDER、UNIONLLM_MODEL、UNIONLLM_API_BASE
换成其他 provider 或地址。
"""
import os

from unionllm import unionchat

provider = os.getenv("UNIONLLM_PROVIDER", "deepseek")
model = os.getenv("UNIONLLM_MODEL", "deepseek-chat")
extra = {"api_base": os.environ["UNIONLLM_API_BASE"]} if os.getenv("UNIONLLM_API_BASE") else {}

stream = unionchat(
    provider=provider,
    model=model,
    messages=[{"role": "user", "content": "写一首四行的小诗。"}],
    stream=True,
    **extra,
)

usage = None
for chunk in stream:
    # 文本、工具调用增量和最终 usage 可能出现在不同的块里，
    # 最后一个只带 usage 的块可能没有 choices。
    if chunk.choices:
        text = getattr(chunk.choices[0].delta, "content", None)
        if text:
            print(text, end="", flush=True)
    usage = getattr(chunk, "usage", None) or usage

print()
if usage:
    print(f"[usage] prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
