"""调用自定义的 OpenAI 兼容端点，例如本地的 vLLM、Ollama 或公司内部网关。

    export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
    export OPENAI_COMPATIBLE_MODEL="my-model"
    export OPENAI_COMPATIBLE_API_KEY="..."   # 服务不鉴权时可以不设
    python examples/custom_endpoint.py

不需要安装 LiteLLM。地址原样使用，不会自动补 /v1；也不会读取 OPENAI_API_KEY。
"""
import os

from unionllm import UnionLLM

base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1")
model = os.getenv("OPENAI_COMPATIBLE_MODEL", "my-model")
auth = {"api_key": os.environ["OPENAI_COMPATIBLE_API_KEY"]} if os.getenv("OPENAI_COMPATIBLE_API_KEY") else {}

client = UnionLLM(custom_llm_provider="openai", api_base=base_url, **auth)
response = client.completion(
    model=model,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
