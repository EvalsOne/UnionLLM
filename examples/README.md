# 示例

每个示例只依赖环境变量，不需要改代码就能运行。先安装 UnionLLM：

```bash
python -m pip install unionllm
```

| 文件 | 内容 | 需要的环境变量 |
| --- | --- | --- |
| [quickstart.py](quickstart.py) | 非流式调用，打印回复与 token 用量 | 默认 `DEEPSEEK_API_KEY`；可用 `UNIONLLM_PROVIDER`、`UNIONLLM_MODEL`、`UNIONLLM_API_BASE` 换成其他 provider |
| [streaming.py](streaming.py) | 流式调用，边生成边打印 | 同上 |
| [custom_endpoint.py](custom_endpoint.py) | 调用自定义的 OpenAI 兼容端点（vLLM、Ollama、内部网关等） | `OPENAI_COMPATIBLE_BASE_URL`、`OPENAI_COMPATIBLE_MODEL`，可选 `OPENAI_COMPATIBLE_API_KEY` |

换 provider 的例子：

```bash
export UNIONLLM_PROVIDER=zhipuai UNIONLLM_MODEL=glm-4 ZHIPU_API_KEY="..."
python examples/quickstart.py
```

各 provider 的环境变量见 [Provider 矩阵](../docs/providers.md)。
