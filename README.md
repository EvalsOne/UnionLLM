# UnionLLM

UnionLLM是一个通过与OpenAI兼容的统一方式调用各种国内外各种大语言模型和Agent编排工具的轻量级开源Python工具包。

我们开发它的起因是因为在实际应用中，我们经常需要使用多个大语言模型，但是每个大语言模型的接口和使用方式都不一样，这给我们的工作带来了很大的困扰。UnionLLM的目标是通过统一且容易扩展的方式连接各种大语言模型，使得我们可以更方便地使用多个大语言模型。

为了不重新造轮子，UnionLLM依赖LiteLLM进行大多数海外模型的调用，而专注于实现国内模型的调用。

UnionLLM目前支持的国内大语言模型包括：
- 智谱AI
- 月之暗面 Moonshot
- 百度文心一言
- 阿里巴巴通义千问
- MiniMax
- 讯飞星火
- 百川智能
- 昆仑天工
- 零一万物
- 阶跃星辰
- 字节豆包
- 深度求索 DeepSeek

UnionLLM目前支持的Agent编排工具包括：
- Coze
- FastGPT
- Dify

UnionLLM支持通过LiteLLM调用100+各种大模型，具体包括以下几类：
- OpenAI, Anthropic, Mistral, Cohere等海外大语言模型开发商
- Azure, Google, Amazon Bedrock, Hugging Face等大模型云服务商
- Ollama, vLLM等开源模型本地部署工具

LiteLLM支持模型厂商的详细列表见[LiteLLM的文档](https://docs.litellm.ai/docs/providers)。


UnionLLM的安装方式：
```bash
pip install unionllm
```

通过UnionLLM调用任何中文大语言模型只需两行代码，以智谱AI的glm-4模型为例：
```python
from unionllm import unionchat
unionchat(provider="zhipuai", model="glm-4", messages=[{"content": "你的开发者是谁？", "role": "user"}], stream=False)
```

通过UnionLLM调用LiteLLM支持的任何其他模型的方式示例如下：
```python
from unionllm import unionchat
unionchat(provider="openai", model="gpt-4o", messages=[{"content": "你的开发者是谁？", "role": "user"}], stream=True)
```

通过UnionLLM调用OpenAI proxy示例如下(借助LiteLLM)：
```python
from unionllm import unionchat
unionchat(custom_llm_provider="openai", model="gpt-4o", api_base="https://your_custom_proxy_domain/v1" messages=[{"content": "你的开发者是谁？", "role": "user"}], stream=True)
```

以下是stream=False的调用方式的返回格式示例：
```python
ModelResponse(id='8635254124951169203', choices=[Choices(finish_reason='stop', index=0, message=Message(content='我是人工智能助手。', role='assistant'))], created=1715570856, model=model, object='chat.completion', system_fingerprint=None, usage=Usage(prompt_tokens=9, completion_tokens=27, total_tokens=36))
```

以下是stream=True的调用方式的chunk增量格式示例：
```python
......
ModelResponse(id='8635254124951169203', choices=[Choices(finish_reason='stop', index=0, message=Message(content='我是人工智能助手。', role='assistant'))], created=1715570856, model=model, object='chat.completion', system_fingerprint=None, usage=Usage(prompt_tokens=9, completion_tokens=27, total_tokens=36))
......
```

UnionLLM的返回结果格式与LiteLLM一致且与OpenAI一致，并在此基础上扩展了Context信息的返回，以实现发起知识库检索的RAG调用时返回相关背景知识。(由于Coze, FastGPT和Dify的接口中返回背景信息的方式和格式经常改变，目前版本可能无法成功获取Context信息)

以下是包含知识库检索背景信息的返回结果示例 (非流式调用)：
```python
ModelResponse(id='8635254124951169203', choices=[Choices(finish_reason='stop', index=0, message=Message(content='我是人工智能助手。', role='assistant'))], created=1715570856, model=model, object='chat.completion', system_fingerprint=None, usage=Usage(prompt_tokens=9, completion_tokens=27, total_tokens=36),context=[Context(id=1, content='retrieved context information 1', score=0.96240234375), Context(id=2, content='retrieved context information 2', score=0.7978515625), Context(id=3, content='retrieved context information 3', score=0.71142578125)])
```

以下是每一种大语言模型的调用方式的详细文档：
- [智谱AI](docs/zhipuai.md)
- [月之暗面 Moonshot](docs/moonshot.md)
- [百度文心一言](docs/baidu.md)
- [阿里通义千问](docs/qwen.md)
- [MiniMax](docs/minimax.md)
- [讯飞星火](docs/xunfei.md)
- [百川智能](docs/baichuan.md)
- [昆仑天工](docs/tiangong.md)
- [零一万物](docs/lingyi.md)
- [阶跃星辰](docs/stepfun.md)
- [Coze](docs/coze.md)
- [FastGPT](docs/fastgpt.md)
- [Dify](docs/dify.md)
- [字节豆包](docs/doubao.md)
- [深度求索 DeepSeek](docs/deepseek.md)
- [通过LiteLLM调用其他大模型](docs/litellm.md)

UnionLLM目前提供的功能包括：
- 支持多种国内大语言模型
- 支持多种Agent编排工具，如Coze、FastGPT、Dify，并会返回知识库检索的相关背景知识（如包含）
- 支持通过LiteLLM调用100+各种大模型
- 支持非流式调用和流式调用
- 支持通过环境变量设置鉴权参数，以及通过直接传入鉴权参数调用

UnionLLM目前存在的功能局限包括：
- 只支持文本输入和生成，不支持视觉、音频等其他模态
- 只支持对话模型
- 不支持Embedding模型
- 暂不支持工具使用/函数调用

我们计划在未来的版本中，在保持轻量级的同时，逐步丰富UnionLLM的功能。也希望社区的朋友们能够一起参与到UnionLLM的开发中来。

在此非常感谢[LiteLLM](https://github.com/BerriAI/litellm)项目的开发者们，UnionLLM的开发离不开你们的工作，我们的日常工作也从中获益匪浅。