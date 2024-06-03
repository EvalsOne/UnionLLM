# 通过LiteLLM调用其他大模型

如果传入的provider参数不在UnionLLM支持的厂商范围内（或者没有传入provider厂商），UnionLLM会将请求转发给LiteLLM，因此可以通过UnionLLM调用LiteLLM支持的任何大模型。

不过，UnionLLM推荐provider和model参数分别传入，而LiteLLM只有model参数，不支持provider参数。因此，推荐将LiteLLM支持的model名称转化为provider和model参数传入UnionLLM。

例如：

调用AWS bedrock上的claude2.1模型时，直接调用LiteLLM的方式是：
```python
import litellm
response = litellm.completion(
    model="bedrock/anthropic.claude-instant-v1",
    model_id="provisioned-model-arn",
    messages=[{"content": "Hello, how are you?", "role": "user"}],
)
```

通过UnionLLM调用AWS bedrock上的claude2.1模型时，推荐将model转化为provider和model参数传入：
```python
from unionllm import unionchat
response = unionchat(
    provider="bedrock",
    model="anthropic.claude-instant-v1",
    messages=[{"content": "Hello, how are you?", "role": "user"}],
)
```

之所以这样推荐是因为我们认为将provider与model分开更加清晰明了。不过，即使你使用LiteLLM的传参方式，不传入provider参数，UnionLLM也可以兼容，并不会导致出错。

以下以Mistral为例，展示如何通过UnionLLM调用LiteLLM支持的大模型。

## 通过环境变量设置鉴权参数

```python
import os 
os.environ["MISTRAL_API_KEY"] = "your-mistral-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="mistral",
    model="mistral-tiny", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=False
)

print(response)
```

### 流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="mistral",
    model="mistral-tiny", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## 直接传入鉴权参数调用

```python
# model call
response = unionchat(
    provider="mistral",
    model="mistral-tiny", 
    app_key="your-mistral-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 兼容不传入provider参数的调用方式

```python
from unionllm import unionchat

# model call
response = unionchat(
    model="mistral/mistral-tiny", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

```

如果你选择传入provider参数，以下是LiteLLM支持厂商的provider列表：
| Provider                                                                            | Code        |
| ----------------------------------------------------------------------------------- | ----------- |
| [openai](https://docs.litellm.ai/docs/providers/openai)                             | openai      |
| [azure](https://docs.litellm.ai/docs/providers/azure)                               | azure       |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)             | sagemaker   |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)                     | bedrock     |
| [google - vertex_ai [Gemini]](https://docs.litellm.ai/docs/providers/vertex)        | vertex_ai   |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)                        | palm        |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)          | gemini      |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)                    | mistral     |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | cloudflare  |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)                             | cohere      |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)                       | anthropic   |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)                   | huggingface |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)                       | replicate   |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)                    | together_ai |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)                     | openrouter  |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)                                 | ai21        |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)                           | baseten     |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)                                 | vllm        |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)                       | nlp_cloud   |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                   | aleph_alpha |
| [petals](https://docs.litellm.ai/docs/providers/petals)                             | petals      |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)                             | ollama      |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)                       | deepinfra   |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)                  | perplexity  |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)                              | groq        |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek)                         | deepseek    |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)                         | anyscale    |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx)                  | watsonx     |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)                          | voyage      |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) | xinference  |

你也可以对照[LiteLLM文档](https://docs.litellm.ai/docs/providers)查看支持的provider列表。