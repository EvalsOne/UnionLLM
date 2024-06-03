# FASTGPT

## 通过环境变量设置鉴权参数

```
import os 
os.environ["FASTGPT_API_KEY"] = "your-fastgpt-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="fastgpt",
    model="fastgpt", 
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
    provider="fastgpt",
    model="fastgpt", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## 接口直接传入鉴权参数调用

```python
# model call
response = unionchat(
    provider="fastgpt",
    model="fastgpt", 
    api_key="your-fastgpt-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 注意事项
- model参数没有实际作用，可以传入任何字符串
- FastGPT的API密钥可以在每个Bot的发布应用页面找到