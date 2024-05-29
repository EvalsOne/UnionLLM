# Dify

## 通过环境变量设置鉴权参数

```
import os 
os.environ["DIFY_API_KEY"] = "your-dify-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="dify",
    model="dify", 
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
    provider="dify",
    model="dify", 
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
    provider="dify",
    model="dify", 
    api_key="your-dify-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 注意事项
- model参数没有实际作用，可以传入任意字符串
- Dify的API key可以在每个Bot的设置页面找到