# 通义千问(dashscope)

## 通过环境变量设置调用参数

```python
import os 
os.environ["DASHSCOPE_API_KEY"] = "your-dashscope-api-key"
```

## 非流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="qwen",
    model="qwen-turbo", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="qwen",
    model="qwen-turbo", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

print(response)
```

## 直接传入API_Key调用

```python
# model call
response = completion(
    apk_key="your-dashscope-api-key",
    provider="qwen",
    model="qwen-turbo", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

for chunk in response:
    print(chunk)
```