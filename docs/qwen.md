# 通义千问(dashscope)

## 通过环境变量设置调用参数

```python
import os 
os.environ["DASHSCOPE_API_KEY"] = "your-dashscope-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="qwen",
    model="qwen-turbo", 
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
    provider="qwen",
    model="qwen-turbo", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## 直接传入API_Key调用

```python
# model call
response = unionchat(
    apk_key="your-dashscope-api-key",
    provider="qwen",
    model="qwen-turbo", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型
支持通义千问的所有模型

参考文档：
- [API调用](https://help.aliyun.com/zh/dashscope/developer-reference/api-details)
- [模型价格](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-thousand-questions-metering-and-billing)
```