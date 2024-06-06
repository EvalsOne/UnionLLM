# 百川AI

## 通过环境变量进行鉴权

```
import os 
os.environ["BAICHUAN_API_KEY"] = "your-baichuan-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="baichuan",
    model="Baichuan2-Turbo", 
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
    provider="baichuan",
    model="Baichuan2-Turbo", 
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
    provider="baichuan",
    model="Baichuan2-Turbo", 
    api_key="your-baichuan-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}],
)
```

## 支持模型
支持百川AI的所有模型

参考文档：
- [百川AI文档](https://platform.baichuan-ai.com/docs/api)
- [模型价格](https://platform.baichuan-ai.com/price)
```