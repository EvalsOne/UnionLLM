# 月之暗面 Moonshot

## 通过环境变量设置调用参数

```python
import os 
os.environ["MOONSHOT_API_KEY"] = "your-moonshot-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="moonshot",
    model="moonshot-v1-32k", 
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
    provider="moonshot",
    model="moonshot-v1-32k", 
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
    provider="moonshot",
    model="moonshot-v1-32k", 
    api_key="your-moonsho-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型
支持Moonshot的所有模型

参考文档：
- [API调用](https://platform.moonshot.cn/docs/api-reference)
- [模型价格](https://platform.moonshot.cn/docs/pricing)
```