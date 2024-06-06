# 零一万物

## 通过环境变量设置调用参数

```python
import os 
os.environ["LINGYI_API_KEY"] = "your-lingyi-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="lingyi",
    model="yi-large", 
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
    provider="lingyi",
    model="yi-large", 
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
    provider="lingyi",
    model="yi-large",
    api_key="your-lingyi-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型
支持零一万物的所有模型

参考文档：
- [零一万物API文档](https://platform.lingyiwanwu.com/docs)
```