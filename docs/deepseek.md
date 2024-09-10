# 深度求索 DeepSeek

## 通过环境变量设置调用参数

```python
import os 
os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="deepseek",
    model="deepseek-chat",
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
    provider="deepseek",  
    model="deepseek-chat", 
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
    provider="deepseek",
    model="deepseek-chat",
    api_key="your-deepseek-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 调用 8K 最大输出长度的Beta版本

```python
# model call
response = unionchat(
    provider="deepseek",
    model="deepseek-chat",
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=False,
    max_tokens=8192,
    beta=True
)
```
## 支持模型
支持深度求索的所有模型

参考文档：
- [深度求索API文档](https://platform.deepseek.com/api-docs)
```