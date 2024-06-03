# 智谱AI

### 通过环境变量设置API KEY

```
import os 
os.environ["ZHIPU_API_KEY"] = "your-zhipu-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="zhipuai",
    model="glm-4", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=False
)

print(response)
```

### 流式调用

```python
from unionllm import unionchat

## set ENV variables
os.environ["COHERE_API_KEY"] = "cohere key"

# model call
response = unionchat(
    provider="zhipuai",
    model="glm-4", 
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
    provider="zhipuai",
    model="glm-4", 
    api_key="your-zhipu-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```


## 支持模型
支持天工的所有文本模型

参考文档：
- [API调用](https://open.bigmodel.cn/dev/api)
- [模型价格](https://open.bigmodel.cn/pricing)
```