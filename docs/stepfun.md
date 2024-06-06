# 阶跃星辰

## 通过环境变量设置调用参数

```python
import os 
os.environ["STEPFUN_API_KEY"] = "your-stepfun-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="stepfun",
    model="step-1-8k", 
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
    provider="stepfun",
    model="step-1-8k",
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
    provider="stepfun",
    model="step-1-8k",
    api_key="your-stepfun-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型
支持的所有模型

参考文档：
- [阶跃星辰API文档](https://platform.stepfun.com/docs/overview/concept)
```