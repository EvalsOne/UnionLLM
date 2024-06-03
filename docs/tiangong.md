# 昆仑天工

## 通过环境变量设置调用参数

```
import os 
os.environ["TIANGONG_APP_KEY"] = "your-tiangong-app-key"
os.environ["TIANGONG_APP_SECRET"] = "your-tiangong-app-secret"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="tiangong",
    model="SkyChat-MegaVerse", 
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
    provider="tiangong",
    model="SkyChat-MegaVerse", 
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
    provider="tiangong",
    model="SkyChat-MegaVerse", 
    app_key="your-baichuan-api-key",
    app_secret="your-baichuan-api-secret",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型
支持天工的所有文本模型

参考文档：
- [API调用](https://model-platform.tiangong.cn/api-reference)
- [模型价格](https://model-platform.tiangong.cn/pricing)
```