# Coze (海外版)

## 通过环境变量设置鉴权参数

```
import os 
os.environ["COZE_API_KEY"] = "your-coze-api-key"
os.environ["COZE_BOT_ID"] = "your-coze-bot-id"
```

## 非流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="coze",
    model="coze", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

print(response)
```

## 流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="tiangong",
    model="SkyChat-MegaVerse", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

for chunk in response:
    print(chunk)
```

## 直接传入鉴权参数调用

```python
# model call
response = completion(
    provider="coze",
    model="coze", 
    app_key="your-coze-api-key",
    app_secret="your-coze-bot-id",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```