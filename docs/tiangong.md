# 昆仑天工

## 设置API KEYS

```
import os 
os.environ["TIANGONG_APP_KEY"] = "your-tiangong-app-key"
os.environ["TIANGONG_APP_SECRET"] = "your-tiangong-app-secret"
```

## 非流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="tiangong",
    model="SkyChat-MegaVerse", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
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

print(response)
```

## 直接传入API_Key调用

```python
# model call
response = completion(
    provider="tiangong",
    model="SkyChat-MegaVerse", 
    app_key="your-baichuan-api-key",
    app_secret="your-baichuan-api-secret",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

for chunk in response:
    print(chunk)
```