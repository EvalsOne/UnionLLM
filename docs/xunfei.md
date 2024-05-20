# 讯飞星火

## 通过环境变量设置调用鉴权参数

```python
import os 
os.environ["XUNFEI_APP_ID"] = "your-app-id"
os.environ["XUNFEI_API_KEY"] = "your-api-key"
os.environ["XUNFEI_API_SECRET"] = "your-api-secret"
```

## 非流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="xunfei",
    model="generalv3.5", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 流式调用

暂不支持

## 直接传入鉴权参数调用

```python
# model call
response = completion(
    provider="xunfei",
    model="generalv3.5", 
    app_id="your-app-id",
    api_key="your-api-key",
    api_secret="your-api-secret",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

for chunk in response:
    print(chunk)
```