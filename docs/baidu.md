# 百度文心一言

## 通过环境变量设置调用参数

```python
import os 
os.environ["ERNIE_CLIENT_ID"] = "your-client-id"
os.environ["ERNIE_CLIENT_SECRET"] = "your-client-secret"
```

## 非流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="wenxin",
    model="ERNIE-3.5-8K", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="wenxin",
    model="ERNIE-3.5-8K", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

print(response)
```

## 直接传入API_Key调用

```python
# model call
response = completion(
    provider="wenxin",
    model="ERNIE-3.5-8K", 
    client_id="your-client-id",
    client_secret="your-client-secret",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

for chunk in response:
    print(chunk)
```