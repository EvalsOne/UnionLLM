# 月之暗面 Moonshot

## 设置API KEYS

```python
import os 
os.environ["MOONSHOT_API_KEY"] = "your-moonshot-api-key"
```

## 非流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="moonshot",
    model="moonshot-v1-32k", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 流式调用

```python
from unionllm import unionchat

## set ENV variables
os.environ["COHERE_API_KEY"] = "cohere key"

# model call
response = completion(
    provider="moonshot",
    model="moonshot-v1-32k", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

print(response)
```

## 直接传入API_Key调用

```python
# model call
response = completion(
    provider="moonshot",
    model="moonshot-v1-32k", 
    api_key="your-moonsho-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

for chunk in response:
    print(chunk)
```