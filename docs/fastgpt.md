# FASTGPT

## 通过环境变量设置鉴权参数

```
import os 
os.environ["FASTGPT_API_KEY"] = "your-fastgpt-api-key"
```

## 非流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="fastgpt",
    model="fastgpt", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

print(response)
```

## 流式调用

```python
from unionllm import unionchat

# model call
response = completion(
    provider="fastgpt",
    model="fastgpt", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)

for chunk in response:
    print(chunk)
```

## 直接传入鉴权参数调用

```python
# model call
response = completion(
    provider="fastgpt",
    model="fastgpt", 
    app_key="your-fastgpt-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```