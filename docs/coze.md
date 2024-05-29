# Coze (海外版)

## 通过环境变量设置鉴权参数

```
import os 
os.environ["COZE_API_KEY"] = "your-coze-api-key"
os.environ["COZE_BOT_ID"] = "your-coze-bot-id"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="coze",
    model="coze", 
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

## 直接传入鉴权参数调用

```python
# model call
response = unionchat(
    provider="coze",
    model="coze", 
    api_key="your-coze-api-key",
    bot_id="your-coze-bot-id",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 注意事项
- model参数没有实际作用，可以传入任意字符串
- Coze API的参考文档：https://www.coze.com/docs/developer_guides/coze_api_overview?_lang=zh