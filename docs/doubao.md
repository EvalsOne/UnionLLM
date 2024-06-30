#  字节豆包
## 通过环境变量进行鉴权

```python
import os 
os.environ["ARK_API_KEY"] = "your-ark-api-key"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="doubao",
    model="ep-20240628173751-*****", 
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
    provider="doubao",
    model="ep-20240628173751-*****", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## 在接口中直接传入鉴权参数调用

```python
# model call
response = unionchat(
    provider="doubao",
    model="ep-20240628173751-*****", 
    api_key="your-ark-api-key",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型
支持字节方舟平台的所有模型，注意豆包需要在需要手动去[方舟后台](https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint )模型推理页面创建推理接入点，以接入点名称作为模型名称，例如：`ep-20240608051426-tkxvl`。


