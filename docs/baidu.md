# 百度文心一言

## 通过环境变量进行鉴权

```python
import os 
os.environ["ERNIE_CLIENT_ID"] = "your-client-id"
os.environ["ERNIE_CLIENT_SECRET"] = "your-client-secret"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="wenxin",
    model="completions_pro", 
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
    provider="wenxin",
    model="completions_pro", 
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
    provider="wenxin",
    model="completions_pro", 
    client_id="your-client-id",
    client_secret="your-client-secret",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型
支持百度文心平台的所有模型，注意传入模型的名称必须是请求地址结尾的部分。

例如：
https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro

那么，需要传入的模型名称是`completions_pro`

## 参考文档：
- [API文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t)
- [模型价格](https://platform.baichuan-ai.com/price)
```