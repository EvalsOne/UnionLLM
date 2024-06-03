# 讯飞星火

## 通过环境变量设置调用鉴权参数

```python
import os 
os.environ["XUNFEI_APP_ID"] = "your-app-id"
os.environ["XUNFEI_API_KEY"] = "your-api-key"
os.environ["XUNFEI_API_SECRET"] = "your-api-secret"
```

### 非流式调用

```python
from unionllm import unionchat

# model call
response = unionchat(
    provider="xunfei",
    model="generalv3.5", 
    messages = [{ "content": "Hello, how are you?","role": "user"}],
    stream=False
)

print(response)
```

### 流式调用

暂不支持

## 直接传入鉴权参数调用

```python
# model call
response = unionchat(
    provider="xunfei",
    model="generalv3.5", 
    app_id="your-app-id",
    api_key="your-api-key",
    api_secret="your-api-secret",
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

## 支持模型

|模型名称|传入model参数名称|
|---|---|
|Spark3.5 Max|generalv3.5|
|Spark Pro|generalv3|
|Spark V2.0|generalv2|
|Spark Lite|general|

## 参考文档：
- [API调用](https://www.xfyun.cn/doc/spark/Web.html)
- [模型价格](https://xinghuo.xfyun.cn/sparkapi)
```