from .base_provider import BaseProvider
from urllib.parse import urlparse
import websocket
import time
import json
import ssl
import threading
import logging
import hmac
import hashlib
import base64
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from cnlitellm.utils import ModelResponse, Message, Choices, Usage

class XunfeiOpenAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

class XunfeiWebSocketClient:
    def __init__(self, app_id, api_key, api_secret, model, **kwargs):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.model = model
        self.ws = None
        self.answer = ""
        self.complete_event = threading.Event()
        self.usage = None
        self.error = None
        self.spark_url = self.get_spark_url(model)
        self.domain = self.model

    # 获取模型对应的spark_url
    def get_spark_url(self, model):
        if model == "generalv3.5":
            return "wss://spark-api.xf-yun.com/v3.5/chat"
        elif model == "generalv3":
            return "wss://spark-api.xf-yun.com/v3.1/chat"
        elif model == "generalv2":
            return "wss://spark-api.xf-yun.com/v2.1/chat"
        elif model == "general":
            return "wss://spark-api.xf-yun.com/v1.1/chat"
        else:
            raise ValueError(f"Unsupported model: {model}")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if data['header']['code'] == 0:
                choices = data["payload"]["choices"]
                for choice in choices["text"]:
                    self.answer += choice["content"]
                
                # 当会话结束（status=2）时处理usage信息
                if data['header']['status'] == 2:  # 会话结束
                    if "usage" in data["payload"]:
                        usage = data["payload"]["usage"]
                        # 假设usage信息中包含text字段，其中有prompt_tokens和completion_tokens
                        if "text" in usage:
                            text_usage = usage["text"]
                            # 更新prompt_tokens和completion_tokens
                            self.prompt_tokens = text_usage.get("prompt_tokens", 0)
                            self.completion_tokens = text_usage.get("completion_tokens", 0)
                    
                    self.complete_event.set()
            else:
                print(f"请求错误: {data['header']['code']}, {data['header']['message']}")
                self.complete_event.set()
                self.error = "connection error"
                raise ValueError(f"请求错误: {data['header']['code']}, {data['header']['message']}")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            self.error = e
            self.complete_event.set()

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")
        self.error = error
        self.complete_event.set()

    def on_close(self, ws, close_status_code, close_msg):
        logging.info("WebSocket connection closed")
        self.complete_event.set()

    def on_open(self, ws, temperature, max_tokens):
        data = {
            "header": {
                "app_id": self.app_id,
            },
            "parameter": {
                "chat": {
                    "domain": self.domain,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            },
            "payload": {
                "message": {
                    "text": self.messages  # 需要在connect之前设置messages
                }
            }
        }
        ws.send(json.dumps(data))

    def connect(self, messages, **kwargs):
        self.messages = messages
        ws_param = Ws_Param(self.app_id, self.api_key, self.api_secret, self.spark_url)
        ws_url = ws_param.create_url()
        self.ws = websocket.WebSocketApp(ws_url, 
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.ws.on_open = lambda ws: self.on_open(ws, kwargs.get('temperature', 0.5), kwargs.get('max_tokens', 2048))
        wst = threading.Thread(target=self.ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
        wst.start()
        self.complete_event.wait()  # 等待消息接收完成或出错
        wst.join()  # 确保WebSocket线程已结束
        return self.answer, {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}, self.error


class XunfeiAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        self.app_id = model_kwargs.get("app_id")
        self.api_key = model_kwargs.get("api_key")
        self.api_secret = model_kwargs.get("api_secret")

    def pre_processing(self, **kwargs):
        if "app_id" in kwargs:
            self.app_id = kwargs.get("app_id")
            kwargs.pop("app_id")
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")
        if "api_secret" in kwargs:
            self.api_secret = kwargs.get("api_secret")
            kwargs.pop("api_secret")

        # 处理参数兼容性问题，不支持的参数全部舍弃
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p", "top_k", 
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)

        return kwargs
    
    def create_model_response_wrapper(self, result: str, usage, model: str) -> ModelResponse:
        choices = []

        message = Message(
            content=result, role="assistant"
        )
        choices.append(
            Choices(
                message=message,
                index=0,
                finish_reason="stop",
            )
        )

        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens+completion_tokens
        )

        response = ModelResponse(
            id="response",
            choices=choices,
            created=int(time.time()),
            model=model,
            usage=usage,
        )
        return response

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise XunfeiOpenAIError(
                    status_code=422, message="Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)

            client = XunfeiWebSocketClient(self.app_id, self.api_key, self.api_secret, model, **new_kwargs)
            answer, usage, error = client.connect(messages)
            if error:
                raise XunfeiOpenAIError(status_code=500, message=error)
            return self.create_model_response_wrapper(answer, usage, model=model)
        
        except Exception as e:
            if hasattr(e, "status_code"):
                raise XunfeiOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise XunfeiOpenAIError(status_code=500, message=str(e))