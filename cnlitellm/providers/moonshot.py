import json
from .base_provider import BaseProvider
from cnlitellm.utils import ResponseModelInterface
from openai import OpenAI
import logging, json


class MoonshotOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class MoonshotAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        self.api_key = model_kwargs.get("api_key")
        self.base_url = "https://api.moonshot.cn/v1"
        self.response_model = ResponseModelInterface()
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def pre_processing(self, **kwargs):
        # 处理参数兼容性问题，不支持的参数全部舍弃
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p", "n",
            "logprobs", "stream", "stop", "presence_penalty", "frequency_penalty",
            "best_of", "logit_bias"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)

        return kwargs
    
    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        result = self.client.chat.completions.create(
            model=model, messages=messages, **new_kwargs
        )
        return self.response_model.post_stream_processing(result)

    def create_model_response_wrapper(self, result, model):
        # 调用 response_model 中的 create_model_response 方法
        return self.response_model.create_model_response(result, model=model)

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise MoonshotOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model=model, messages=messages, **new_kwargs)
            else:
                result = self.client.chat.completions.create(
                    model=model, messages=messages, **new_kwargs
                )
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise MoonshotOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise MoonshotOpenAIError(status_code=500, message=str(e))
