import json
from .base_provider import BaseProvider
from cnlitellm.utils import ResponseModelInterface
from zhipuai import ZhipuAI
import logging

class ZhiPuOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class ZhipuAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        self.api_key = model_kwargs.get("api_key")
        self.response_model = ResponseModelInterface()
        self.client = ZhipuAI(api_key=self.api_key)

    def pre_processing(self, **kwargs):
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "n",
            "logprobs", "stream", "stop", "presence_penalty", "frequency_penalty",
            "best_of", "logit_bias"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def post_stream_processing_wrapper(self, response):
        return self.response_model.post_stream_processing(response)

    def create_model_response_wrapper(self, result, model):
        return self.response_model.create_model_response(result, model=model)

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise ZhiPuOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = new_kwargs.get("stream", False)

            if stream:
                response = self.client.chat.completions.create(
                    model=model, messages=messages, **new_kwargs
                )
                return self.post_stream_processing_wrapper(response)
            else:
                result = self.client.chat.completions.create(
                    model=model, messages=messages, **new_kwargs
                )
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise ZhiPuOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise ZhiPuOpenAIError(status_code=500, message=str(e))
