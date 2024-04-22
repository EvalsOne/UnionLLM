import json
from .base_provider import BaseProvider
from cnlitellm.utils import ModelResponse, Message, Choices, Usage
import requests
import logging

class BaiChuanOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class BaiChuanAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        self.api_key = model_kwargs.get("api_key")
        self.endpoint_url = "https://api.baichuan-ai.com/v1/chat/completions"

    def pre_processing(self, **kwargs):
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
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint_url, headers=headers, data=payload)
        for line in response.iter_lines():
            if line:
                new_line = line.decode("utf-8").replace("data: ", "")
                if new_line == "[DONE]":
                    break
                data = json.loads(new_line)
                chunk_message = data["choices"][0]["delta"]
                chunk_line = {
                    "choices": [
                        {
                            "delta": {
                                "role": chunk_message["role"],
                                "content": chunk_message["content"],
                            }
                        }
                    ]
                }
                if "usage" in data:
                    usage_info = data["usage"]
                    chunk_line["usage"] = {
                        "total_tokens": usage_info["total_tokens"],
                        "prompt_tokens": usage_info["prompt_tokens"],
                        "completion_tokens": usage_info["completion_tokens"],
                    }
                yield json.dumps(chunk_line) + "\n\n"

    def create_model_response_wrapper(self, result, model):
        response_dict = result.json()
        choices = []
        for choice in response_dict["choices"]:
            message = Message(
                content=choice["message"]["content"], role=choice["message"]["role"]
            )
            choices.append(
                Choices(
                    message=message,
                    index=choice["index"],
                    finish_reason=choice["finish_reason"],
                )
            )
        usage = Usage(
            prompt_tokens=response_dict["usage"]["prompt_tokens"],
            completion_tokens=response_dict["usage"]["completion_tokens"],
            total_tokens=response_dict["usage"]["total_tokens"],
        )
        response = ModelResponse(
            id=response_dict["id"],
            choices=choices,
            created=response_dict["created"],
            model=model,
            usage=usage,
        )
        return response


    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise BaiChuanOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                result = requests.post(self.endpoint_url, headers=headers, data=payload)
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise BaiChuanOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise BaiChuanOpenAIError(status_code=500, message=str(e))
