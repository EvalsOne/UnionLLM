from .base_provider import BaseProvider
from cnlitellm.utils import create_baichuan_model_response
import requests
import json


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
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def pre_processing(self, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        if "temperature" in kwargs:
            temperature = kwargs.get("temperature")
            if temperature > 1 or temperature < 0:
                raise BaiChuanOpenAIError(
                    status_code=422, message="Temperature must be between 0 and 1"
                )
        return kwargs

    def post_stream_processing(self, model, messages, **new_kwargs):
        url = "https://api.baichuan-ai.com/v1/chat/completions"
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        result = requests.post(url, headers=headers, data=payload)
        for line in result.iter_lines():
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
                yield chunk_line

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise BaiChuanOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing(model, messages, **new_kwargs)
            else:
                url = "https://api.baichuan-ai.com/v1/chat/completions"
                payload = json.dumps({"model": model, "messages": messages, **kwargs})
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                result = requests.post(url, headers=headers, data=payload)
                if result.status_code == 200:
                    return create_baichuan_model_response(result, model=model)
                else:
                    raise BaiChuanOpenAIError(
                        status_code=result.status_code,
                        message=f"Failed to complete request: {result.text}",
                    )
        except Exception as e:
            if hasattr(e, "status_code"):
                raise BaiChuanOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise BaiChuanOpenAIError(status_code=500, message=str(e))
