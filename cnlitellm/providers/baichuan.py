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
        lines = [line.strip() for line in result.text.split("\n") if line.strip()]
        parsed_data = []
        for line in lines:
            new_line = line.replace("data: ", "")
            parsed_data.append(json.loads(new_line))
        for chunk in enumerate(parsed_data):
            chunk_message = chunk["choices"][0]["delta"]
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
            if hasattr(chunk, "usage") and chunk["choices"][0]["usage"] is not None:
                chunk_line["usage"] = {
                    "total_tokens": chunk["usage"]["total_tokens"],
                    "prompt_tokens": chunk["usage"]["prompt_tokens"],
                    "completion_tokens": chunk["usage"]["completion_tokens"],
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
