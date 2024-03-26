import time
from .base_provider import BaseProvider
from cnlitellm.utils import create_tiangong_model_response
import requests
import json
import hashlib


class TianGongOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class TianGongAIProvider(BaseProvider):
    def __init__(self, api_key: str = None, app_secret: str = None):
        self.app_key = api_key
        self.app_secret = app_secret

    def pre_processing(self, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        if "temperature" in kwargs:
            temperature = kwargs.get("temperature")
            if temperature > 1 or temperature < 0:
                raise TianGongOpenAIError(
                    status_code=422, message="Temperature must be between 0 and 1"
                )
        return kwargs

    def post_stream_processing(self, model, messages, **new_kwargs):
        url = "https://sky-api.singularity-ai.com/saas/api/v4/generate"
        timestamp = str(int(time.time()))
        sign_content = self.app_key + self.app_secret + timestamp
        sign_result = hashlib.md5(sign_content.encode("utf-8")).hexdigest()
        payload = {"model": model, "messages": messages, **new_kwargs}
        headers = {
            "app_key": self.app_key,
            "timestamp": timestamp,
            "sign": sign_result,
            "Content-Type": "application/json",
            "stream": "true",
        }
        result = requests.post(url, headers=headers, json=payload, stream=True)
        for line in result.iter_lines():
            if line:
                new_line = json.loads(line.decode('utf-8'))
                chunk_line = {
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": new_line["resp_data"]["reply"],
                            }
                        }
                    ]
                }
                if "usage" in new_line["resp_data"]:
                    chunk_usage = new_line["resp_data"]["usage"]
                    chunk_line["usage"] = {
                        "prompt_tokens": chunk_usage["prompt_tokens"],
                        "completion_tokens": chunk_usage["completion_tokens"],
                        "total_tokens": chunk_usage["total_tokens"],
                    }
                yield chunk_line

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise TianGongOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing(model, messages, **new_kwargs)
            else:
                url = "https://sky-api.singularity-ai.com/saas/api/v4/generate"
                timestamp = str(int(time.time()))
                sign_content = self.app_key + self.app_secret + timestamp
                sign_result = hashlib.md5(sign_content.encode("utf-8")).hexdigest()
                payload = {"model": model, "messages": messages, **new_kwargs}
                headers = {
                    "app_key": self.app_key,
                    "timestamp": timestamp,
                    "sign": sign_result,
                    "Content-Type": "application/json",
                    "stream": "false",
                }
                result = requests.post(url, headers=headers, json=payload, stream=False)
                if result.json()['code'] == 200:
                    return create_tiangong_model_response(result, model=model)
                else:
                    raise TianGongOpenAIError(
                        status_code=result.json()['code'],
                        message=f"{result.json()["code_msg"]}",
                    )
        except Exception as e:
            if hasattr(e, "status_code"):
                raise TianGongOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise TianGongOpenAIError(status_code=500, message=str(e))
