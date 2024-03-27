import time
from .base_provider import BaseProvider
from cnlitellm.utils import create_wenxin_model_response
import requests
import json
import hashlib


class WenXinOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class WenXinAIProvider(BaseProvider):
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key
        self.secret_key = secret_key

    def get_access_token(self, api_key, secret_key):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key,
        }
        response = requests.post(url, params=params)
        return str(response.json().get("access_token"))

    def pre_processing(self, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        if "temperature" in kwargs:
            temperature = kwargs.get("temperature")
            if temperature > 1 or temperature < 0:
                raise WenXinOpenAIError(
                    status_code=422, message="Temperature must be between 0 and 1"
                )

        return kwargs

    def post_stream_processing(self, model, messages, **new_kwargs):
        url = (
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"
            + model
            + "?access_token="
            + self.get_access_token(self.api_key, self.secret_key)
        )
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {"Content-Type": "application/json"}
        result = requests.request("POST", url, headers=headers, data=payload)
        for line in result.iter_lines():
            if line:
                new_line = line.decode("utf-8").replace("data: ", "")
                data = json.loads(new_line)
                chunk_line = {
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": data["result"],
                            }
                        }
                    ]
                }
                if "usage" in data:
                    chunk_usage = data["usage"]
                    chunk_line["usage"] = {
                        "prompt_tokens": chunk_usage["prompt_tokens"],
                        "completion_tokens": chunk_usage["completion_tokens"],
                        "total_tokens": chunk_usage["total_tokens"],
                    }
                yield chunk_line

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise WenXinOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing(model, messages, **new_kwargs)
            else:
                url = (
                    "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"
                    + model
                    + "?access_token="
                    + self.get_access_token(self.api_key, self.secret_key)
                )
                payload = json.dumps(
                    {"model": model, "messages": messages, **new_kwargs}
                )
                headers = {"Content-Type": "application/json"}
                result = requests.request("POST", url, headers=headers, data=payload)
                return create_wenxin_model_response(result.text, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise WenXinOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise WenXinOpenAIError(status_code=500, message=str(e))
