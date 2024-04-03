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
        if "client_id" in kwargs:
            self.client_id = kwargs.get("client_id")
            kwargs.pop("client_id")

        if "client_secret" in kwargs:
            self.client_secret = kwargs.get("client_secret")
            kwargs.pop("client_secret")

        if "temperature" in kwargs:
            temperature = kwargs.get("temperature")
            if temperature > 1 or temperature < 0:
                raise WenXinOpenAIError(
                    status_code=422, message="Temperature must be between 0 and 1"
                )
        if "top_p" in kwargs:
            kwargs.pop("top_p")

        return kwargs

    def post_stream_processing(self, model, messages, **new_kwargs):
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {"Content-Type": "application/json"}
        # result = requests.request("POST", self.url, headers=headers, data=payload)
        for line in requests.request("POST", self.url, headers=headers, data=payload).iter_lines():
            print("line: ", line)
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
                print("chunk_line: ", json.dumps(chunk_line))
                yield json.dumps(chunk_line) + "\n\n"

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise WenXinOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            access_token = self.get_access_token(self.client_id, self.client_secret)

            if model == "ERNIE-4.0":
                model_path = "completions_pro"
            elif model == "ERNIE-3.5-8K":
                model_path = "completions"
            elif model == "ERNIE-Bot-8K":
                model_path = "ernie_bot_8k"
            else:
                model_path = model

            self.url = (
                "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"
                + model_path
                + "?access_token="
                + access_token
            )

            if stream:
                return self.post_stream_processing(model, messages, **new_kwargs)
            else:
                print("url: ", self.url)
                payload = json.dumps(
                    {"model": model, "messages": messages, **new_kwargs}
                )
                print("payload: ", payload)
                headers = {"Content-Type": "application/json"}
                result = requests.post(self.url, headers=headers, data=payload)
                print("result: ", result.json())
                return create_wenxin_model_response(result.text, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise WenXinOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise WenXinOpenAIError(status_code=500, message=str(e))
