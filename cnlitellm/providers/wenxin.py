import time
import requests
import json
import logging
import hashlib
from .base_provider import BaseProvider
from cnlitellm.utils import ModelResponse, Message, Choices, Usage

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
    def __init__(self, **model_kwargs):
        self.client_id = model_kwargs.get("client_id")
        self.client_secret = model_kwargs.get("client_secret")
        self.access_token = self.get_access_token()

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        response = requests.post(url, params=params)
        return str(response.json().get("access_token"))

    def pre_processing(self, **kwargs):
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "logprobs", "stream", "stop",
            "presence_penalty", "frequency_penalty", "best_of", "logit_bias"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def to_formatted_prompt(self, messages):
        if messages and messages[0].get('role') == 'system':
            system = messages.pop(0).get('content')
        else:
            system = None
        return messages, system

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {"Content-Type": "application/json"}
        for line in requests.post(self.endpoint_url, headers=headers, data=payload, stream=True).iter_lines():
            if line:
                try:
                    # Remove the "data: " prefix before decoding the JSON
                    line_without_prefix = line.decode('utf-8').removeprefix('data: ')
                    new_line = json.loads(line_without_prefix)
                    chunk_line = {
                        "choices": [
                            {
                                "delta": {
                                    "role": "assistant",
                                    "content": new_line.get("result", ""),
                                }
                            }
                        ]
                    }
                    if "usage" in new_line:
                        chunk_usage = new_line["usage"]
                        chunk_line["usage"] = {
                            "prompt_tokens": chunk_usage.get("prompt_tokens", 0),
                            "completion_tokens": chunk_usage.get("completion_tokens", 0),
                            "total_tokens": chunk_usage.get("total_tokens", 0),
                        }
                    yield json.dumps(chunk_line) + "\n\n"
                except json.JSONDecodeError:
                    # Log the error or handle it as needed
                    continue


    def create_model_response_wrapper(self, result, model):
        response_dict = json.loads(result)
        choices = []

        message = Message(content=response_dict["result"], role="assistant")
        choices.append(
            Choices(
                message=message,
                index=0,
                finish_reason="unknown",  # The finish_reason is not provided by the API
            )
        )

        usage = Usage(
            prompt_tokens=0,  # The usage details are not provided by the API
            completion_tokens=0,
            total_tokens=0,
        )

        response = ModelResponse(
            id="unknown",  # The request_id is not provided by the API
            choices=choices,
            created=int(time.time()),
            model=model,
            usage=usage,
        )
        return response

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise WenXinOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            messages, system = self.to_formatted_prompt(messages)
            if system:
                new_kwargs["system"] = system

            # 模型与url中model_path的对应关系
            if model == "ERNIE-4.0":
                self.model_path = "completions_pro"
            elif model == "ERNIE-3.5-8K":
                self.model_path = "completions"
            elif model == "ERNIE-Bot-8K":
                self.model_path = "ernie_bot_8k"
            else:
                self.model_path = model

            self.endpoint_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self.model_path}?access_token={self.access_token}"

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
                headers = {"Content-Type": "application/json"}
                result = requests.post(self.endpoint_url, headers=headers, data=payload)
                return self.create_model_response_wrapper(result.text, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise WenXinOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise WenXinOpenAIError(status_code=500, message=str(e))
