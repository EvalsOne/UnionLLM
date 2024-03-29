import json
from .base_provider import BaseProvider
from cnlitellm.utils import create_model_response
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
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        # self.base_url = base_url
        self.base_url = "https://api.moonshot.cn/v1"        

    def pre_processing(self, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        if "temperature" in kwargs:
            temperature = kwargs.get("temperature")
            if temperature > 1 or temperature < 0:
                raise MoonshotOpenAIError(
                    status_code=422, message="Temperature must be between 0 and 1"
                )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return kwargs

    def post_stream_processing(self, model, messages, **new_kwargs):
        print("start stream processing... step 1")
        response = self.client.chat.completions.create(
            model=model, messages=messages, **new_kwargs
        )
        print("start stream processing... step ")

        for chunk in response:
            # print("chunk: ", chunk)
            chunk_message = chunk.choices[0].delta
            line = {
                "choices": [
                    {
                        "delta": {
                            "role": chunk_message.role,
                            "content": chunk_message.content,
                        }
                    }
                ]
            }
            if (
                hasattr(chunk.choices[0], "usage")
                and chunk.choices[0].usage is not None
            ):
                chunk_usage = chunk.choices[0].usage
                line["usage"] = {
                    "prompt_tokens": chunk_usage["prompt_tokens"],
                    "completion_tokens": chunk_usage["completion_tokens"],
                    "total_tokens": chunk_usage["total_tokens"],
                }
            # print("line: ", line)
            yield json.dumps(line) + "\n\n"

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise MoonshotOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                print("start stream processing...")
                return self.post_stream_processing(model, messages, **new_kwargs)
            else:
                result = self.client.chat.completions.create(
                    model=model, messages=messages, **new_kwargs
                )
                return create_model_response(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise MoonshotOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise MoonshotOpenAIError(status_code=500, message=str(e))
