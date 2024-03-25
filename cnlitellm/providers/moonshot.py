import json
from .base_provider import BaseProvider
from cnlitellm.utils import create_model_response
from openai import OpenAI
import logging, json


class MoonshotAIProvider(BaseProvider):
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url

    def completion(self, model: str, messages: list, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")
        self.base_url = "https://api.moonshot.cn/v1"        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        stream = kwargs.get("stream", False)
        if stream:

            def generate_stream():
                response = self.client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                for chunk in response:
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
                    yield json.dumps(line) + "\n\n"

            return generate_stream()

        else:
            result = self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return create_model_response(result, model=model)
