import json
from .base_provider import BaseProvider
from cnlitellm.utils import create_model_response
from zhipuai import ZhipuAI


class ZhipuAIProvider(BaseProvider):
    def __init__(self):
        pass

    def completion(self, model: str, messages: list, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        self.client = ZhipuAI(api_key=self.api_key)

        stream = kwargs.get("stream", False)

        if stream:

            def generate_stream():
                for chunk in self.client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                ):
                    delta = chunk.choices[0].delta
                    line = {
                        "choices": [
                            {"delta": {"role": delta.role, "content": delta.content}}
                        ]
                    }
                    if chunk.usage is not None:
                        line["usage"] = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    yield json.dumps(line) + "\n\n"

            return generate_stream()

        else:
            result = self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return create_model_response(result, model=model)