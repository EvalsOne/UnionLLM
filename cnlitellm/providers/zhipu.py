import json
from .base_provider import BaseProvider
from cnlitellm.utils import create_model_response
from zhipuai import ZhipuAI


class ZhiPuOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class ZhipuAIProvider(BaseProvider):
    def __init__(self):
        pass

    def pre_processing(self, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        if "temperature" in kwargs:
            temperature = kwargs.get("temperature")
            if temperature > 1 or temperature < 0:
                raise ZhiPuOpenAIError(
                    status_code=422, message="Temperature must be between 0 and 1"
                )

        self.client = ZhipuAI(api_key=self.api_key)
        return kwargs

    def post_stream_processing(self, model, messages, **new_kwargs):
        for chunk in self.client.chat.completions.create(
            model=model, messages=messages, **new_kwargs
        ):
            delta = chunk.choices[0].delta
            line = {
                "choices": [
                    {
                        "delta": {
                            "role": delta.role,
                            "content": delta.content,
                        }
                    }
                ]
            }
            if chunk.usage is not None:
                line["usage"] = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            yield json.dumps(line) + "\n\n"

    def completion(self, model: str, messages: list, **kwargs):
        print("model: ", model)
        print("messages: ", messages)
        print("kwargs: ", kwargs)
        try:
            if model is None or messages is None:
                raise ZhiPuOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)

            stream = new_kwargs.get("stream", False)
            print("new_kwargs: ", new_kwargs)
            if stream:
                return self.post_stream_processing(model, messages, **new_kwargs)
            else:
                result = self.client.chat.completions.create(
                    model=model, messages=messages, **new_kwargs
                )
                return create_model_response(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise ZhiPuOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise ZhiPuOpenAIError(status_code=500, message=str(e))
