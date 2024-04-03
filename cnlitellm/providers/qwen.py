import json

import dashscope
from .base_provider import BaseProvider
from cnlitellm.utils import create_qwen_model_response
from http import HTTPStatus
from dashscope import Generation


class QwenOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class QwenAIProvider(BaseProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def pre_processing(self, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        if "result_format" in kwargs:
            kwargs.pop("result_format")

        dashscope.api_key = self.api_key
        return kwargs

    def post_stream_processing(self, model, messages, **new_kwargs):
        responses = Generation.call(
            model=model,
            messages=messages,
            result_format="message",
            incremental_output=True,
            **new_kwargs,
        )
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                chunk_message = response.output.choices[0].message
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
                if hasattr(response, "usage") and response.usage is not None:
                    chunk_usage = response.usage
                    line["usage"] = {
                        "prompt_tokens": chunk_usage["input_tokens"],
                        "completion_tokens": chunk_usage["output_tokens"],
                        "total_tokens": chunk_usage["total_tokens"],
                    }
                yield json.dumps(line) + "\n\n"
            else:
                raise QwenOpenAIError(
                    status_code=response.status_code,
                    message=f"Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}",
                )

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise QwenOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)

            stream = new_kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing(model, messages, **new_kwargs)
            else:
                response = Generation.call(
                    model=model,
                    messages=messages,
                    result_format="message",
                    **new_kwargs,
                )
                if response.status_code == HTTPStatus.OK:
                    return create_qwen_model_response(response, model=model)
                else:
                    raise QwenOpenAIError(
                        status_code=response.status_code,
                        message=f"Request failed with status code: {response.status_code}",
                    )
        except Exception as e:
            if hasattr(e, "status_code"):
                raise QwenOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise QwenOpenAIError(status_code=500, message=str(e))
