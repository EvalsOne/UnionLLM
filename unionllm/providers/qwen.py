from openai import OpenAI
from .base_provider import BaseProvider
from http import HTTPStatus
from dashscope import Generation
from unionllm.utils import ModelResponse, Message, Choices, Usage, Delta, StreamingChoices
import json, time, os

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
    def __init__(self, **model_kwargs):
        # Get DASHSCOPE_API_KEY from environment variables
        _env_api_key = os.environ.get("DASHSCOPE_API_KEY")
        self.api_key = model_kwargs.get("api_key") if model_kwargs.get("api_key") else _env_api_key
        if not self.api_key:
            raise QwenOpenAIError(
                status_code=422, message=f"Missing API key"
            )        
        
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def pre_processing(self, **kwargs):
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p", "n",
            "logprobs", "stream", "stop", "presence_penalty", "frequency_penalty",
            "best_of", "logit_bias", "tools", "tool_choice"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        result = self.client.chat.completions.create(
            model=model, messages=messages, **new_kwargs
        )
        return self.post_stream_processing(result, model=model)

    def create_model_response_wrapper(self, result, model):
        return self.create_model_response(result, model=model)

    def create_model_response(self, result: dict, model: str) -> ModelResponse:
        choices = []
        for index, choice in enumerate(result.output.choices):
            message = Message(content=choice.message.content, role=choice.message.role)
            choices.append(
                Choices(message=message, index=index, finish_reason=choice.finish_reason)
            )
        usage = Usage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        )
        response = ModelResponse(
            id=result.request_id,
            choices=choices,
            created=int(time.time()),
            model=model,
            usage=usage,
        )
        return response

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise QwenOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )

            message_check_result = self.check_prompt("qwen", model, messages)   
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise QwenOpenAIError(
                    status_code=422, message=message_check_result['reason']
                )                
            new_kwargs = self.pre_processing(**kwargs)
            stream = new_kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model=model, messages=messages, **new_kwargs)
            else:
                result = self.client.chat.completions.create(
                    model=model, messages=messages, **new_kwargs
                )
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise QwenOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise QwenOpenAIError(status_code=500, message=str(e))
        
