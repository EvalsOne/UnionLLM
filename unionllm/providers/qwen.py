import dashscope
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
        dashscope.api_key = self.api_key

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
        responses = Generation.call(
            model=model,
            messages=messages,
            result_format="message",
            incremental_output=True,
            **new_kwargs,
        )
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                # chunk_message = response.output.choices[0].message
                chunk_choices = []
                index = 0
                for choice in response.output.choices:
                    chunk_message = choice.message
                    chunk_delta = Delta()
                    if chunk_message:
                        if "role" in chunk_message:
                            chunk_delta.role = chunk_message["role"]
                        if "content" in chunk_message:
                            chunk_delta.content = chunk_message["content"]
                        chunk_choices.append(StreamingChoices(index=str(index), delta=chunk_delta))

                if hasattr(response, "usage") and response.usage is not None:
                    chunk_usage = Usage()
                    if "input_tokens" in response.usage:
                        chunk_usage.prompt_tokens = response.usage["input_tokens"]
                    if "output_tokens" in response.usage:
                        chunk_usage.completion_tokens = response.usage["output_tokens"]
                    if "total_tokens" in response.usage:
                        chunk_usage.total_tokens = response.usage["total_tokens"]

                chunk_response = ModelResponse(
                    id=response.request_id,
                    choices=chunk_choices,
                    created=int(time.time()),
                    model=model,
                    usage=chunk_usage if chunk_usage else None,
                    stream=True
                )
                index += 1
                yield chunk_response

            else:
                raise QwenOpenAIError(
                    status_code=response.status_code,
                    message=f"error message: {response.message}",
                )
                

    def create_model_response_wrapper(self, response, model):
        if response.status_code == HTTPStatus.OK:
            response_dict = response.output
            choices = []

            for index, choice in enumerate(response_dict.choices):
                message = Message(
                    content=choice.message.content,
                    role=choice.message.role
                )
                choices.append(
                    Choices(
                        message=message,
                        index=index,
                        finish_reason=choice.finish_reason,
                    )
                )

            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            )

            return ModelResponse(
                id=response.request_id,
                choices=choices,
                created=int(time.time()),
                model=model,
                usage=usage,
            )
        else:
            raise QwenOpenAIError(
                status_code=response.status_code,
                message=f"Request failed with status code: {response.status_code}",
            )


    def create_qwen_model_response(result: dict, model: str) -> ModelResponse:
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
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                response = Generation.call(
                    model=model,
                    messages=messages,
                    result_format="message",
                    **new_kwargs,
                )
                if response.status_code == HTTPStatus.OK:
                    return self.create_model_response_wrapper(response, model=model)
                else:
                    return {'success': False, 'error': {'status_code': response.status_code, 'message': response.message}}
        except Exception as e:
            if hasattr(e, "status_code"):
                raise QwenOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise QwenOpenAIError(status_code=500, message=str(e))
        
