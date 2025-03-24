import time
import requests
import json, os
import logging
import hashlib
from .base_provider import BaseProvider
from unionllm.utils import ModelResponse, Message, Choices, Usage, Delta, StreamingChoices

class DouBaoOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class DouBaoAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        # Get ERNIE_CLIENT_ID and ERNIE_CLIENT_ID from environment variables
        _env_api_key= os.environ.get("ARK_API_KEY")
        self.api_key = model_kwargs.get("api_key") if model_kwargs.get("api_key") else _env_api_key
        if not self.api_key:
            raise DouBaoOpenAIError(
                status_code=422, message=f"Missing api_key"
            )


    def pre_processing(self, **kwargs):
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "logprobs", "stream", "stop",
            "presence_penalty", "frequency_penalty", "best_of", "logit_bias", "tools", "tool_choice"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def to_formatted_prompt(self, messages):
        return messages

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {"Content-Type": "application/json","Authorization": f"Bearer {self.api_key}"}
        index = 0
        for line in requests.post(self.endpoint_url, headers=headers, data=payload, stream=True).iter_lines():
            if line:
                try:
                    # Remove the "data: " prefix before decoding the JSON
                    line_without_prefix = line.decode('utf-8').removeprefix('data: ')
                    new_line = json.loads(line_without_prefix)
                    chunk_choices = []
                    chunk_delta = Delta()
                    if new_line.get("result"):
                        chunk_delta.role = "assistant"
                        chunk_delta.content=new_line.get("result", "")
                        chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta))
                    if 'usage' in new_line:
                        chunk_usage = Usage()
                        if "input_tokens" in chunk_usage:
                            chunk_usage.prompt_tokens = chunk_usage.get("prompt_tokens", 0),
                        if "output_tokens" in chunk_usage:
                            chunk_usage.completion_tokens = chunk_usage.get("completion_tokens", 0),
                        if "total_tokens" in chunk_usage:
                            chunk_usage.total_tokens = chunk_usage.get("total_tokens", 0)
                    else:
                        chunk_usage = None

                    chunk_response = ModelResponse(
                        id="hello",
                        choices=chunk_choices,
                        created=int(time.time()),
                        model=model,
                        usage=chunk_usage if chunk_usage else None,
                        stream=True
                    )
                    index += 1
                    yield chunk_response

                except json.JSONDecodeError:
                    # Log the error or handle it as needed
                    continue


    def create_model_response_wrapper(self, result, model):
        response_dict = json.loads(result)
        choices = []

        # message = Message(content=response_dict["result"], role="assistant")
        choices_dict = response_dict['choices'][0]
        choices.append(
            Choices(
                message=choices_dict['message'],
                index=0,
                finish_reason=choices_dict.get("finish_reason", ""),
                logprobs=choices_dict.get("logprobs", None),
            )
        )

        usage = Usage(
            prompt_tokens=response_dict['usage']['prompt_tokens'],
            completion_tokens=response_dict['usage']['completion_tokens'],
            total_tokens=response_dict['usage']['total_tokens']
        )

        response = ModelResponse(
            id= response_dict["id"],  # The request_id is not provided by the API
            choices=choices,
            created=int(time.time()),
            model=model,
            usage=usage,
        )
        return response

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise DouBaoOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
                
            message_check_result = self.check_prompt("doubao", model, messages)            
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise DouBaoOpenAIError(
                    status_code=422, message=message_check_result['reason']
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            messages = self.to_formatted_prompt(messages)

            self.model_path = model

            self.endpoint_url = f"https://ark.cn-beijing.volces.com/api/v3/chat/completions"

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                result = requests.post(self.endpoint_url, headers=headers, data=payload)
                return self.create_model_response_wrapper(result.text, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise DouBaoOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise DouBaoOpenAIError(status_code=500, message=str(e))
