import time
import requests
import json
import hashlib
from .base_provider import BaseProvider
from cnlitellm.utils import ModelResponse, Message, Choices, Usage, Delta, StreamingChoices

class TianGongOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class TianGongAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        self.app_key = model_kwargs.get("app_key")
        self.app_secret = model_kwargs.get("app_secret")
        self.endpoint_url = "https://sky-api.singularity-ai.com/saas/api/v4/generate"

    def pre_processing(self, **kwargs):
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p", "n",
            "logprobs", "stream", "stop", "presence_penalty", "frequency_penalty",
            "best_of", "logit_bias"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def to_formatted_prompt(self, messages):
        # replace the role of assistant to bot
        for message in messages:
            if message.get("role") == "assistant":
                message["role"] = "bot"
        return messages
    
    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        timestamp = str(int(time.time()))
        sign_content = self.app_key + self.app_secret + timestamp
        sign_result = hashlib.md5(sign_content.encode("utf-8")).hexdigest()
        payload = {"model": model, "messages": messages, **new_kwargs}
        headers = {
            "app_key": self.app_key,
            "timestamp": timestamp,
            "sign": sign_result,
            "Content-Type": "application/json",
            "stream": "true",
        }
        result = requests.post(self.endpoint_url, headers=headers, json=payload, stream=True)
        index = 0
        for line in result.iter_lines():
            if line:
                new_line = json.loads(line.decode('utf-8'))
                chunk_choices = []
                chunk_delta = Delta()
                if 'reply' in new_line["resp_data"]:
                    chunk_delta.role = "assistant"
                    chunk_delta.content=new_line["resp_data"]["reply"]
                    chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta))

                if 'usage' in new_line["resp_data"]:
                    chunk_usage = Usage()
                    if "input_tokens" in new_line["resp_data"]["usage"]:
                        chunk_usage.prompt_tokens = new_line["resp_data"]["usage"]["prompt_tokens"],
                    if "output_tokens" in new_line["resp_data"]["usage"]:
                        chunk_usage.completion_tokens = new_line["resp_data"]["usage"]["output_tokens"]
                    if "total_tokens" in new_line["resp_data"]["usage"]:
                        chunk_usage.total_tokens = new_line["resp_data"]["usage"]["total_tokens"]
                else:
                    chunk_usage = None

                chunk_response = ModelResponse(
                    id=new_line['trace_id'],
                    choices=chunk_choices,
                    created=int(time.time()),
                    model=model,
                    usage=chunk_usage if chunk_usage else None,
                    stream=True
                )
                index += 1
                yield chunk_response

    def create_model_response_wrapper(self, result, model):
        response_dict = result.json()
        choices = []
        message = Message(content=response_dict["resp_data"]["reply"], role="assistant")
        choices.append(
            Choices(
                message=message,
                index=0,
                finish_reason=response_dict["resp_data"]["finish_reason"],
            )
        )
        usage = Usage(
            prompt_tokens=response_dict["resp_data"]["usage"]["prompt_tokens"],
            completion_tokens=response_dict["resp_data"]["usage"]["completion_tokens"],
            total_tokens=response_dict["resp_data"]["usage"]["total_tokens"],
        )
        response = ModelResponse(
            id=response_dict["trace_id"],
            choices=choices,
            created=int(time.time()),
            model=model,
            usage=usage,
        )
        return response


    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise TianGongOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            messages = self.to_formatted_prompt(messages)

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                timestamp = str(int(time.time()))
                sign_content = self.app_key + self.app_secret + timestamp
                sign_result = hashlib.md5(sign_content.encode("utf-8")).hexdigest()
                payload = {"model": model, "messages": messages, **new_kwargs}
                print("payload", payload)
                headers = {
                    "app_key": self.app_key,
                    "timestamp": timestamp,
                    "sign": sign_result,
                    "Content-Type": "application/json",
                    "stream": "false",
                }
                result = requests.post(self.endpoint_url, headers=headers, json=payload)
                print(result, result.json())
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise TianGongOpenAIError(status_code=e.status)
