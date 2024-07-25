from .base_provider import BaseProvider
from unionllm.utils import ModelResponse, Message, Choices, Usage, Context, Delta, StreamingChoices
from openai import OpenAI
import logging, json, time, requests, os


class FastGPTError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class FastGPTProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        # Get FASTGPT_API_KEY from environment variables
        _env_api_key = os.environ.get("FASTGPT_API_KEY")
        self.api_key = model_kwargs.get("api_key") if model_kwargs.get("api_key") else _env_api_key
        if not self.api_key:
            raise FastGPTError(
                status_code=422, message=f"Missing API key"
            )
        self.base_url = model_kwargs.get("api_base") if model_kwargs.get("api_base") else "https://api.fastgpt.in/api/v1"
        self.endpoint_url = self.base_url + "/chat/completions"

    def pre_processing(self, **kwargs):
        # process the compatibility issue of parameters, all unsupported parameters are discarded
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p", "n",
            "logprobs", "stream", "stop", "presence_penalty", "frequency_penalty",
            "best_of", "logit_bias"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        # 获取详细信息，如果希望得到RAG的Context则必须开启
        kwargs.update({"detail": True})
        return kwargs

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint_url, headers=headers, data=payload)
        index = 0
        # 解析stream返回信息并生成OpenAI兼容格式
        for line in response.iter_lines():
            if line:
                chunk_choices = []
                chunk_context = []
                if line.startswith(b"data:"):
                    new_line = line.decode("utf-8").replace("data: ", "")
                    if new_line == "[DONE]":
                        continue
                    try:
                        data = json.loads(new_line)
                    except Exception as e:
                        continue

                    if "choices" in data:
                        chunk_message = data["choices"][0]["delta"]
                        chunk_delta = Delta()
                        if chunk_message:
                            if "role" in chunk_message:
                                chunk_delta.role = chunk_message['role']
                            if "content" in chunk_message:
                                chunk_delta.content = chunk_message['content']
                            chunk_choices.append(StreamingChoices(index=str(index), delta=chunk_delta))

                    if isinstance(data, list):
                        summary = data[0]
                        if "tokens" in summary:
                            chunk_usage = Usage()
                            chunk_usage.total_tokens = summary['tokens']
                        else:
                            chunk_usage = None
                    else:
                        chunk_usage = None

                    if isinstance(data, list):
                        for module in data:
                            if "quoteList" in module:
                                for quote in module["quoteList"]:
                                    content = f'question:[{quote["q"]}], answer:[{quote["a"]}]'
                                    chunk_context.append(
                                        {
                                            "id": quote["id"],
                                            "content": content,
                                        }    
                                    )
                
                    chunk_response = ModelResponse(
                        id=data['id'] if 'id' in data else None,
                        choices=chunk_choices,
                        context=chunk_context,
                        created=int(time.time()),
                        model=model,
                        usage=chunk_usage if chunk_usage else None,
                        stream=True
                    )
                    index += 1
                    yield chunk_response

    def create_model_response_wrapper(self, result):
        response_dict = result.json()
        choices = []
        context = []
        message = Message(
            content=response_dict['choices'][0]['message']['content'],
            role="assistant"
        )
        choices.append(
            Choices(
                message=message,
                index=0,
                finish_reason="stop",
            )
        )

        usage = Usage(
            prompt_tokens=response_dict['usage']['prompt_tokens'],
            completion_tokens=response_dict['usage']['completion_tokens'],
            total_tokens=response_dict['usage']['total_tokens'],
        )

        # 解析quoteList中包含的q&a背景信息并合成为字符串
        if "responseData" in response_dict:
            for module in response_dict["responseData"]:
                if "quoteList" in module:
                    for quote in module["quoteList"]:
                        content = f'question:[{quote["q"]}], answer:[{quote["a"]}]'
                        context.append(Context(id=quote["id"], content=content))

        response = ModelResponse(
            id=response_dict.get("id", ""),
            choices=choices,
            context=context,
            created=int(time.time()),
            model='',
            usage=usage,
        )
        return response

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise FastGPTError(
                    status_code=422, message="Missing model or messages"
                )
                
            message_check_result = self.check_prompt("fastgpt", model, messages)    
            
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise FastGPTError(
                    status_code=422, message=message_check_result['reason']
                )
                
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                result = requests.post(self.endpoint_url, headers=headers, data=payload)
                return self.create_model_response_wrapper(result)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise FastGPTError(status_code=e.status_code, message=str(e))
            else:
                raise FastGPTError(status_code=500, message=str(e))