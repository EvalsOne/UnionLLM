import json
from .base_provider import BaseProvider
from cnlitellm.utils import ModelResponse, Message, Choices, Usage, Context
import requests
import logging, json, time

class DifyOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class DifyAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        self.api_key = model_kwargs.get("api_key")
        self.endpoint_url = "https://api.dify.ai/v1/chat-messages"

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
        message = messages[-1]
        # if last message role is not user, return error
        if message["role"] != "user":
            raise DifyOpenAIError(
                status_code=422, message=f"Last message role should be user"
            )
        query = message["content"]
        return messages, query


    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        # 预处理对话内容并返回最近的用户问题
        messages, query = self.to_formatted_prompt(messages)
        stream = new_kwargs.get("stream", False)
        mode = "streaming" if stream else "blocking"
        payload = json.dumps({"query": query, "response_mode": mode, "user": "abc-123","conversation_id": "","inputs":{}})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint_url, headers=headers, data=payload)

        for line in response.iter_lines():
            if line:
                # judge if the new_line begins with "data:"
                if line.startswith(b"data:"):
                    new_line = line.decode("utf-8").replace("data: ", "")
                    if new_line == "[DONE]":
                        continue

                    data = json.loads(new_line)
                    chunk_line = {}
                    if "answer" in data:
                        chunk_message = data["answer"]
                        if chunk_message:
                            chunk_line["choices"] = [
                                {
                                    "delta": {
                                        "role": "assistant",
                                        "content": chunk_message,
                                    }
                                }
                            ]

                    if "metadata" in data:
                        context = []
                        metadata = data["metadata"]
                        if "usage" in metadata:
                            usage_info = metadata["usage"]
                            chunk_line["usage"] = {
                                "total_tokens": usage_info["total_tokens"],
                                "prompt_tokens": usage_info["prompt_tokens"],
                                "completion_tokens": usage_info["completion_tokens"],
                            }

                        if 'retriever_resources' in metadata:
                            context = []
                            for resource in metadata['retriever_resources']:
                                context.append({
                                    "id": resource["position"],
                                    "content": resource["content"],
                                    "score": resource["score"],    
                                })
                            chunk_line["context"] = context

                    if chunk_line:
                        yield json.dumps(chunk_line) + "\n\n"

    def create_model_response_wrapper(self, result, model):
        response_dict = result.json()
        choices = []
        context = []

        message = Message(
            content=response_dict['answer'], role="assistant"
        )
        choices.append(
            Choices(
                message=message,
                index=0,
                finish_reason="stop",
            )
        )

        usage = Usage(
            prompt_tokens=response_dict['metadata']["usage"]["prompt_tokens"],
            completion_tokens=response_dict['metadata']["usage"]["completion_tokens"],
            total_tokens=response_dict['metadata']["usage"]["total_tokens"],
        )

        retrieved_resources = response_dict['metadata']["retriever_resources"]
        if retrieved_resources is not None:
            for resource in retrieved_resources:
                context.append(
                    Context(
                        id=resource["position"],
                        content=resource["content"],
                        score=resource["score"],
                    )
                )

        response = ModelResponse(
            id=response_dict["id"],
            choices=choices,
            context=context,
            created=int(time.time()),            
            model=model,
            usage=usage,
        )
        return response


    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise DifyOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                messages, query = self.to_formatted_prompt(messages)
                mode = "streaming" if stream == 'streaming' else "blocking"
                payload = json.dumps({"query": query, "response_mode": mode, "user": "abc-123","conversation_id": "","inputs":{}})
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                result = requests.post(self.endpoint_url, headers=headers, data=payload)
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise DifyOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise DifyOpenAIError(status_code=500, message=str(e))