import json
from .base_provider import BaseProvider
from unionllm.utils import ModelResponse, Message, Choices, Usage, Context, Delta, StreamingChoices
import requests
import logging, json, time, os

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
        # Get DIFY_API_KEY from environment variables
        _env_api_key = os.environ.get("DIFY_API_KEY")
        self.api_key = model_kwargs.get("api_key") if model_kwargs.get("api_key") else _env_api_key
        self.conversation_id = model_kwargs.get("conversation_id", "")
        self.user = model_kwargs.get("user", "guest-user")
        if not self.api_key:
            raise DifyOpenAIError(
                status_code=422, message=f"Missing API key"
            )
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
        mode = "streaming"
        payload = {"query": query, "response_mode": mode, "user": self.user,"conversation_id": "","inputs":{}}
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
        payload = json.dumps(payload)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint_url, headers=headers, data=payload)

        chunk_usage = Usage()
        index = 0
        for line in response.iter_lines():
            if line:
                chunk_choices = []
                chunk_context = []
                chunk_delta = Delta()
                # judge if the new_line begins with "data:"
                if line.startswith(b"data:"):
                    new_line = line.decode("utf-8").replace("data: ", "")
                    if new_line == "[DONE]":
                        continue
                                        
                    data = json.loads(new_line)
                    event = data.get("event")
                    msg_id = data.get("message_id")
                    conversation_id = data.get("conversation_id")
                    if event == "message":
                        if 'answer' in data:
                            chunk_message = data["answer"]
                            if chunk_message:
                                chunk_delta.role = "assistant"
                                chunk_delta.content = chunk_message
                                chunk_choices.append(StreamingChoices(index=str(index), delta=chunk_delta))

                    if event == "message_end" and "metadata" in data:
                        metadata = data["metadata"]
                        if "usage" in metadata:
                            usage_info = metadata["usage"]
                            if "prompt_tokens" in usage_info:
                                chunk_usage.prompt_tokens = usage_info["prompt_tokens"]
                            if "completion_tokens" in usage_info:
                                chunk_usage.completion_tokens = usage_info["completion_tokens"]
                            if "total_tokens" in usage_info:
                                chunk_usage.total_tokens = usage_info["total_tokens"]

                        if 'retriever_resources' in metadata:
                            for resource in metadata['retriever_resources']:
                                chunk_context.append({
                                    "id": resource["position"],
                                    "content": resource["content"],
                                    "score": resource["score"],    
                                })
                    chunk_response = ModelResponse(
                        id=msg_id,
                        conversation_id=conversation_id,
                        choices=chunk_choices,
                        context=chunk_context,
                        created=int(time.time()),
                        model=model,
                        usage=chunk_usage,
                        stream=True
                    )
                    index += 1
                    yield chunk_response

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

        if 'retriever_resources' in response_dict['metadata']:
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
        else:
            context = []

        response = ModelResponse(
            id=response_dict["id"],
            conversation_id=self.conversation_id,
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

            message_check_result = self.check_prompt("dify", model, messages)            
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise DifyOpenAIError(
                    status_code=422, message=message_check_result['reason']
                )
                
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                messages, query = self.to_formatted_prompt(messages)
                mode = "blocking"

                payload = {"query": query, "response_mode": mode, "user": self.user,"conversation_id": "","inputs":{}}
                if self.conversation_id:
                    payload["conversation_id"] = self.conversation_id
                payload = json.dumps(payload)
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                result = requests.post(self.endpoint_url, headers=headers, data=payload)
                # print("Dify response")
                # print(result.json())
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise DifyOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise DifyOpenAIError(status_code=500, message=str(e))
