import dashscope
from .base_provider import BaseProvider
from http import HTTPStatus
from dashscope import Generation, MultiModalConversation
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
        # Configure DashScope base URL per latest sample usage
        # Allow override via env DASHSCOPE_BASE_URL if provided, otherwise use the default API endpoint
        dashscope.base_http_api_url = os.environ.get(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
        )
        # Keep global api_key for backward compatibility, though we will also pass api_key explicitly on each call
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
        if_vision_model = new_kwargs.get("if_vision_model", False)
        has_vision_input = new_kwargs.get("has_vision_input", False)
        use_multimodal = new_kwargs.get("use_multimodal", False)
        # 清理内部控制字段，避免传入SDK
        new_kwargs.pop("if_vision_model", None)
        new_kwargs.pop("has_vision_input", None)
        new_kwargs.pop("use_multimodal", None)
        try:
            # 清理与SDK不兼容的参数，并显式打开 stream
            clean_kwargs = dict(new_kwargs)
            clean_kwargs["stream"] = True
            if use_multimodal:
                responses = MultiModalConversation.call(
                    api_key=self.api_key,
                    model=model,
                    messages=messages,
                    incremental_output=True,
                    **clean_kwargs
                )
            else:
                responses = Generation.call(
                    api_key=self.api_key,
                    model=model,
                    messages=messages,
                    result_format="message",
                    incremental_output=True,
                    **clean_kwargs,
                )
            # 在部分错误场景，SDK可能直接返回dict错误响应而非迭代器
            if isinstance(responses, dict):
                status_code = responses.get("status_code", 500)
                code = responses.get("code")
                req_id = responses.get("request_id")
                msg = responses.get("message")
                raise QwenOpenAIError(
                    status_code=status_code,
                    message=f"DashScope stream error (code={code}, request_id={req_id}): {msg}"
                )
        except Exception as e:
            # 提前捕获 DashScope SDK 抛出的异常，包含异常类型，便于诊断（鉴权/参数/网络等）
            raise QwenOpenAIError(
                status_code=500,
                message=f"DashScope streaming call failed: {type(e).__name__}: {str(e)}"
            )
        for response in responses:
            if not hasattr(response, "status_code"):
                # 兼容性保护：如果SDK返回了意外的字符串/字节流，避免AttributeError并给出清晰诊断
                preview = None
                try:
                    preview = (response[:200] if isinstance(response, str) else str(response))
                except Exception:
                    preview = "<unrepresentable>"
                raise QwenOpenAIError(
                    status_code=500,
                    message=f"Unexpected streaming chunk type: {type(response).__name__}; preview={preview}"
                )
            if response.status_code == HTTPStatus.OK:
                # chunk_message = response.output.choices[0].message
                chunk_choices = []
                index = 0
                chunk_usage = None
                for choice in response.output.choices:
                    chunk_message = choice.message
                    chunk_delta = Delta()
                    if chunk_message:
                        if "role" in chunk_message:
                            chunk_delta.role = chunk_message["role"]
                        if "content" in chunk_message:
                            if isinstance(chunk_message["content"], list) and chunk_message["content"] and isinstance(chunk_message["content"][0], dict):
                                chunk_delta.content = chunk_message["content"][0]["text"]
                            else:
                                chunk_delta.content = chunk_message["content"]

                        if 'tool_calls' in chunk_message and chunk_message['tool_calls']:
                            tool_calls = []
                            for tool_call in chunk_message['tool_calls']:
                                tool_calls.append(
                                    {
                                        "id": tool_call['id'] if 'id' in tool_call and tool_call['id'] else None,
                                        "index": tool_call['index'],
                                        "type": "function",
                                        "function": {
                                            "name": tool_call['function']['name'] if 'name' in tool_call['function'] else None,
                                            "arguments": tool_call['function']['arguments'] if 'arguments' in tool_call['function'] else None
                                        }
                                    }
                                )
                            chunk_delta.tool_calls = tool_calls
                        stream_choices = StreamingChoices(index=index, delta=chunk_delta)
                        if 'finish_reason' in choice:
                            if choice['finish_reason'] is not None and choice['finish_reason'] != 'null':
                                stream_choices.finish_reason = choice['finish_reason']
                            else:
                                stream_choices.finish_reason = None
                        chunk_choices.append(stream_choices)

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
                # 将DashScope响应中的错误详情透出，包含请求ID与错误码（若有）
                err_code = getattr(response, "code", None)
                req_id = getattr(response, "request_id", None)
                msg = getattr(response, "message", None)
                raise QwenOpenAIError(
                    status_code=response.status_code,
                    message=f"DashScope stream error (code={err_code}, request_id={req_id}): {msg}",
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

    def reformat_messages(self, messages: list) -> list:
        # 根据qwen的格式要求，重新组织messages
        new_messages = []
        for message in messages:
            if message['role'] == 'user':
                if isinstance(message['content'], list):
                    new_content = []
                    for content in message['content']:
                        if content.get('type') == 'text':
                            new_content.append({
                                "text": content.get('text')
                            })
                        elif content.get('type') == 'image_url':
                            new_content.append({
                                "image": content.get('image_url').get('url')
                            })
                    new_messages.append({
                        "role": "user",
                        "content": new_content
                    })
                else:
                    new_messages.append({
                        "role": "user",
                        "content": message['content']
                    })
            else:
                new_messages.append({
                    "role": message['role'],
                    "content": message['content']
                })
        return new_messages
    
    def check_if_vision_model(self, model: str) -> bool:
        # 视觉/多模态模型：qwen-vl-*, qwen3-omni-*
        m = (model or "").lower()
        return m.startswith("qwen-vl") or m.startswith("qwen3-omni")

    def should_use_multimodal_api(self, model: str, has_vision_input: bool) -> bool:
        m = (model or "").lower()
        # qwen3-omni 系列统一走 MultiModalConversation；qwen-vl 系列默认也是多模接口
        if m.startswith("qwen3-omni") or m.startswith("qwen-vl"):
            return True
        # 其他模型仅在确有视觉输入时才考虑多模接口
        return has_vision_input

    def ensure_mm_text_format(self, messages: list) -> list:
        # 确保在调用 MultiModalConversation 时，纯文本消息也符合多模消息格式
        fixed = []
        for msg in messages:
            content = msg.get('content')
            if isinstance(content, list):
                fixed.append(msg)
            else:
                fixed.append({
                    'role': msg.get('role'),
                    'content': [{ 'type': 'text', 'text': content }]
                })
        return fixed

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

            has_vision_input = message_check_result['multimodal_info']['has_vision_input']
            if_vision_model = self.check_if_vision_model(model)
            use_multimodal = self.should_use_multimodal_api(model, has_vision_input)
            if use_multimodal:
                # 如果包含视觉输入，按需重排；如果是 omni 系列但纯文本，也保证多模文本格式
                if has_vision_input:
                    messages = self.reformat_messages(messages)
                else:
                    messages = self.ensure_mm_text_format(messages)

            new_kwargs = self.pre_processing(**kwargs)
            new_kwargs['if_vision_model'] = if_vision_model
            new_kwargs['has_vision_input'] = has_vision_input
            new_kwargs['use_multimodal'] = use_multimodal
            stream = new_kwargs.get("stream", False)
            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                try:
                    call_kwargs = dict(new_kwargs)
                    call_kwargs.pop("stream", None)
                    if use_multimodal:
                        response = MultiModalConversation.call(
                            api_key=self.api_key,
                            model=model,
                            messages=messages,
                            **call_kwargs,
                        )
                    else:
                        response = Generation.call(
                            api_key=self.api_key,
                            model=model,
                            messages=messages,
                            result_format="message",
                            **call_kwargs,
                        )
                except Exception as e:
                    raise QwenOpenAIError(
                        status_code=500,
                        message=f"DashScope call failed: {type(e).__name__}: {str(e)}"
                    )
                if not hasattr(response, "status_code"):
                    raise QwenOpenAIError(
                        status_code=500,
                        message=f"Unexpected response type: {type(response).__name__}. The DashScope SDK may have returned an incompatible object."
                    )
                if response.status_code == HTTPStatus.OK:
                    return self.create_model_response_wrapper(response, model=model)
                else:
                    err_code = getattr(response, "code", None)
                    req_id = getattr(response, "request_id", None)
                    msg = getattr(response, "message", None)
                    raise QwenOpenAIError(
                        status_code=response.status_code,
                        message=f"DashScope error (code={err_code}, request_id={req_id}): {msg}"
                    )
        except Exception as e:
            if hasattr(e, "status_code"):
                raise QwenOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise QwenOpenAIError(status_code=500, message=str(e))
        
