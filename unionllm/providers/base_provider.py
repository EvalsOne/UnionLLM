from abc import ABC, abstractmethod
from ..models import ResponseModel
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, List
from unionllm.utils import ModelResponse, Message, Choices, Usage, Context, StreamingChoices, Delta, check_object_input_support, check_video_input_support, check_vision_input_support, reformat_object_content, check_file_input_support

import openai
import json
from openai._models import BaseModel as OpenAIObject

# if TYPE_CHECKING:
from dataclasses import dataclass

@dataclass
class BaseProvider(ABC):
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None

    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message

    @abstractmethod
    def completion(self, model: str, messages: list) -> ResponseModel:
        pass

    def check_prompt(self, provider, model, messages):
        # 遍历messages列表，判断消息中间是否存在system消息，是否所有消息content都是string类型, 是否消息中包含图片和文件类型
        is_invalid_format = False
        has_middle_system = False
        has_object_content = False
        has_vision_input = False
        has_video_input = False
        has_file_input = False
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                if i != 0:
                    has_middle_system = True
            if not isinstance(message.get("content"), str):
                if isinstance(message.get("content"), list):
                    message_contents = message.get("content")
                    for content in message_contents:
                        if not isinstance(content, dict):
                            is_invalid_format = True
                        else:
                            has_object_content = True
                            if content.get("type") == "image_url":
                                if content.get("image_url").get("url"):
                                    has_vision_input = True
                                else:
                                    is_invalid_format = True
                            elif content.get("type") == "video_url":
                                if content.get("video_url").get("url"):
                                    has_video_input = True
                                else:
                                    is_invalid_format = True
                            elif content.get("type") == "file_url":
                                if content.get("file_url").get("url"):
                                    has_file_input = True
                                else:
                                    is_invalid_format = True
                                                                      
        if is_invalid_format:
            return {"pass_check": False, "reformatted": False, "reason": "Invalid message format"}

        reformated = 0
        if has_object_content:
            # 判断object content是否支持
            object_support = check_object_input_support(provider)
            if object_support == "NONE":
                # 如果不支持
                if has_vision_input or has_file_input:
                    # 如果包含图片或文件，则返回错误信息
                    return {"pass_check": False, "reformatted": False, "reason": "Object content is not supported"}
                else:
                    # 如果不包含，则将object content转为文本
                    messages = reformat_object_content(messages, False, False, False)
                    reformated = 1
            else:
                # 如果支持
                reformat_image = 0
                reformat_file = 0
                reformat_video = 0
                if object_support == "PARTIAL":
                    # 如果部分支持，则需要将object content转为文本
                    reformated = 1
                if has_vision_input:
                    # 如果包含图片
                    vision_input_support = check_vision_input_support(provider, model)
                    if vision_input_support == "PARTIAL":
                        # 如果图片是部分支持，则需要将图片转为文本
                        reformat_image = 1
                        reformated = 1
                    elif vision_input_support == "NONE":
                        # 如果图片不支持，则返回错误信息
                        return {"pass_check": False, "reformatted": False, "reason": "Vision input is not supported"}
                
                if has_video_input:
                    video_input_support = check_video_input_support(provider, model)
                    if video_input_support == "PARTIAL":
                        # 如果视频是部分支持，则需要将视频转为文本
                        reformat_video = 1
                        reformated = 1
                    elif video_input_support == "NONE":
                        # 如果视频不支持，则返回错误信息
                        return {"pass_check": False, "reformatted": False, "reason": "Video input is not supported"}
                    
                if has_file_input:
                    # 如果包含文件
                    file_input_support = check_file_input_support(provider, model)
                    if file_input_support == "PARTIAL":
                        # 如果文件是部分支持，则需要将文件转为文本
                        reformat_file= 1
                        reformated = 1
                    elif file_input_support == "NONE":
                        # 如果文件不支持，则返回错误信息
                        return {"pass_check": False, "reformatted": False, "messages": messages, "reason": "File input is not supported"}
                if reformated or reformat_image or reformat_file or reformat_video:
                    messages = reformat_object_content(messages, 1, reformat_image, reformat_file, reformat_video)
                
        return {"pass_check": True, "reformatted": reformated, "messages": messages}         

    def create_model_response(
        self, openai_response: openai.ChatCompletion, model: str
    ) -> ModelResponse:
        choices = []
        
        for choice in openai_response.choices:
            # 假设choice.message.tool_calls存在
            tool_calls = getattr(choice.message, 'tool_calls', None)

            # 创建Message对象时，仅当tool_calls存在时才添加tool_calls属性
            if tool_calls:
                message = Message(content=choice.message.content, role=choice.message.role, tool_calls=tool_calls)
            else:
                message = Message(content=choice.message.content, role=choice.message.role)
            choices.append(
                Choices(
                    message=message, index=choice.index, finish_reason=choice.finish_reason
                )
            )

        # 把Usage对象里面的其他属性也添加到Usage对象中
        usage_dict = openai_response.usage.model_dump()
        usage = Usage(**usage_dict)

        response = ModelResponse(
            id=openai_response.id,
            choices=choices,
            created=openai_response.created,
            model=model,
            usage=usage,
        )
        return response
    
    def post_stream_processing(self, response, model=None):
        for chunk in response:
            data = chunk.json()
            # print(data)
            if isinstance(data, str):
                data = json.loads(data)
            if 'choices' in data:
                choices = data['choices']
                chunk_choices = []
                for choice in choices:
                    # 判断如果choice是StreamingChoices类型的对象，则直接添加到chunk_choices中
                    if isinstance(choice, StreamingChoices):
                        chunk_choices.append(choice)
                    else:
                        delta = choice.get("delta")
                        if choice.get("finish_reason") == "stop":
                            stream_choices = StreamingChoices(index=choice['index'], finish_reason="stop")
                            chunk_choices.append(stream_choices)
                        elif delta:
                            delta = choice.get("delta")
                            chunk_delta = Delta()
                            if "role" in choice['delta']:
                                chunk_delta.role = choice['delta']["role"]
                            if "content" in choice['delta']:
                                chunk_delta.content = choice['delta']["content"]
                            if "tool_calls" in choice['delta']:
                                chunk_delta.tool_calls = choice['delta']["tool_calls"]
                            
                            stream_choices = StreamingChoices(index=choice['index'], delta=chunk_delta, finish_reason=choice.get("finish_reason"))
                            # 遍历choices字典中的key, 如果在特定的列表中，则添加到stream_choices中
                            for key in choice.keys():
                                if key in ["content_filter_results", "content_filter_offsets", "logprobs"]:
                                    setattr(stream_choices, key, choice[key])
                            chunk_choices.append(stream_choices)
                    
            if "usage" in data:
                chunk_usage = Usage()
                if data["usage"]:
                    # 把usage字典中的数据添加到chunk_usage中
                    for key, value in data["usage"].items():
                        setattr(chunk_usage, key, value)
            
            chunk_response = ModelResponse(
                id=data["id"],
                choices=chunk_choices,
                created=data["created"],
                model=model,
                usage=chunk_usage if "usage" in data else None,
                stream=True,
                system_fingerprint=data.get("system_fingerprint") if "system_fingerprint" in data else None
            )
            yield chunk_response