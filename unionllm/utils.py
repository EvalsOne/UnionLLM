import json
from abc import ABC, abstractmethod
from openai import OpenAIError as OriginalError
from typing import List, Union, Optional

import uuid, time, openai, random, requests
from openai._models import BaseModel as OpenAIObject

class Message(OpenAIObject):
    def __init__(
        self,
        content="default",
        role="assistant",
        logprobs=None,
        function_call=None,
        **params
    ):
        super(Message, self).__init__(**params)
        self.content = content
        self.role = role
        self._logprobs = logprobs
        if function_call:
            self.function_call = function_call

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Delta(OpenAIObject):
    def __init__(self, content=None, role=None, **params):
        super(Delta, self).__init__(**params)
        if content is not None:
            self.content = content
        if role:
            self.role = role

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


class Choices(OpenAIObject):
    def __init__(self, finish_reason=None, index=0, message=None, **params):
        super(Choices, self).__init__(**params)
        if finish_reason:
            self.finish_reason = map_finish_reason(finish_reason)
        else:
            self.finish_reason = "stop"
        self.index = index
        if message is None:
            self.message = Message(content=None)
        else:
            self.message = message

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

class Usage(OpenAIObject):
    def __init__(
        self, prompt_tokens=None, completion_tokens=None, total_tokens=None, **params
    ):
        super(Usage, self).__init__(**params)
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens
        if completion_tokens:
            self.completion_tokens = completion_tokens
        if total_tokens:
            self.total_tokens = total_tokens
        
    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

class Context(OpenAIObject):
    def __init__(self, id=None, content=None, score=None, **params):
        super(Context, self).__init__(**params)
        if id:
            self.id = id
        if content:
            self.content = content
        if score:
            self.score = score

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

class StreamingChoices(OpenAIObject):
    def __init__(
        self, finish_reason=None, index=0, delta: Optional[Delta] = None, **params
    ):
        super(StreamingChoices, self).__init__(**params)

        self.finish_reason = finish_reason
        self.index = index
        if delta:
            self.delta = delta
        else:
            self.delta = Delta()

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)


def _generate_id():  # private helper function
    return "chatcmpl-" + str(uuid.uuid4())


def map_finish_reason(
    finish_reason: str,
):  # openai supports 5 stop sequences - 'stop', 'length', 'function_call', 'content_filter', 'null'
    # anthropic mapping
    if finish_reason == "stop_sequence":
        return "stop"
    return finish_reason


class ModelResponse(OpenAIObject):
    id: str
    """A unique identifier for the completion."""

    choices: List[Union[Choices, StreamingChoices]]
    """The list of completion choices the model generated for the input prompt."""

    created: int
    """The Unix timestamp (in seconds) of when the completion was created."""

    model: Optional[str] = None
    """The model used for completion."""

    object: str
    """The object type, which is always "text_completion" """

    system_fingerprint: Optional[str] = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: Optional[Usage] = None
    """Usage statistics for the completion request."""  

    _hidden_params: dict = {}

    def __init__(
        self,
        id=None,
        choices=None,
        created=None,
        model=None,
        object=None,
        system_fingerprint=None,
        usage=None,
        conversation_id=None,
        stream=False,
        response_ms=None,
        hidden_params=None,
        **params
    ):
        if stream:
            object = "chat.completion.chunk"
        else:
            object = "chat.completion"
        if id is None:
            id = _generate_id()
        if created is None:
            created = int(time.time())
        if response_ms:
            _response_ms = response_ms
        if not usage:
            usage = Usage()
        
        if hidden_params:
            self._hidden_params = hidden_params

        filtered_params = {}      
        if conversation_id:
            filtered_params['conversation_id'] = conversation_id

        # 只有当 conversation_id 不为 None 时才添加到 filtered_params
        if 'conversation_id'in params and conversation_id is not None:
            filtered_params['conversation_id'] = conversation_id
        
        super().__init__(
            id=id,
            choices=choices,
            created=created,
            model=model,
            object=object,
            system_fingerprint=system_fingerprint,
            usage=usage,
            **filtered_params
        )

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

class EmbeddingResponse(OpenAIObject):
    def __init__(
        self,
        id=None,
        choices=None,
        created=None,
        model=None,
        usage=None,
        stream=False,
        response_ms=None,
        **params
    ):
        self.object = "list"
        if response_ms:
            self._response_ms = response_ms
        else:
            self._response_ms = None
        self.data = []
        self.model = model

    def to_dict_recursive(self):
        d = super().to_dict_recursive()
        return d

def generate_unique_uid():
    # Get the current time in microseconds
    microseconds = int(time.time() * 1000000)
    
    # Get a random number between 0 and 0xFFFF
    rand_num = random.randint(0, 0xFFFF)
    
    # Combine them to get a unique ID
    unique_uid = f"{microseconds:x}{rand_num:04x}"
    
    return unique_uid

def check_object_input_support(provider):
    not_supported_providers = ["wenxin", "baichuan", "minimax", "xunfei", "tiangong", "lingyi", "fastgpt", "doubao", "moonshot"]
    if provider in not_supported_providers:
        return "NONE"
    elif provider == "coze":
        return "PARTIAL"
    else:
        return "FULL"

def check_file_input_support(provider, model):
    # supported_providers = ["openai", "anthropic", "claude"]
    if provider == "coze":
        return "PARTIAL"
    # elif provider in supported_providers:
    #     return "PARTIAL"
    else:
        return "PARTIAL"

def check_vision_input_support(provider, model):
    # supported_providers = ['zhipuai']
    if provider == "coze":
        return "PARTIAL"
    elif provider == "dify":
        return "FULL"
    else:
        # 默认支持，不做限制
        return "FULL"
    
    # supported_models = [
    #     {"zhipuai": ["glm-4v", "glm-4v-plus"]}
    # ]
    
    # for entry in supported_models:
    #     if provider in entry:
    #         if model in entry[provider]:
    #             return "FULL"    
    # return "NONE"

def check_video_input_support(provider, model):
    # supported_providers = ['zhipuai']
    if provider == "coze":
        return "PARTIAL"
    # elif provider not in supported_providers:
    #     return "NONE"
    else:
        # 默认支持，不做限制
        return "FULL"

def reformat_object_content(messages, reformat=False, reformat_image=False, reformat_file=False, reformat_video=False):
    formatted_messages = []
    for message in messages:
        # 保留原始消息的所有字段（包括tool_call_id和name）
        new_formatted_message = {k: v for k, v in message.items() if k != 'content'}
        
        if isinstance(message.get("content"), list):
            new_formatted_message["content"] = ""
            message_contents = message.get("content")            
            for content in message_contents:
                if not isinstance(content, dict):
                    return False
                content_type = content.get("type")
                if content_type == "text":
                    if isinstance(content.get("text"), str):
                        new_formatted_message["content"] += content.get("text")
                elif content_type in ["image_url","image"]:
                    if not reformat_image:
                        formatted_messages.append(message)
                        continue
                    elif content.get("image_url") and content.get("image_url").get("url"):
                        new_formatted_message["content"] += f"![image]({content.get('image_url').get('url')})"
                    else:
                        return False
                elif content_type == "video_url":
                    if not reformat_video:
                        formatted_messages.append(message)
                        continue
                    elif content.get("video_url") and content.get("video_url").get("url"):
                        new_formatted_message["content"] += f"![video]({content.get('video_url').get('url')})"
                    else:
                        return False
                elif content_type == "file_url":
                    if not reformat_file:
                        formatted_messages.append(message)
                        continue
                    elif content.get("file_url") and content.get("file_url").get("url"):
                        new_formatted_message["content"] += f"[file]({content.get('file_url').get('url')})"
                    else:
                        return False
            formatted_messages.append(new_formatted_message)   
        else:
            formatted_messages.append(message)
    return formatted_messages

class Function(OpenAIObject):
    def __init__(
        self,
        name=None,
        description=None,
        parameters=None,
        **params
    ):
        super(Function, self).__init__(**params)
        if name:
            self.name = name
        if description:
            self.description = description
        if parameters:
            self.parameters = parameters

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

class ChatCompletionMessageToolCall(OpenAIObject):
    def __init__(
        self,
        id=None,
        type=None,
        function=None,
        **params
    ):
        super(ChatCompletionMessageToolCall, self).__init__(**params)
        if id:
            self.id = id
        if type:
            self.type = type
        if function:
            self.function = function

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)