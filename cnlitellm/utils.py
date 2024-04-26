import json
from abc import ABC, abstractmethod
from openai import OpenAIError as OriginalError
from typing import List, Union, Optional

import uuid, time, openai, random, requests
from openai._models import BaseModel as OpenAIObject

# from .exceptions import (
#     AuthenticationError,
#     BadRequestError,
#     RateLimitError,
#     ServiceUnavailableError,
#     OpenAIError,
#     ContextWindowExceededError,
#     Timeout,
#     APIConnectionError,
#     APIError,
#     BudgetExceededError
# )

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
        super().__init__(
            id=id,
            choices=choices,
            created=created,
            model=model,
            object=object,
            system_fingerprint=system_fingerprint,
            usage=usage,
            **params
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

class ResponseModelInterface:
    def post_stream_processing(self, response, model=None):
        for chunk in response:
            print("chunk: ", chunk)
            data = chunk.json()
            if isinstance(data, str):
                data = json.loads(data)
            if 'choices' in data:
                choices = data['choices']
                chunk_choices = []
                for choice in choices:
                    delta = choice.get("delta")
                    if delta:
                        chunk_delta = Delta()
                        if "role" in choice['delta']:
                            chunk_delta.role = choice['delta']["role"]
                        if "content" in choice['delta']:
                            chunk_delta.content = choice['delta']["content"]
                        chunk_choices.append(StreamingChoices(index=choice['index'], delta=chunk_delta))

            if "usage" in data:
                chunk_usage = Usage()
                if data["usage"]:
                    if "prompt_tokens" in data["usage"]:
                        chunk_usage.prompt_tokens = data["usage"]["prompt_tokens"]
                    if "completion_tokens" in data["usage"]:
                        chunk_usage.completion_tokens = data["usage"]["completion_tokens"]
                    if "total_tokens" in data["usage"]:
                        chunk_usage.total_tokens = data["usage"]["total_tokens"]
            
            chunk_response = ModelResponse(
                id=data["id"],
                choices=chunk_choices,
                created=data["created"],
                model=model,
                usage=chunk_usage if "usage" in data else None,
                stream=True
            )
            yield chunk_response

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        payload = json.dumps({"model": model, "messages": messages, **new_kwargs})
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint_url, headers=headers, data=payload)
        for line in response.iter_lines():
            print("line: ", line)
            if line:
                new_line = line.decode("utf-8").replace("data: ", "")
                if new_line == "[DONE]":
                    break
                data = json.loads(new_line)
                chunk_choices = []
                for choice in data["choices"]:
                    chunk_delta = Delta()
                    delta = choice.get("delta")
                    if delta:
                        if "role" in choice['delta']:
                            chunk_delta.role = choice['delta']["role"]
                        if "content" in choice['delta']:
                            chunk_delta.content = choice['delta']["content"]
                        chunk_choices.append(StreamingChoices(index=choice['index'], delta=chunk_delta))

                if "usage" in data:
                    chunk_usage = Usage()
                    if "prompt_tokens" in data["usage"]:
                        chunk_usage.prompt_tokens = data["usage"]["prompt_tokens"]
                    if "completion_tokens" in data["usage"]:
                        chunk_usage.completion_tokens = data["usage"]["completion_tokens"]
                    if "total_tokens" in data["usage"]:
                        chunk_usage.total_tokens = data["usage"]["total_tokens"]
                
                chunk_response = ModelResponse(
                    id=data["id"],
                    choices=chunk_choices,
                    created=data["created"],
                    model=model,
                    usage=chunk_usage if "usage" in data else None,
                    stream=True
                )
                yield chunk_response


    def create_model_response(
        self, openai_response: openai.ChatCompletion, model: str
    ) -> ModelResponse:
        # print("openai_response: ", openai_response)
        choices = []

        for choice in openai_response.choices:
            message = Message(content=choice.message.content, role=choice.message.role)
            choices.append(
                Choices(
                    message=message, index=choice.index, finish_reason=choice.finish_reason
                )
            )

        usage = Usage(
            prompt_tokens=openai_response.usage.prompt_tokens,
            completion_tokens=openai_response.usage.completion_tokens,
            total_tokens=openai_response.usage.total_tokens,
        )
        response = ModelResponse(
            id=openai_response.id,
            choices=choices,
            created=openai_response.created,
            model=model,
            usage=usage,
        )
        return response

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