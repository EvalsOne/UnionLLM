from abc import ABC, abstractmethod
from ..models import ResponseModel
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, List
from unionllm.utils import ModelResponse, Message, Choices, Usage, Context, StreamingChoices, Delta

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
    
    def post_stream_processing(self, response, model=None):
        for chunk in response:
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
            # print(chunk_response)
            yield chunk_response