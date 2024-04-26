from abc import ABC, abstractmethod
from ..models import ResponseModel
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, List
from cnlitellm.utils import Message, Choices, Usage, ModelResponse

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass

@dataclass
class BaseProvider(ABC):
    cls: str
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None

    @abstractmethod
    def completion(self, model: str, messages: list) -> ResponseModel:
        pass

    def create_model_response(self, openai_response: Any, model: str) -> ResponseModel:
        print("openai_response: ", openai_response)
        choices = []

        for choice in openai_response.choices:
            print("choice.message.content: ", choice.message.content)
            message = Message(content=choice.message.content, role=choice.message.role)
            print("message: ", message)
            choices.append(Choices(message=message, index=choice.index, finish_reason=choice.finish_reason))

        usage = Usage(
            prompt_tokens=openai_response.usage.prompt_tokens,
            completion_tokens=openai_response.usage.completion_tokens,
            total_tokens=openai_response.usage.total_tokens
        )
        response =  ModelResponse(
            id=openai_response.id,
            choices=choices,
            created=openai_response.created,
            model=model,
            usage=usage
        )
        pass