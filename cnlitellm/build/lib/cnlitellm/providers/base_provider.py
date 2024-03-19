from abc import ABC, abstractmethod
from ..models import ResponseModel
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, List

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