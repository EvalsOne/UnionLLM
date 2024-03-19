from .base_provider import BaseProvider
from ..models import ResponseModel
from ..utils import preprocess_prompt, convert_parameters

class ExampleProvider(BaseProvider):
    def completion(self, model: str, messages: list) -> ResponseModel:
        # 对每条消息进行预处理
        preprocessed_messages = [preprocess_prompt(msg) for msg in messages]
        
        # 转换参数为提供商所需的格式
        provider_parameters = convert_parameters({"model": model, "messages": preprocessed_messages})
        
        # 调用具体API（使用假数据作为示例）
        return ResponseModel(
            prompt_tokens=provider_parameters["messages"],
            completion_tokens=["天气不错。"],
            total_tokens=5,
            total_attempts=1
        )
