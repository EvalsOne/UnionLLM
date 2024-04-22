import logging
from typing import Any, List
from .providers import base_provider
from .providers import zhipu, moonshot, minimax, qwen, tiangong, baichuan, wenxin, xunfei, dify, fastgpt, coze
from .exceptions import ProviderError

logger = logging.getLogger(__name__)

class CNLiteLLM:
    def __init__(self, provider: str, **kwargs):
        self.provider = provider.lower()
        if self.provider == "zhipuai":
            self.provider_instance = zhipu.ZhipuAIProvider(**kwargs)
        elif self.provider == "moonshot":
            self.provider_instance = moonshot.MoonshotAIProvider(**kwargs)
        elif self.provider == "minimax":
            self.provider_instance = minimax.MinimaxAIProvider(**kwargs)
        elif self.provider == "qwen":
            self.provider_instance = qwen.QwenAIProvider(**kwargs)
        elif self.provider == "tiangong":
            self.provider_instance = tiangong.TianGongAIProvider(**kwargs)
        elif self.provider == "baichuan":
            self.provider_instance = baichuan.BaiChuanAIProvider(**kwargs)
        elif self.provider == "wenxin":
            self.provider_instance = wenxin.WenXinAIProvider(**kwargs)
        elif self.provider == "xunfei":
            self.provider_instance = xunfei.XunfeiAIProvider(**kwargs)
        elif self.provider == "dify":
            self.provider_instance = dify.DifyAIProvider(**kwargs)
        elif self.provider == "fastgpt":
            self.provider_instance = fastgpt.FastGPTProvider(**kwargs)
        elif self.provider == "coze":
            self.provider_instance = coze.CozeAIProvider(**kwargs)
        else:
            raise ProviderError(f"Provider '{self.provider}' is not supported.")

    def completion(self, model: str, messages: List[str], **kwargs) -> Any:
        if not self.provider_instance:
            raise ProviderError(f"Provider '{self.provider}' is not initialized.")
        return self.provider_instance.completion(model, messages, **kwargs)