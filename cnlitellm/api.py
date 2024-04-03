from .providers import base_provider
from .providers import zhipu, moonshot, minimax, qwen, tiangong, baichuan, wenxin, xunfei
from .providers.example_provider import ExampleProvider
from .exceptions import ProviderError

class CNLiteLLM:
    def __init__(self, provider: str):
        self.provider = provider.lower()
        print(f"Provider: {self.provider}")
        if self.provider == "zhipuai":
            self.provider_instance = zhipu.ZhipuAIProvider()
        elif self.provider == "moonshot":
            self.provider_instance = moonshot.MoonshotAIProvider()
        elif self.provider == "minimax":
            self.provider_instance = minimax.MinimaxAIProvider()
        elif self.provider == "qwen":
            self.provider_instance = qwen.QwenAIProvider()
        elif self.provider == "tiangong":
            self.provider_instance = tiangong.TianGongAIProvider()
        elif self.provider == "baichuan":
            self.provider_instance = baichuan.BaiChuanAIProvider()
        elif self.provider == "wenxin":
            self.provider_instance = wenxin.WenXinAIProvider()
        elif self.provider == "xunfei":
            self.provider_instance = xunfei.XunfeiAIProvider()
        else:
            raise ProviderError(f"Provider '{self.provider}' is not supported.")

    def completion(self, model: str, messages: list, **kwargs):
        return self.provider_instance.completion(model, messages, **kwargs)
