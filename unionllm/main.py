import logging
import asyncio
from functools import partial

from typing import Any, List
from .providers import zhipu, moonshot, minimax, qwen, tiangong, baichuan, wenxin, xunfei, xunfei_http, dify, fastgpt, coze, litellm, lingyi, stepfun, doubao, deepseek
from .exceptions import ProviderError
# from litellm import completion as litellm_completion
from typing import Optional

logger = logging.getLogger(__name__)

class UnionLLM:
    def __init__(self, provider: Optional[str] = None, **kwargs):
        self.provider = provider.lower() if provider else None
        self.litellm_call_type = None
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
        elif self.provider == "xunfei_http":
            self.provider_instance = xunfei_http.XunfeiHTTPProvider(**kwargs)
        elif self.provider == "dify":
            self.provider_instance = dify.DifyAIProvider(**kwargs)
        elif self.provider == "fastgpt":
            self.provider_instance = fastgpt.FastGPTProvider(**kwargs)
        elif self.provider == "coze":
            self.provider_instance = coze.CozeAIProvider(**kwargs)
        elif self.provider == "lingyi":
            self.provider_instance = lingyi.LingyiAIProvider(**kwargs)
        elif self.provider == "stepfun":
            self.provider_instance = stepfun.StepfunAIProvider(**kwargs)
        elif self.provider == "doubao":
            self.provider_instance = doubao.DouBaoAIProvider(**kwargs)
        elif self.provider == "deepseek":
            self.provider_instance = deepseek.DeepSeekAIProvider(**kwargs)
        elif self.provider:
            if_litellm_support, support_type = self.check_litellm_providers(provider=self.provider)
            if if_litellm_support:
                self.provider_instance = litellm.LiteLLMProvider(**kwargs)
                if support_type == 1:
                    self.litellm_call_type = 1
                elif support_type == 2:
                    self.litellm_call_type = 2
            else:
                raise ProviderError(f"Provider '{self.provider}' is not supported.")
        else:
            self.provider_instance = litellm.LiteLLMProvider(**kwargs)

    def completion(self, model: str, messages: List[str], **kwargs) -> Any:
        if not self.provider_instance:
            raise ProviderError(f"Provider '{self.provider}' is not initialized.")
        if self.litellm_call_type:
            if self.litellm_call_type == 1:
                # Jugde whether the model starts with self.provider, if not, add it
                if not model.startswith(self.provider+"/"):
                    model = f"{self.provider}/{model}"
                    
                return self.provider_instance.completion(model, messages, **kwargs)
            elif self.litellm_call_type == 2:
                return self.provider_instance.completion(model, messages, **kwargs)
        else:
            return self.provider_instance.completion(model, messages, **kwargs)
        
    async def acompletion(self, model: str, messages: List[str], **kwargs) -> Any:
        loop = asyncio.get_event_loop()
        if not self.provider_instance:
            raise ProviderError(f"Provider '{self.provider}' is not initialized.")
        
        if self.litellm_call_type:
            if self.litellm_call_type == 1:
                if not model.startswith(self.provider + "/"):
                    model = f"{self.provider}/{model}"
                func = partial(self.provider_instance.completion, model, messages, **kwargs)
                return await loop.run_in_executor(None, func)
            elif self.litellm_call_type == 2:
                func = partial(self.provider_instance.completion, model, messages, **kwargs)
                return await loop.run_in_executor(None, func)
        else:
            func = partial(self.provider_instance.completion, model, messages, **kwargs)
            return await loop.run_in_executor(None, func)

    def check_litellm_providers(self, provider: str) -> bool:
        # Judge whether the provider is supported by LiteLLM, and if provider name should be added to the model name
        if provider in ['azure', 'anthropic', 'sagemaker', 'bedrock', 'vertex_ai', 'vertex_ai_beta', 'palm', 'gemini', 'mistral', 'cloudflare', 'huggingface', 'replicate', 'together_ai', 'openrouter', 'baseten', 'nlp_cloud', 'petals', 'ollama', 'perplexity', 'groq', 'anyscale', 'watsonx', 'voyage', 'xinference']:
            # provider name should be added to the model name
            return True, 1
        elif provider in ['openai', 'cohere', 'ai21', 'deepseek', 'deepinfra', 'ai21', 'alpha_alpha']:
            # provider name should not be added to the model name
            return True, 2
        else:
            return False, 0
        
# This is the new function to simplify calling
def unionchat(model: str, messages: List[dict], **kwargs) -> Any:
    client = UnionLLM(**kwargs)
    return client.completion(model=model, messages=messages, **kwargs)