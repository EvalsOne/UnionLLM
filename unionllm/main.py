import logging
import asyncio
import os
from functools import partial

from typing import Any, List, Optional
from .providers import zhipu, moonshot, xai, minimax, qwen, tiangong, baichuan, wenxin, xunfei, xunfei_http, dify, fastgpt, coze, lingyi, stepfun, doubao, deepseek, gemini, azure, anthropic, openai_native
from .exceptions import ProviderError

logger = logging.getLogger(__name__)

# Vendors that speak the OpenAI Chat Completions API: called through the openai
# SDK at their own base URL (they used to go through LiteLLM). Values are the
# default base URL and the environment variable holding the API key.
OPENAI_COMPATIBLE_PROVIDERS = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}
# Reserved Qwen models served through DashScope's OpenAI-compatible mode.
QWEN_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _litellm_provider(route, **kwargs):
    """The LiteLLM route, imported on first use; LiteLLM is an optional extra
    (``unionllm[litellm]``) and a clear error says so when it is missing."""
    from .providers import litellm as litellm_route

    return litellm_route.LiteLLMProvider(route=route, **kwargs)

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
            # 根据model判断是否需要使用dashscope的api, 否则使用openai的兼容api
            model = kwargs.get("model")
            # 预留特殊模型走 DashScope 的 OpenAI 兼容接口
            if model is not None and model not in ["qwen-special-model"]:
                # 使用 DashScope 兼容模式（需要 dashscope key）
                self.provider_instance = qwen.QwenAIProvider(**kwargs)
            else:
                self.provider_instance = openai_native.OpenAICompatibleProvider(
                    default_base_url=QWEN_COMPATIBLE_BASE_URL,
                    api_key_env="DASHSCOPE_API_KEY",
                    **kwargs,
                )

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
        # elif self.provider == "xai":
        #     self.provider_instance = xai.XAIHTTPProvider(**kwargs)
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
        elif self.provider == "gemini": 
            self.provider_instance = gemini.GeminiAIProvider(**kwargs)    
        elif self.provider == "anthropic":
            use_litellm_anthropic = kwargs.get("use_litellm_anthropic")
            if use_litellm_anthropic is None:
                use_litellm_anthropic = os.getenv("UNIONLLM_USE_LITELLM_ANTHROPIC", "").lower() in ("1", "true", "yes")
            if use_litellm_anthropic:
                self.provider_instance = _litellm_provider(self.provider, **kwargs)
                self.litellm_call_type = 1
            else:
                self.provider_instance = anthropic.AnthropicAIProvider(**kwargs)
        elif self.provider == "azure":
            # Claude 部署走 Azure 原生（Anthropic Foundry），其余模型（GPT、o 系列等）走
            # openai SDK 的 AzureOpenAI。按每次调用的模型选择：unionchat 等调用方构造
            # UnionLLM(provider="azure") 时不带 model，只在 completion() 里给出。
            self._azure_kwargs = kwargs
            self._azure_instances = {}
            model = kwargs.get("model")
            self.provider_instance = self._azure_instance(model) if model else None
        elif self.provider == "openai":
            self.provider_instance = openai_native.OpenAINativeProvider(**kwargs)
        elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            base_url, key_env = OPENAI_COMPATIBLE_PROVIDERS[self.provider]
            self.provider_instance = openai_native.OpenAICompatibleProvider(
                default_base_url=base_url,
                api_key_env=key_env,
                model_prefix=self.provider,
                **kwargs,
            )
        elif self.provider:
            if_litellm_support, support_type = self.check_litellm_providers(provider=self.provider)
            if if_litellm_support:
                self.provider_instance = _litellm_provider(self.provider, **kwargs)
                if support_type == 1:
                    self.litellm_call_type = 1
                elif support_type == 2:
                    self.litellm_call_type = 2
            else:
                raise ProviderError(f"Provider '{self.provider}' is not supported.")
        elif kwargs.get("custom_llm_provider") == "openai":
            # A user-defined OpenAI-compatible endpoint (api_base/base_url).
            self.provider_instance = openai_native.OpenAICompatibleProvider(
                require_api_key=False,
                **kwargs,
            )
        else:
            self.provider_instance = _litellm_provider(None, **kwargs)

    def _azure_instance(self, model: Optional[str]):
        """The Azure provider for this model (Claude or not), built once per kind
        from the constructor's arguments."""
        claude = azure.is_claude_deployment(model)
        if claude not in self._azure_instances:
            provider_class = azure.AzureAnthropicProvider if claude else azure.AzureOpenAIProvider
            self._azure_instances[claude] = provider_class(**self._azure_kwargs)
        return self._azure_instances[claude]

    def completion(self, model: str, messages: List[str], **kwargs) -> Any:
        if self.provider == "azure":
            return self._azure_instance(model).completion(model, messages, **kwargs)
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
        if self.provider == "azure":
            func = partial(self._azure_instance(model).completion, model, messages, **kwargs)
            return await loop.run_in_executor(None, func)
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
        if provider in ['azure', 'azure_ai', 'anthropic', 'deepseek', 'sagemaker', 'bedrock', 'vertex_ai', 'vertex_ai_beta', 'palm', 'gemini', 'mistral', 'cloudflare', 'huggingface', 'replicate', 'together_ai', 'openrouter', 'baseten', 'nlp_cloud', 'petals', 'ollama', 'perplexity', 'groq', 'anyscale', 'watsonx', 'voyage', 'xinference']:
            # provider name should be added to the model name
            return True, 1
        elif provider in ['openai', 'cohere', 'ai21', 'deepinfra', 'ai21', 'alpha_alpha']:
            # provider name should not be added to the model name
            return True, 2
        else:
            return False, 0
        
# This is the new function to simplify calling
def unionchat(model: str, messages: List[dict], **kwargs) -> Any:
    client = UnionLLM(**kwargs)
    return client.completion(model=model, messages=messages, **kwargs)
