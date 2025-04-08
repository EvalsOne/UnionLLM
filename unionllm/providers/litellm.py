import json, time
import dashscope
from .base_provider import BaseProvider
import litellm
from litellm import completion

class LiteLLMError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class LiteLLMProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        litellm.drop_params = True
        # litellm.set_verbose = True
        pass

    def pre_processing(self, **kwargs):
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p", "n",
            "logprobs", "stream", "stop", "presence_penalty", "frequency_penalty",
            "best_of", "logit_bias", "api_key", "api_secret", "api_url", "provider", "api_version", "api_base", "extra_headers"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                # kwargs.pop(key) # This line is commented out to avoid removing the key from the kwargs
                pass
        return kwargs

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        result = completion(
            model=model, messages=messages, **new_kwargs
        )
        return self.post_stream_processing(result, model=model)

    def create_model_response_wrapper(self, result, model):
        return self.create_model_response(result, model=model)

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if 'provider' in kwargs:
                provider = kwargs['provider']
                kwargs.pop('provider')
            else:
                # 如果provider没有传入，则从model中提取provider
                provider = model.split('/')[0]
            if model is None or messages is None:
                raise LiteLLMError(
                    status_code=422, message=f"Missing model or messages"
                )
            # 检查消息格式
            message_check_result = self.check_prompt(provider, model, messages)       
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise LiteLLMError(
                    status_code=422, message=message_check_result['reason']
                )
            
            new_kwargs = self.pre_processing(**kwargs)
                        
            stream = kwargs.get("stream", False)

            if stream:
                if provider not in ['azure_ai']:
                    new_kwargs['stream_options'] = {"include_usage": True}
                return self.post_stream_processing_wrapper(model=model, messages=messages, **new_kwargs)
            else:
                result = completion(
                    model=model, messages=messages, **new_kwargs
                )
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise LiteLLMError(status_code=e.status_code, message=str(e))
            else:
                raise LiteLLMError(status_code=500, message=str(e))