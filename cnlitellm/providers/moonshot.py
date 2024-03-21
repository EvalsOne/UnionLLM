from .base_provider import BaseProvider
from cnlitellm.utils import create_model_response
from openai import OpenAI


class MoonshotAIProvider(BaseProvider):
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url

    def completion(self, model: str, messages: list, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        stream = kwargs.get("stream", False)

        if stream:

            def generate_stream():
                response = self.client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                collected_messages = []
                for chunk in response:
                    chunk_message = chunk.choices[0].delta
                    if not chunk_message.content:
                        continue
                    collected_messages.append(chunk_message)
                return collected_messages

            return generate_stream()

        else:
            result = self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return create_model_response(result, model=model)
