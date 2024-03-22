from .base_provider import BaseProvider
from cnlitellm.utils import create_minimax_model_response
import requests
import json


class MinimaxAIProvider(BaseProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def completion(self, model: str, messages: list, **kwargs):
        if "api_key" in kwargs:
            self.api_key = kwargs.get("api_key")
            kwargs.pop("api_key")

        stream = kwargs.get("stream", False)

        if stream:

            def generate_stream():
                url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
                payload = json.dumps({"model": model, "messages": messages, **kwargs})
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                result = requests.post(url, headers=headers, data=payload)
                lines = [
                    line.strip() for line in result.text.split("\n") if line.strip()
                ]
                parsed_data = []
                for line in lines:
                    line = line.replace("data: ", "")
                    parsed_data.append(json.loads(line))
                for index, chunk in enumerate(parsed_data):
                    if index == len(parsed_data) - 1:
                        chunk_message = chunk["choices"][0]["message"]
                    else:
                        chunk_message = chunk["choices"][0]["delta"]
                    chunk_line = {
                        "choices": [
                            {
                                "delta": {
                                    "role": chunk_message["role"],
                                    "content": chunk_message["content"],
                                }
                            }
                        ]
                    }
                    if hasattr(chunk, "usage") and chunk.choices[0].usage is not None:
                        chunk_line["usage"] = {
                            "total_tokens": chunk["usage"]["total_tokens"],
                        }
                yield chunk_line

            return generate_stream()

        else:
            url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
            payload = json.dumps({"model": model, "messages": messages, **kwargs})
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            result = requests.post(url, headers=headers, data=payload)
            return create_minimax_model_response(result, model=model)
