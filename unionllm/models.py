from typing import List, Optional

class ResponseModel:
    def __init__(self, prompt_tokens: List[str], completion_tokens: List[str], total_tokens: int, total_attempts: int, raw_response: Optional[dict] = None, usage: Optional[dict] = None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.total_attempts = total_attempts
        self.raw_response = raw_response
        self.usage = usage

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_attempts": self.total_attempts,
            "raw_response": self.raw_response,
            "usage": self.usage
        }

    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_response and "choices" in self.raw_response:
            for choice in self.raw_response["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    completions.append(choice["message"]["content"])
        return completions
