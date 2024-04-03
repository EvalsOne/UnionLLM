import json
import sys
import os
import unittest

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.providers.moonshot import MoonshotAIProvider


class TestZhipuAIProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = MoonshotAIProvider(
            api_key="sk-EUZowzL6zYhlBi9lbeBHnaWV2yloiooI0Hec0WpiqCC1n0Ar",
            base_url="https://api.moonshot.cn/v1",
        )

    # def test_completion(self):
    #     model = "moonshot-v1-8k"
    #     messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
    #     response = self.provider.completion(model=model, messages=messages, stream=True)
    #     for chunk in response:
    #         new_chunk = json.loads(chunk)
    #         delta = new_chunk["choices"][0]["delta"]
    #         line = {
    #             "choices": [
    #                 {
    #                     "delta": {
    #                         "role": delta["role"],
    #                         "content": delta["content"],
    #                     }
    #                 }
    #             ]
    #         }
    #         if "usage" in new_chunk:
    #             usage_info = new_chunk["usage"]
    #             line["usage"] = {
    #                 "prompt_tokens": usage_info["prompt_tokens"],
    #                 "completion_tokens": usage_info["completion_tokens"],
    #                 "total_tokens": usage_info["total_tokens"],
    #             }
    #         print("line: ", line)
    #     self.assertIsNotNone(response)

    def test_completion(self):
        model = "moonshot-v1-8k"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        response = self.provider.completion(model=model, messages=messages, stream=True)
        for chunk in response:
            print("chunk: ", chunk)
            chunk_message = chunk.choices[0].delta
            line = {
                "choices": [
                    {
                        "delta": {
                            "role": chunk_message.role,
                            "content": chunk_message.content,
                        }
                    }
                ]
            }
            if (
                hasattr(chunk.choices[0], "usage")
                and chunk.choices[0].usage is not None
            ):
                line["usage"] = {
                    "prompt_tokens": chunk.choices[0].usage.prompt_tokens,
                    "completion_tokens": chunk.choices[0].usage.completion_tokens,
                    "total_tokens": chunk.choices[0].usage.total_tokens,
                }
        print("response: ", response)
        self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
