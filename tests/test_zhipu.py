import sys
import os
import unittest
import json

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.providers.zhipu import ZhipuAIProvider


class TestZhipuAIProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = ZhipuAIProvider(
            api_key="38dc046c622b6ec1b9bfa0413e6ca2ee.2Yn3uFr8j74pMOvK"
        )

    def test_completion(self):
        model = "GLM-4"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        response = self.provider.completion(model=model, messages=messages)
        for chunk in response:
            print("chunk: ", chunk)
            delta = chunk.choices[0].delta
            line = {
                "choices": [
                    {
                        "delta": {
                            "role": delta.role,
                            "content": delta.content,
                        }
                    }
                ]
            }
            if chunk.usage is not None:
                line["usage"] = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            print("line内容2: ", line)
        print("response: ", response)
        self.assertIsNotNone(response)

if __name__ == "__main__":
    unittest.main()
