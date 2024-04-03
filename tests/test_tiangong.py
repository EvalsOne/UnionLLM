import json
import sys
import os
import unittest

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.providers.tiangong import TianGongAIProvider


class TestTianGongProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = TianGongAIProvider(
            api_key="a253c3d3bb00495a6f1f6572367e20b7",
            app_secret="df6b91ff0e400b8114122de882ff16597ab3539eb71afee5",
        )

    def test_completion(self):
        model = "SkyChat-MegaVerse"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        response = self.provider.completion(model=model, messages=messages)
        print("response: ", response)
        self.assertIsNotNone(response)

    # def test_completion(self):
    #     model = "SkyChat-MegaVerse"
    #     messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
    #     response = self.provider.completion(model=model, messages=messages, stream=True)
    #     for chunk in response:
    #         delta = chunk["choices"][0]["delta"]
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
    #         if "usage" in chunk:
    #             line["usage"] = {
    #                 "prompt_tokens": chunk["usage"]["prompt_tokens"],
    #                 "completion_tokens": chunk["usage"]["completion_tokens"],
    #                 "total_tokens": chunk["usage"]["total_tokens"],
    #             }
    #         print("line: ", line)
    #     self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
