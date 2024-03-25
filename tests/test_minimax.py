import json
import sys
import os
import unittest

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.utils import create_minimax_model_response
from cnlitellm.providers.minimax import MinimaxAIProvider


class TestMinimaxProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = MinimaxAIProvider(
            api_key="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJ0aWFtbyIsIlVzZXJOYW1lIjoidGlhbW8iLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTc2ODUzNjQzNzMxNTA4MDQxMCIsIlBob25lIjoiMTk1MTAzNjA5MzQiLCJHcm91cElEIjoiMTc2ODUzNjQzNzMwNjY5MTgwMiIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IiIsIkNyZWF0ZVRpbWUiOiIyMDI0LTAzLTIxIDE1OjAyOjI0IiwiaXNzIjoibWluaW1heCJ9.LVXY3zEwzpNQ7yei--ZP7JtG7Xs_vfeADmGd_JcooTOlzUJ3PmDQFssDkY-8DKj_Pm9dthI4Dd3bI2ELdyYbmcA_VU-qNWBfqtB0Gtfb_azdzAdTbkftUxgSm6t44gHZCgymKfblIHk-hdAISJ5YeE1ETdFww9gGREnY8T54Zhe8OsCdy7pjFXktlAEFWWPNBfumcUORSHtiIX3SAJRHTafgMWuv0tAxvIO1Vzz0OoWIBdpYoslHJx1aIOCwuaohsuXmfUVdmiWZyQJKvxl5L6ix9IsNQW9e1m9gJ_1M3UnJh9i6I2UC3gkNLwm9zYFqG2v20nWLtchu9vBShEpFQA",
        )

    def test_completion(self):
        model = "abab6-chat"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        response = self.provider.completion(model=model, messages=messages)
        print("response: ", response)
        self.assertIsNotNone(response)

    # def test_completion(self):
    #     model = "abab6-chat"
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
    #                 "total_tokens": chunk["usage"]["total_tokens"],
    #             }
    #         print("line: ", line)
    #     self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
