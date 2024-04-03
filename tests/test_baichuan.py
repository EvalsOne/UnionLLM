import sys
import os
import unittest

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.providers.baichuan import BaiChuanAIProvider, BaiChuanOpenAIError


class TestBaiChuanProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = BaiChuanAIProvider(
            api_key="sk-42c0206e207bc39c2367964e76da72ff",
        )

    def test_completion(self):
        model = "Baichuan2-Turbo"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        try:
            response = self.provider.completion(model=model, messages=messages)
            print("response: ", response)
            self.assertIsNotNone(response)
        except BaiChuanOpenAIError as e:
            raise e

    # def test_completion(self):
    #     model = "Baichuan2-Turbo"
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
    #             usage_info = chunk["usage"]
    #             line["usage"] = {
    #                 "total_tokens": usage_info["total_tokens"],
    #                 "prompt_tokens": usage_info["prompt_tokens"],
    #                 "completion_tokens": usage_info["completion_tokens"],
    #             }
    #         print("line: ", line)
    #     self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
