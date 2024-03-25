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

    def test_completion(self):
        model = "moonshot-v1-8k"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        response = self.provider.completion(model=model, messages=messages)
        print("response: ", response)
        self.assertIsNotNone(response)

    # def test_completion(self):
    #     model = "moonshot-v1-8k"
    #     messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
    #     response = self.provider.completion(model=model, messages=messages, stream=True)
    #     print("response: ", response)
    #     self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
