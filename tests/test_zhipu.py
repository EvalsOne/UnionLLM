import sys
import os
import unittest

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.utils import create_model_response, ModelResponse
from cnlitellm.providers.zhipu import ZhipuAIProvider


class TestZhipuAIProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = ZhipuAIProvider(
            api_key="e3833c6712b25eb4d89babd15c8134f5.Do904yhEYxDjewxI"
        )

    def test_completion(self):
        model = "GLM-4"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        response = self.provider.completion(model=model, messages=messages)
        self.assertIsNotNone(response)

    # def test_completion(self):
    #     model = "GLM-4"
    #     messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
    #     response = self.provider.completion(model=model, messages=messages, stream=True)
    #     print("response: ", response)
    #     self.assertIsNotNone(response)


if __name__ == "__main__":
    unittest.main()
