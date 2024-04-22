import sys
import os
import unittest
import json
from unittest.mock import patch, MagicMock

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.providers.zhipu import ZhipuAIProvider, ZhiPuOpenAIError

class TestZhipuAIProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = ZhipuAIProvider(api_key="38dc046c622b6ec1b9bfa0413e6ca2ee.2Yn3uFr8j74pMOvK")

    def test_completion_non_stream_success(self):
        # Test non-stream completion
        model = "GLM-4"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
        response = self.provider.completion(model=model, messages=messages, stream=True)

        print("stream response: ", response)

        # 检查捕获的异常是否符合预期
        self.assertIsNotNone(response)

    def test_completion_non_stream_failure(self):

        # 定义模型和消息
        model = "GLM-4"
        messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]

        response = self.provider.completion(model=model, messages=messages, stream=False)

        print("response: ", response)

        # 检查捕获的异常是否符合预期
        self.assertIsNotNone(response)

if __name__ == "__main__":
    unittest.main()
