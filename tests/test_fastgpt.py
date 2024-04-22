import sys
import os
import unittest
import json

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.providers.fastgpt import FastGPTProvider


class TestFastGPTProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = FastGPTProvider(
            api_key="fastgpt-iiCmNijSSN6Mvkr3LOv1tICdlVeVVhUlgSP7QUaiZMgXeXBhM4SzJPvyx4vVT"
        )

    # def test_completion_non_stream_success(self):
    #     # Test non-stream completion
    #     model = "fastgpt"
    #     messages = [{"content": "你好，今天天气怎么样？", "role": "user"}]
    #     response = self.provider.completion(model=model, messages=messages, stream=True, details=True)

    #     print("stream response: ", response)

    #     # 检查捕获的异常是否符合预期
    #     self.assertIsNotNone(response)

    def test_completion_non_stream_failure(self):

        # 定义模型和消息
        model = "fastgpt"
        messages = [{"content": "财商测试", "role": "user"}]

        response = self.provider.completion(model=model, messages=messages, stream=False, details=True)

        print("response: ", response)

        # 检查捕获的异常是否符合预期
        self.assertIsNotNone(response)

if __name__ == "__main__":
    unittest.main()
