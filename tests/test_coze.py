import sys
import os
import unittest
import json

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cnlitellm.providers.coze import CozeAIProvider, CozeAIError


class TestFastGPTProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = CozeAIProvider(
            api_key="pat_hodyfcRog0CfCAfEMtKAueI1I8yD2tDuDBSrvSRoXhKm1xLCw0HDiePtEVgRFHzt"
        )

    def test_completion_non_stream_failure(self):

        # 定义模型和消息
        model = "7355829428481769477"
        messages = [{"content": "智商测试", "role": "user"}]

        response = self.provider.completion(model=model, messages=messages, stream=False, details=True)

        print("response: ", response)

        # 检查捕获的异常是否符合预期
        self.assertIsNotNone(response)

if __name__ == "__main__":
    unittest.main()
