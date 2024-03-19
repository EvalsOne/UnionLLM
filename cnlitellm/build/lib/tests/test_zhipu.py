import sys
import os
import unittest

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnlitellm.utils import create_model_response

from cnlitellm.providers.zhipu import ZhipuAIProvider

class TestZhipuAIProvider(unittest.TestCase):
    def setUp(self):
        # 请将'your_api_key'替换为您的Zhipu AI API密钥
        self.provider = ZhipuAIProvider(api_key='38dc046c622b6ec1b9bfa0413e6ca2ee.2Yn3uFr8j74pMOvK')

    def test_completion(self):
        model = 'your_model_name'
        messages = [{'content': '你好，今天天气怎么样？', 'role': 'user'}]
        response = self.provider.completion(model='GLM-4', messages=messages)

        model_response = create_model_response(response, model='GLM-4')
        print(model_response.model_dump())

        # print(response)
        # print(response.get_completions())
        # self.assertIsNotNone(response)
        # self.assertIsInstance(response.get_completions(), list)

if __name__ == '__main__':
    unittest.main()
