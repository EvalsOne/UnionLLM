import sys
import os
import json
import pytest
from dotenv import load_dotenv
load_dotenv()

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unionllm.providers.xunfei import XunfeiAIProvider, XunfeiOpenAIError

class TestXunfeiAIProvider:
    @pytest.fixture(autouse=True)
    def setup_provider(self):
        # 从环境变量中导入API密钥
        app_id=os.getenv("XUNFEI_APP_ID")
        self.provider = XunfeiAIProvider(app_id=os.getenv("XUNFEI_APP_ID"), api_key=os.getenv("XUNFEI_API_KEY"), api_secret=os.getenv("XUNFEI_API_SECRET"))

    def test_completion_non_stream(self):
        # 定义模型和消息
        model = "generalv3"
        messages = [{"content": "introduce yourself briefly.", "role": "user"}]
        try:
            response = self.provider.completion(model=model, messages=messages, stream=False)
            print(response)
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
