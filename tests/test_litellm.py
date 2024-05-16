import sys
import os
import json
import pytest
import litellm
from dotenv import load_dotenv
load_dotenv()

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unionllm.providers.litellm import LiteLLMProvider, LiteLLMError
# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unionllm.providers.base_provider import BaseProvider

class TestLiteLLMProvider:
    @pytest.fixture(autouse=True)
    def setup_provider(self):
        # 从环境变量中导入API密钥
        self.provider = LiteLLMProvider()
    
    def test_completion(self):
        # provider = "openai"
        model = "gpt-3.5-turbo"
        messages = [{"content": "Introduce yourself briefly", "role": "user"}]
        kwargs = {
            # "provider": provider,
            "model": model,
            "temperature": 0.5,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        response = self.provider.completion(model=model, messages=messages, stream=False)
        print("response: ", response)