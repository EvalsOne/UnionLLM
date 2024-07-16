import pytest
import os, sys
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

import os

# baichuan
os.environ['BAICHUAN_API_KEY'] = 'your-baichuan-api-key'

# zhipu
os.environ['ZHIPU_API_KEY'] = 'your-zhipu-api-key'

# moonshot
os.environ['MOONSHOT_API_KEY'] = 'your-moonshot-api-key'

# qwen
os.environ['DASHSCOPE_API_KEY'] = 'your-dashscope-api-key'

# minimax
os.environ['MINIMAX_API_KEY'] = 'your-minimax-api-key'

# coze
os.environ['COZE_API_KEY'] = 'your-coze-api-key'
os.environ['COZE_BOT_ID'] = 'your-coze-bot-id'

# dify
os.environ['DIFY_API_KEY'] = 'your-dify-api-key'

# fastgpt
os.environ['FASTGPT_API_KEY'] = 'your-fastgpt-api-key'

# wenxin
os.environ['ERNIE_CLIENT_ID'] = 'your-ernie-client-id'
os.environ['ERNIE_CLIENT_SECRET'] = 'your-ernie-client-secret'

# tiangong
os.environ['TIANGONG_APP_KEY'] = 'your-tiangong-app-key'
os.environ['TIANGONG_APP_SECRET'] = 'your-tiangong-app-secret'

# xunfei
os.environ['XUNFEI_APP_ID'] = 'your-xunfei-app-id'
os.environ['XUNFEI_API_KEY'] = 'your-xunfei-api-key'
os.environ['XUNFEI_API_SECRET'] = 'your-xunfei-api-secret'

# openai
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'

# azure
os.environ['AZURE_API_KEY'] = 'your-azure-api-key'
os.environ['AZURE_API_BASE'] = 'your-azure-api-base'
os.environ['AZURE_API_VERSION'] = 'your-azure-api-version'

# groq
os.environ['GROQ_API_KEY'] = 'your-groq-api-key'

# bedrock
os.environ['AWS_ACCESS_KEY_ID'] = 'your-aws-access-key-id'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your-aws-secret-access-key'
os.environ['AWS_REGION_NAME'] = 'your-aws-region-name'

# mistral
os.environ['MISTRAL_API_KEY'] = 'your-mistral-api-key'

# cohere
os.environ['COHERE_API_KEY'] = 'your-cohere-api-key'

# gemini
os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key'

# openai
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'

# anthropic
os.environ['ANTHROPIC_API_KEY'] = 'your-anthropic-api-key'

# 01 yi-large
os.environ['LINGYI_API_KEY'] = 'your-lingyi-api-key'

# jieyuexingchen step-1-8k
os.environ['STEPFUN_API_KEY'] = 'your-stepfun-api-key'

# doubao
os.environ['ARK_API_KEY'] = 'your-ark-api-key'


# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unionllm.exceptions import ProviderError
from unionllm import unionchat
from unionllm.utils import ModelResponse, Message, Choices, Usage, Context

common_messages = [{"content": "你的开发者是谁？", "role": "user"}]

@pytest.mark.parametrize("provider, model, messages, expected_exception, stream_mode", [
    # non stream mode
    # This line is part of a parameterized test in Python using pytest. It defines a set of input
    # parameters for the test function `test_unionchat`.
    # ("azure", "azure/gpt-35-turbo", common_messages, None, False),
    # ("zhipuai", "glm-4", common_messages, None, False),
    # ("tiangong", "SkyChat-MegaVerse", common_messages, None, False),
    # ("wenxin", "ERNIE-3.5-8K", common_messages, None, False),
    # ("qwen", "qwen-plus", common_messages, None, False),
    # ("doubao", "ep-20240630165653-phz9t", common_messages, None, False),
    # ("moonshot", "moonshot-v1-8k", common_messages, None, False),
    # ("minimax", "abab5.5-chat", common_messages, None, False),
    # ("baichuan", "Baichuan2-Turbo", common_messages, None, False),
    # ("xunfei", "generalv3", common_messages, None, False),
    # ("lingyi", "yi-large", common_messages, None, False),
    # ("stepfun", "step-1-8k", common_messages, None, False),
    # ("coze", "coze", common_messages, None, False),
    # ("dify", "dify", common_messages, None, False),
    # ("fastgpt", "fastgpt", common_messages, None, False),    
    # {"ollama", "ollama/llama3", common_messages, None, False}
    # ("mistral", "mistral-large-latest", common_messages, None, False),
    # ("cohere", "command-r", common_messages, None, False),
    # ("groq", "groq/mixtral-8x7b-32768", common_messages, None, False),
    # ("bedrock", "ai21.j2-ultra-v1", common_messages, None, False),
    # ("", "groq/mixtral-8x7b-32768", common_messages, None, False), #no provider case
    # ("nonexistent", "model", common_messages, ProviderError, False),
    # stream mode
    # ("doubao", "ep-20240630165653-phz9t", common_messages, None, True),
    # ("azure", "azure/gpt-35-turbo", common_messages, None, True),
    # ("zhipuai", "glm-4", common_messages, None, True),
    # ("nonexistent", "model", common_messages, ProviderError, True),
    # ("tiangong", "SkyChat-MegaVerse", common_messages, None, True),
    # ("wenxin", "ERNIE-3.5-8K", common_messages, None, True),
    # ("qwen", "qwen-plus", common_messages, None, True),
    # ("moonshot", "moonshot-v1-8k", common_messages, None, True),
    # ("minimax", "abab5.5-chat", common_messages, None, True),
    # ("baichuan", "Baichuan2-Turbo", common_messages, None, True),
    # ("coze", "coze", common_messages, None, True),
    # ("dify", "dify", common_messages, None, True),
    # ("fastgpt", "fastgpt", common_messages, None, True),    
    # {"ollama", "ollama/llama3", common_messages, None, False}
    # ("mistral", "mistral-large-latest", common_messages, None, True),
    # ("cohere", "command-r", common_messages, None, True),
    # ("groq", "groq/mixtral-8x7b-32768", common_messages, None, True),
    # ("bedrock", "ai21.j2-ultra-v1", common_messages, None, True),
    # ("nonexistent", "model", common_messages, ProviderError, True),
    # ("lingyi", "yi-large", common_messages, None, True),
    # ("stepfun", "step-1-8k", common_messages, None, True),
])

def test_unionchat(provider, model, messages, expected_exception, stream_mode, mocker):
    mock_unionllm = mocker.patch('unionllm.UnionLLM')
    if expected_exception:
        mock_unionllm.return_value.completion.side_effect = ProviderError("Provider not supported")
    else:
        if stream_mode:
            def stream_response():
                yield ModelResponse(id='8635254124951169203', choices=[Choices(finish_reason='stop', index=0, message=Message(content='我是人工智能助手。', role='assistant'))], created=1715570856, model=model, object='chat.completion', system_fingerprint=None, usage=Usage(prompt_tokens=9, completion_tokens=27, total_tokens=36))
            mock_unionllm.return_value.completion.return_value = stream_response()
        else:
            mock_unionllm.return_value.completion.return_value = ModelResponse(id='8635254124951169203', choices=[Choices(finish_reason='stop', index=0, message=Message(content='我是人工智能助手。', role='assistant'))], created=1715570856, model=model, object='chat.completion', system_fingerprint=None, usage=Usage(prompt_tokens=9, completion_tokens=27, total_tokens=36))

    if expected_exception:
        with pytest.raises(expected_exception):
            unionchat(provider=provider, model=model, messages=messages, stream=stream_mode)
    else:
        result = unionchat(provider=provider, model=model, messages=messages, stream=stream_mode)
        if stream_mode:
            for response in result:
                # print(response)
                assert isinstance(response, ModelResponse), "Each item in the stream should be an instance of ModelResponse"
        else:
            # print(result)
            assert isinstance(result, ModelResponse), "The result should be an instance of ModelResponse"