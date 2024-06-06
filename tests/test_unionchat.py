import pytest
import os, sys
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
load_dotenv()

# 将项目根目录添加到sys.path中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unionllm.exceptions import ProviderError
from unionllm import unionchat
from unionllm.utils import ModelResponse, Message, Choices, Usage, Context

common_messages = [{"content": "你的开发者是谁？", "role": "user"}]

@pytest.mark.parametrize("provider, model, messages, expected_exception, stream_mode", [
    # non stream mode
    ("azure", "azure/gpt-35-turbo", common_messages, None, False),
    ("zhipuai", "glm-4", common_messages, None, False),
    ("tiangong", "SkyChat-MegaVerse", common_messages, None, False),
    ("wenxin", "ERNIE-3.5-8K", common_messages, None, False),
    ("qwen", "qwen-plus", common_messages, None, False),
    ("moonshot", "moonshot-v1-8k", common_messages, None, False),
    ("minimax", "abab5.5-chat", common_messages, None, False),
    ("baichuan", "Baichuan2-Turbo", common_messages, None, False),
    ("xunfei", "generalv3", common_messages, None, False),
    ("lingyi", "yi-large", common_messages, None, False),
    ("stepfun", "step-1-8k", common_messages, None, False),
    ("coze", "coze", common_messages, None, False),
    ("dify", "dify", common_messages, None, False),
    ("fastgpt", "fastgpt", common_messages, None, False),    
    {"ollama", "ollama/llama3", common_messages, None, False}
    ("mistral", "mistral-large-latest", common_messages, None, False),
    ("cohere", "command-r", common_messages, None, False),
    ("groq", "groq/mixtral-8x7b-32768", common_messages, None, False),
    ("bedrock", "ai21.j2-ultra-v1", common_messages, None, False),
    ("", "groq/mixtral-8x7b-32768", common_messages, None, False), #no provider case
    ("nonexistent", "model", common_messages, ProviderError, False),
    # stream mode
    ("azure", "azure/gpt-35-turbo", common_messages, None, True),
    ("zhipuai", "glm-4", common_messages, None, True),
    ("nonexistent", "model", common_messages, ProviderError, True),
    ("tiangong", "SkyChat-MegaVerse", common_messages, None, True),
    ("wenxin", "ERNIE-3.5-8K", common_messages, None, True),
    ("qwen", "qwen-plus", common_messages, None, True),
    ("moonshot", "moonshot-v1-8k", common_messages, None, True),
    ("minimax", "abab5.5-chat", common_messages, None, True),
    ("baichuan", "Baichuan2-Turbo", common_messages, None, True),
    ("coze", "coze", common_messages, None, True),
    ("dify", "dify", common_messages, None, True),
    ("fastgpt", "fastgpt", common_messages, None, True),    
    {"ollama", "ollama/llama3", common_messages, None, False}
    ("mistral", "mistral-large-latest", common_messages, None, True),
    ("cohere", "command-r", common_messages, None, True),
    ("groq", "groq/mixtral-8x7b-32768", common_messages, None, True),
    ("bedrock", "ai21.j2-ultra-v1", common_messages, None, True),
    ("nonexistent", "model", common_messages, ProviderError, True),
    ("lingyi", "yi-large", common_messages, None, True),
    ("stepfun", "step-1-8k", common_messages, None, True),
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
                print(response)
                assert isinstance(response, ModelResponse), "Each item in the stream should be an instance of ModelResponse"
        else:
            print(result)
            assert isinstance(result, ModelResponse), "The result should be an instance of ModelResponse"