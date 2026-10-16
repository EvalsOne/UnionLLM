import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from unionllm.exceptions import ProviderError
from unionllm.main import UnionLLM
from unionllm.providers.litellm import LiteLLMUnavailableError
from unionllm.providers.openai_native import OpenAICompatibleProvider, OpenAINativeError

from tests.test_openai_native import build_chat_response, build_mock_client

MESSAGES = [{"role": "user", "content": "Hi"}]


@pytest.fixture
def openai_ctor(monkeypatch):
    for name in ("XAI_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # Must never be sent to a third-party endpoint.
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
    client = build_mock_client(chat_response=build_chat_response("compatible hello"))
    ctor = MagicMock(return_value=client)
    monkeypatch.setattr("unionllm.providers.openai_native.OpenAI", ctor)
    return ctor


def test_xai_uses_its_base_url_and_strips_the_provider_prefix(openai_ctor):
    client = UnionLLM(provider="xai", api_key="xai-key")
    assert isinstance(client.provider_instance, OpenAICompatibleProvider)
    assert client.litellm_call_type is None

    result = client.completion(model="xai/grok-3", messages=MESSAGES, api_key="xai-key", provider="xai",
                               thinking={"type": "enabled"}, top_k=5)

    openai_ctor.assert_called_once_with(api_key="xai-key", base_url="https://api.x.ai/v1")
    sent = openai_ctor.return_value.chat.completions.create.call_args.kwargs
    assert sent["model"] == "grok-3"
    assert "thinking" not in sent and "top_k" not in sent and "provider" not in sent
    assert result.choices[0].message.content == "compatible hello"


def test_vendor_key_comes_from_its_own_environment_variable(openai_ctor, monkeypatch):
    with pytest.raises(OpenAINativeError) as excinfo:
        UnionLLM(provider="xai")
    assert excinfo.value.status_code == 422

    monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
    UnionLLM(provider="xai")
    assert openai_ctor.call_args.kwargs["api_key"] == "xai-env-key"


def test_openrouter_keeps_vendor_model_ids(openai_ctor):
    client = UnionLLM(provider="openrouter", api_key="or-key")
    client.completion(model="perplexity/llama-3.1-sonar-large-128k-online", messages=MESSAGES)

    assert openai_ctor.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    sent = openai_ctor.return_value.chat.completions.create.call_args.kwargs
    assert sent["model"] == "perplexity/llama-3.1-sonar-large-128k-online"


def test_groq_base_url_path_is_kept_verbatim(openai_ctor):
    UnionLLM(provider="groq", api_key="groq-key")
    assert openai_ctor.call_args.kwargs["base_url"] == "https://api.groq.com/openai/v1"


def test_custom_endpoint_uses_api_base_verbatim(openai_ctor):
    # ConsoleX CUSTOM models: no provider, custom_llm_provider="openai".
    client = UnionLLM(
        custom_llm_provider="openai",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        api_key="user-key",
        model="glm-4.5",
    )
    client.completion(model="glm-4.5", messages=MESSAGES, stream=False, custom_llm_provider="openai",
                      api_base="https://open.bigmodel.cn/api/paas/v4/", api_key="user-key")

    openai_ctor.assert_called_once_with(api_key="user-key", base_url="https://open.bigmodel.cn/api/paas/v4")
    sent = openai_ctor.return_value.chat.completions.create.call_args.kwargs
    assert sent["model"] == "glm-4.5"
    assert "custom_llm_provider" not in sent


def test_custom_endpoint_without_key_uses_placeholder_not_openai_env(openai_ctor):
    UnionLLM(custom_llm_provider="openai", base_url="http://localhost:8000/v1")
    openai_ctor.assert_called_once_with(api_key="unused", base_url="http://localhost:8000/v1")


def test_custom_endpoint_without_base_url_fails_clearly(openai_ctor):
    with pytest.raises(OpenAINativeError) as excinfo:
        UnionLLM(custom_llm_provider="openai", api_key="user-key")
    assert "base_url" in str(excinfo.value)


def test_reserved_qwen_model_uses_dashscope_compatible_mode(openai_ctor, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    client = UnionLLM(provider="qwen", model="qwen-special-model")
    client.completion(model="qwen-special-model", messages=MESSAGES)

    openai_ctor.assert_called_once_with(
        api_key="dashscope-key", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert openai_ctor.return_value.chat.completions.create.call_args.kwargs["model"] == "qwen-special-model"


def test_litellm_routes_fail_clearly_without_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)  # import litellm -> ImportError

    with pytest.raises(LiteLLMUnavailableError) as excinfo:
        UnionLLM(provider="replicate", api_key="r8-key")
    assert isinstance(excinfo.value, ProviderError)
    assert excinfo.value.status_code == 501
    assert "Provider 'replicate'" in str(excinfo.value)
    assert "unionllm[litellm]" in str(excinfo.value)

    with pytest.raises(LiteLLMUnavailableError) as excinfo:
        UnionLLM(api_key="key")
    assert "without a provider" in str(excinfo.value)

    monkeypatch.setenv("UNIONLLM_USE_LITELLM_ANTHROPIC", "1")
    with pytest.raises(LiteLLMUnavailableError):
        UnionLLM(provider="anthropic")


def test_importing_unionllm_does_not_need_litellm():
    code = (
        "import sys; sys.modules['litellm'] = None\n"
        "import unionllm.main\n"
        "from unionllm.main import UnionLLM, unionchat\n"
        "from unionllm.providers import litellm as route\n"
        "assert 'litellm' not in [m for m, v in sys.modules.items() if v is not None]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
