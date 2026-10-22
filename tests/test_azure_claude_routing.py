"""provider="azure" sends Claude deployments to Anthropic Foundry and the rest to AzureOpenAI.

The choice follows the model of each call. unionchat, and callers such as
ConsoleX's agent loop, construct UnionLLM(provider="azure") without a model and
only name it in completion(); model names may carry LiteLLM's "azure/" prefix.
Claude requests go through a real AnthropicFoundry client over a mock transport,
so the URL, headers and body checked here are what the SDK would send.
"""

import asyncio
from unittest.mock import MagicMock

import anthropic
import pytest

from unionllm.main import UnionLLM, unionchat
from unionllm.providers import azure as azure_module
from unionllm.providers.azure import AzureAnthropicProvider, AzureOpenAIProvider

from tests.test_anthropic_sampling_params import on_mock_transport
from tests.test_openai_native import build_chat_response, build_mock_client

FOUNDRY_KWARGS = {
    "api_key": "mock-only",
    # The form ConsoleX uses for shared Claude models, trailing slash included.
    "api_base": "https://mock.services.ai.azure.com/anthropic/",
    "api_version": "2025-04-01-preview",
}
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]


@pytest.fixture
def foundry(monkeypatch):
    """Records the requests sent to Anthropic Foundry; fails the test if AzureOpenAI is built."""
    for name in ("AZURE_ANTHROPIC_VERSION", "AZURE_OPENAI_API_KEY", "AZURE_API_KEY",
                 "AZURE_OPENAI_ENDPOINT", "AZURE_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    sent, seen = [], []
    monkeypatch.setattr("anthropic.AnthropicFoundry", on_mock_transport(anthropic.AnthropicFoundry, sent, seen))
    azure_openai = MagicMock(side_effect=AssertionError("Claude went to AzureOpenAI"))
    monkeypatch.setattr(azure_module, "AzureOpenAI", azure_openai)
    return sent, seen


@pytest.fixture
def azure_openai(monkeypatch):
    """A mock AzureOpenAI client; fails the test if Anthropic Foundry is built."""
    client = build_mock_client(chat_response=build_chat_response("azure hello"))
    ctor = MagicMock(return_value=client)
    monkeypatch.setattr(azure_module, "AzureOpenAI", ctor)
    monkeypatch.setattr("anthropic.AnthropicFoundry", MagicMock(side_effect=AssertionError("GPT went to Foundry")))
    return client, ctor


def call(style, model, **kwargs):
    if style == "unionchat":
        return unionchat(provider="azure", model=model, messages=MESSAGES, **FOUNDRY_KWARGS, **kwargs)
    if style == "UnionLLM(model=...)":
        client = UnionLLM(provider="azure", model=model, **FOUNDRY_KWARGS)
    else:
        client = UnionLLM(provider="azure", **FOUNDRY_KWARGS)
    return client.completion(model=model, messages=MESSAGES, **kwargs)


STYLES = pytest.mark.parametrize("style", ["unionchat", "UnionLLM(model=...)", "UnionLLM()"])
CLAUDE = pytest.mark.parametrize("model", ["claude-sonnet-4-5", "azure/claude-sonnet-4-5", "Azure/Claude-Sonnet-4-5"])


@STYLES
@CLAUDE
def test_claude_reaches_foundry(foundry, style, model):
    sent, seen = foundry

    result = call(style, model, max_tokens=64, temperature=0.3)

    assert seen[0].url.path == "/anthropic/v1/messages"
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert sent[0]["model"] == model.split("/", 1)[-1]
    assert sent[0]["system"] == "Be brief."
    assert sent[0]["temperature"] == 0.3
    assert result.choices[0].message.content == "hello"


@STYLES
def test_claude_stream_reaches_foundry(foundry, style):
    sent, seen = foundry

    chunks = list(call(style, "azure/claude-sonnet-4-5", stream=True))

    assert seen[0].url.path == "/anthropic/v1/messages"
    assert sent[0]["model"] == "claude-sonnet-4-5"
    assert sent[0]["stream"] is True
    assert "".join(getattr(c.choices[0].delta, "content", None) or "" for c in chunks) == "hello"


@STYLES
@pytest.mark.parametrize("model", ["gpt-4o-mini", "azure/gpt-4o-mini"])
def test_other_models_reach_azure_openai(azure_openai, style, model):
    client, ctor = azure_openai

    result = call(style, model)

    assert ctor.call_count == 1
    assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"
    assert result.choices[0].message.content == "azure hello"


def test_one_client_routes_each_call_and_reuses_providers(monkeypatch):
    built = []

    def recording(provider_class):
        def build(**kwargs):
            instance = MagicMock(spec=provider_class)
            instance.completion.return_value = provider_class.__name__
            built.append(provider_class)
            return instance
        return build

    monkeypatch.setattr(azure_module, "AzureAnthropicProvider", recording(AzureAnthropicProvider))
    monkeypatch.setattr(azure_module, "AzureOpenAIProvider", recording(AzureOpenAIProvider))
    client = UnionLLM(provider="azure", **FOUNDRY_KWARGS)
    assert built == []

    assert client.completion(model="azure/gpt-4o-mini", messages=MESSAGES) == "AzureOpenAIProvider"
    assert client.completion(model="azure/claude-sonnet-4-5", messages=MESSAGES) == "AzureAnthropicProvider"
    assert client.completion(model="claude-haiku-4-5", messages=MESSAGES) == "AzureAnthropicProvider"
    assert asyncio.run(client.acompletion(model="gpt-4o", messages=MESSAGES)) == "AzureOpenAIProvider"
    assert built == [AzureOpenAIProvider, AzureAnthropicProvider]


def test_consolex_agent_loop_call_shape(foundry):
    """ConsoleX's agent loop builds UnionLLM(provider="azure", **credentials) and passes
    the prefixed model and the credentials again to completion()."""
    sent, seen = foundry

    client = UnionLLM(provider="azure", **FOUNDRY_KWARGS)
    chunks = list(client.completion(model="azure/claude-sonnet-4-5", messages=MESSAGES, stream=True,
                                    stream_options={"include_usage": True}, **FOUNDRY_KWARGS))

    assert seen[0].url.path == "/anthropic/v1/messages"
    assert sent[0]["model"] == "claude-sonnet-4-5"
    assert chunks


def test_constructor_model_still_builds_eagerly(foundry):
    client = UnionLLM(provider="azure", model="azure/claude-sonnet-4-5", **FOUNDRY_KWARGS)
    assert isinstance(client.provider_instance, AzureAnthropicProvider)


def test_is_claude_deployment():
    assert azure_module.is_claude_deployment("claude-opus-4-5")
    assert azure_module.is_claude_deployment("azure/claude-opus-4-5")
    assert azure_module.is_claude_deployment("AZURE/CLAUDE-OPUS-4-5")
    for model in ("gpt-4o", "azure/gpt-4o", "azure/o3", None, ""):
        assert not azure_module.is_claude_deployment(model)
