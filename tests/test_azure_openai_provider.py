from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice as ChatChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from unionllm.main import UnionLLM, unionchat
from unionllm.providers import azure as azure_module
from unionllm.providers.azure import AzureAnthropicProvider, AzureOpenAIProvider
from unionllm.providers.openai_native import OpenAINativeError

from tests.test_openai_native import build_chat_chunk, build_chat_response, build_mock_client

AZURE_KWARGS = {
    "api_key": "azure-key",
    "api_base": "https://example-resource.openai.azure.com/",
    "api_version": "2025-01-01-preview",
}
MESSAGES = [{"role": "user", "content": "Hi"}]


@pytest.fixture
def azure_client(monkeypatch):
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_API_BASE",
                 "AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    client = build_mock_client(chat_response=build_chat_response("azure hello"))
    azure_ctor = MagicMock(return_value=client)
    openai_ctor = MagicMock(return_value=client)
    monkeypatch.setattr(azure_module, "AzureOpenAI", azure_ctor)
    monkeypatch.setattr(azure_module, "OpenAI", openai_ctor)
    return client, azure_ctor, openai_ctor


def sent_kwargs(client):
    return client.chat.completions.create.call_args.kwargs


def test_unionllm_routes_azure_gpt_to_openai_sdk_and_claude_to_foundry(azure_client, monkeypatch):
    assert isinstance(UnionLLM(provider="azure", model="gpt-4o-mini", **AZURE_KWARGS).provider_instance,
                      AzureOpenAIProvider)
    # Without a model the choice waits for completion(), which names it.
    assert UnionLLM(provider="AZURE", **AZURE_KWARGS).provider_instance is None

    monkeypatch.setattr("anthropic.AnthropicFoundry", MagicMock())
    client = UnionLLM(provider="azure", model="claude-sonnet-4-5", **AZURE_KWARGS)
    assert isinstance(client.provider_instance, AzureAnthropicProvider)
    assert client.litellm_call_type is None


def test_helper_call_strips_prefix_and_keeps_chat_model_params(azure_client):
    client, azure_ctor, _ = azure_client

    result = unionchat(
        model="azure/gpt-4o-mini",
        messages=MESSAGES,
        provider="azure",
        temperature=0.0,
        response_format={"type": "json_object"},
        reasoning_effort="low",
        max_tokens=100,
        **AZURE_KWARGS,
    )

    azure_ctor.assert_called_once_with(
        api_key="azure-key",
        azure_endpoint="https://example-resource.openai.azure.com",
        api_version="2025-01-01-preview",
    )
    sent = sent_kwargs(client)
    assert sent["model"] == "gpt-4o-mini"
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 100
    assert sent["response_format"] == {"type": "json_object"}
    # gpt-4o has no reasoning; LiteLLM's drop_params dropped it too.
    assert "reasoning_effort" not in sent
    for internal in ("api_key", "api_base", "api_version", "provider"):
        assert internal not in sent
    assert result.choices[0].message.content == "azure hello"
    assert result.usage.total_tokens == 5


def test_memory_call_shape_uses_base_url_and_default_api_version(azure_client):
    # working_memory/misc_task pass base_url and an empty api_version when unset.
    client, azure_ctor, _ = azure_client

    unionchat(
        model="gpt-4.1-mini",
        messages=MESSAGES,
        provider="azure",
        api_key="azure-key",
        base_url="https://example-resource.openai.azure.com/",
        api_version="",
    )

    assert azure_ctor.call_args.kwargs["api_version"] == AzureOpenAIProvider.DEFAULT_API_VERSION
    assert azure_ctor.call_args.kwargs["azure_endpoint"] == "https://example-resource.openai.azure.com"
    assert sent_kwargs(client)["model"] == "gpt-4.1-mini"


def test_reasoning_model_gets_max_completion_tokens_and_loses_sampling_params(azure_client):
    client, _, _ = azure_client
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]

    UnionLLM(provider="azure", **AZURE_KWARGS).completion(
        model="gpt-5.4",
        messages=MESSAGES,
        max_tokens=2048,
        top_p=0.9,
        temperature=0.2,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        stop=["END"],
        reasoning_effort="high",
        tools=tools,
        tool_choice="auto",
        **AZURE_KWARGS,
    )

    sent = sent_kwargs(client)
    assert sent["model"] == "gpt-5.4"
    assert sent["max_completion_tokens"] == 2048
    assert sent["reasoning_effort"] == "high"
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"
    for dropped in ("max_tokens", "top_p", "temperature", "presence_penalty", "frequency_penalty", "stop"):
        assert dropped not in sent


def test_reasoning_off_keeps_sampling_only_where_the_model_allows_it(azure_client):
    client, _, _ = azure_client
    provider = UnionLLM(provider="azure", **AZURE_KWARGS)

    provider.completion(model="gpt-5.1", messages=MESSAGES, reasoning_effort="none", temperature=0.3, top_p=0.8)
    sent = sent_kwargs(client)
    assert sent["reasoning_effort"] == "none"
    assert sent["temperature"] == 0.3
    assert sent["top_p"] == 0.8

    provider.completion(model="gpt-5", messages=MESSAGES, reasoning_effort="none", temperature=0.3, top_p=0.8)
    sent = sent_kwargs(client)
    assert "reasoning_effort" not in sent
    assert "temperature" not in sent
    assert "top_p" not in sent

    provider.completion(model="gpt-5", messages=MESSAGES, temperature=1)
    assert sent_kwargs(client)["temperature"] == 1


def test_o_series_and_gpt_5_chat_families(azure_client):
    client, _, _ = azure_client
    provider = UnionLLM(provider="azure", **AZURE_KWARGS)

    provider.completion(model="o3-mini", messages=MESSAGES, max_tokens=50, temperature=0.0,
                        reasoning_effort="medium")
    sent = sent_kwargs(client)
    assert sent["max_completion_tokens"] == 50
    assert sent["reasoning_effort"] == "medium"
    assert "temperature" not in sent and "max_tokens" not in sent

    provider.completion(model="azure/gpt-5-chat", messages=MESSAGES, max_tokens=50, temperature=0.4,
                        reasoning_effort="high")
    sent = sent_kwargs(client)
    assert sent["max_tokens"] == 50
    assert sent["temperature"] == 0.4
    assert "reasoning_effort" not in sent

    assert AzureOpenAIProvider.model_family("gpt-5.2-chat") == "gpt5"
    assert AzureOpenAIProvider.model_family("gpt-6-sol") == "gpt5"
    assert AzureOpenAIProvider.model_family("gpt-4o-azure") == "chat"
    assert not AzureOpenAIProvider.supports_no_reasoning("gpt-5.2")
    assert AzureOpenAIProvider.supports_no_reasoning("gpt-5.6-sol")
    assert not AzureOpenAIProvider.supports_no_reasoning("gpt-5.5-pro")


def test_gpt_5_6_tools_rule_still_applies_on_azure(azure_client):
    client, _, _ = azure_client
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]

    UnionLLM(provider="azure", **AZURE_KWARGS).completion(
        model="azure/gpt-5.6-sol", messages=MESSAGES, tools=tools, reasoning_effort="high", top_p=0.5,
    )

    sent = sent_kwargs(client)
    assert sent["model"] == "gpt-5.6-sol"
    assert sent["reasoning_effort"] == "none"
    assert sent["top_p"] == 0.5


def test_stream_skips_prompt_filter_chunk_and_keeps_tool_calls_and_usage(azure_client):
    client, _, _ = azure_client
    filter_chunk = ChatCompletionChunk(id="", choices=[], created=0, model="", object="chat.completion.chunk")
    tool_chunk = ChatCompletionChunk(
        id="chatcmpl-1",
        choices=[
            ChatChunkChoice(
                index=0,
                finish_reason=None,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name="ping", arguments='{"a":1}'),
                        )
                    ]
                ),
            )
        ],
        created=1,
        model="gpt-5.4",
        object="chat.completion.chunk",
    )
    usage = CompletionUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10,
                            prompt_tokens_details={"cached_tokens": 4})
    client.chat.completions.create.return_value = iter([
        filter_chunk,
        build_chat_chunk(content="hel", role="assistant"),
        tool_chunk,
        build_chat_chunk(finish_reason="tool_calls"),
        build_chat_chunk(finish_reason="usage", usage=usage),
    ])

    chunks = list(UnionLLM(provider="azure", **AZURE_KWARGS).completion(
        model="azure/gpt-5.4", messages=MESSAGES, stream=True, **AZURE_KWARGS,
    ))

    assert sent_kwargs(client)["stream_options"] == {"include_usage": True}
    assert len(chunks) == 4
    assert chunks[0].choices[0].delta.content == "hel"
    tool_call = chunks[1].choices[0].delta.tool_calls[0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "ping"
    assert chunks[2].choices[0].finish_reason == "tool_calls"
    assert chunks[3].usage.total_tokens == 10
    assert chunks[3].usage.cache_read_input_tokens == 4


def test_byok_endpoint_field_is_accepted(azure_client):
    _, azure_ctor, _ = azure_client
    UnionLLM(provider="azure", api_key="azure-key", endpoint="https://byok.openai.azure.com/",
             api_version="2024-10-21").completion(model="gpt-4o", messages=MESSAGES)
    assert azure_ctor.call_args.kwargs["azure_endpoint"] == "https://byok.openai.azure.com"


def test_v1_endpoint_uses_plain_openai_client(azure_client):
    _, azure_ctor, openai_ctor = azure_client

    UnionLLM(
        provider="azure",
        api_key="azure-key",
        api_base="https://example.services.ai.azure.com/openai/v1/",
    ).completion(model="gpt-4o", messages=MESSAGES)

    openai_ctor.assert_called_once_with(
        api_key="azure-key", base_url="https://example.services.ai.azure.com/openai/v1/"
    )
    azure_ctor.assert_not_called()


def test_call_time_credentials_build_a_new_client(azure_client):
    client, azure_ctor, _ = azure_client
    provider = UnionLLM(provider="azure", **AZURE_KWARGS)

    provider.completion(model="gpt-4o", messages=MESSAGES, **AZURE_KWARGS)
    assert azure_ctor.call_count == 1

    provider.completion(model="gpt-4o", messages=MESSAGES, **{**AZURE_KWARGS, "api_version": "2025-04-01-preview"})
    assert azure_ctor.call_count == 2
    assert azure_ctor.call_args.kwargs["api_version"] == "2025-04-01-preview"


def test_missing_credentials_fail_clearly(azure_client):
    # At construction when the model is given, otherwise at the first call.
    with pytest.raises(OpenAINativeError) as excinfo:
        UnionLLM(provider="azure", model="gpt-4o", api_base="https://example-resource.openai.azure.com/")
    assert excinfo.value.status_code == 422

    with pytest.raises(OpenAINativeError) as excinfo:
        UnionLLM(provider="azure", api_key="azure-key").completion(model="gpt-4o", messages=MESSAGES)
    assert "endpoint" in str(excinfo.value)
