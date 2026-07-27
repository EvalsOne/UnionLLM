from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import AuthenticationError as OpenAIAuthenticationError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as ChatChoice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChatChunkChoice,
    ChoiceDelta,
)
from openai.types.completion import Completion, CompletionChoice
from openai.types.completion_usage import CompletionUsage

import unionllm.main as main_module
from unionllm.main import UnionLLM
from unionllm.providers.openai_native import OpenAINativeError, OpenAINativeProvider


class FakeCompletionChunk:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def build_mock_client(*, chat_response=None, completion_response=None, responses_response=None):
    client = MagicMock()
    client.chat.completions.create.return_value = chat_response
    client.completions.create.return_value = completion_response
    client.responses.create.return_value = responses_response
    return client


def build_chat_response(content="hello", usage=None):
    return ChatCompletion(
        id="chatcmpl-1",
        choices=[
            ChatChoice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=1,
        model="gpt-4o",
        object="chat.completion",
        usage=usage or CompletionUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )


def build_chat_chunk(*, content=None, role=None, finish_reason=None, usage=None):
    return ChatCompletionChunk(
        id="chatcmpl-1",
        choices=[
            ChatChunkChoice(
                index=0,
                finish_reason=finish_reason,
                delta=ChoiceDelta(role=role, content=content),
            )
        ]
        if finish_reason != "usage"
        else [],
        created=1,
        model="gpt-4o",
        object="chat.completion.chunk",
        usage=usage,
    )


def build_completion_response(text="hello"):
    return Completion(
        id="cmpl-1",
        choices=[
            CompletionChoice(
                index=0,
                finish_reason="stop",
                text=text,
                logprobs=None,
            )
        ],
        created=1,
        model="gpt-3.5-turbo-instruct",
        object="text_completion",
        usage=CompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )


def build_completion_chunk(*, text="", finish_reason=None, usage=None):
    return FakeCompletionChunk(
        {
            "id": "cmpl-1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                    "text": text,
                }
            ]
            if finish_reason != "usage"
            else [],
            "created": 1,
            "model": "gpt-3.5-turbo-instruct",
            "object": "text_completion",
            "usage": usage.model_dump() if usage else None,
        }
    )


def build_responses_usage():
    return SimpleNamespace(
        input_tokens=8,
        output_tokens=5,
        total_tokens=13,
        input_tokens_details=SimpleNamespace(
            cached_tokens=3,
            cache_write_tokens=2,
        ),
        output_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )


def build_responses_response(*, text=None, function_call=None):
    output = []
    if text is not None:
        output.append(
            SimpleNamespace(
                id="msg_1",
                type="message",
                status="completed",
                role="assistant",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text=text,
                        annotations=[],
                    )
                ],
            )
        )
    if function_call is not None:
        output.append(
            SimpleNamespace(
                id="fc_1",
                type="function_call",
                status="completed",
                call_id="call_1",
                name=function_call["name"],
                arguments=function_call["arguments"],
            )
        )
    return SimpleNamespace(
        id="resp_1",
        object="response",
        created_at=2,
        status="completed",
        incomplete_details=None,
        output=output,
        usage=build_responses_usage(),
    )


def test_unionllm_routes_openai_to_native_provider(monkeypatch):
    provider_instance = MagicMock()
    provider_ctor = MagicMock(return_value=provider_instance)
    monkeypatch.setattr(main_module.openai_native, "OpenAINativeProvider", provider_ctor)

    client = UnionLLM(provider="openai", api_key="call-key")

    provider_ctor.assert_called_once_with(api_key="call-key")
    assert client.provider_instance is provider_instance
    assert client.litellm_call_type is None


def test_chat_non_stream_uses_env_key_default_base_and_forwards_openai_params(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    usage = CompletionUsage(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        prompt_tokens_details={"cached_tokens": 1},
    )
    mock_client = build_mock_client(chat_response=build_chat_response("native hello", usage=usage))
    openai_ctor = MagicMock(return_value=mock_client)
    monkeypatch.setattr("unionllm.providers.openai_native.OpenAI", openai_ctor)

    provider = OpenAINativeProvider()
    result = provider.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}],
        tool_choice="auto",
        response_format={"type": "json_object"},
    )

    openai_ctor.assert_called_once_with(api_key="env-key", base_url="https://api.openai.com/v1")
    create_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-4o"
    assert create_kwargs["messages"] == [{"role": "user", "content": "Hi"}]
    assert create_kwargs["tool_choice"] == "auto"
    assert create_kwargs["response_format"] == {"type": "json_object"}
    assert result.choices[0].message.content == "native hello"
    assert result.usage.total_tokens == 5
    assert result.usage.cache_read_input_tokens == 1


@pytest.mark.parametrize("reasoning_effort", [None, "low", "high"])
def test_gpt_5_6_function_tools_default_to_chat_with_reasoning_none(
    monkeypatch,
    reasoning_effort,
):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    mock_client = build_mock_client(chat_response=build_chat_response("chat fallback"))
    monkeypatch.setattr(
        "unionllm.providers.openai_native.OpenAI",
        MagicMock(return_value=mock_client),
    )

    provider = OpenAINativeProvider()
    kwargs = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ]
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    result = provider.completion(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Use the tool"}],
        **kwargs,
    )

    sent = mock_client.chat.completions.create.call_args.kwargs
    assert sent["reasoning_effort"] == "none"
    assert not mock_client.responses.create.called
    assert result.choices[0].message.content == "chat fallback"


def test_explicit_responses_mode_preserves_reasoning_and_maps_function_tools(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    responses_response = build_responses_response(
        function_call={"name": "lookup", "arguments": '{"query":"weather"}'}
    )
    mock_client = build_mock_client(responses_response=responses_response)
    monkeypatch.setattr(
        "unionllm.providers.openai_native.OpenAI",
        MagicMock(return_value=mock_client),
    )

    provider = OpenAINativeProvider()
    result = provider.completion(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "What is the weather?"},
        ],
        api_mode="responses",
        reasoning_effort="high",
        max_completion_tokens=4096,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice={
            "type": "function",
            "function": {"name": "lookup"},
        },
    )

    sent = mock_client.responses.create.call_args.kwargs
    assert sent["model"] == "gpt-5.6-luna"
    assert sent["input"] == [
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "What is the weather?"},
    ]
    assert sent["reasoning"] == {"effort": "high"}
    assert sent["max_output_tokens"] == 4096
    assert sent["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            "strict": False,
        }
    ]
    assert sent["tool_choice"] == {"type": "function", "name": "lookup"}
    assert not mock_client.chat.completions.create.called
    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls[0].id == "call_1"
    assert result.choices[0].message.tool_calls[0].function.name == "lookup"
    assert result.usage.prompt_tokens == 8
    assert result.usage.completion_tokens == 5
    assert result.usage.cache_read_input_tokens == 3
    assert result.usage.cache_creation_input_tokens == 2


def test_provider_level_responses_mode_is_default_and_call_can_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    mock_client = build_mock_client(
        chat_response=build_chat_response("chat"),
        responses_response=build_responses_response(text="responses"),
    )
    monkeypatch.setattr(
        "unionllm.providers.openai_native.OpenAI",
        MagicMock(return_value=mock_client),
    )

    provider = OpenAINativeProvider(api_mode="responses")
    responses_result = provider.completion(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hi"}],
    )
    chat_result = provider.completion(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": "Hi again"}],
        api_mode="chat",
    )

    assert mock_client.responses.create.call_count == 1
    assert mock_client.chat.completions.create.call_count == 1
    assert responses_result.choices[0].message.content == "responses"
    assert chat_result.choices[0].message.content == "chat"


def test_responses_mode_converts_tool_call_history_and_tool_outputs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    mock_client = build_mock_client(
        responses_response=build_responses_response(text="It is sunny.")
    )
    monkeypatch.setattr(
        "unionllm.providers.openai_native.OpenAI",
        MagicMock(return_value=mock_client),
    )

    provider = OpenAINativeProvider()
    result = provider.completion(
        model="gpt-5.6-luna",
        api_mode="responses",
        messages=[
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"weather"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": {"condition": "sunny"},
            },
        ],
        reasoning_effort="medium",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    sent_input = mock_client.responses.create.call_args.kwargs["input"]
    assert sent_input == [
        {"role": "user", "content": "What is the weather?"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"query":"weather"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"condition": "sunny"}',
        },
    ]
    assert result.choices[0].message.content == "It is sunny."
    assert result.choices[0].finish_reason == "stop"


def test_chat_stream_maps_deltas_and_normalizes_usage_only_chunk(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    usage = CompletionUsage(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        prompt_tokens_details={"cached_tokens": 1},
    )
    mock_client = build_mock_client(
        chat_response=iter(
            [
                build_chat_chunk(content="hel", role="assistant"),
                build_chat_chunk(content="lo"),
                build_chat_chunk(finish_reason="stop"),
                build_chat_chunk(finish_reason="usage", usage=usage),
            ]
        )
    )
    openai_ctor = MagicMock(return_value=mock_client)
    monkeypatch.setattr("unionllm.providers.openai_native.OpenAI", openai_ctor)

    provider = OpenAINativeProvider()
    chunks = list(
        provider.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
    )

    create_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert create_kwargs["stream"] is True
    assert create_kwargs["stream_options"] == {"include_usage": True}
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content == "hel"
    assert chunks[1].choices[0].delta.content == "lo"
    assert chunks[2].choices[0].finish_reason == "stop"
    assert chunks[3].usage.total_tokens == 5
    assert chunks[3].usage.cache_read_input_tokens == 1
    assert chunks[3].choices[0].delta.content == ""


def test_responses_stream_maps_text_finish_reason_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    completed = build_responses_response(text="hello")
    response_events = iter(
        [
            SimpleNamespace(
                type="response.created",
                response=SimpleNamespace(id="resp_1", created_at=2),
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="hel",
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                delta="lo",
            ),
            SimpleNamespace(
                type="response.completed",
                response=completed,
            ),
        ]
    )
    mock_client = build_mock_client(responses_response=response_events)
    monkeypatch.setattr(
        "unionllm.providers.openai_native.OpenAI",
        MagicMock(return_value=mock_client),
    )

    provider = OpenAINativeProvider()
    chunks = list(
        provider.completion(
            model="gpt-5.6-luna",
            api_mode="responses",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            reasoning_effort="low",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
    )

    sent = mock_client.responses.create.call_args.kwargs
    assert sent["stream"] is True
    assert "stream_options" not in sent
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[0].choices[0].delta.content == "hel"
    assert chunks[1].choices[0].delta.content == "lo"
    assert chunks[2].choices[0].finish_reason == "stop"
    assert chunks[2].usage.total_tokens == 13
    assert chunks[2].usage.cache_read_input_tokens == 3


def test_call_time_api_key_and_base_url_override_default_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    default_client = build_mock_client(chat_response=build_chat_response("default"))
    override_client = build_mock_client(chat_response=build_chat_response("override"))
    openai_ctor = MagicMock(side_effect=[default_client, override_client])
    monkeypatch.setattr("unionllm.providers.openai_native.OpenAI", openai_ctor)

    provider = OpenAINativeProvider()
    result = provider.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        api_key="override-key",
        api_base="https://proxy.example.com",
    )

    assert openai_ctor.call_args_list[1].kwargs == {
        "api_key": "override-key",
        "base_url": "https://proxy.example.com/v1",
    }
    assert override_client.chat.completions.create.called
    assert result.choices[0].message.content == "override"


def test_instruct_model_uses_completions_endpoint_and_prompt_conversion(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    mock_client = build_mock_client(completion_response=build_completion_response("completion hello"))
    openai_ctor = MagicMock(return_value=mock_client)
    monkeypatch.setattr("unionllm.providers.openai_native.OpenAI", openai_ctor)

    provider = OpenAINativeProvider()
    result = provider.completion(
        model="gpt-3.5-turbo-instruct",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Say hi"},
        ],
    )

    create_kwargs = mock_client.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-3.5-turbo-instruct"
    assert create_kwargs["prompt"] == "System: You are terse.\n\nUser: Say hi"
    assert result.choices[0].message.content == "completion hello"
    assert result.usage.total_tokens == 6


def test_completions_stream_maps_text_chunks_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    usage = CompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
    mock_client = build_mock_client(
        completion_response=iter(
            [
                build_completion_chunk(text="com"),
                build_completion_chunk(text="plete"),
                build_completion_chunk(finish_reason="stop"),
                build_completion_chunk(finish_reason="usage", usage=usage),
            ]
        )
    )
    openai_ctor = MagicMock(return_value=mock_client)
    monkeypatch.setattr("unionllm.providers.openai_native.OpenAI", openai_ctor)

    provider = OpenAINativeProvider()
    chunks = list(
        provider.completion(
            model="gpt-3.5-turbo-instruct",
            messages=[{"role": "user", "content": "Finish this"}],
            stream=True,
        )
    )

    create_kwargs = mock_client.completions.create.call_args.kwargs
    assert create_kwargs["stream"] is True
    assert create_kwargs["stream_options"] == {"include_usage": True}
    assert chunks[0].choices[0].delta.content == "com"
    assert chunks[1].choices[0].delta.content == "plete"
    assert chunks[2].choices[0].finish_reason == "stop"
    assert chunks[3].usage.total_tokens == 6


def test_maps_openai_status_errors_with_request_id(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    mock_client = build_mock_client()
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request, headers={"x-request-id": "req_123"})
    mock_client.chat.completions.create.side_effect = OpenAIAuthenticationError(
        "Invalid API key",
        response=response,
        body={"message": "Invalid API key"},
    )
    openai_ctor = MagicMock(return_value=mock_client)
    monkeypatch.setattr("unionllm.providers.openai_native.OpenAI", openai_ctor)

    provider = OpenAINativeProvider()

    with pytest.raises(OpenAINativeError) as excinfo:
        provider.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )

    assert excinfo.value.status_code == 401
    assert excinfo.value.request_id == "req_123"
    assert "req_123" in str(excinfo.value)
