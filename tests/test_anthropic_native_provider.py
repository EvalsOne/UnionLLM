import json

from unionllm.main import UnionLLM
from unionllm.providers.anthropic import AnthropicAIProvider


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DummyMessagesAPI:
    def __init__(self):
        self.calls = []
        self.response = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class DummyClient:
    def __init__(self):
        self.messages = DummyMessagesAPI()


def test_anthropic_init_uses_env_api_key_and_default_base_url(monkeypatch):
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    provider = AnthropicAIProvider()
    assert provider.api_key == "env-key"
    assert captured["api_key"] == "env-key"
    assert captured["base_url"] == "https://api.anthropic.com"
    assert captured["default_headers"]["anthropic-version"] == "2023-06-01"


def test_anthropic_init_prefers_call_api_key_and_custom_base_url(monkeypatch):
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    provider = AnthropicAIProvider(
        api_key="call-key",
        base_url="https://proxy.example.com",
        anthropic_version="2024-01-01",
    )
    assert provider.api_key == "call-key"
    assert captured["api_key"] == "call-key"
    assert captured["base_url"] == "https://proxy.example.com"
    assert captured["default_headers"]["anthropic-version"] == "2024-01-01"


def test_convert_openai_messages_supports_system_image_and_tools(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicAIProvider()

    messages = [
        {"role": "system", "content": "You are helpful"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": "{\"city\":\"shanghai\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"ok\":true}"},
    ]

    system_message, converted = provider._convert_openai_to_anthropic_messages(messages)
    assert system_message == "You are helpful"
    assert converted[0]["role"] == "user"
    assert converted[0]["content"][1]["type"] == "image"
    assert converted[1]["content"][1]["type"] == "tool_use"
    assert converted[1]["content"][1]["name"] == "lookup_weather"
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0]["type"] == "tool_result"


def test_prepare_request_keeps_cache_control_top_level(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicAIProvider()

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "this is a long prompt prefix for cache"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "lookup weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    converted_messages, params = provider._prepare_request(
        messages,
        tools=tools,
        tool_choice="required",
        cache_control={"type": "ephemeral"},
    )

    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["system"] == "system prompt"
    assert params["tool_choice"]["type"] == "any"
    assert all("cache_control" not in tool for tool in params["tools"])
    assert converted_messages[0]["content"] == "this is a long prompt prefix for cache"


def test_prepare_request_uses_sensible_anthropic_max_tokens_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicAIProvider()

    _, params = provider._prepare_request([{"role": "user", "content": "hello"}])

    assert params["max_tokens"] == 8192


def test_prepare_request_removes_consolex_message_bookkeeping(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicAIProvider()

    converted_messages, _ = provider._prepare_request(
        [
            {
                "role": "user",
                "content": "hello",
                "messageIndex": 0,
                "name": "ConsoleX",
            }
        ]
    )

    assert converted_messages == [{"role": "user", "content": "hello"}]


def test_create_model_response_maps_tool_calls_and_cache_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicAIProvider()

    anthropic_resp = Obj(
        id="msg_1",
        stop_reason="tool_use",
        content=[
            Obj(type="text", text="Need a tool"),
            Obj(type="tool_use", id="call_1", name="lookup_weather", input={"city": "shanghai"}),
        ],
        usage=Obj(
            input_tokens=11,
            output_tokens=7,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=3,
        ),
    )

    response = provider.create_model_response_wrapper(anthropic_resp, model="claude-3-5-sonnet")
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.content == "Need a tool"
    assert response.choices[0].message.tool_calls[0]["function"]["name"] == "lookup_weather"
    assert json.loads(response.choices[0].message.tool_calls[0]["function"]["arguments"]) == {"city": "shanghai"}
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 7
    assert response.usage.cache_creation_input_tokens == 10
    assert response.usage.cache_read_input_tokens == 3


def test_streaming_maps_tool_and_text_deltas(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicAIProvider()
    provider.client = DummyClient()

    events = [
        Obj(type="message_start", message=Obj(id="msg_stream", usage=Obj(input_tokens=30))),
        Obj(
            type="content_block_start",
            index=0,
            content_block=Obj(type="tool_use", id="call_1", name="lookup_weather"),
        ),
        Obj(
            type="content_block_delta",
            index=0,
            delta=Obj(type="input_json_delta", partial_json='{"city":"sha'),
        ),
        Obj(
            type="content_block_delta",
            index=0,
            delta=Obj(type="input_json_delta", partial_json='nghai"}'),
        ),
        Obj(type="content_block_start", index=1, content_block=Obj(type="text")),
        Obj(type="content_block_delta", index=1, delta=Obj(type="text_delta", text="Processing")),
        Obj(type="message_delta", delta=Obj(stop_reason="tool_use"), usage=Obj(output_tokens=5)),
        Obj(type="message_stop"),
    ]
    provider.client.messages.response = events

    stream = provider.completion(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        max_tokens=128,
    )
    chunks = list(stream)

    first_call = provider.client.messages.calls[0]
    assert first_call["stream"] is True
    assert first_call["model"] == "claude-3-5-sonnet"

    tool_chunks = [c for c in chunks if getattr(c.choices[0].delta, "tool_calls", None)]
    text_chunks = [c for c in chunks if getattr(c.choices[0].delta, "content", None)]
    usage_chunks = [
        c
        for c in chunks
        if getattr(c.usage, "prompt_tokens", None) is not None
        or getattr(c.usage, "completion_tokens", None) is not None
    ]

    assert len(tool_chunks) >= 2
    assert tool_chunks[0].choices[0].delta.tool_calls[0]["function"]["name"] == "lookup_weather"
    assert len(text_chunks) >= 1
    assert text_chunks[0].choices[0].delta.content == "Processing"
    assert len(usage_chunks) >= 1
    assert usage_chunks[0].usage.prompt_tokens == 30
    assert usage_chunks[0].usage.completion_tokens == 5
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_non_stream_completion_sends_tools_and_returns_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicAIProvider()
    provider.client = DummyClient()
    provider.client.messages.response = Obj(
        id="msg_non_stream",
        stop_reason="end_turn",
        content=[Obj(type="text", text="hello world")],
        usage=Obj(input_tokens=4, output_tokens=3),
    )

    response = provider.completion(
        model="claude-3-5-sonnet",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ],
        tool_choice="required",
        cache_control={"type": "ephemeral"},
        max_tokens=256,
    )

    sent = provider.client.messages.calls[0]
    assert sent["tool_choice"] == {"type": "any"}
    assert sent["tools"][0]["name"] == "lookup"
    assert sent["system"] == "sys"
    assert sent["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"][0]["content"] == "hello"
    assert response.choices[0].message.content == "hello world"
    assert response.choices[0].finish_reason == "stop"


def test_unionllm_routes_anthropic_to_native_provider(monkeypatch):
    class FakeAnthropicProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def completion(self, model, messages, **kwargs):
            return None

    monkeypatch.setattr("unionllm.main.anthropic.AnthropicAIProvider", FakeAnthropicProvider)
    client = UnionLLM(provider="anthropic")
    assert isinstance(client.provider_instance, FakeAnthropicProvider)
    assert client.litellm_call_type is None


def test_unionllm_passes_top_level_cache_control_to_native_provider(monkeypatch):
    class FakeAnthropicProvider:
        def __init__(self, **kwargs):
            self.calls = []

        def completion(self, model, messages, **kwargs):
            self.calls.append((model, messages, kwargs))
            return "ok"

    monkeypatch.setattr("unionllm.main.anthropic.AnthropicAIProvider", FakeAnthropicProvider)
    client = UnionLLM(provider="anthropic")

    result = client.completion(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        cache_control={"type": "ephemeral"},
    )

    assert result == "ok"
    assert client.provider_instance.calls[0][2]["cache_control"] == {"type": "ephemeral"}


def test_unionllm_can_fallback_to_litellm_for_anthropic(monkeypatch):
    class FakeLiteLLMProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def completion(self, model, messages, **kwargs):
            return None

    monkeypatch.setenv("UNIONLLM_USE_LITELLM_ANTHROPIC", "true")
    monkeypatch.setattr("unionllm.main.litellm.LiteLLMProvider", FakeLiteLLMProvider)

    client = UnionLLM(provider="anthropic")
    assert isinstance(client.provider_instance, FakeLiteLLMProvider)
    assert client.litellm_call_type == 1
