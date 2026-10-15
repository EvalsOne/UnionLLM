from unionllm.providers.azure import AzureAnthropicProvider


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DummyMessagesAPI:
    def __init__(self, response=None, events=None):
        self.calls = []
        self.response = response
        self.events = events or []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events) if kwargs.get("stream") else self.response


def make_provider(**api_kwargs):
    # Skip __init__: it builds a real AnthropicFoundry client with Azure credentials.
    provider = AzureAnthropicProvider.__new__(AzureAnthropicProvider)
    provider.client = Obj(messages=DummyMessagesAPI(**api_kwargs))
    return provider


MESSAGES = [
    {"role": "system", "content": "Agent instructions"},
    {"role": "system", "content": "# Long-Term Memory Context\n\n- prefers short answers"},
    {"role": "user", "content": "hello"},
]


def test_completion_passes_cache_control_and_keeps_every_system_message():
    response = Obj(
        id="msg_1",
        content=[Obj(type="text", text="hi")],
        stop_reason="end_turn",
        usage=Obj(input_tokens=3, output_tokens=2, cache_creation_input_tokens=0, cache_read_input_tokens=2032),
    )
    provider = make_provider(response=response)

    result = provider.completion("claude-sonnet-4-5", MESSAGES, cache_control={"type": "ephemeral"})

    call = provider.client.messages.calls[0]
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["system"] == "Agent instructions\n\n# Long-Term Memory Context\n\n- prefers short answers"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 2
    assert result.usage.cache_read_input_tokens == 2032
    assert result.usage.cache_creation_input_tokens == 0


def test_stream_passes_cache_control_and_merged_system():
    provider = make_provider(events=[])

    list(provider.completion("claude-sonnet-4-5", MESSAGES, stream=True, cache_control={"type": "ephemeral"}))

    call = provider.client.messages.calls[0]
    assert call["stream"] is True
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["system"].startswith("Agent instructions\n\n# Long-Term Memory Context")


def test_no_cache_control_unless_requested():
    provider = make_provider(response=Obj(id="m", content=[], stop_reason="end_turn", usage=None))

    provider.completion("claude-sonnet-4-5", [{"role": "user", "content": "hi"}])

    call = provider.client.messages.calls[0]
    assert "cache_control" not in call
    assert "system" not in call
