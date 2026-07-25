from types import SimpleNamespace

from unionllm.providers.gemini import GeminiAIProvider
from unionllm.utils import Usage
from google.genai import types


class _RecordingModels:
    def __init__(self):
        self.contents = None
        self.config = None

    def generate_content_stream(self, *, model, contents, config):
        self.contents = contents
        self.config = config
        return []


def test_streaming_tool_result_uses_user_role_and_call_id():
    """Gemini function responses must be supplied as user content, not role=function."""
    provider = object.__new__(GeminiAIProvider)
    models = _RecordingModels()
    provider.client = SimpleNamespace(models=models)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Shanghai"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_1",
            "name": "get_weather",
            "content": '{"temperature": 30}',
        },
    ]

    assert list(provider.post_stream_processing_wrapper("gemini-3.6-flash", messages)) == []

    tool_result = models.contents[1]
    assert tool_result.role == "user"
    assert tool_result.parts[0].function_response.name == "get_weather"
    assert tool_result.parts[0].function_response.id == "call_weather_1"


def test_gemini_usage_keeps_cached_content_token_count():
    usage = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120)

    GeminiAIProvider._attach_cached_content_usage(
        usage,
        SimpleNamespace(cached_content_token_count=64),
    )

    assert usage.model_dump()["cached_content_token_count"] == 64


class _RecordingCaches:
    def __init__(self):
        self.calls = []

    def create(self, *, model, config):
        self.calls.append({"model": model, "config": config})
        return SimpleNamespace(name="cachedContents/long-horizon")


def test_long_gemini_context_uses_explicit_cached_content():
    provider = object.__new__(GeminiAIProvider)
    provider.client = SimpleNamespace(caches=_RecordingCaches())
    provider._context_caches = {}
    prefix = "a" * 18000
    contents = [
        types.Content(role="user", parts=[types.Part(text=prefix)]),
        types.Content(role="user", parts=[types.Part(text="What should I do next?")]),
    ]
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        tools=[types.Tool(function_declarations=[])],
    )

    request_contents = provider._prepare_context_cache(
        "gemini-3.6-flash",
        contents,
        config,
        {"enable_context_cache": True, "context_cache_min_tokens": 4096},
    )

    assert request_contents == contents[-1:]
    assert config.cached_content == "cachedContents/long-horizon"
    assert config.system_instruction is None
    assert config.tools is None
    assert provider.client.caches.calls[0]["config"].contents == contents[:-1]


def test_streaming_long_context_sends_only_uncached_suffix():
    provider = object.__new__(GeminiAIProvider)
    models = _RecordingModels()
    caches = _RecordingCaches()
    provider.client = SimpleNamespace(models=models, caches=caches)
    provider._context_caches = {}
    messages = [
        {"role": "user", "content": "a" * 18000},
        {"role": "user", "content": "Answer the latest question."},
    ]

    assert list(
        provider.post_stream_processing_wrapper(
            "gemini-3.6-flash",
            messages,
            enable_context_cache=True,
            context_cache_min_tokens=4096,
        )
    ) == []

    assert len(caches.calls) == 1
    assert len(models.contents) == 1
    assert models.contents[0].parts[0].text == "Answer the latest question."
    assert models.config.cached_content == "cachedContents/long-horizon"
