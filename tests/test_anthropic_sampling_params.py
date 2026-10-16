"""Sampling parameters reach Claude without breaking the anthropic SDK.

anthropic 1.x removed temperature/top_p/top_k from Messages.create, so passing
one as a keyword is a TypeError before any request is sent. The fake clients in
the other provider tests accept any keyword, so these tests go through real SDK
clients (Anthropic and AnthropicFoundry) over a mock transport and check the
JSON the SDK actually sends. They pass on anthropic 0.x and 1.x.
"""

import importlib
import json

import anthropic
import pytest
from anthropic import Anthropic, AnthropicFoundry

from unionllm.providers.anthropic import AnthropicAIProvider, place_sampling_params
from unionllm.providers.azure import AzureAnthropicProvider

# anthropic 1.x is built on httpx2; 0.x on httpx. The transport must match.
http = importlib.import_module("httpx2" if int(anthropic.__version__.split(".")[0]) >= 1 else "httpx")

MESSAGES = [{"role": "user", "content": "hi"}]


def _message(model):
    return {
        "id": "msg_mock", "type": "message", "role": "assistant", "model": model,
        "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _sse(model):
    message = _message(model)
    events = [
        {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
         "usage": {"input_tokens": 1, "output_tokens": 1,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _mock_http(sent):
    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        if body.get("stream"):
            return http.Response(200, text=_sse(body["model"]), headers={"content-type": "text/event-stream"})
        return http.Response(200, json=_message(body["model"]))

    return http.Client(transport=http.MockTransport(handle))


def _anthropic_provider(sent):
    provider = AnthropicAIProvider(api_key="mock-only")
    provider.client = Anthropic(api_key="mock-only", base_url="https://mock.local", http_client=_mock_http(sent))
    return provider


def _azure_provider(sent):
    # Skip __init__: the real one is fine, but this keeps the Foundry client on the mock transport.
    provider = AzureAnthropicProvider.__new__(AzureAnthropicProvider)
    provider.client = AnthropicFoundry(
        api_key="mock-only", base_url="https://mock.local/anthropic", http_client=_mock_http(sent),
    )
    return provider


def _send(make_provider, model, stream, **kwargs):
    sent = []
    provider = make_provider(sent)
    # Explicit max_tokens: Azure's 30000 default makes the SDK refuse a
    # non-streaming request ("Streaming is required ...").
    kwargs.setdefault("max_tokens", 2048)
    result = provider.completion(model=model, messages=MESSAGES, stream=stream, **kwargs)
    if stream:
        list(result)
    assert len(sent) == 1
    return sent[0]


PROVIDERS = pytest.mark.parametrize("make_provider", [_anthropic_provider, _azure_provider], ids=["anthropic", "azure"])
STREAM = pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])


@PROVIDERS
@STREAM
def test_older_claude_keeps_temperature(make_provider, stream):
    body = _send(make_provider, "claude-sonnet-4-5", stream, temperature=0.3)
    assert body["temperature"] == 0.3


@PROVIDERS
@STREAM
def test_reasoning_drops_sampling_instead_of_forcing_temperature(make_provider, stream):
    body = _send(make_provider, "claude-sonnet-4-5", stream, temperature=0.3, top_p=0.9, reasoning_effort="high")
    assert body["thinking"]["type"] == "enabled"
    assert not {"temperature", "top_p", "top_k"} & body.keys()


@STREAM
def test_adaptive_only_claude_gets_effort_and_no_sampling(stream):
    body = _send(_anthropic_provider, "claude-haiku-5-5", stream, top_p=0.9, reasoning_effort="low")
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"}
    assert "top_p" not in body


def test_anthropic_top_k_reaches_api():
    body = _send(_anthropic_provider, "claude-sonnet-4-5", False, top_k=40)
    assert body["top_k"] == 40


def test_azure_prefers_temperature_over_top_p():
    body = _send(_azure_provider, "claude-sonnet-4-5", False, temperature=0.3, top_p=0.9)
    assert body["temperature"] == 0.3
    assert "top_p" not in body


def test_place_sampling_params_rules():
    # Older models: moved into extra_body; a caller's extra_body value wins.
    assert place_sampling_params(
        {"temperature": 0.3, "top_p": None, "extra_body": {"temperature": 0.7, "x": 1}}, "claude-sonnet-4-6"
    ) == {"extra_body": {"temperature": 0.7, "x": 1}}
    # Thinking on (either mode) or an adaptive-only model: dropped.
    assert place_sampling_params({"temperature": 1, "thinking": {"type": "enabled"}}, "claude-sonnet-4-5") == {
        "thinking": {"type": "enabled"}
    }
    for model in ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert place_sampling_params({"top_p": 0.9}, model) == {}
    # Nothing to move: params unchanged, no empty extra_body.
    assert place_sampling_params({"max_tokens": 10}, "claude-sonnet-4-5") == {"max_tokens": 10}


def test_4_6_and_older_keep_budget_thinking():
    for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5"):
        assert not AnthropicAIProvider._uses_adaptive_thinking(model)
