import pytest
import base64


from unionllm.providers.moonshot import MoonshotAIProvider, MoonshotOpenAIError
from unionllm.providers import moonshot as moonshot_module


@pytest.fixture()
def provider():
    return MoonshotAIProvider(api_key="test-moonshot-key")


def test_check_prompt_allows_image_url_parts(provider):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    check = provider.check_prompt("moonshot", "kimi-k2.5", messages)
    assert check["pass_check"] is True
    assert check["messages"][0]["content"][0]["type"] == "image_url"


def test_check_prompt_keeps_video_url_parts_for_moonshot(provider):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    check = provider.check_prompt("moonshot", "kimi-k2.5", messages)
    assert check["pass_check"] is True
    assert check["messages"][0]["content"][0]["type"] == "video_url"


def test_moonshot_rejects_non_base64_image_urls(provider):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    with pytest.raises(MoonshotOpenAIError) as excinfo:
        provider._ensure_base64_multimodal("moonshot-v1-8k", messages)
    assert excinfo.value.status_code == 422


def test_moonshot_rejects_non_base64_video_urls(provider):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    with pytest.raises(MoonshotOpenAIError) as excinfo:
        provider._ensure_base64_multimodal("moonshot-v1-8k", messages)
    assert excinfo.value.status_code == 422


def test_moonshot_rejects_data_uri_without_base64(provider):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png,AAAA"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    with pytest.raises(MoonshotOpenAIError) as excinfo:
        provider._ensure_base64_multimodal("kimi-k2.5", messages)
    assert excinfo.value.status_code == 422


def test_k25_converts_image_http_url_to_data_uri(provider, monkeypatch):
    raw = b"fake-image-bytes"

    class Resp:
        def __init__(self, content, headers):
            self.content = content
            self.headers = headers

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        assert url == "https://example.com/a.png"
        assert timeout == 30
        return Resp(raw, {"Content-Type": "image/png"})

    monkeypatch.setattr(moonshot_module.requests, "get", fake_get)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    normalized = provider._ensure_base64_multimodal("kimi-k2.5", messages)
    url = normalized[0]["content"][0]["image_url"]["url"]
    assert url == f"data:image/png;base64,{base64.b64encode(raw).decode('utf-8')}"


def test_k25_converts_video_http_url_to_data_uri(provider, monkeypatch):
    raw = b"fake-video-bytes"

    class Resp:
        def __init__(self, content, headers):
            self.content = content
            self.headers = headers

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        assert url == "https://example.com/a.mp4"
        assert timeout == 60
        return Resp(raw, {"Content-Type": "video/mp4"})

    monkeypatch.setattr(moonshot_module.requests, "get", fake_get)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    normalized = provider._ensure_base64_multimodal("kimi-k2.5", messages)
    url = normalized[0]["content"][0]["video_url"]["url"]
    assert url == f"data:video/mp4;base64,{base64.b64encode(raw).decode('utf-8')}"
