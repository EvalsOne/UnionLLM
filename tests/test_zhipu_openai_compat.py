from unionllm.providers.zhipu import ZhipuAIProvider


class DummyCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyClient:
    def __init__(self):
        self.chat = DummyChat()


def test_zhipu_uses_bigmodel_openai_compatible_base_url():
    provider = ZhipuAIProvider(api_key="test-zhipu-key")

    assert provider.base_url == "https://open.bigmodel.cn/api/paas/v4"


def test_zhipu_allows_custom_api_base():
    provider = ZhipuAIProvider(
        api_key="test-zhipu-key",
        api_base="https://example.com/custom/v1",
    )

    assert provider.base_url == "https://example.com/custom/v1"


def test_zhipu_pre_processing_keeps_openai_compatible_params():
    provider = ZhipuAIProvider(api_key="test-zhipu-key")

    kwargs = provider.pre_processing(
        provider="zhipuai",
        api_key="test-zhipu-key",
        stream=True,
        temperature=1.0,
        top_p=0.9,
        response_format={"type": "json_object"},
        extra_body={"request_id": "abc"},
        timeout=30,
    )

    assert kwargs == {
        "stream": True,
        "temperature": 1.0,
        "top_p": 0.9,
        "response_format": {"type": "json_object"},
        "extra_body": {"request_id": "abc"},
        "timeout": 30,
    }


def test_zhipu_completion_calls_openai_compatible_client():
    provider = ZhipuAIProvider(api_key="test-zhipu-key")
    provider.client = DummyClient()
    provider.create_model_response_wrapper = lambda result, model: {
        "result": result,
        "model": model,
    }

    response = provider.completion(
        model="glm-5.1",
        messages=[{"role": "user", "content": "你好"}],
        provider="zhipuai",
        api_key="test-zhipu-key",
        temperature=1.0,
        stream=False,
    )

    assert response == {"result": {"ok": True}, "model": "glm-5.1"}
    assert provider.client.chat.completions.calls == [
        {
            "model": "glm-5.1",
            "messages": [{"role": "user", "content": "你好"}],
            "temperature": 1.0,
            "stream": False,
        }
    ]
