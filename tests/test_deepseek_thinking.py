import pytest


from unionllm.providers.deepseek import DeepSeekAIProvider, DeepSeekError


class TestDeepSeekThinking:
    @pytest.fixture()
    def provider(self):
        return DeepSeekAIProvider(api_key="test-deepseek-key")

    def test_defaults_thinking_to_disabled_in_extra_body(self, provider):
        kwargs = provider.pre_processing(stream=False)

        assert "thinking" not in kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    def test_preserves_explicit_thinking_in_extra_body(self, provider):
        kwargs = provider.pre_processing(
            stream=False,
            thinking={"type": "enabled"},
            extra_body={"foo": "bar"},
        )

        assert "thinking" not in kwargs
        assert kwargs["extra_body"]["foo"] == "bar"
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    def test_reasoning_effort_enables_thinking(self, provider):
        kwargs = provider.pre_processing(
            stream=False,
            reasoning_effort="high",
        )

        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    def test_invalid_extra_body_raises(self, provider):
        with pytest.raises(DeepSeekError) as excinfo:
            provider.pre_processing(extra_body=["not-a-dict"])

        assert excinfo.value.status_code == 422