import pytest


from unionllm.providers.moonshot import MoonshotAIProvider, MoonshotOpenAIError


class TestMoonshotK25Temperature:
    @pytest.fixture()
    def provider(self):
        return MoonshotAIProvider(api_key="test-moonshot-key")

    def test_k25_defaults_to_thinking_temperature(self, provider):
        kwargs = provider.pre_processing(model="kimi-k2.5", stream=False)
        assert kwargs["temperature"] == 1.0

    def test_k25_non_thinking_temperature_and_extra_body(self, provider):
        kwargs = provider.pre_processing(
            model="kimi-k2.5",
            thinking={"type": "disabled"},
        )
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.6
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    def test_k25_overrides_mismatched_temperature(self, provider):
        kwargs = provider.pre_processing(
            model="kimi-k2.5",
            thinking={"type": "disabled"},
            temperature=1.0,
        )
        assert kwargs["temperature"] == 0.6

    def test_merges_existing_extra_body(self, provider):
        kwargs = provider.pre_processing(
            model="kimi-k2.5",
            thinking={"type": "disabled"},
            extra_body={"foo": "bar"},
        )
        assert kwargs["extra_body"]["foo"] == "bar"
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}

    def test_invalid_extra_body_raises(self, provider):
        with pytest.raises(MoonshotOpenAIError) as excinfo:
            provider.pre_processing(
                model="kimi-k2.5",
                thinking={"type": "disabled"},
                extra_body=["not-a-dict"],
            )
        assert excinfo.value.status_code == 422

