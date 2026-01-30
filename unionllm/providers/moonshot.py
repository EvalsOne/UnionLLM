from .base_provider import BaseProvider
from openai import OpenAI
import base64
import logging, json, os
import mimetypes
from urllib.parse import urlparse

import requests


class MoonshotOpenAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class MoonshotAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        # Get MOONSHOT_API_KEY from environment variables
        _env_api_key = os.environ.get("MOONSHOT_API_KEY")
        self.api_key = model_kwargs.get("api_key") if model_kwargs.get("api_key") else _env_api_key
        if not self.api_key:
            raise MoonshotOpenAIError(
                status_code=422, message=f"Missing API key"
            )
        self.base_url = model_kwargs.get("base_url") or "https://api.moonshot.cn/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _merge_extra_body(self, kwargs: dict, extra: dict) -> None:
        existing = kwargs.get("extra_body")
        if existing is None:
            kwargs["extra_body"] = dict(extra)
            return
        if not isinstance(existing, dict):
            raise MoonshotOpenAIError(
                status_code=422,
                message="Invalid extra_body: must be a dict/object when using Moonshot-specific params.",
            )
        merged = dict(existing)
        merged.update(extra)
        kwargs["extra_body"] = merged

    def _parse_float(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_data_uri_base64(self, part_type: str, url: str) -> tuple[str, str]:
        header, sep, data = url.partition(",")
        if not sep or not data:
            raise MoonshotOpenAIError(
                status_code=422,
                message=f"Invalid {part_type} data URI: missing base64 payload",
            )
        if ";base64" not in header.lower():
            raise MoonshotOpenAIError(
                status_code=422,
                message=f"Invalid {part_type} data URI: must be base64-encoded",
            )
        return header, data

    def _guess_mime_type(self, url: str, fallback: str) -> str:
        try:
            parsed = urlparse(url)
            mime, _ = mimetypes.guess_type(parsed.path or url)
            return mime or fallback
        except Exception:
            return fallback

    def _encode_bytes_as_data_uri(self, media_type: str, raw: bytes) -> str:
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:{media_type};base64,{encoded}"

    def _fetch_url_as_data_uri(self, url: str, *, fallback_mime: str, timeout_s: int) -> str:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        content_type = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip()
        if not content_type or content_type in ("application/octet-stream", "binary/octet-stream"):
            content_type = self._guess_mime_type(url, fallback_mime)
        return self._encode_bytes_as_data_uri(content_type, resp.content)

    def _read_file_as_data_uri(self, path: str, *, fallback_mime: str) -> str:
        if not os.path.exists(path):
            raise MoonshotOpenAIError(status_code=422, message=f"File not found: {path}")
        with open(path, "rb") as f:
            raw = f.read()
        mime = self._guess_mime_type(path, fallback_mime)
        return self._encode_bytes_as_data_uri(mime, raw)

    def _ensure_base64_multimodal(self, model: str, messages: list) -> list:
        """
        Moonshot's multimodal inputs require base64 data URIs. For kimi-k2.5, we
        accept http(s) URLs (and local file paths) and convert them to base64 data URIs
        right before calling the endpoint.
        """
        model_name = str(model or "").lower()
        allow_url_fetch = model_name.startswith("kimi-k2.5")

        normalized_messages = []
        for message in messages or []:
            content = message.get("content")
            if not isinstance(content, list):
                normalized_messages.append(message)
                continue

            new_message = dict(message)
            new_content = []

            for part in content:
                if not isinstance(part, dict):
                    new_content.append(part)
                    continue

                part_type = part.get("type")
                if part_type not in ("image_url", "video_url"):
                    new_content.append(part)
                    continue

                payload_key = "image_url" if part_type == "image_url" else "video_url"
                payload = part.get(payload_key) or {}
                url = payload.get("url") if isinstance(payload, dict) else None
                if not isinstance(url, str) or not url:
                    raise MoonshotOpenAIError(
                        status_code=422,
                        message=f"Invalid {part_type} input: missing '{payload_key}.url'",
                    )

                if url.startswith("data:"):
                    self._parse_data_uri_base64(part_type, url)
                    new_content.append(part)
                    continue

                if not allow_url_fetch:
                    raise MoonshotOpenAIError(
                        status_code=422,
                        message=(
                            f"Moonshot {part_type} requires base64 data URI for model={model}; "
                            f"received non-data URL: {url}"
                        ),
                    )

                if url.startswith(("http://", "https://")):
                    if part_type == "image_url":
                        new_url = self._fetch_url_as_data_uri(url, fallback_mime="image/jpeg", timeout_s=30)
                    else:
                        new_url = self._fetch_url_as_data_uri(url, fallback_mime="video/mp4", timeout_s=60)
                else:
                    if part_type == "image_url":
                        new_url = self._read_file_as_data_uri(url, fallback_mime="image/jpeg")
                    else:
                        new_url = self._read_file_as_data_uri(url, fallback_mime="video/mp4")

                new_part = dict(part)
                new_payload = dict(payload) if isinstance(payload, dict) else {}
                new_payload["url"] = new_url
                new_part[payload_key] = new_payload
                new_content.append(new_part)

            new_message["content"] = new_content
            normalized_messages.append(new_message)

        return normalized_messages

    def pre_processing(self, model: str, **kwargs):
        # process the compatibility issue of parameters, all unsupported parameters are discarded
        # Moonshot-specific fields (e.g. thinking) must be passed via OpenAI SDK's extra_body.
        thinking = kwargs.pop("thinking", None)

        supported_params = [
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "n",
            "logprobs",
            "top_logprobs",
            "stream",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "best_of",
            "logit_bias",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "seed",
            "response_format",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)

        if thinking is not None:
            self._merge_extra_body(kwargs, {"thinking": thinking})

        # kimi-k2.5 temperature constraints:
        # - thinking enabled (default) => temperature must be 1.0
        # - thinking disabled => temperature must be 0.6
        model_name = str(model or "").lower()
        if model_name.startswith("kimi-k2.5"):
            thinking_cfg = None
            extra_body = kwargs.get("extra_body")
            if isinstance(extra_body, dict):
                thinking_cfg = extra_body.get("thinking")

            thinking_type = None
            if isinstance(thinking_cfg, dict):
                thinking_type = thinking_cfg.get("type")
            elif isinstance(thinking_cfg, str):
                thinking_type = thinking_cfg

            is_non_thinking = str(thinking_type or "").lower() == "disabled"
            required_temperature = 0.6 if is_non_thinking else 1.0

            provided_temperature = kwargs.get("temperature")
            provided_temperature_f = self._parse_float(provided_temperature)
            if provided_temperature is not None and provided_temperature_f is not None:
                if abs(provided_temperature_f - required_temperature) > 1e-9:
                    logging.getLogger(__name__).warning(
                        "Moonshot model %s enforces temperature %.1f for thinking=%s; overriding provided temperature=%s",
                        model,
                        required_temperature,
                        "disabled" if is_non_thinking else "enabled",
                        provided_temperature,
                    )

            kwargs["temperature"] = required_temperature

        return kwargs
    
    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        result = self.client.chat.completions.create(
            model=model, messages=messages, **new_kwargs
        )
        return self.post_stream_processing(result)

    def create_model_response_wrapper(self, result, model):
        # 调用 response_model 中的 create_model_response 方法
        return self.create_model_response(result, model=model)

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise MoonshotOpenAIError(
                    status_code=422, message=f"Missing model or messages"
                )
                
            message_check_result = self.check_prompt("moonshot", model, messages)            
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise MoonshotOpenAIError(
                    status_code=422, message=message_check_result['reason']
                )

            messages = self._ensure_base64_multimodal(model, messages)
                
            new_kwargs = self.pre_processing(model=model, **kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model=model, messages=messages, **new_kwargs)
            else:
                result = self.client.chat.completions.create(
                    model=model, messages=messages, **new_kwargs
                )
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise MoonshotOpenAIError(status_code=e.status_code, message=str(e))
            else:
                raise MoonshotOpenAIError(status_code=500, message=str(e))
