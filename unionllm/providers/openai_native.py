import os
from urllib.parse import urlparse, urlunparse

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError

from unionllm.utils import Choices, Delta, Message, ModelResponse, StreamingChoices, Usage

from .base_provider import BaseProvider


class OpenAINativeError(Exception):
    def __init__(self, status_code, message, request_id=None, model=None):
        self.status_code = status_code
        self.request_id = request_id
        self.model = model
        if request_id and "request_id" not in str(message):
            message = f"{message} (request_id: {request_id})"
        self.message = str(message)
        super().__init__(self.message)


class OpenAINativeProvider(BaseProvider):
    DEFAULT_BASE_URL = "https://api.openai.com"
    COMPLETION_MODELS = {"gpt-3.5-turbo-instruct", "davinci-002", "babbage-002"}
    CHAT_SUPPORTED_PARAMS = {
        "audio",
        "extra_body",
        "extra_headers",
        "extra_query",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "timeout",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
    }
    COMPLETIONS_SUPPORTED_PARAMS = {
        "best_of",
        "echo",
        "extra_body",
        "extra_headers",
        "extra_query",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "suffix",
        "temperature",
        "timeout",
        "top_p",
        "user",
    }

    def __init__(self, **model_kwargs):
        self.api_key = model_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAINativeError(status_code=422, message="Missing API key")

        raw_base_url = (
            model_kwargs.get("base_url")
            or model_kwargs.get("api_base")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or self.DEFAULT_BASE_URL
        )
        self.base_url = self._normalize_base_url(raw_base_url)
        self.client = self._build_client(self.api_key, self.base_url)

    def _build_client(self, api_key: str, base_url: str) -> OpenAI:
        return OpenAI(api_key=api_key, base_url=base_url)

    def _normalize_base_url(self, base_url: str) -> str:
        raw = str(base_url or self.DEFAULT_BASE_URL).strip()
        parsed = urlparse(raw)
        if not parsed.scheme:
            parsed = urlparse(f"https://{raw}")

        path = (parsed.path or "").rstrip("/")
        if not path:
            path = "/v1"
        elif not path.endswith("/v1"):
            path = f"{path}/v1"

        return urlunparse(parsed._replace(path=path))

    def _resolve_client(self, **kwargs) -> OpenAI:
        api_key = kwargs.get("api_key") or self.api_key
        if not api_key:
            raise OpenAINativeError(status_code=422, message="Missing API key")

        base_url = self._normalize_base_url(
            kwargs.get("base_url") or kwargs.get("api_base") or self.base_url
        )
        if api_key == self.api_key and base_url == self.base_url:
            return self.client
        return self._build_client(api_key, base_url)

    def _resolve_api_mode(self, model: str, **kwargs) -> str:
        requested_mode = kwargs.get("api_mode") or kwargs.get("api_type") or kwargs.get("endpoint")
        if requested_mode:
            normalized = str(requested_mode).strip().lower().replace("-", "_")
            if normalized in {"completion", "completions", "text_completion", "text_completions"}:
                return "completions"
            return "chat"

        if kwargs.get("prompt") is not None:
            return "completions"

        if str(model or "").lower() in self.COMPLETION_MODELS:
            return "completions"

        return "chat"

    def _include_stream_usage(self, params: dict) -> dict:
        stream_options = params.get("stream_options")
        if stream_options is None:
            params["stream_options"] = {"include_usage": True}
        elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
            merged = dict(stream_options)
            merged["include_usage"] = True
            params["stream_options"] = merged
        return params

    def _preprocess_chat_params(self, **kwargs) -> dict:
        return {key: value for key, value in kwargs.items() if key in self.CHAT_SUPPORTED_PARAMS}

    def _preprocess_completion_params(self, **kwargs) -> dict:
        return {key: value for key, value in kwargs.items() if key in self.COMPLETIONS_SUPPORTED_PARAMS}

    def _normalize_messages(self, model: str, messages: list) -> list:
        check = self.check_prompt("openai", model, messages)
        if not check.get("pass_check"):
            raise OpenAINativeError(status_code=422, message=str(check.get("reason")), model=model)
        return check.get("messages", messages)

    def _stringify_message_content(self, content, model: str):
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                    raise OpenAINativeError(
                        status_code=422,
                        message="Completions API only supports text messages",
                        model=model,
                    )
                text_parts.append(part["text"])
            return "\n".join(text_parts)

        raise OpenAINativeError(
            status_code=422,
            message="Completions API only supports text messages",
            model=model,
        )

    def _messages_to_prompt(self, messages: list, model: str) -> str:
        prompt_parts = []
        for message in messages or []:
            role = str(message.get("role") or "user")
            content = self._stringify_message_content(message.get("content"), model)
            if not content:
                continue

            if role == "tool":
                name = message.get("name")
                label = f"Tool ({name})" if name else "Tool"
            elif role == "assistant":
                label = "Assistant"
            elif role == "system":
                label = "System"
            else:
                label = "User"

            prompt_parts.append(f"{label}: {content}")

        return "\n\n".join(prompt_parts)

    def _create_completion_response(self, result, model: str) -> ModelResponse:
        choices = []
        for choice in result.choices:
            message = Message(content=choice.text, role="assistant")
            choices.append(
                Choices(
                    message=message,
                    index=choice.index,
                    finish_reason=choice.finish_reason,
                )
            )

        usage_dict = result.usage.model_dump() if getattr(result, "usage", None) else {}
        usage = Usage(**usage_dict) if usage_dict else Usage()

        return ModelResponse(
            id=result.id,
            choices=choices,
            created=result.created,
            model=model,
            usage=usage,
            object=getattr(result, "object", "text_completion"),
            system_fingerprint=getattr(result, "system_fingerprint", None),
        )

    def _post_completion_stream_processing(self, response, model: str):
        for chunk in response:
            data = chunk.model_dump()
            chunk_choices = []

            for choice in data.get("choices", []):
                text = choice.get("text")
                finish_reason = choice.get("finish_reason")
                index = choice.get("index", 0)

                if text:
                    chunk_delta = Delta(content=text)
                    stream_choice = StreamingChoices(
                        index=index,
                        delta=chunk_delta,
                        finish_reason=finish_reason,
                    )
                else:
                    stream_choice = StreamingChoices(
                        index=index,
                        finish_reason=finish_reason,
                        delta=Delta(),
                    )

                if choice.get("logprobs") is not None:
                    stream_choice.logprobs = choice.get("logprobs")
                chunk_choices.append(stream_choice)

            usage_data = data.get("usage") or None
            chunk_usage = Usage(**usage_data) if usage_data else None

            yield ModelResponse(
                id=data["id"],
                choices=chunk_choices,
                created=data["created"],
                model=model,
                usage=chunk_usage,
                stream=True,
                system_fingerprint=data.get("system_fingerprint"),
            )

    def _raise_mapped_error(self, exc: Exception, model: str):
        status_code = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        response = getattr(exc, "response", None)

        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if request_id is None and response is not None and getattr(response, "headers", None):
            request_id = response.headers.get("x-request-id")

        raise OpenAINativeError(
            status_code=status_code or 500,
            message=getattr(exc, "message", str(exc)),
            request_id=request_id,
            model=model,
        ) from exc

    def completion(self, model: str, messages: list, **kwargs):
        if not model:
            raise OpenAINativeError(status_code=422, message="Missing model", model=model)

        api_mode = self._resolve_api_mode(model, **kwargs)
        client = self._resolve_client(**kwargs)

        try:
            if api_mode == "completions":
                prompt = kwargs.get("prompt")
                if prompt is None:
                    if messages is None:
                        raise OpenAINativeError(
                            status_code=422,
                            message="Missing messages or prompt",
                            model=model,
                        )
                    messages = self._normalize_messages(model, messages)
                    prompt = self._messages_to_prompt(messages, model)

                params = self._preprocess_completion_params(**kwargs)
                stream = bool(params.get("stream", False))
                if stream:
                    params = self._include_stream_usage(params)
                    response = client.completions.create(
                        model=model,
                        prompt=prompt,
                        **params,
                    )
                    return self._post_completion_stream_processing(response, model=model)

                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    **params,
                )
                return self._create_completion_response(response, model=model)

            if messages is None:
                raise OpenAINativeError(
                    status_code=422,
                    message="Missing messages",
                    model=model,
                )

            messages = self._normalize_messages(model, messages)
            params = self._preprocess_chat_params(**kwargs)
            stream = bool(params.get("stream", False))
            if stream:
                params = self._include_stream_usage(params)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **params,
                )
                return self.post_stream_processing(response, model=model)

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **params,
            )
            return self.create_model_response(response, model=model)
        except OpenAINativeError:
            raise
        except (APIStatusError, APIConnectionError, APITimeoutError, OpenAIError) as exc:
            self._raise_mapped_error(exc, model=model)
        except Exception as exc:
            self._raise_mapped_error(exc, model=model)
