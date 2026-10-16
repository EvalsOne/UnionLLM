import json
import os
import re
from urllib.parse import urlparse, urlunparse

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, OpenAIError

from unionllm.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    StreamingChoices,
    Usage,
)

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
    GPT_5_6_MODEL_PATTERN = re.compile(r"(?:^|/)gpt-5\.6(?:$|[-.])", re.IGNORECASE)
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
    RESPONSES_SUPPORTED_PARAMS = {
        "background",
        "conversation",
        "extra_body",
        "extra_headers",
        "extra_query",
        "include",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "text",
        "timeout",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
    RESPONSES_UNSUPPORTED_CHAT_PARAMS = {
        "audio",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "modalities",
        "n",
        "prediction",
        "presence_penalty",
        "seed",
        "stop",
        "web_search_options",
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
        self.api_mode = model_kwargs.get("api_mode")
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
        requested_mode = (
            kwargs.get("api_mode")
            or self.api_mode
            or kwargs.get("api_type")
            or kwargs.get("endpoint")
        )
        if requested_mode:
            normalized = str(requested_mode).strip().lower().replace("-", "_")
            if normalized in {"completion", "completions", "text_completion", "text_completions"}:
                return "completions"
            if normalized in {"response", "responses", "responses_api"} or normalized.rstrip(
                "/"
            ).endswith("/responses"):
                return "responses"
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

    @staticmethod
    def _map_cached_tokens_to_usage(usage):
        """Expose OpenAI prompt-cache hits in UnionLLM's top-level usage contract."""
        if usage is None:
            return usage

        prompt_token_details = getattr(usage, "prompt_tokens_details", None)
        if hasattr(prompt_token_details, "model_dump"):
            prompt_token_details = prompt_token_details.model_dump()
        if not isinstance(prompt_token_details, dict):
            return usage

        cached_tokens = prompt_token_details.get("cached_tokens")
        if cached_tokens is not None:
            usage.cache_read_input_tokens = cached_tokens
        return usage

    def _preprocess_chat_params(self, **kwargs) -> dict:
        return {key: value for key, value in kwargs.items() if key in self.CHAT_SUPPORTED_PARAMS}

    def _preprocess_completion_params(self, **kwargs) -> dict:
        return {key: value for key, value in kwargs.items() if key in self.COMPLETIONS_SUPPORTED_PARAMS}

    @classmethod
    def _is_gpt_5_6_model(cls, model: str) -> bool:
        return bool(model and cls.GPT_5_6_MODEL_PATTERN.search(str(model)))

    @staticmethod
    def _has_function_tools(params: dict) -> bool:
        if params.get("functions"):
            return True
        return any(
            isinstance(tool, dict) and tool.get("type") == "function"
            for tool in (params.get("tools") or [])
        )

    def _ensure_chat_tool_reasoning_compatibility(self, model: str, params: dict) -> dict:
        if not self._is_gpt_5_6_model(model) or not self._has_function_tools(params):
            return params

        # GPT-5.6 Chat Completions only supports function tools when the
        # effective reasoning effort is none. Responses remains available via
        # api_mode="responses" for callers that need both reasoning and tools.
        params["reasoning_effort"] = "none"
        return params

    @staticmethod
    def _convert_response_format(response_format):
        if not isinstance(response_format, dict):
            return response_format
        if response_format.get("type") != "json_schema":
            return dict(response_format)

        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            return dict(response_format)
        return {"type": "json_schema", **json_schema}

    @staticmethod
    def _convert_tools_to_responses(tools, functions=None):
        converted = []
        for tool in tools or []:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                converted.append(tool)
                continue

            function = tool.get("function") or {}
            converted_tool = {
                "type": "function",
                "name": function.get("name"),
                "parameters": function.get("parameters") or {},
                # Chat Completions functions are non-strict by default.
                "strict": function.get("strict", False),
            }
            if function.get("description") is not None:
                converted_tool["description"] = function["description"]
            converted.append(converted_tool)

        for function in functions or []:
            if not isinstance(function, dict):
                continue
            converted_tool = {
                "type": "function",
                "name": function.get("name"),
                "parameters": function.get("parameters") or {},
                "strict": function.get("strict", False),
            }
            if function.get("description") is not None:
                converted_tool["description"] = function["description"]
            converted.append(converted_tool)
        return converted

    @staticmethod
    def _convert_tool_choice_to_responses(tool_choice, function_call=None):
        if tool_choice is None and function_call is not None:
            tool_choice = function_call
        if not isinstance(tool_choice, dict):
            return tool_choice

        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "name": function["name"]}
        if tool_choice.get("name"):
            return {"type": "function", "name": tool_choice["name"]}
        return tool_choice

    def _preprocess_responses_params(self, **kwargs) -> dict:
        unsupported = sorted(
            key
            for key in self.RESPONSES_UNSUPPORTED_CHAT_PARAMS
            if kwargs.get(key) is not None
        )
        if unsupported:
            raise OpenAINativeError(
                status_code=422,
                message=(
                    "Responses API does not support these Chat Completions parameters: "
                    + ", ".join(unsupported)
                ),
            )

        params = {
            key: value
            for key, value in kwargs.items()
            if key in self.RESPONSES_SUPPORTED_PARAMS
        }

        if "max_output_tokens" not in params:
            max_output_tokens = kwargs.get("max_completion_tokens")
            if max_output_tokens is None:
                max_output_tokens = kwargs.get("max_tokens")
            if max_output_tokens is not None:
                params["max_output_tokens"] = max_output_tokens

        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is not None and "reasoning" not in params:
            params["reasoning"] = {"effort": reasoning_effort}

        text = dict(params.get("text") or {})
        if kwargs.get("response_format") is not None and "format" not in text:
            text["format"] = self._convert_response_format(kwargs["response_format"])
        if kwargs.get("verbosity") is not None and "verbosity" not in text:
            text["verbosity"] = kwargs["verbosity"]
        if text:
            params["text"] = text

        tools = self._convert_tools_to_responses(
            kwargs.get("tools"),
            functions=kwargs.get("functions"),
        )
        if tools:
            params["tools"] = tools

        tool_choice = self._convert_tool_choice_to_responses(
            kwargs.get("tool_choice"),
            function_call=kwargs.get("function_call"),
        )
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        return params

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

    @staticmethod
    def _stringify_tool_output(content):
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _convert_content_to_responses(content, role: str):
        if not isinstance(content, list):
            return content

        converted = []
        for part in content:
            if not isinstance(part, dict):
                converted.append(part)
                continue

            part_type = part.get("type")
            if part_type == "text":
                converted.append(
                    {
                        "type": "input_text",
                        "text": part.get("text", ""),
                    }
                )
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    converted_part = {
                        "type": "input_image",
                        "image_url": image_url.get("url"),
                    }
                    if image_url.get("detail") is not None:
                        converted_part["detail"] = image_url["detail"]
                    converted.append(converted_part)
                else:
                    converted.append({"type": "input_image", "image_url": image_url})
            else:
                converted.append(dict(part))
        return converted

    def _messages_to_responses_input(self, messages: list, model: str) -> list:
        items = []
        for raw_message in messages or []:
            message = self.convert_object_to_dict(raw_message)
            role = message.get("role")

            if role == "tool":
                call_id = message.get("tool_call_id")
                if not call_id:
                    raise OpenAINativeError(
                        status_code=422,
                        message="Responses API tool messages require tool_call_id",
                        model=model,
                    )
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": self._stringify_tool_output(message.get("content")),
                    }
                )
                continue

            content = message.get("content")
            if content not in (None, "", []):
                items.append(
                    {
                        "role": role,
                        "content": self._convert_content_to_responses(content, role),
                    }
                )

            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id"),
                            "name": function.get("name"),
                            "arguments": arguments,
                        }
                    )

        return items

    @staticmethod
    def _get_response_value(payload, key, default=None):
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)

    def _create_responses_usage(self, usage_payload):
        if usage_payload is None:
            return Usage()
        usage_data = self.convert_object_to_dict(usage_payload)
        input_details = usage_data.get("input_tokens_details") or {}
        output_details = usage_data.get("output_tokens_details") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens"),
            completion_tokens=usage_data.get("output_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            prompt_tokens_details=input_details,
            completion_tokens_details=output_details,
        )
        cached_tokens = input_details.get("cached_tokens")
        if cached_tokens is not None:
            usage.cache_read_input_tokens = cached_tokens
        cache_write_tokens = input_details.get("cache_write_tokens")
        if cache_write_tokens is not None:
            usage.cache_creation_input_tokens = cache_write_tokens
        return usage

    def _responses_finish_reason(self, response, has_tool_calls: bool) -> str:
        if has_tool_calls:
            return "tool_calls"
        if self._get_response_value(response, "status") != "incomplete":
            return "stop"

        details = self._get_response_value(response, "incomplete_details")
        reason = self._get_response_value(details, "reason")
        if reason == "max_output_tokens":
            return "length"
        return reason or "stop"

    def _create_responses_model_response(self, response, model: str) -> ModelResponse:
        text_parts = []
        tool_calls = []
        annotations = []

        for item in self._get_response_value(response, "output", []) or []:
            item_type = self._get_response_value(item, "type")
            if item_type == "message":
                for part in self._get_response_value(item, "content", []) or []:
                    part_type = self._get_response_value(part, "type")
                    if part_type == "output_text":
                        text_parts.append(self._get_response_value(part, "text", ""))
                        annotations.extend(
                            self._get_response_value(part, "annotations", []) or []
                        )
                    elif part_type == "refusal":
                        text_parts.append(self._get_response_value(part, "refusal", ""))
            elif item_type == "function_call":
                function = Function(
                    name=self._get_response_value(item, "name"),
                    arguments=self._get_response_value(item, "arguments", ""),
                )
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=self._get_response_value(item, "call_id")
                        or self._get_response_value(item, "id"),
                        type="function",
                        function=function,
                    )
                )

        message = Message(
            content="".join(text_parts) if text_parts else None,
            role="assistant",
            tool_calls=tool_calls or None,
        )
        if annotations:
            message.annotations = self.convert_object_to_dict(annotations)

        return ModelResponse(
            id=self._get_response_value(response, "id"),
            choices=[
                Choices(
                    index=0,
                    finish_reason=self._responses_finish_reason(
                        response,
                        has_tool_calls=bool(tool_calls),
                    ),
                    message=message,
                )
            ],
            created=self._get_response_value(response, "created_at"),
            model=model,
            usage=self._create_responses_usage(
                self._get_response_value(response, "usage")
            ),
            object="chat.completion",
        )

    def _post_responses_stream_processing(self, response, model: str):
        response_id = None
        created_at = None
        role_emitted = False

        for event in response:
            event_type = self._get_response_value(event, "type")

            if event_type == "response.created":
                created_response = self._get_response_value(event, "response")
                response_id = self._get_response_value(created_response, "id")
                created_at = self._get_response_value(created_response, "created_at")
                continue

            if event_type == "response.output_text.delta":
                delta = Delta(
                    content=self._get_response_value(event, "delta", ""),
                    role=None if role_emitted else "assistant",
                )
                role_emitted = True
                yield ModelResponse(
                    id=response_id,
                    choices=[StreamingChoices(index=0, delta=delta)],
                    created=created_at,
                    model=model,
                    usage=None,
                    stream=True,
                )
                continue

            if event_type == "response.output_item.added":
                item = self._get_response_value(event, "item")
                if self._get_response_value(item, "type") != "function_call":
                    continue
                index = self._get_response_value(event, "output_index", 0)
                delta = Delta(role=None if role_emitted else "assistant")
                role_emitted = True
                delta.tool_calls = [
                    {
                        "index": index,
                        "id": self._get_response_value(item, "call_id")
                        or self._get_response_value(item, "id"),
                        "type": "function",
                        "function": {
                            "name": self._get_response_value(item, "name"),
                            "arguments": self._get_response_value(item, "arguments", ""),
                        },
                    }
                ]
                yield ModelResponse(
                    id=response_id,
                    choices=[StreamingChoices(index=0, delta=delta)],
                    created=created_at,
                    model=model,
                    usage=None,
                    stream=True,
                )
                continue

            if event_type == "response.function_call_arguments.delta":
                index = self._get_response_value(event, "output_index", 0)
                delta = Delta()
                delta.tool_calls = [
                    {
                        "index": index,
                        "function": {
                            "arguments": self._get_response_value(event, "delta", ""),
                        },
                    }
                ]
                yield ModelResponse(
                    id=response_id,
                    choices=[StreamingChoices(index=0, delta=delta)],
                    created=created_at,
                    model=model,
                    usage=None,
                    stream=True,
                )
                continue

            if event_type == "response.completed":
                completed = self._get_response_value(event, "response")
                response_id = self._get_response_value(completed, "id", response_id)
                created_at = self._get_response_value(completed, "created_at", created_at)
                has_tool_calls = any(
                    self._get_response_value(item, "type") == "function_call"
                    for item in self._get_response_value(completed, "output", []) or []
                )
                yield ModelResponse(
                    id=response_id,
                    choices=[
                        StreamingChoices(
                            index=0,
                            finish_reason=self._responses_finish_reason(
                                completed,
                                has_tool_calls=has_tool_calls,
                            ),
                            delta=Delta(),
                        )
                    ],
                    created=created_at,
                    model=model,
                    usage=self._create_responses_usage(
                        self._get_response_value(completed, "usage")
                    ),
                    stream=True,
                )
                continue

            if event_type in {"response.failed", "error"}:
                response = self._get_response_value(event, "response")
                error = self._get_response_value(event, "error") or (
                    self._get_response_value(response, "error") if response else None
                )
                raise OpenAINativeError(
                    status_code=500,
                    message=self._get_response_value(error, "message", str(error)),
                    model=model,
                )

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
        self._map_cached_tokens_to_usage(usage)

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
            self._map_cached_tokens_to_usage(chunk_usage)

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

    def _post_chat_stream_processing(self, response, model: str):
        for chunk in self.post_stream_processing(response, model=model):
            self._map_cached_tokens_to_usage(chunk.usage)
            # OpenAI sends a final usage-only chunk with choices=[] when
            # stream_options.include_usage is enabled. UnionLLM's streaming
            # contract keeps choices[0] safe for existing consumers.
            if not chunk.choices and chunk.usage is not None:
                chunk.choices = [
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=""),
                    )
                ]
            yield chunk

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

            if api_mode == "responses":
                params = self._preprocess_responses_params(**kwargs)
                response_input = self._messages_to_responses_input(messages, model=model)
                stream = bool(params.get("stream", False))
                response = client.responses.create(
                    model=model,
                    input=response_input,
                    **params,
                )
                if stream:
                    return self._post_responses_stream_processing(response, model=model)
                return self._create_responses_model_response(response, model=model)

            params = self._preprocess_chat_params(**kwargs)
            params = self._ensure_chat_tool_reasoning_compatibility(model, params)
            stream = bool(params.get("stream", False))
            if stream:
                params = self._include_stream_usage(params)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **params,
                )
                return self._post_chat_stream_processing(response, model=model)

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **params,
            )
            result = self.create_model_response(response, model=model)
            self._map_cached_tokens_to_usage(result.usage)
            return result
        except OpenAINativeError:
            raise
        except (APIStatusError, APIConnectionError, APITimeoutError, OpenAIError) as exc:
            self._raise_mapped_error(exc, model=model)
        except Exception as exc:
            self._raise_mapped_error(exc, model=model)


class OpenAICompatibleProvider(OpenAINativeProvider):
    """A vendor or self-hosted server that speaks the OpenAI Chat Completions API.

    Replaces the LiteLLM "openai/<model>" route for xAI, OpenRouter, Groq and
    user-defined endpoints. Unlike `OpenAINativeProvider`, the base URL is used
    verbatim (vendors put the API under their own path, e.g. Groq's
    ``/openai/v1``) and the OpenAI environment key is never used, so a
    ConsoleX OpenAI key cannot leak to a third-party endpoint.
    """

    # Self-hosted servers often run without authentication, but the openai SDK
    # refuses to build a client without some key.
    KEYLESS_PLACEHOLDER = "unused"

    def __init__(self, default_base_url=None, api_key_env=None, model_prefix=None, require_api_key=True, **model_kwargs):
        self.model_prefix = model_prefix
        self.api_key = model_kwargs.get("api_key") or (os.getenv(api_key_env) if api_key_env else None)
        if not self.api_key:
            if require_api_key:
                raise OpenAINativeError(status_code=422, message="Missing API key")
            self.api_key = self.KEYLESS_PLACEHOLDER

        raw_base_url = model_kwargs.get("base_url") or model_kwargs.get("api_base") or default_base_url
        if not raw_base_url:
            raise OpenAINativeError(
                status_code=422,
                message="Missing base_url or api_base for the OpenAI-compatible endpoint",
            )
        self.base_url = self._normalize_base_url(raw_base_url)
        self.api_mode = model_kwargs.get("api_mode")
        self.client = self._build_client(self.api_key, self.base_url)

    def _normalize_base_url(self, base_url: str) -> str:
        raw = str(base_url).strip()
        parsed = urlparse(raw)
        if not parsed.scheme:
            parsed = urlparse(f"https://{raw}")
        return urlunparse(parsed._replace(path=(parsed.path or "").rstrip("/")))

    def completion(self, model: str, messages: list, **kwargs):
        # Callers that used the LiteLLM route may still send "<provider>/<model>".
        if self.model_prefix and model and model.lower().startswith(f"{self.model_prefix}/"):
            model = model.split("/", 1)[1]
        return super().completion(model, messages, **kwargs)
