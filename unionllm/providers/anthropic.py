import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base_provider import BaseProvider
from unionllm.utils import Choices, Delta, Message, ModelResponse, StreamingChoices, Usage


class AnthropicProviderError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class AnthropicAIProvider(BaseProvider):
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
    # Anthropic requires max_tokens.  ConsoleX normally omits it for a regular
    # chat turn, so 1024 here silently cut otherwise normal answers short.
    # Keep the provider fallback aligned with ConsoleX's general chat default.
    DEFAULT_MAX_TOKENS = 8192
    def __init__(self, **kwargs):
        try:
            from anthropic import Anthropic
        except Exception as e:
            raise AnthropicProviderError(
                status_code=500,
                message=(
                    "Missing required dependency for Anthropic provider. "
                    "Please install 'anthropic'. "
                    f"Import error: {type(e).__name__}: {e}"
                ),
            )

        env_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.api_key = kwargs.get("api_key") if kwargs.get("api_key") else env_api_key
        if not self.api_key:
            raise AnthropicProviderError(status_code=422, message="Missing API key")

        self.base_url = (
            kwargs.get("base_url")
            or kwargs.get("api_base")
            or os.getenv("ANTHROPIC_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        self.anthropic_version = (
            kwargs.get("anthropic_version")
            or os.getenv("ANTHROPIC_VERSION")
            or self.DEFAULT_ANTHROPIC_VERSION
        )
        self.anthropic_beta = kwargs.get("anthropic_beta") or os.getenv("ANTHROPIC_BETA")

        default_headers = {}
        if self.anthropic_version:
            default_headers["anthropic-version"] = self.anthropic_version
        if self.anthropic_beta:
            default_headers["anthropic-beta"] = self.anthropic_beta

        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=default_headers if default_headers else None,
        )

    def _preprocess(self, **kwargs) -> Dict[str, Any]:
        supported = {
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "metadata",
            "tools",
            "tool_choice",
            "thinking",
            "reasoning_effort",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
        clean = {k: v for k, v in kwargs.items() if k in supported}

        if "max_completion_tokens" in clean and "max_tokens" not in clean:
            clean["max_tokens"] = clean.pop("max_completion_tokens")
        else:
            clean.pop("max_completion_tokens", None)

        if "stop" in kwargs and "stop_sequences" not in clean:
            stop_val = kwargs.get("stop")
            if stop_val is not None:
                if isinstance(stop_val, (list, tuple)):
                    clean["stop_sequences"] = list(stop_val)
                else:
                    clean["stop_sequences"] = [stop_val]

        if not clean.get("max_tokens"):
            clean["max_tokens"] = self.DEFAULT_MAX_TOKENS

        if "reasoning_effort" in kwargs and "thinking" not in clean:
            # Anthropic thinking requires token budget and works best with temperature=1.
            max_tokens = int(clean.get("max_tokens", self.DEFAULT_MAX_TOKENS))
            if max_tokens <= 1024:
                max_tokens = 1025
                clean["max_tokens"] = max_tokens
            clean["thinking"] = {
                "type": "enabled",
                "budget_tokens": max_tokens - 1,
            }
            clean["temperature"] = 1

        clean.pop("reasoning_effort", None)
        return clean

    def _get(self, payload: Any, key: str, default: Any = None) -> Any:
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)

    def _map_stop_reason(self, stop_reason: Optional[str]) -> Optional[str]:
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            None: None,
        }
        return mapping.get(stop_reason, stop_reason)

    def _normalize_tool_result_content(self, content: Any) -> Any:
        if content is None:
            return ""
        return content

    def _parse_tool_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except Exception:
                return {}
        return arguments if arguments is not None else {}

    def _convert_openai_to_anthropic_messages(self, messages: List[dict]) -> Tuple[Optional[str], List[dict]]:
        system_text_parts: List[str] = []
        converted_messages: List[dict] = []

        for message in messages:
            role = message.get("role")

            if role == "system":
                content = message.get("content", "")
                if isinstance(content, str):
                    system_text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_text_parts.append(block.get("text", ""))
                continue

            if role == "tool":
                converted_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.get("tool_call_id", ""),
                                "content": self._normalize_tool_result_content(message.get("content")),
                            }
                        ],
                    }
                )
                continue

            if role == "assistant" and message.get("tool_calls"):
                converted_content = []
                content = message.get("content")
                if isinstance(content, str) and content:
                    converted_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    converted_content.extend(content)

                for tool_call in message.get("tool_calls", []):
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": self._get(tool_call.get("function"), "name", ""),
                        "input": self._parse_tool_arguments(
                            self._get(tool_call.get("function"), "arguments", "{}")
                        ),
                    }
                    converted_content.append(tool_use_block)

                converted_messages.append({"role": "assistant", "content": converted_content})
                continue

            # ConsoleX history adds UI bookkeeping fields (for example
            # ``messageIndex``).  Anthropic's Messages API only accepts
            # ``role`` and ``content`` at the message level, so never forward
            # arbitrary OpenAI/UI fields here.
            converted_msg = {"role": role}
            content = message.get("content")
            if isinstance(content, str):
                converted_msg["content"] = content
            elif isinstance(content, list):
                converted_content = []
                for content_block in content:
                    block_type = content_block.get("type")
                    if block_type == "text":
                        converted_content.append({"type": "text", "text": content_block.get("text", "")})
                    elif block_type == "image_url":
                        converted_content.append(self._convert_image_url_to_anthropic(content_block))
                    elif block_type == "image":
                        converted_content.append(content_block)
                    elif block_type == "file_url":
                        converted_content.append(self._convert_file_url_to_anthropic(content_block))
                    elif block_type == "file":
                        converted_content.append(self._convert_file_to_anthropic(content_block))
                    elif block_type in ["document", "tool_result", "tool_use"]:
                        converted_content.append(content_block)
                    else:
                        converted_content.append(content_block)

                converted_msg["content"] = converted_content

            converted_messages.append(converted_msg)

        system_message = "\n".join([p for p in system_text_parts if isinstance(p, str) and p != ""]).strip()
        return (system_message if system_message else None), converted_messages

    def _convert_image_url_to_anthropic(self, image_block: dict) -> dict:
        image_url = image_block.get("image_url", {})
        url = image_url.get("url", "")
        if not url:
            raise AnthropicProviderError(status_code=422, message="Image URL is empty")

        if url.startswith("data:"):
            header_part, _, data_part = url.partition(",")
            if not data_part:
                raise AnthropicProviderError(status_code=422, message="Invalid base64 image data URI")
            media_type = "image/jpeg"
            if ":" in header_part and ";" in header_part:
                media_type = header_part.split(":", 1)[1].split(";", 1)[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data_part,
                },
            }

        if url.startswith("http://") or url.startswith("https://"):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                media_type = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0].strip()
                if media_type == "application/octet-stream":
                    url_lower = url.lower()
                    if ".png" in url_lower:
                        media_type = "image/png"
                    elif ".gif" in url_lower:
                        media_type = "image/gif"
                    elif ".webp" in url_lower:
                        media_type = "image/webp"
                    else:
                        media_type = "image/jpeg"

                image_data_base64 = base64.b64encode(response.content).decode("utf-8")
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data_base64,
                    },
                }
            except Exception as e:
                raise AnthropicProviderError(
                    status_code=500, message=f"Failed to download image from URL: {str(e)}"
                )

        # Treat non-http non-data URI as local file path.
        try:
            with open(url, "rb") as f:
                file_data = f.read()
            media_type = "image/jpeg"
            lower = url.lower()
            if lower.endswith(".png"):
                media_type = "image/png"
            elif lower.endswith(".gif"):
                media_type = "image/gif"
            elif lower.endswith(".webp"):
                media_type = "image/webp"
            image_data_base64 = base64.b64encode(file_data).decode("utf-8")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data_base64,
                },
            }
        except Exception as e:
            raise AnthropicProviderError(
                status_code=422, message=f"Invalid image input (URL/path): {str(e)}"
            )

    def _convert_file_url_to_anthropic(self, file_block: dict) -> dict:
        file_url = file_block.get("file_url", {})
        url = file_url.get("url", "")
        if not url:
            raise AnthropicProviderError(status_code=422, message="File URL is empty")
        return {
            "type": "document",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    def _convert_file_to_anthropic(self, file_block: dict) -> dict:
        file_info = file_block.get("file", {})
        file_data = file_info.get("file_data", "")
        if not file_data:
            raise AnthropicProviderError(status_code=422, message="File data is empty")

        media_type = file_info.get("media_type", "application/pdf")
        if file_data.startswith("data:"):
            header_part, _, data_part = file_data.partition(",")
            if not data_part:
                raise AnthropicProviderError(status_code=422, message="Invalid file data URI")
            if ":" in header_part and ";" in header_part:
                media_type = header_part.split(":", 1)[1].split(";", 1)[0]
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data_part,
                },
            }

        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": file_data,
            },
        }

    def _convert_tools_to_anthropic(self, tools: List[dict], tool_choice: Optional[Any] = None) -> Dict[str, Any]:
        if not tools:
            return {}

        result: Dict[str, Any] = {}
        converted_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                converted_tools.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
            else:
                converted_tools.append(tool)

        if converted_tools:
            result["tools"] = converted_tools

        if tool_choice is not None:
            if tool_choice == "auto":
                result["tool_choice"] = {"type": "auto"}
            elif tool_choice in ("required", "any"):
                result["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, str) and tool_choice.startswith("function="):
                result["tool_choice"] = {"type": "tool", "name": tool_choice.split("=", 1)[1]}
            elif isinstance(tool_choice, str):
                result["tool_choice"] = {"type": "tool", "name": tool_choice}
            elif isinstance(tool_choice, dict):
                if tool_choice.get("type") == "function":
                    fn = self._get(tool_choice.get("function"), "name", "")
                    if fn:
                        result["tool_choice"] = {"type": "tool", "name": fn}
                else:
                    result["tool_choice"] = tool_choice

        return result

    def _build_usage(self, usage_payload: Any, prompt_tokens_hint: Optional[int] = None) -> Optional[Usage]:
        if usage_payload is None and prompt_tokens_hint is None:
            return None

        usage = Usage()
        has_usage = False

        prompt_tokens = self._get(usage_payload, "input_tokens", None)
        completion_tokens = self._get(usage_payload, "output_tokens", None)
        cache_creation_tokens = self._get(usage_payload, "cache_creation_input_tokens", None)
        cache_read_tokens = self._get(usage_payload, "cache_read_input_tokens", None)

        if prompt_tokens is None and prompt_tokens_hint is not None:
            prompt_tokens = prompt_tokens_hint

        if prompt_tokens is not None:
            usage.prompt_tokens = prompt_tokens
            has_usage = True
        if completion_tokens is not None:
            usage.completion_tokens = completion_tokens
            has_usage = True
        if prompt_tokens is not None and completion_tokens is not None:
            usage.total_tokens = prompt_tokens + completion_tokens
            has_usage = True
        if cache_creation_tokens is not None:
            usage.cache_creation_input_tokens = cache_creation_tokens
            has_usage = True
        if cache_read_tokens is not None:
            usage.cache_read_input_tokens = cache_read_tokens
            has_usage = True

        return usage if has_usage else None

    def _prepare_request(self, messages: List[dict], **kwargs) -> Tuple[List[dict], Dict[str, Any]]:
        try:
            system_message, anthropic_messages = self._convert_openai_to_anthropic_messages(messages)
        except AnthropicProviderError:
            raise
        except Exception as e:
            raise AnthropicProviderError(
                status_code=500, message=f"Failed to convert messages format: {str(e)}"
            )

        params = self._preprocess(**kwargs)
        cache_control = kwargs.get("cache_control")
        if cache_control:
            # Anthropic's current automatic prompt-caching API receives this
            # control at the top level of Messages.create().
            params["cache_control"] = cache_control
        if system_message:
            params["system"] = system_message

        tools = kwargs.get("tools")
        if tools:
            try:
                tools_config = self._convert_tools_to_anthropic(tools, kwargs.get("tool_choice"))
                params.update(tools_config)
            except AnthropicProviderError:
                raise
            except Exception as e:
                raise AnthropicProviderError(
                    status_code=500, message=f"Failed to convert tools format: {str(e)}"
                )

        return anthropic_messages, params

    def create_model_response_wrapper(self, anthropic_resp: Any, model: str) -> ModelResponse:
        text_parts = []
        tool_calls = []

        try:
            content_blocks = self._get(anthropic_resp, "content", []) or []
            for index, block in enumerate(content_blocks):
                block_type = self._get(block, "type")
                if block_type == "text":
                    block_text = self._get(block, "text", "")
                    if block_text:
                        text_parts.append(block_text)
                elif block_type == "tool_use":
                    raw_input = self._get(block, "input", {})
                    if isinstance(raw_input, str):
                        arguments = raw_input
                    else:
                        arguments = json.dumps(raw_input or {}, ensure_ascii=False)

                    tool_calls.append(
                        {
                            "id": self._get(block, "id", f"call_{index}"),
                            "index": index,
                            "type": "function",
                            "function": {
                                "name": self._get(block, "name", ""),
                                "arguments": arguments,
                            },
                        }
                    )
        except Exception as e:
            raise AnthropicProviderError(
                status_code=500,
                message=f"Failed to parse Anthropic response content: {str(e)}",
            )

        finish_reason = self._map_stop_reason(self._get(anthropic_resp, "stop_reason", "stop"))
        message = Message(content=("".join(text_parts) if text_parts else None), role="assistant")
        if tool_calls:
            message.tool_calls = tool_calls

        choice = Choices(
            message=message,
            index=0,
            finish_reason=finish_reason or "stop",
        )

        usage_obj = self._build_usage(self._get(anthropic_resp, "usage"))
        return ModelResponse(
            id=self._get(anthropic_resp, "id", None),
            choices=[choice],
            created=int(time.time()),
            model=model,
            usage=usage_obj,
        )

    def post_stream_processing_wrapper(self, model: str, messages: List[dict], params: Dict[str, Any]):
        try:
            stream = self.client.messages.create(
                model=model,
                messages=messages,
                stream=True,
                **params,
            )
        except Exception as e:
            status = getattr(e, "status_code", 500)
            raise AnthropicProviderError(status_code=status, message=str(e))

        msg_id = None
        finish_reason = None
        prompt_tokens_hint = None
        tool_calls = {}
        content_blocks = {}

        for event in stream:
            event_type = self._get(event, "type")

            if event_type == "message_start":
                event_message = self._get(event, "message")
                event_message_id = self._get(event_message, "id")
                if event_message_id:
                    msg_id = event_message_id
                start_usage = self._get(event_message, "usage")
                start_input_tokens = self._get(start_usage, "input_tokens")
                if start_input_tokens is not None:
                    prompt_tokens_hint = start_input_tokens
                continue

            if msg_id is None:
                maybe_id = self._get(event, "id")
                if maybe_id:
                    msg_id = maybe_id

            if event_type == "content_block_start":
                index = self._get(event, "index", 0)
                content_block = self._get(event, "content_block")
                block_type = self._get(content_block, "type")

                if block_type == "tool_use":
                    tool_id = self._get(content_block, "id", f"call_{index}")
                    tool_name = self._get(content_block, "name", "")
                    tool_calls[index] = {
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": ""},
                    }
                    content_blocks[index] = {"type": "tool_use", "partial_json": ""}

                    chunk_delta = Delta(
                        tool_calls=[
                            {
                                "index": index,
                                "id": tool_id,
                                "type": "function",
                                "function": {"name": tool_name, "arguments": ""},
                            }
                        ]
                    )
                    yield ModelResponse(
                        id=msg_id,
                        choices=[StreamingChoices(index=0, delta=chunk_delta)],
                        created=int(time.time()),
                        model=model,
                        usage=None,
                        stream=True,
                    )
                elif block_type == "text":
                    content_blocks[index] = {"type": "text"}
                continue

            if event_type == "content_block_delta":
                index = self._get(event, "index", 0)
                delta_obj = self._get(event, "delta")
                delta_type = self._get(delta_obj, "type")

                if delta_type == "text_delta":
                    text = self._get(delta_obj, "text", "")
                    if text:
                        chunk_delta = Delta(content=text)
                        yield ModelResponse(
                            id=msg_id,
                            choices=[StreamingChoices(index=0, delta=chunk_delta)],
                            created=int(time.time()),
                            model=model,
                            usage=None,
                            stream=True,
                        )
                elif delta_type == "input_json_delta":
                    partial_json = self._get(delta_obj, "partial_json", "")
                    if partial_json and index in tool_calls:
                        tool_calls[index]["function"]["arguments"] += partial_json
                        chunk_delta = Delta(
                            tool_calls=[
                                {
                                    "index": index,
                                    "id": tool_calls[index]["id"],
                                    "type": "function",
                                    "function": {"arguments": partial_json},
                                }
                            ]
                        )
                        yield ModelResponse(
                            id=msg_id,
                            choices=[StreamingChoices(index=0, delta=chunk_delta)],
                            created=int(time.time()),
                            model=model,
                            usage=None,
                            stream=True,
                        )
                elif delta_type == "thinking_delta":
                    reasoning_content = self._get(delta_obj, "thinking", "")
                    if reasoning_content:
                        chunk_delta = Delta(reasoning_content=reasoning_content)
                        yield ModelResponse(
                            id=msg_id,
                            choices=[StreamingChoices(index=0, delta=chunk_delta)],
                            created=int(time.time()),
                            model=model,
                            usage=None,
                            stream=True,
                        )
                elif delta_type == "signature_delta":
                    signature = self._get(delta_obj, "signature", "")
                    if signature:
                        chunk_delta = Delta(thought_signature=signature)
                        yield ModelResponse(
                            id=msg_id,
                            choices=[StreamingChoices(index=0, delta=chunk_delta)],
                            created=int(time.time()),
                            model=model,
                            usage=None,
                            stream=True,
                        )
                continue

            if event_type == "message_delta":
                delta_payload = self._get(event, "delta")
                stop_reason = self._map_stop_reason(self._get(delta_payload, "stop_reason"))
                if stop_reason:
                    finish_reason = stop_reason
                chunk_usage = self._build_usage(
                    self._get(event, "usage"),
                    prompt_tokens_hint=prompt_tokens_hint,
                )
                yield ModelResponse(
                    id=msg_id,
                    choices=[
                        StreamingChoices(
                            index=0,
                            finish_reason=finish_reason,
                            delta=Delta(),
                        )
                    ],
                    created=int(time.time()),
                    model=model,
                    usage=chunk_usage,
                    stream=True,
                )
                continue

            if event_type == "message_stop":
                yield ModelResponse(
                    id=msg_id,
                    choices=[
                        StreamingChoices(
                            index=0,
                            finish_reason=finish_reason or "stop",
                            delta=Delta(),
                        )
                    ],
                    created=int(time.time()),
                    model=model,
                    usage=None,
                    stream=True,
                )

    def completion(self, model: str, messages: List[dict], **kwargs) -> ModelResponse:
        try:
            if not model or messages is None:
                raise AnthropicProviderError(status_code=422, message="Missing model or messages")

            check = self.check_prompt("anthropic", model, messages)
            if not check.get("pass_check"):
                raise AnthropicProviderError(status_code=422, message=str(check.get("reason")))
            norm_messages = check.get("messages", messages)

            anthropic_messages, params = self._prepare_request(norm_messages, **kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model, anthropic_messages, params)

            resp = self.client.messages.create(
                model=model,
                messages=anthropic_messages,
                **params,
            )
            return self.create_model_response_wrapper(resp, model=model)
        except AnthropicProviderError:
            raise
        except Exception as e:
            status = getattr(e, "status_code", 500)
            raise AnthropicProviderError(status_code=status, message=str(e))
