import os
import re
import time
import base64
import requests
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from openai import AzureOpenAI, OpenAI

from .base_provider import BaseProvider
from .openai_native import OpenAINativeError, OpenAINativeProvider
from unionllm.utils import ModelResponse, Message, Choices, Usage, Delta, StreamingChoices


class AzureProviderError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class AzureOpenAIProvider(OpenAINativeProvider):
    """Azure OpenAI (non-Claude) deployments through the openai SDK's AzureOpenAI client.

    Replaces the LiteLLM route. Model names may carry LiteLLM's "azure/" prefix;
    the deployment is the rest, and its name decides the model family. As
    LiteLLM did with ``drop_params``, parameters that family rejects are dropped
    instead of failing the request:

    - o-series and GPT-5+ reasoning models get ``max_completion_tokens`` instead
      of ``max_tokens``, and lose the sampling parameters they reject;
    - other models (gpt-4o, gpt-4.1, gpt-5-chat, ...) lose ``reasoning_effort``.

    A base URL ending in ``/openai/v1`` is Azure's v1 API, which takes the plain
    OpenAI client and no api-version.
    """

    DEFAULT_API_VERSION = "2025-04-01-preview"
    GPT_PATTERN = re.compile(r"^gpt-(\d+)(?:\.(\d+))?", re.IGNORECASE)
    O_SERIES_PATTERN = re.compile(r"^o\d", re.IGNORECASE)
    # Not accepted by any reasoning model (LiteLLM's o-series and GPT-5 configs);
    # GPT-5 and later also reject stop sequences.
    REASONING_UNSUPPORTED_PARAMS = ("presence_penalty", "frequency_penalty", "logit_bias")
    # Accepted by reasoning models only when reasoning is off (effort "none").
    SAMPLING_PARAMS = ("top_p", "logprobs", "top_logprobs")

    def __init__(self, **model_kwargs):
        self.api_key = (
            model_kwargs.get("api_key")
            or os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("AZURE_API_KEY")
        )
        if not self.api_key:
            raise OpenAINativeError(status_code=422, message="Missing Azure OpenAI API key")

        raw_base_url = (
            model_kwargs.get("api_base")
            or model_kwargs.get("base_url")
            or model_kwargs.get("azure_endpoint")
            or model_kwargs.get("endpoint")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_API_BASE")
        )
        if not raw_base_url:
            raise OpenAINativeError(
                status_code=422,
                message="Missing Azure OpenAI endpoint: pass api_base or set AZURE_OPENAI_ENDPOINT",
            )
        self.api_version = self._resolve_api_version(**model_kwargs)
        self.base_url = self._normalize_base_url(raw_base_url)
        self.api_mode = model_kwargs.get("api_mode")
        self.client = self._build_client(self.api_key, self.base_url, self.api_version)

    def _resolve_api_version(self, **kwargs) -> str:
        return (
            kwargs.get("api_version")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("AZURE_API_VERSION")
            or self.DEFAULT_API_VERSION
        )

    def _normalize_base_url(self, base_url: str) -> str:
        raw = str(base_url).strip()
        parsed = urlparse(raw)
        if not parsed.scheme:
            parsed = urlparse(f"https://{raw}")
        return urlunparse(parsed._replace(path=(parsed.path or "").rstrip("/")))

    def _build_client(self, api_key: str, base_url: str, api_version: Optional[str] = None):
        if urlparse(base_url).path.endswith("/openai/v1"):
            return OpenAI(api_key=api_key, base_url=f"{base_url}/")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version or self.api_version,
        )

    def _resolve_client(self, **kwargs):
        api_key = kwargs.get("api_key") or self.api_key
        base_url = self._normalize_base_url(
            kwargs.get("api_base")
            or kwargs.get("base_url")
            or kwargs.get("azure_endpoint")
            or kwargs.get("endpoint")
            or self.base_url
        )
        api_version = kwargs.get("api_version") or self.api_version
        if (api_key, base_url, api_version) == (self.api_key, self.base_url, self.api_version):
            return self.client
        return self._build_client(api_key, base_url, api_version)

    @staticmethod
    def deployment_name(model: str) -> str:
        name = str(model or "")
        if name.lower().startswith("azure/"):
            return name.split("/", 1)[1]
        return name

    @classmethod
    def model_family(cls, model: str) -> str:
        """Return "o_series", "gpt5" (GPT-5 and later reasoning models) or "chat"."""
        name = cls.deployment_name(model)
        if cls.O_SERIES_PATTERN.match(name):
            return "o_series"
        match = cls.GPT_PATTERN.match(name)
        # gpt-5-chat* are plain chat models; versioned ones (gpt-5.2-chat) reason.
        if match and int(match.group(1)) >= 5 and not name.lower().startswith("gpt-5-chat"):
            return "gpt5"
        return "chat"

    @classmethod
    def supports_no_reasoning(cls, model: str) -> bool:
        """Whether reasoning_effort="none" (and then sampling) is accepted.

        Mirrors LiteLLM's model map: gpt-5.1, gpt-5.4+ and gpt-6+, except the
        -chat, -codex and -pro variants.
        """
        name = cls.deployment_name(model).lower()
        if any(tag in name for tag in ("-chat", "-codex", "-pro")):
            return False
        match = cls.GPT_PATTERN.match(name)
        if not match:
            return False
        major, minor = int(match.group(1)), int(match.group(2) or 0)
        return major >= 6 or (major == 5 and (minor == 1 or minor >= 4))

    def _adapt_params_to_model(self, model: str, params: dict) -> dict:
        family = self.model_family(model)
        if family == "chat":
            params.pop("reasoning_effort", None)
            return params

        if "max_tokens" in params:
            max_tokens = params.pop("max_tokens")
            params.setdefault("max_completion_tokens", max_tokens)
        for key in self.REASONING_UNSUPPORTED_PARAMS:
            params.pop(key, None)

        if family == "gpt5":
            params.pop("stop", None)

        effort = params.get("reasoning_effort")
        if isinstance(effort, dict):
            effort = effort.get("effort")
        reasoning_off = family == "gpt5" and self.supports_no_reasoning(model) and effort in (None, "none")
        if effort == "none" and not reasoning_off:
            params.pop("reasoning_effort", None)
        if not reasoning_off:
            for key in self.SAMPLING_PARAMS:
                params.pop(key, None)
            if params.get("temperature") not in (None, 1):
                params.pop("temperature", None)
        return params

    def _ensure_chat_tool_reasoning_compatibility(self, model: str, params: dict) -> dict:
        # The per-model hook of the Chat Completions path; runs after the GPT-5.6
        # tools rule so that a forced reasoning_effort="none" is seen here.
        params = super()._ensure_chat_tool_reasoning_compatibility(model, params)
        return self._adapt_params_to_model(model, params)

    def _post_chat_stream_processing(self, response, model: str):
        # Azure opens a stream with a prompt_filter_results chunk that has no
        # choices and no usage; it carries nothing for UnionLLM's callers.
        def without_filter_only_chunks(chunks):
            for chunk in chunks:
                if not getattr(chunk, "choices", None) and getattr(chunk, "usage", None) is None:
                    continue
                yield chunk

        return super()._post_chat_stream_processing(without_filter_only_chunks(response), model=model)

    def completion(self, model: str, messages: list, **kwargs):
        return super().completion(self.deployment_name(model), messages, **kwargs)


# Backwards-compatible name; it used to delegate to LiteLLM.
AzureAIProvider = AzureOpenAIProvider


class AzureAnthropicProvider(BaseProvider):
    """
    Azure native provider for Anthropic Claude models via AnthropicFoundry.

    Required configuration:
    - endpoint/base_url: Azure Cognitive Services Anthropic endpoint, e.g.
      https://<your-resource>.openai.azure.com/anthropic
      Pass via kwarg `endpoint` or env `AZURE_ANTHROPIC_ENDPOINT`.

    Authentication:
    - Uses DefaultAzureCredential and AAD bearer token scope
      "https://cognitiveservices.azure.com/.default".

    Usage:
        provider = AzureAnthropicProvider(endpoint="https://.../anthropic")
        provider.completion(model="<deployment_name>", messages=[...])
    """

    SCOPE = "https://cognitiveservices.azure.com/.default"

    def __init__(self, **kwargs):
        try:
            from anthropic import AnthropicFoundry
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except Exception as e:
            raise AzureProviderError(
                status_code=500,
                message=(
                    "Missing required dependencies for AzureAnthropicProvider. "
                    "Please install 'anthropic' and 'azure-identity'. "
                    f"Import error: {type(e).__name__}: {e}"
                ),
            )

        api_base = (
            kwargs.get("api_base")
            or kwargs.get("endpoint")
            or os.getenv("AZURE_ANTHROPIC_API_BASE")
            or os.getenv("AZURE_ANTHROPIC_ENDPOINT")
        )
        if not api_base:
            raise AzureProviderError(
                status_code=422,
                message=(
                    "Missing Azure Anthropic endpoint. Provide kwarg 'api_base' or 'endpoint', "
                    "or set env AZURE_ANTHROPIC_API_BASE / AZURE_ANTHROPIC_ENDPOINT."
                ),
            )

        api_key = kwargs.get("api_key")

        # Create AnthropicFoundry client
        self.client = AnthropicFoundry(api_key=api_key, base_url=api_base)

    def _preprocess(self, **kwargs) -> Dict[str, Any]:
        # Keep only commonly-supported params for Claude messages.create
        supported = {
            "max_tokens",
            "temperature",
            "top_p",
            "stop_sequences",
            "metadata",
            "tools",
            "tool_choice",
            "thinking",
            "reasoning_effort",
            # Top-level automatic prompt caching, as in AnthropicAIProvider; the
            # Foundry endpoint accepts it (verified 2026-10-10: second call read the prefix).
            "cache_control",
            # stream intentionally not passed (non-stream implementation here)
        }
        clean = {k: v for k, v in kwargs.items() if k in supported}
        # Map OpenAI-style 'stop' to Anthropic 'stop_sequences' if provided
        if "stop" in kwargs and "stop_sequences" not in clean:
            stop_val = kwargs.get("stop")
            if stop_val is not None:
                if isinstance(stop_val, (list, tuple)):
                    clean["stop_sequences"] = list(stop_val)
                else:
                    clean["stop_sequences"] = [stop_val]
                    
        if 'temperature' in clean:
            clean.pop('top_p', None)

        # Custom mapping: thinking_effort -> thinking dict
        # Requirement: thinking = {"type": "enabled", "budget_tokens": <max_tokens or 2048>}
        # budget_tokens 使用原始传入的 max_tokens 值 (不受默认补充值影响), 若未传则 2048
        if not clean.get('max_tokens'):
            clean['max_tokens'] = 30000
        
        if "reasoning_effort" in kwargs:
            if clean.get("max_tokens") <= 1024:
                clean['max_tokens'] = 1025
            original_max = clean['max_tokens']    
            budget_tokens = original_max - 1 if original_max is not None else 2048
            # 覆盖/设置 thinking 字段
            clean["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
            clean['temperature'] = 1
            clean.pop("reasoning_effort", None)

        return clean

    def _convert_openai_to_anthropic_messages(self, messages: List[dict]) -> tuple[Optional[str], List[dict]]:
        """
        Convert OpenAI-style messages to Anthropic-style messages.
        
        Main conversions:
        - role: "system" -> extracted and returned as separate system parameter
        - image_url blocks -> image blocks with base64 data
        - video_url blocks -> video blocks with base64 data
        - tool_calls -> tool_use blocks in content array
        - role: "tool" -> role: "user" with tool_result blocks
        - Functions parameter -> tools parameter
        
        Anthropic format:
        - Image: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
        - Video: {"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "..."}}
        - Text: {"type": "text", "text": "..."}
        - Tool use: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        - Tool result: {"type": "tool_result", "tool_use_id": "...", "content": "..."}
        
        Returns:
            tuple: (system_message, converted_messages)
                - system_message: str or None - the system message content
                - converted_messages: list - non-system messages in Anthropic format
        """
        system_parts: List[str] = []
        converted_messages = []
        
        for message in messages:
            # Handle system role: Anthropic doesn't support system role in messages array.
            # Every system message is kept (callers send more than one, e.g. instructions
            # plus a memory block), joined into the single system parameter.
            if message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    system_parts.append("\n".join(text_parts))
                continue
            
            # Handle tool role conversion: OpenAI's role="tool" -> Anthropic's role="user" with tool_result
            if message.get("role") == "tool":
                converted_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": message.get("content", "")
                        }
                    ]
                }
                converted_messages.append(converted_msg)
                continue
            
            # Handle assistant messages with tool_calls
            if message.get("role") == "assistant" and "tool_calls" in message:
                converted_content = []
                
                # Add text content if present
                if message.get("content"):
                    if isinstance(message["content"], str):
                        converted_content.append({
                            "type": "text",
                            "text": message["content"]
                        })
                    elif isinstance(message["content"], list):
                        # Already in content block format
                        converted_content.extend(message["content"])
                
                # Convert tool_calls to tool_use blocks
                for tool_call in message.get("tool_calls", []):
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("function", {}).get("name", ""),
                    }
                    
                    # Parse arguments from string to dict
                    arguments = tool_call.get("function", {}).get("arguments", "{}")
                    if isinstance(arguments, str):
                        try:
                            import json
                            tool_use_block["input"] = json.loads(arguments)
                        except Exception:
                            # If parsing fails, use empty dict
                            tool_use_block["input"] = {}
                    else:
                        tool_use_block["input"] = arguments
                    
                    converted_content.append(tool_use_block)
                
                converted_msg = {
                    "role": "assistant",
                    "content": converted_content
                }
                converted_messages.append(converted_msg)
                continue
            
            # Handle regular messages
            converted_msg = {k: v for k, v in message.items() if k not in ["content", "tool_calls"]}
            
            # Handle string content
            if isinstance(message.get("content"), str):
                converted_msg["content"] = message["content"]
            # Handle list content (multimodal)
            elif isinstance(message.get("content"), list):
                converted_content = []
                for content_block in message.get("content", []):
                    if content_block.get("type") == "text":
                        converted_content.append({
                            "type": "text",
                            "text": content_block.get("text", "")
                        })
                    elif content_block.get("type") == "image_url":
                        # Convert OpenAI image_url to Anthropic image
                        converted_content.append(
                            self._convert_image_url_to_anthropic(content_block)
                        )
                    elif content_block.get("type") == "image":
                        # Already Anthropic format, pass through
                        converted_content.append(content_block)
                    elif content_block.get("type") == "video_url":
                        # Convert OpenAI video_url to Anthropic video
                        converted_content.append(
                            self._convert_video_url_to_anthropic(content_block)
                        )
                    elif content_block.get("type") == "video":
                        # Already Anthropic format, pass through
                        converted_content.append(content_block)
                    elif content_block.get("type") == "file_url":
                        # Convert file_url to document format
                        converted_content.append(
                            self._convert_file_url_to_anthropic(content_block)
                        )
                    elif content_block.get("type") == "file":
                        # Convert file to document format
                        converted_content.append(
                            self._convert_file_to_anthropic(content_block)
                        )
                    elif content_block.get("type") == "document":
                        # Already Anthropic format, pass through
                        converted_content.append(content_block)
                    elif content_block.get("type") == "tool_result":
                        # Already Anthropic format, pass through
                        converted_content.append(content_block)
                    elif content_block.get("type") == "tool_use":
                        # Already Anthropic format, pass through
                        converted_content.append(content_block)
                    else:
                        # Unknown type, pass through as-is
                        converted_content.append(content_block)
                
                converted_msg["content"] = converted_content
            
            converted_messages.append(converted_msg)
        
        system_message = "\n\n".join(part for part in system_parts if part and part.strip()) or None
        return system_message, converted_messages

    def _convert_image_url_to_anthropic(self, image_block: dict) -> dict:
        """
        Convert OpenAI-style image_url block to Anthropic image block.
        Supports:
        - URL-based images (downloads and encodes to base64)
        - Base64-encoded images (passes through)
        """
        image_url = image_block.get("image_url", {})
        url = image_url.get("url", "")
        
        if not url:
            raise AzureProviderError(
                status_code=422,
                message="Image URL is empty"
            )
        
        # If already base64, extract the data part
        if url.startswith("data:"):
            # Format: data:image/jpeg;base64,/9j/4AAQSkZJRgABA...
            parts = url.split(",", 1)
            if len(parts) == 2:
                header_part = parts[0]  # "data:image/jpeg;base64"
                data_part = parts[1]
                
                # Extract media type
                media_type = "image/jpeg"  # default
                if ":" in header_part and ";" in header_part:
                    media_type = header_part.split(":")[1].split(";")[0]
                
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data_part
                    }
                }
        
        # Download image from URL
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine media type from Content-Type header
            media_type = response.headers.get("Content-Type", "image/jpeg")
            
            # If unable to determine from header, infer from URL
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
            
            # Encode to base64
            image_data_base64 = base64.b64encode(response.content).decode("utf-8")
            
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data_base64
                }
            }
        except Exception as e:
            raise AzureProviderError(
                status_code=500,
                message=f"Failed to download image from URL: {str(e)}"
            )

    def _convert_video_url_to_anthropic(self, video_block: dict) -> dict:
        """
        Convert OpenAI-style video_url block to Anthropic video block.
        Supports URL-based videos (downloads and encodes to base64).
        """
        video_url = video_block.get("video_url", {})
        url = video_url.get("url", "")
        
        if not url:
            raise AzureProviderError(
                status_code=422,
                message="Video URL is empty"
            )
        
        # Download video from URL
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Determine media type from Content-Type header
            media_type = response.headers.get("Content-Type", "video/mp4")
            
            # If unable to determine from header, infer from URL
            if media_type.startswith("application/") or media_type == "application/octet-stream":
                url_lower = url.lower()
                if ".mp4" in url_lower:
                    media_type = "video/mp4"
                elif ".mpeg" in url_lower:
                    media_type = "video/mpeg"
                elif ".webm" in url_lower:
                    media_type = "video/webm"
                elif ".mov" in url_lower:
                    media_type = "video/quicktime"
                else:
                    media_type = "video/mp4"
            
            # Encode to base64
            video_data_base64 = base64.b64encode(response.content).decode("utf-8")
            
            return {
                "type": "video",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": video_data_base64
                }
            }
        except Exception as e:
            raise AzureProviderError(
                status_code=500,
                message=f"Failed to download video from URL: {str(e)}"
            )

    def _convert_file_url_to_anthropic(self, file_block: dict) -> dict:
        """
        Convert file_url block to Anthropic document block.
        Supports URL-based files.

        Input format:
        {
            "type": "file_url",
            "file_url": {
                "url": "https://..."
            }
        }

        Output format:
        {
            "type": "document",
            "source": {
                "type": "url",
                "url": "https://..."
            }
        }
        """
        file_url = file_block.get("file_url", {})
        url = file_url.get("url", "")

        if not url:
            raise AzureProviderError(
                status_code=422,
                message="File URL is empty"
            )

        return {
            "type": "document",
            "source": {
                "type": "url",
                "url": url
            }
        }

    def _convert_file_to_anthropic(self, file_block: dict) -> dict:
        """
        Convert file block to Anthropic document block.
        Supports base64-encoded files.

        Input format:
        {
            "type": "file",
            "file": {
                "file_data": "base64_encoded_data...",
                "media_type": "application/pdf"  # optional
            }
        }

        Output format:
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "base64_encoded_data..."
            }
        }
        """
        file_info = file_block.get("file", {})
        file_data = file_info.get("file_data", "")

        if not file_data:
            raise AzureProviderError(
                status_code=422,
                message="File data is empty"
            )

        # Determine media type
        media_type = file_info.get("media_type", "application/pdf")

        # Handle data URL format (e.g., "data:application/pdf;base64,JVBERi0...")
        if file_data.startswith("data:"):
            parts = file_data.split(",", 1)
            if len(parts) == 2:
                header_part = parts[0]  # "data:application/pdf;base64"
                data_part = parts[1]

                # Extract media type from header
                if ":" in header_part and ";" in header_part:
                    media_type = header_part.split(":")[1].split(";")[0]

                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data_part
                    }
                }

        # Plain base64 data
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": file_data
            }
        }

    def _convert_tools_to_anthropic(self, tools: List[dict], tool_choice: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert OpenAI-style tools to Anthropic-style tools.
        
        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "function_name",
                "description": "...",
                "parameters": {...}
            }
        }
        
        Anthropic format:
        {
            "name": "function_name",
            "description": "...",
            "input_schema": {...}
        }
        
        tool_choice mapping:
        - "auto" -> {"type": "auto"}
        - "required" -> {"type": "any"}
        - specific tool name -> {"type": "tool", "name": "tool_name"}
        - None -> not included
        """
        if not tools:
            return {}
        
        result = {}
        converted_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                # Anthropic tools do NOT have "type": "function" at the top level
                anthropic_tool = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})  # Map 'parameters' to 'input_schema'
                }
                converted_tools.append(anthropic_tool)
            else:
                # Pass through non-function tools as-is
                converted_tools.append(tool)
        
        if converted_tools:
            result["tools"] = converted_tools
        
        # Convert tool_choice
        if tool_choice is not None:
            if tool_choice == "auto":
                result["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required" or tool_choice == "any":
                # For "required", use "any" which forces tool use
                result["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, str) and tool_choice.startswith("function="):
                # Handle format like "function=my_function"
                func_name = tool_choice.split("=", 1)[1]
                result["tool_choice"] = {"type": "tool", "name": func_name}
            elif isinstance(tool_choice, str):
                # Assume it's a function name
                result["tool_choice"] = {"type": "tool", "name": tool_choice}
            elif isinstance(tool_choice, dict):
                # Already Anthropic format, pass through
                result["tool_choice"] = tool_choice
        return result

    def create_model_response_wrapper(self, foundry_resp, model: str) -> ModelResponse:
        # Anthropic responses typically have: id, content (list), model, stop_reason, usage (optional)
        # We normalize content list -> single text string
        text_parts: List[str] = []
        try:
            for block in getattr(foundry_resp, "content", []) or []:
                # block could be dict-like or object with .text
                if isinstance(block, dict):
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(block["text"])
                else:
                    # object-style
                    t = getattr(block, "text", None)
                    if t:
                        text_parts.append(t)
        except Exception:
            # best-effort fallback
            text_parts = [str(getattr(foundry_resp, "content", ""))]
        content = "".join(text_parts).strip()

        choice = Choices(
            message=Message(content=content, role="assistant"),
            index=0,
            finish_reason=getattr(foundry_resp, "stop_reason", "stop"),
        )

        usage_obj = None
        try:
            usage = getattr(foundry_resp, "usage", None)
            if usage:
                def read(key):
                    return usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)

                # Same contract as the stream path and AnthropicAIProvider: prompt_tokens
                # excludes cached tokens, which are reported in their own two fields.
                prompt_tokens = read("input_tokens")
                completion_tokens = read("output_tokens")
                total_tokens = None
                if prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens
                usage_obj = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                usage_obj.cache_creation_input_tokens = read("cache_creation_input_tokens")
                usage_obj.cache_read_input_tokens = read("cache_read_input_tokens")
        except Exception:
            usage_obj = None

        return ModelResponse(
            id=getattr(foundry_resp, "id", None),
            choices=[choice],
            created=int(time.time()),
            model=model,
            usage=usage_obj,
        )

    def post_stream_processing_wrapper(self, model: str, messages: List[dict], **kwargs):
        """
        Stream AnthropicFoundry events and yield OpenAI-like streaming chunks.
        """        
        # Convert messages to Anthropic format and extract system message
        try:
            system_message, messages = self._convert_openai_to_anthropic_messages(messages)
        except AzureProviderError:
            raise
        except Exception as e:
            raise AzureProviderError(
                status_code=500,
                message=f"Failed to convert messages format: {str(e)}"
            )
        
        params = self._preprocess(**kwargs)
        params["stream"] = True
        
        # Add system parameter if system message exists
        if system_message:
            params["system"] = system_message
        
        # Convert tools from OpenAI format to Anthropic format
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        if tools:
            try:
                tools_config = self._convert_tools_to_anthropic(tools, tool_choice)
                params.update(tools_config)
            except AzureProviderError:
                raise
            except Exception as e:
                raise AzureProviderError(
                    status_code=500,
                    message=f"Failed to convert tools format: {str(e)}"
                )
        
        stream = self.client.messages.create(
            model=model,
            messages=messages,
            **params,
        )

        msg_id = None
        finish_reason = None
        tool_calls = {}  # Track tool calls by index
        content_blocks = {}  # Track content blocks by index
        
        for event in stream:            
            # Extract message ID from message_start event
            if msg_id is None:
                if hasattr(event, "message") and hasattr(event.message, "id"):
                    msg_id = event.message.id
                elif hasattr(event, "id"):
                    msg_id = event.id

            event_type = getattr(event, "type", None)

            # Handle different event types according to Anthropic's streaming spec
            if event_type == "message_start":
                # Message started, extract ID and initial data
                if hasattr(event, "message") and hasattr(event.message, "id"):
                    msg_id = event.message.id

            elif event_type == "content_block_start":
                # New content block started (text or tool_use)
                index = getattr(event, "index", 0)
                content_block = getattr(event, "content_block", None)
                
                if content_block:
                    block_type = getattr(content_block, "type", None)
                    if block_type == "tool_use":
                        # Tool use block started
                        tool_id = getattr(content_block, "id", f"call_{index}")
                        tool_name = getattr(content_block, "name", "")
                        tool_calls[index] = {
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": ""}
                        }
                        content_blocks[index] = {"type": "tool_use", "partial_json": ""}
                        
                        # Yield initial tool call with name (no arguments yet)
                        chunk_delta = Delta(
                            tool_calls=[{
                                "index": index,
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": ""
                                }
                            }]
                        )
                        stream_choice = StreamingChoices(index=0, delta=chunk_delta)
                        yield ModelResponse(
                            id=msg_id,
                            choices=[stream_choice],
                            created=int(time.time()),
                            model=model,
                            usage=None,
                            stream=True,
                        )
                    elif block_type == "text":
                        # Text block started
                        content_blocks[index] = {"type": "text", "text": ""}

            elif event_type == "content_block_delta":
                # Content block delta (text or tool input)
                index = getattr(event, "index", 0)
                delta_obj = getattr(event, "delta", None)
                
                if delta_obj:
                    delta_type = getattr(delta_obj, "type", None)                    
                    if delta_type == "text_delta":
                        # Text content delta
                        text = getattr(delta_obj, "text", "")
                        if text and index in content_blocks and content_blocks[index]["type"] == "text":
                            content_blocks[index]["text"] += text
                            
                            # Yield text delta
                            chunk_delta = Delta(content=text)
                            stream_choice = StreamingChoices(index=0, delta=chunk_delta)
                            yield ModelResponse(
                                id=msg_id,
                                choices=[stream_choice],
                                created=int(time.time()),
                                model=model,
                                usage=None,
                                stream=True,
                            )
                    
                    elif delta_type == "input_json_delta":
                        # Tool input JSON delta
                        partial_json = getattr(delta_obj, "partial_json", "")
                        if index in tool_calls and index in content_blocks:
                            content_blocks[index]["partial_json"] += partial_json
                            tool_calls[index]["function"]["arguments"] += partial_json
                            
                            # Yield tool call delta (only arguments, name already sent in content_block_start)
                            chunk_delta = Delta(
                                tool_calls=[{
                                    "index": index,
                                    "id": tool_calls[index]["id"],
                                    "type": "function",
                                    "function": {
                                        "arguments": partial_json  # Only send the incremental arguments
                                    }
                                }]
                            )
                            stream_choice = StreamingChoices(index=0, delta=chunk_delta)
                            yield ModelResponse(
                                id=msg_id,
                                choices=[stream_choice],
                                created=int(time.time()),
                                model=model,
                                usage=None,
                                stream=True,
                            )
                    elif delta_type == "thinking_delta":
                        reasoning_content = getattr(delta_obj, "thinking", "")
                        if reasoning_content:
                            # Yield reasoning_content delta
                            chunk_delta = Delta(reasoning_content=reasoning_content)
                            stream_choice = StreamingChoices(index=0, delta=chunk_delta)
                            yield ModelResponse(
                                id=msg_id,
                                choices=[stream_choice],
                                created=int(time.time()),
                                model=model,
                                usage=None,
                                stream=True,
                            )      
                    elif delta_type == "signature_delta":
                        signature_content = getattr(delta_obj, "signature", "")
                        if signature_content:
                            # Yield signature_content delta
                            chunk_delta = Delta(thought_signature=signature_content)
                            stream_choice = StreamingChoices(index=0, delta=chunk_delta)
                            yield ModelResponse(
                                id=msg_id,
                                choices=[stream_choice],
                                created=int(time.time()),
                                model=model,
                                usage=None,
                                stream=True,
                            )

            elif event_type == "content_block_stop":
                # Content block completed
                index = getattr(event, "index", 0)
                # Could emit final tool call here if needed

            elif event_type == "message_stop":
                # Message completed
                stream_choice = StreamingChoices(
                    index=0, 
                    finish_reason=finish_reason or "stop",
                    delta=Delta()
                )
                yield ModelResponse(
                    id=msg_id,
                    choices=[stream_choice],
                    created=int(time.time()),
                    model=model,
                    usage=None,
                    stream=True,
                )
            elif event_type == "message_delta":
                stream_choice = StreamingChoices(
                    index=0, 
                    finish_reason=event.finish_reason if hasattr(event, "finish_reason") else None,
                    delta=Delta()
                )
                
                if hasattr(event, "usage"):
                    chunk_usage = Usage()
                    chunk_usage.prompt_tokens = event.usage.input_tokens
                    chunk_usage.completion_tokens = event.usage.output_tokens
                    chunk_usage.cache_creation_input_tokens = event.usage.cache_creation_input_tokens
                    chunk_usage.cache_read_input_tokens = event.usage.cache_read_input_tokens
                    chunk_usage.total_tokens = chunk_usage.prompt_tokens + chunk_usage.completion_tokens

                model_response = ModelResponse(
                    id=msg_id,
                    choices=[stream_choice],
                    created=int(time.time()),
                    model=model,
                    usage=chunk_usage,
                    stream=True,
                )
                yield model_response

    def completion(self, model: str, messages: List[dict], **kwargs) -> ModelResponse:
        if not model or messages is None:
            raise AzureProviderError(status_code=422, message="Missing model or messages")

        # Ensure message format is acceptable, and check multimodal constraints
        check = self.check_prompt("anthropic", model, messages)
        if not check.get("pass_check"):
            raise AzureProviderError(status_code=422, message=str(check.get("reason")))
        checked_messages = check.get("messages", messages)

        stream = kwargs.get("stream", False)

        if stream:
            # The stream wrapper does its own conversion. It must get the messages
            # before conversion: converted ones have no system message left, so the
            # system prompt used to be dropped from every streamed request.
            try:
                return self.post_stream_processing_wrapper(model, checked_messages, **kwargs)
            except AzureProviderError:
                raise
            except Exception as e:
                status = getattr(e, "status_code", 500)
                raise AzureProviderError(status_code=status, message=str(e))
        else:
            # Convert OpenAI-style messages to Anthropic-style messages and extract system message
            # This handles image_url -> image, video_url -> video, system -> system parameter, etc.
            try:
                system_message, norm_messages = self._convert_openai_to_anthropic_messages(checked_messages)
            except AzureProviderError:
                raise
            except Exception as e:
                raise AzureProviderError(
                    status_code=500,
                    message=f"Failed to convert messages format: {str(e)}"
                )

            params = self._preprocess(**kwargs)
            
            # Add system parameter if system message exists
            if system_message:
                params["system"] = system_message
            
            # Convert tools from OpenAI format to Anthropic format
            tools = kwargs.get("tools")
            tool_choice = kwargs.get("tool_choice")
            if tools:
                try:
                    tools_config = self._convert_tools_to_anthropic(tools, tool_choice)
                    params.update(tools_config)
                except AzureProviderError:
                    raise
                except Exception as e:
                    raise AzureProviderError(
                        status_code=500,
                        message=f"Failed to convert tools format: {str(e)}"
                    )

            try:
                # AnthropicFoundry expects messages list with {role, content}
                resp = self.client.messages.create(
                    model=model,
                    messages=norm_messages,
                    **params,
                )
            except Exception as e:
                status = getattr(e, "status_code", 500)
                raise AzureProviderError(status_code=status, message=str(e))

            return self.create_model_response_wrapper(resp, model=model)
