from .base_provider import BaseProvider
from unionllm.utils import ModelResponse, Message, Choices, Usage, Delta, StreamingChoices
from google import genai
import os, json, time
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import requests
import re

class GeminiError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class GeminiAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        _env_api_key = os.environ.get("GEMINI_API_KEY")
        self.api_key = model_kwargs.get("api_key") if model_kwargs.get("api_key") else _env_api_key
        if not self.api_key:
            raise GeminiError(
                status_code=422, message=f"Missing API key"
            )
        self.client = genai.Client(api_key=self.api_key) 

    def pre_processing(self, **kwargs):
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p",
            "stream", "stop", "presence_penalty", "frequency_penalty",
            "system_instruction",
            "image_url", "tools", "tool_choice",
            "file_url", "video_url", "audio_url", "reasoning_effort",
            # 兼容思考相关参数
            "thinking_level", "thinking",
            "aspect_ratio", "resolution",
            "google_search_grounding"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def _build_config(self, new_kwargs: dict, stream: bool = False, multimodal: bool = False, has_image: bool = False):
        """抽象出在流式与非流式里重复的 GenerateContentConfig 构建逻辑。
        统一处理: response_modalities / system_instruction / tools / thinking_config / image_config / google_search_grounding。
        说明:
        - 文件模态(File)与后续基于具体 contents 的图像追加, 仍在调用处补充。
        - stream 参数区分工具声明的写法差异。
        """
        try:
            # 初始模态
            modalities = ['Text']
            if multimodal or has_image or ('image_url' in new_kwargs) or ('aspect_ratio' in new_kwargs) or ('resolution' in new_kwargs):
                modalities = ['Text', 'Image']
            config = types.GenerateContentConfig(response_modalities=modalities)

            # system instruction
            if 'system_instruction' in new_kwargs:
                try:
                    config.system_instruction = new_kwargs['system_instruction']
                except Exception:
                    pass

            # 工具处理
            if 'tools' in new_kwargs:
                if stream:
                    # 流式: 直接使用用户提供的 function 对象
                    try:
                        gemini_tools = [tool['function'] for tool in new_kwargs['tools']]
                        tool_obj = types.Tool(function_declarations=gemini_tools)
                        config.tools = [tool_obj]
                    except Exception:
                        pass
                else:
                    # 非流式: 构建 FunctionDeclaration 并提供 tool_config
                    fdecls = []
                    for tool in new_kwargs['tools']:
                        f = tool.get('function', {})
                        try:
                            fd = types.FunctionDeclaration(
                                name=f.get('name'),
                                description=f.get('description', ''),
                                parameters=f.get('parameters', {})
                            )
                            fdecls.append(fd)
                        except Exception:
                            fdecls.append(f)  # 透传回退
                    try:
                        config.tools = [types.Tool(function_declarations=fdecls)]
                    except Exception:
                        pass
                    try:
                        allowed_names = [tool['function']['name'] for tool in new_kwargs['tools'] if 'function' in tool and 'name' in tool['function']]
                        mode = 'AUTO' if new_kwargs.get('tool_choice') == 'auto' else 'ANY'
                        config.tool_config = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                allowed_function_names=allowed_names,
                                mode=mode
                            )
                        )
                    except Exception:
                        pass

            # thinking config
            try:
                thinking_level = new_kwargs.get('reasoning_effort') or new_kwargs.get('thinking_level')
                if thinking_level is None and 'thinking' in new_kwargs:
                    val = new_kwargs.get('thinking')
                    if isinstance(val, str) and val:
                        thinking_level = val
                    elif bool(val):
                        thinking_level = 'low'
                if thinking_level:
                    try:
                        config.thinking_config = types.ThinkingConfig(thinking_level=str(thinking_level), include_thoughts=True)
                    except Exception:
                        try:
                            config.thinking_config = types.ThinkingConfig(thinking_level=str(thinking_level))
                        except Exception:
                            pass
            except Exception:
                pass

            # image config
            try:
                aspect_ratio = new_kwargs.get('aspect_ratio')
                resolution = new_kwargs.get('resolution')
                if aspect_ratio or resolution:
                    try:
                        img_cfg = types.ImageConfig()
                        if aspect_ratio:
                            img_cfg.aspect_ratio = aspect_ratio
                        if resolution:
                            img_cfg.image_size = resolution
                        config.image_config = img_cfg
                    except Exception:
                        try:
                            igc = types.ImageGenerationConfig()
                            igc.aspect_ratio = aspect_ratio
                            igc.image_size = resolution
                            config.image_generation_config = igc
                        except Exception:
                            pass
                    try:
                        if 'Image' not in getattr(config, 'response_modalities', []):
                            config.response_modalities = list({*getattr(config, 'response_modalities', []), 'Image'})
                    except Exception:
                        pass
            except Exception:
                pass

            # google_search_grounding
            try:
                if new_kwargs.get('google_search_grounding') == 'enabled':
                    try:
                        if not getattr(config, 'tools', None):
                            config.tools = []
                    except Exception:
                        config.tools = []
                    try:
                        config.tools.append({'google_search': {}})
                    except Exception:
                        try:
                            gs_tool = types.Tool()
                            setattr(gs_tool, 'google_search', {})
                            config.tools.append(gs_tool)
                        except Exception:
                            pass
            except Exception:
                pass

            return config
        except Exception as e:
            raise GeminiError(status_code=500, message=f"Error building config: {str(e)}")

    def _extract_markdown_image_url(self, text: str):
        """
        如果 text 完全是 Markdown 图片语法，如: ![alt](https://example.com/x.png)
        则返回其中的 URL；否则返回 None。
        """

        if not isinstance(text, str):
            return None
        
        s = text.strip()
        # 尝试去除可能包裹的三引号（针对某些输入场景）
        if len(s) >= 6:
            if s.startswith('"""') and s.endswith('"""'):
                s = s[3:-3].strip()
            elif s.startswith("'''") and s.endswith("'''"):
                s = s[3:-3].strip()
        
        pattern = r"^!\[[^\]]*\]\((https?://[^\s)]+)\)$"
        m = re.match(pattern, s)
        return m.group(1) if m else None

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        # 处理所有消息
        processed_messages = []
        stream_has_image = False
        for msg in messages:
            if msg["role"] == "user":
                # 支持 OpenAI 风格的多模态输入（content 为数组，含多张 image_url）
                parts = []
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    has_image = False
                    for item in content:
                        # 支持直接传 genai.types.Part 对象（非 dict）
                        if not isinstance(item, dict):
                            # 处理 Part 对象中的 text / inline_data
                            try:
                                if hasattr(item, 'text') and item.text is not None:
                                    txt = item.text
                                    converted = self._try_convert_markdown_image_to_part(str(txt))
                                    if converted:
                                        parts.append(converted)
                                        has_image = True
                                    else:
                                        text_parts.append(str(txt))
                                elif hasattr(item, 'inline_data') and item.inline_data is not None:
                                    # 直接保留已有 inline_data
                                    parts.append(item)
                                    has_image = True
                                else:
                                    # 未识别的 Part，尝试转成文本
                                    parts.append(types.Part(text=str(item)))
                                continue
                            except Exception:
                                continue
                        item_type = item.get("type")
                        
                        if item_type == "text":
                            txt = item.get("text", "")
                            converted = self._try_convert_markdown_image_to_part(txt)
                            if converted:
                                parts.append(converted)
                                has_image = True
                            else:
                                text_parts.append(txt)
                        elif item_type == "image_url":
                            image_url = (item.get("image_url") or {}).get("url")
                            if image_url:
                                try:
                                    resp = requests.get(image_url)
                                    img_bytes = resp.content
                                    # 通过 header 或 PIL 推断 mime
                                    mime_type = resp.headers.get('Content-Type', None)
                                    if not mime_type:
                                        try:
                                            from PIL import Image as _Img
                                            im = _Img.open(BytesIO(img_bytes))
                                            fmt = (im.format or 'JPEG').lower()
                                            mime_type = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
                                        except Exception:
                                            mime_type = 'image/jpeg'
                                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                                    has_image = True
                                except Exception as e:
                                    raise GeminiError(status_code=500, message=f"Error processing image URL in stream: {str(e)}")
                        elif item_type == "audio_url":
                            audio_url = (item.get("audio_url") or {}).get("url")
                            if audio_url:
                                try:
                                    a_resp = requests.get(audio_url)
                                    a_bytes = a_resp.content
                                    a_mime = a_resp.headers.get('Content-Type', None)
                                    if not a_mime:
                                        # 根据扩展名简单推断
                                        lower = audio_url.lower()
                                        if lower.endswith('.wav'):
                                            a_mime = 'audio/wav'
                                        elif lower.endswith('.mp3'):
                                            a_mime = 'audio/mp3'
                                        elif lower.endswith('.aiff') or lower.endswith('.aif'):
                                            a_mime = 'audio/aiff'
                                        elif lower.endswith('.aac'):
                                            a_mime = 'audio/aac'
                                        elif lower.endswith('.ogg') or lower.endswith('.oga'):
                                            a_mime = 'audio/ogg'
                                        elif lower.endswith('.flac'):
                                            a_mime = 'audio/flac'
                                        else:
                                            a_mime = 'audio/mpeg'
                                    parts.append(types.Part.from_bytes(data=a_bytes, mime_type=a_mime))
                                except Exception as e:
                                    raise GeminiError(status_code=500, message=f"Error processing audio URL in stream: {str(e)}")
                        elif item_type == "file":
                            if "file" in item:
                                file_data = item["file"]["file_data"]
                                # 从file_data中获取文件类型
                                content_type = file_data.split(",")[0].split(";")[0].replace("data:", "")
                                try:
                                    # 解析base64部分
                                    base64_data = file_data.split(",")[1]
                                    file_content = base64.b64decode(base64_data)
                                    
                                    # 使用 types.Part.from_bytes 创建文件部分
                                    file_part = types.Part.from_bytes(
                                        data=file_content,
                                        mime_type=content_type
                                    )
                                    parts.append(file_part)
                                except Exception as e:
                                    raise GeminiError(status_code=500, message=f"Error processing file data in stream: {str(e)}")
                        elif item_type == "video_url":
                            video_url = item.get("video_url", {}).get("url", "")
                            if video_url and video_url.startswith("https://www.youtube.com/"):
                                try:
                                    parts.append(types.Part(file_data=types.FileData(file_uri=video_url)))
                                except Exception as e:
                                    raise GeminiError(status_code=500, message=f"Error processing video URL in stream: {str(e)}")
                    # 文本优先放在前面,便于上下文理解
                    if text_parts:
                        parts.insert(0, types.Part(text=" ".join(text_parts)))
                    # 如果既没有文本也没有有效图片，至少传一个空文本，避免 SDK 报错
                    if not parts:
                        parts = [types.Part(text="")]
                    if has_image:
                        stream_has_image = True
                else:
                    # 纯文本消息，支持 "markdown 图片" 转图片 part
                    txt = str(content)
                    converted = self._try_convert_markdown_image_to_part(txt)
                    if converted:
                        parts = [converted]
                        stream_has_image = True
                    else:
                        parts = [types.Part(text=txt)]

                processed_messages.append(types.Content(
                    role="user",
                    parts=parts
                ))
            elif msg["role"] == "assistant":
                parts = []
                # 处理助手消息，包括工具调用
                
                # 处理文本内容
                if msg.get("content"):
                    converted = self._try_convert_markdown_image_to_part(msg["content"])
                    if converted:
                        parts.append(converted)
                        if msg.get("thought_signature") and msg['thought_signature']:
                            try:
                                parts[-1].thought_signature = msg['thought_signature']
                            except Exception:
                                pass

                    else:
                        if msg.get("thought_signature") and msg['thought_signature']:
                            parts.append(types.Part(text=msg["content"], thought_signature=msg["thought_signature"]))
                        else:
                            parts.append(types.Part(text=msg["content"]))
                              
                # 处理工具调用
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        if tool_call.get("type") == "function":
                            function_info = tool_call.get("function", {})
                            try:
                                args = json.loads(function_info.get("arguments", "{}"))
                                parts.append(types.Part(
                                    function_call=types.FunctionCall(
                                        name=function_info.get("name"),
                                        args=args
                                    )
                                ))
                                if "thought_signature" in tool_call:
                                    raw_ts = tool_call["thought_signature"]
                                    parts[-1].thought_signature = raw_ts
                            except Exception as e:
                                raise GeminiError(status_code=500, message=f"Error processing tool_calls in stream: {str(e)}")

                if not parts:
                    parts = [types.Part(text="")]
                processed_messages.append(types.Content(
                    role="model",
                    parts=parts
                ))
            elif msg["role"] == "tool":
                # 处理工具响应消息
                try:
                    function_response = types.FunctionResponse(
                        name=msg.get("name", ""),
                        response={"result": msg.get("content", "")}
                    )
                    processed_messages.append(types.Content(
                        role="function",
                        parts=[types.Part(function_response=function_response)]
                    ))
                except Exception as e:
                    raise GeminiError(status_code=500, message=f"Error processing tool response in stream: {str(e)}")
            elif msg["role"] == "system":
                # 系统消息作为配置项处理
                new_kwargs["system_instruction"] = msg["content"]

        # 使用统一构建方法创建 config
        config = self._build_config(new_kwargs, stream=True, multimodal=False, has_image=stream_has_image)

        # 处理多模态内容
        last_message = messages[-1]["content"]
        if "audio_url" in new_kwargs:
            try:
                a_resp = requests.get(new_kwargs["audio_url"])
                a_bytes = a_resp.content
                a_mime = a_resp.headers.get('Content-Type', None)
                if not a_mime:
                    lower = new_kwargs["audio_url"].lower()
                    if lower.endswith('.wav'):
                        a_mime = 'audio/wav'
                    elif lower.endswith('.mp3'):
                        a_mime = 'audio/mp3'
                    elif lower.endswith('.aiff') or lower.endswith('.aif'):
                        a_mime = 'audio/aiff'
                    elif lower.endswith('.aac'):
                        a_mime = 'audio/aac'
                    elif lower.endswith('.ogg') or lower.endswith('.oga'):
                        a_mime = 'audio/ogg'
                    elif lower.endswith('.flac'):
                        a_mime = 'audio/flac'
                    else:
                        a_mime = 'audio/mpeg'
                # 音频理解仅需要文本输出
                processed_messages[-1].parts.append(types.Part.from_bytes(data=a_bytes, mime_type=a_mime))
            except Exception as e:
                raise GeminiError(status_code=500, message=f"Error processing audio URL: {str(e)}")
        if "image_url" in new_kwargs:
            config.response_modalities = ['Image', 'Text']
            try:
                response = requests.get(new_kwargs["image_url"])
                img_bytes = response.content
                mime_type = response.headers.get('Content-Type', 'image/jpeg')
                processed_messages[-1].parts.append(
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                )
            except Exception as e:
                raise GeminiError(status_code=500, message=f"Error processing image URL: {str(e)}")
            stream_has_image = True
        
        if "file_url" in new_kwargs:
            file_url = new_kwargs["file_url"]
            config.response_modalities = ['Text', 'File']
            try:
                response = requests.get(file_url)
                file_content = response.content
                
                content_type = response.headers.get('Content-Type', 'application/octet-stream')
                if file_url.endswith('.pdf'):
                    content_type = 'application/pdf'
                elif file_url.endswith('.docx'):
                    content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                
                file_part = types.Part.from_bytes(
                    data=file_content,
                    mime_type=content_type
                )
                processed_messages[-1].parts.append(file_part)
            except Exception as e:
                raise GeminiError(status_code=500, message=f"Error processing file URL: {str(e)}")

        # 如果消息中包含图片（无论来自 content 数组还是 image_url 参数），确保响应模态包含图像
        if stream_has_image:
            try:
                # 有些情况下 config 可能未设置 response_modalities（例如启用 tools），此处补齐
                existing = getattr(config, 'response_modalities', None)
                if not existing:
                    config.response_modalities = ['Text', 'Image']
                elif 'Image' not in existing:
                    config.response_modalities = list({*existing, 'Image'})
            except Exception:
                # 如果出现属性不可写等问题，忽略，仍可正常流式返回文本
                pass

        try:
            # 使用 generate_content_stream 方法
            response = self.client.models.generate_content_stream(
                model=model,
                contents=processed_messages,
                config=config
            )
            
            index = 0
            final_usage_obj = None
            for chunk in response:
                chunk_choices = []
                # 不在中间chunk直接输出usage, 仅收集到最后
                usage_obj = None
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    # 检查candidates[0]是否有content属性
                    if hasattr(chunk.candidates[0], 'content'):
                        # 检查content是否有parts属性且不为None
                        if hasattr(chunk.candidates[0].content, 'parts') and chunk.candidates[0].content.parts is not None:
                            for part in chunk.candidates[0].content.parts:
                                chunk_delta = Delta()
                                if part.text is not None:
                                    chunk_delta.role = "assistant"
                                    # 如果是思考内容（Gemini 思考部分），映射到 reasoning_content
                                    if hasattr(part, 'thought') and getattr(part, 'thought', False):
                                        chunk_delta.reasoning_content = part.text
                                    else:
                                        chunk_delta.content = part.text
                                    # 捕获任意文本 part 上的 thought_signature（不再仅限于 function_call）
                                    if hasattr(part, 'thought_signature') and getattr(part, 'thought_signature'):
                                        try:
                                            chunk_delta.thought_signature = part.thought_signature
                                        except Exception:
                                            pass
                                    chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta))
                                elif part.inline_data is not None:
                                    try:
                                        # 将图片转换为base64字符串
                                        image = Image.open(BytesIO(part.inline_data.data))
                                        buffered = BytesIO()
                                        image.save(buffered, format="PNG")
                                        img_str = base64.b64encode(buffered.getvalue()).decode()
                                        
                                        # 添加markdown格式的图片
                                        chunk_delta.role = "assistant"
                                        chunk_delta.content = f"\n![generated_image](data:image/png;base64,{img_str})\n"
                                        # 如果图片 part 上也有 thought_signature，同步加入
                                        if hasattr(part, 'thought_signature') and part.thought_signature:
                                            try:
                                                chunk_delta.thought_signature = part.thought_signature
                                            except Exception:
                                                pass
                                        chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta))
                                    except Exception as e:
                                        raise GeminiError(status_code=500, message=f"Error processing image in stream: {str(e)}")
                                # 处理函数调用
                                elif hasattr(part, 'function_call') and part.function_call:
                                    try:
                                        chunk_delta.role = "assistant"
                                        this_tool_call = {
                                            "id": f"call_{time.time()}",
                                            "type": "function",
                                            "function": {
                                                "name": part.function_call.name,
                                                "arguments": json.dumps(part.function_call.args)
                                            }
                                        }
                                        chunk_delta.tool_calls = []
                                        
                                        # 处理 thought_signature: 原样透传，不做编码或解码（保持可逆性）
                                        if hasattr(part, 'thought_signature') and part.thought_signature:
                                            raw_ts = part.thought_signature
                                            # 直接挂载原始对象（通常为bytes）。调用方需自行处理序列化。
                                            chunk_delta.thought_signature = raw_ts
                                            this_tool_call['thought_signature'] = raw_ts
                                        # 若 function_call part 也附带普通文本 thought（SDK 行为不确定），继续兼容
                                        if hasattr(part, 'thought') and getattr(part, 'thought', False) and part.text:
                                            chunk_delta.reasoning_content = part.text
                                        chunk_delta.tool_calls.append(this_tool_call)
                                        chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta, finish_reason="tool_calls"))
                                    except Exception as e:
                                        raise GeminiError(status_code=500, message=f"Error processing function call in stream: {str(e)}")
                                    
                        else:
                            raise GeminiError(status_code=500, message="Candidate content has no parts attribute or parts is None")
                    else:
                        raise GeminiError(status_code=500, message="Candidate has no content attribute")
                
                # 处理 usage_metadata (仅在最后一个/包含统计的 chunk 上出现)
                if hasattr(chunk, 'usage_metadata') and getattr(chunk, 'usage_metadata') is not None:
                    try:
                        um = chunk.usage_metadata
                        # 基础计数
                        prompt_tokens = getattr(um, 'prompt_token_count', None)
                        candidates_tokens = getattr(um, 'candidates_token_count', 0) or 0
                        total_tokens = getattr(um, 'total_token_count', None)
                        thoughts_tokens = getattr(um, 'thoughts_token_count', 0) or 0

                        # 细分模态统计（prompt 与 completion）
                        text_prompt_tokens = 0
                        image_prompt_tokens = 0
                        text_completion_tokens = 0
                        image_completion_tokens = 0
                        try:
                            for d in getattr(um, 'prompt_tokens_details', []) or []:
                                modality = getattr(d, 'modality', None) or getattr(d, 'media_type', None)
                                count = getattr(d, 'token_count', 0) or 0
                                if modality == 'TEXT':
                                    text_prompt_tokens += count
                                elif modality == 'IMAGE':
                                    image_prompt_tokens += count
                            for d in getattr(um, 'candidates_tokens_details', []) or []:
                                modality = getattr(d, 'modality', None) or getattr(d, 'media_type', None)
                                count = getattr(d, 'token_count', 0) or 0
                                if modality == 'TEXT':
                                    text_completion_tokens += count
                                elif modality == 'IMAGE':
                                    image_completion_tokens += count
                        except Exception:
                            # 忽略细分统计错误，保持健壮
                            pass

                        # 原 completion_tokens 保持与现有逻辑兼容 (候选+思考)
                        completion_tokens = candidates_tokens + thoughts_tokens if (candidates_tokens is not None or thoughts_tokens is not None) else None
                        if image_completion_tokens:
                            completion_tokens -= image_completion_tokens
                            if completion_tokens < 0:
                                completion_tokens = 0
                        final_usage_obj = Usage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens
                        )
                        # 附加新字段到 usage 对象（动态属性）
                        try:
                            final_usage_obj.thought_tokens = thoughts_tokens
                            final_usage_obj.text_prompt_tokens = text_prompt_tokens
                            final_usage_obj.image_prompt_tokens = image_prompt_tokens
                            final_usage_obj.text_completion_tokens = text_completion_tokens
                            final_usage_obj.image_completion_tokens = image_completion_tokens
                        except Exception:
                            pass
                    except Exception as e:
                        raise GeminiError(status_code=500, message=f"Error extracting usage metadata in stream: {str(e)}")

                # 只有当有内容时才生成响应
                if chunk_choices:
                    chunk_response = ModelResponse(
                        id=f"gemini-{time.time()}",
                        choices=chunk_choices,
                        created=int(time.time()),
                        model=model,
                        stream=True,
                        usage=None
                    )
                    index += 1
                    yield chunk_response
            # 循环结束后如果收集到usage, 发送一个最终仅包含usage的chunk
            if final_usage_obj is not None:
                chunk_choices = [StreamingChoices(index=index, delta='', finish_reason=None)]
                final_chunk = ModelResponse(
                    id=f"gemini-{time.time()}",
                    choices=chunk_choices,
                    created=int(time.time()),
                    model=model,
                    stream=True,
                    usage=final_usage_obj
                )
                yield final_chunk
        except Exception as e:
            raise GeminiError(status_code=500, message=f"Error in stream processing: {str(e)}")

    def create_model_response_wrapper(self, response, model):
        choices = []
        content = ""
        tool_calls = []
        collected_thought_signature = None
                    
        # 处理所有输出部分(文本、图片和函数调用)
        for part in response.candidates[0].content.parts:            
            if part.text is not None:
                content += part.text
                if hasattr(part, 'thought_signature') and part.thought_signature and collected_thought_signature is None:
                    collected_thought_signature = part.thought_signature
            elif part.inline_data is not None:
                try:
                    # 将图片转换为base64字符串
                    image = Image.open(BytesIO(part.inline_data.data))                    
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # 添加markdown格式的图片
                    content += f"\n![generated_image](data:image/png;base64,{img_str})\n"
                    # 捕获图片 part 的 thought_signature
                    if hasattr(part, 'thought_signature') and part.thought_signature and collected_thought_signature is None:
                        collected_thought_signature = part.thought_signature
                except Exception as e:
                    raise GeminiError(status_code=500, message=f"Error processing image: {str(e)}")
            elif hasattr(part, 'function_call'):
                try:
                    # 添加工具调用
                    tool_calls.append({
                        "id": f"call_{time.time()}",
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(part.function_call.args)
                        }
                    })
                    if hasattr(part, 'thought_signature') and part.thought_signature and collected_thought_signature is None:
                        collected_thought_signature = part.thought_signature
                except Exception as e:
                    raise GeminiError(status_code=500, message=f"Error processing function call: {str(e)}")
        
        message = Message(
            content=content,
            role="assistant"
        )
        if collected_thought_signature is not None:
            # 动态附加 thought_signature 到 message，供上层调用方使用
            message.thought_signature = collected_thought_signature
        
        # 如果有工具调用，添加到消息中
        if tool_calls:
            message.tool_calls = tool_calls
        
        choices.append(
            Choices(
                message=message,
                index=0,
                finish_reason="stop",
            )
        )
        # 提取 usage metadata
        usage_obj = None
        if hasattr(response, 'usage_metadata') and getattr(response, 'usage_metadata') is not None:
            try:
                um = response.usage_metadata
                prompt_tokens = getattr(um, 'prompt_token_count', None)
                candidates_tokens = getattr(um, 'candidates_token_count', 0) or 0
                total_tokens = getattr(um, 'total_token_count', None)
                thoughts_tokens = getattr(um, 'thoughts_token_count', 0) or 0

                text_prompt_tokens = 0
                image_prompt_tokens = 0
                text_completion_tokens = 0
                image_completion_tokens = 0
                try:
                    for d in getattr(um, 'prompt_tokens_details', []) or []:
                        modality = getattr(d, 'modality', None) or getattr(d, 'media_type', None)
                        count = getattr(d, 'token_count', 0) or 0
                        if modality == 'TEXT':
                            text_prompt_tokens += count
                        elif modality == 'IMAGE':
                            image_prompt_tokens += count
                    for d in getattr(um, 'candidates_tokens_details', []) or []:
                        modality = getattr(d, 'modality', None) or getattr(d, 'media_type', None)
                        count = getattr(d, 'token_count', 0) or 0
                        if modality == 'TEXT':
                            text_completion_tokens += count
                        elif modality == 'IMAGE':
                            image_completion_tokens += count
                except Exception:
                    pass

                completion_tokens = candidates_tokens + thoughts_tokens if (candidates_tokens is not None or thoughts_tokens is not None) else None
                usage_obj = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
                try:
                    usage_obj.thought_tokens = thoughts_tokens
                    usage_obj.text_prompt_tokens = text_prompt_tokens
                    usage_obj.image_prompt_tokens = image_prompt_tokens
                    usage_obj.text_completion_tokens = text_completion_tokens
                    usage_obj.image_completion_tokens = image_completion_tokens
                except Exception:
                    pass
            except Exception as e:
                raise GeminiError(status_code=500, message=f"Error extracting usage metadata: {str(e)}")

        model_response = ModelResponse(
            id=f"gemini-{time.time()}",
            choices=choices,
            created=int(time.time()),
            model=model,
            usage=usage_obj
        )
        return model_response

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise GeminiError(
                    status_code=422, message=f"Missing model or messages"
                )
            message_check_result = self.check_prompt("gemini", model, messages)            
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise GeminiError(
                    status_code=422, message=message_check_result['reason']
                )

            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)            

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                # 获取最后一条消息内容
                last_msg_obj = messages[-1]
                last_message = last_msg_obj["content"]
                # 通过抽象方法统一创建 config
                config = self._build_config(new_kwargs, stream=False, multimodal=("multimodal" in kwargs), has_image=False)
                
                # 处理图片URL
                contents = []

                if isinstance(last_message, list):
                    # 处理OpenAI格式的多模态消息
                    text_parts = []
                    image_parts = []
                    video_url = None
                    
                    for content in last_message:
                        if not isinstance(content, dict):
                            # 处理 Part 对象
                            if hasattr(content, 'text') and content.text is not None:
                                txt = content.text
                                converted = self._try_convert_markdown_image_to_part(str(txt))
                                if converted:
                                    image_parts.append(converted)
                                    try:
                                        converted.thought_signature = content.get("thought_signature")
                                    except Exception:
                                        pass
                                else:
                                    text_parts.append(str(txt))
                            elif hasattr(content, 'inline_data') and content.inline_data is not None:
                                # 透传元素级 thought_signature
                                if hasattr(content, 'thought_signature') and content.thought_signature:
                                    try:
                                        content.thought_signature = content.thought_signature
                                    except Exception:
                                        pass
                                image_parts.append(content)
                            continue
                        if content.get("type") == "text":
                            txt = content.get("text", "")
                            converted = self._try_convert_markdown_image_to_part(txt)
                            if converted:
                                # 透传内容项 thought_signature
                                if content.get("thought_signature"):
                                    try:
                                        converted.thought_signature = content.get("thought_signature")
                                    except Exception:
                                        pass
                                image_parts.append(converted)
                            else:
                                text_parts.append(txt)
                        elif content.get("type") == "audio_url":
                            try:
                                audio_url = content.get("audio_url", {}).get("url", "")
                                if audio_url:
                                    a_resp = requests.get(audio_url)
                                    a_bytes = a_resp.content
                                    a_mime = a_resp.headers.get('Content-Type', None)
                                    if not a_mime:
                                        lower = audio_url.lower()
                                        if lower.endswith('.wav'):
                                            a_mime = 'audio/wav'
                                        elif lower.endswith('.mp3'):
                                            a_mime = 'audio/mp3'
                                        elif lower.endswith('.aiff') or lower.endswith('.aif'):
                                            a_mime = 'audio/aiff'
                                        elif lower.endswith('.aac'):
                                            a_mime = 'audio/aac'
                                        elif lower.endswith('.ogg') or lower.endswith('.oga'):
                                            a_mime = 'audio/ogg'
                                        elif lower.endswith('.flac'):
                                            a_mime = 'audio/flac'
                                        else:
                                            a_mime = 'audio/mpeg'
                                    image_parts.append(types.Part.from_bytes(data=a_bytes, mime_type=a_mime))
                            except Exception as e:
                                raise GeminiError(status_code=500, message=f"Error downloading audio: {str(e)}")
                        elif content.get("type") == "image_url":
                            try:
                                image_url = content.get("image_url", {}).get("url", "")
                                if image_url:
                                    response = requests.get(image_url)
                                    img_bytes = response.content
                                    mime_type = response.headers.get('Content-Type', None)
                                    if not mime_type:
                                        try:
                                            from PIL import Image as _Img
                                            im = _Img.open(BytesIO(img_bytes))
                                            fmt = (im.format or 'JPEG').lower()
                                            mime_type = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
                                        except Exception:
                                            mime_type = 'image/jpeg'
                                    image_parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                            except Exception as e:
                                raise GeminiError(status_code=500, message=f"Error downloading image: {str(e)}")
                        elif content.get("type") == "file":
                            if "file" in content:
                                file_data = content["file"]["file_data"]
                                # 从file_data中获取文件类型
                                content_type = file_data.split(",")[0].split(";")[0].replace("data:", "")
                                
                                config.response_modalities = ['Text']
                                try:
                                    # 解析base64部分
                                    base64_data = file_data.split(",")[1]
                                    file_content = base64.b64decode(base64_data)
                                    
                                    # 使用 types.Part.from_bytes 创建文件部分
                                    file_part = types.Part.from_bytes(
                                        data=file_content,
                                        mime_type=content_type
                                    )
                                    
                                    # 添加文本部分和文件部分
                                    contents = [file_part, " ".join(text_parts) if text_parts else ""]
                                except Exception as e:
                                    raise GeminiError(status_code=500, message=f"Error processing file data: {str(e)}")
                        elif content.get("type") == "video_url":
                            video_url = content.get("video_url", {}).get("url", "")
                    
                    # 如果有视频URL，则优先处理视频
                    if video_url and video_url.startswith("https://www.youtube.com/"):
                        try:
                            # 创建Content对象
                            contents = types.Content(
                                parts=[
                                    types.Part(text=" ".join(text_parts) if text_parts else ""),
                                    types.Part(
                                        file_data=types.FileData(file_uri=video_url)
                                    )
                                ]
                            )
                            
                            # 使用 generate_content 方法处理视频内容
                            result = self.client.models.generate_content(
                                model=model,
                                contents=contents,
                                config=config
                            )
                            
                            return self.create_model_response_wrapper(result, model=model)
                        except Exception as e:
                            raise GeminiError(status_code=500, message=f"Error processing video URL: {str(e)}")
                    else:
                        # 添加文本部分
                        if text_parts:
                            base_text = " ".join(text_parts)
                            # 如果顶层消息有 thought_signature，用一个 Part 包裹文本以附加该字段
                            if last_msg_obj.get("thought_signature"):
                                txt_part = types.Part(text=base_text)
                                try:
                                    txt_part.thought_signature = last_msg_obj.get("thought_signature")
                                except Exception:
                                    pass
                                contents.append(txt_part)
                            else:
                                contents.append(base_text)
                        
                        # 添加图片部分
                        contents.extend(image_parts)
                        
                        # 设置响应模态
                        if image_parts:
                            config.response_modalities = ['Text', 'Image']
                elif "audio_url" in new_kwargs:
                    try:
                        audio_url = new_kwargs["audio_url"]
                        contents.append(last_message)
                        a_resp = requests.get(audio_url)
                        a_bytes = a_resp.content
                        a_mime = a_resp.headers.get('Content-Type', None)
                        if not a_mime:
                            lower = audio_url.lower()
                            if lower.endswith('.wav'):
                                a_mime = 'audio/wav'
                            elif lower.endswith('.mp3'):
                                a_mime = 'audio/mp3'
                            elif lower.endswith('.aiff') or lower.endswith('.aif'):
                                a_mime = 'audio/aiff'
                            elif lower.endswith('.aac'):
                                a_mime = 'audio/aac'
                            elif lower.endswith('.ogg') or lower.endswith('.oga'):
                                a_mime = 'audio/ogg'
                            elif lower.endswith('.flac'):
                                a_mime = 'audio/flac'
                            else:
                                a_mime = 'audio/mpeg'
                        contents.append(types.Part.from_bytes(data=a_bytes, mime_type=a_mime))
                        # 音频 + 文本
                        config.response_modalities = ['Text']
                    except Exception as e:
                        raise GeminiError(status_code=500, message=f"Error downloading audio: {str(e)}")
                elif "image_url" in new_kwargs:
                    # 处理image_url参数
                    try:
                        image_url = new_kwargs["image_url"]
                        # 添加文本部分
                        contents.append(last_message)
                        response = requests.get(image_url)
                        img_bytes = response.content
                        mime_type = response.headers.get('Content-Type', None)
                        if not mime_type:
                            try:
                                from PIL import Image as _Img
                                im = _Img.open(BytesIO(img_bytes))
                                fmt = (im.format or 'JPEG').lower()
                                mime_type = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
                            except Exception:
                                mime_type = 'image/jpeg'
                        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
                        config.response_modalities = ['Text', 'Image']
                    except Exception as e:
                        raise GeminiError(status_code=500, message=f"Error downloading image: {str(e)}")
                elif "video_url" in new_kwargs:
                    # 处理YouTube视频URL
                    try:
                        video_url = new_kwargs["video_url"]
                        # 创建Content对象
                        contents = types.Content(
                            parts=[
                                types.Part(text=last_message),
                                types.Part(
                                    file_data=types.FileData(file_uri=video_url)
                                )
                            ]
                        )
                        
                        # 使用 generate_content 方法处理视频内容
                        result = self.client.models.generate_content(
                            model=model,
                            contents=contents,
                            config=config
                        )
                        
                        return self.create_model_response_wrapper(result, model=model)
                    except Exception as e:
                        raise GeminiError(status_code=500, message=f"Error processing video URL: {str(e)}")
                else:
                    # 普通文本消息，支持 "markdown 图片" 转图片 part
                    if isinstance(last_message, str):
                        converted = self._try_convert_markdown_image_to_part(last_message)
                        if converted:
                            if last_msg_obj.get("thought_signature"):
                                try:
                                    converted.thought_signature = last_msg_obj.get("thought_signature")
                                except Exception:
                                    pass
                            contents = [converted]
                            config.response_modalities = ['Text', 'Image']
                        else:
                            if last_msg_obj.get("thought_signature"):
                                tp = types.Part(text=last_message)
                                try:
                                    tp.thought_signature = last_msg_obj.get("thought_signature")
                                except Exception:
                                    pass
                                contents = [tp]
                            else:
                                contents = last_message
                    else:
                        contents = last_message
                    
                # 直接使用 generate_content 方法
                result = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                return self.create_model_response_wrapper(result, model=model)
                
        except Exception as e:
            if hasattr(e, "status_code"):
                raise GeminiError(status_code=e.status_code, message=str(e))
            else:
                raise GeminiError(status_code=500, message=str(e))

    def _try_convert_markdown_image_to_part(self, text: str):
        md_url = self._extract_markdown_image_url(text)
        if md_url:
            try:
                resp = requests.get(md_url)
                img_bytes = resp.content
                mime_type = resp.headers.get('Content-Type', None)
                if not mime_type:
                    try:
                        from PIL import Image as _Img
                        im = _Img.open(BytesIO(img_bytes))
                        fmt = (im.format or 'JPEG').lower()
                        mime_type = f"image/{'jpeg' if fmt == 'jpg' else fmt}"
                    except Exception:
                        mime_type = 'image/jpeg'
                return types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            except Exception as e:
                raise GeminiError(status_code=500, message=f"Error processing markdown image: {str(e)}")
        return None
