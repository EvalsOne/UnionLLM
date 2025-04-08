from .base_provider import BaseProvider
from unionllm.utils import ModelResponse, Message, Choices, Usage, Delta, StreamingChoices
from google import genai
import os, json, time
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import requests

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
            "file_url", "video_url"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        # 处理所有消息
        processed_messages = []
        for msg in messages:
            if msg["role"] == "user":
                processed_messages.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                processed_messages.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg["content"])]
                ))
            elif msg["role"] == "system":
                # 系统消息作为配置项处理
                new_kwargs["system_instruction"] = msg["content"]

        # 处理工具函数
        if "tools" in new_kwargs:
            gemini_tools = []
            for tool in new_kwargs["tools"]:
                gemini_tools.append(tool['function'])

            tools = types.Tool(function_declarations=gemini_tools)
            config = types.GenerateContentConfig(tools=[tools])
        else:
            config = types.GenerateContentConfig(response_modalities=['Text', 'Image'])

        # 如果有system instruction,添加到配置中
        if "system_instruction" in new_kwargs:
            config.system_instruction = new_kwargs["system_instruction"]

        # 处理多模态内容
        last_message = messages[-1]["content"]
        if "image_url" in new_kwargs:
            config.response_modalities = ['Image', 'Text']
            try:
                response = requests.get(new_kwargs["image_url"])
                image = Image.open(BytesIO(response.content))
                processed_messages[-1].parts.append(image)
            except Exception as e:
                print(f"Error processing image URL: {str(e)}")
        
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
                print(f"Error processing file URL: {str(e)}")

        try:
            # 使用 generate_content_stream 方法
            response = self.client.models.generate_content_stream(
                model=model,
                contents=processed_messages,
                config=config
            )
            
            index = 0
            for chunk in response:
                chunk_choices = []
                chunk_delta = Delta()
                if chunk.text:
                    chunk_delta.role = "assistant"
                    chunk_delta.content = chunk.text
                    chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta))
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    # 检查candidates[0]是否有content属性
                    if hasattr(chunk.candidates[0], 'content'):
                        # 检查content是否有parts属性且不为None
                        if hasattr(chunk.candidates[0].content, 'parts') and chunk.candidates[0].content.parts is not None:
                            for part in chunk.candidates[0].content.parts:
                                if part.text is not None:
                                    chunk_delta.role = "assistant"
                                    chunk_delta.content = part.text
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
                                        chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta))
                                    except Exception as e:
                                        print("Error processing image in stream:", str(e))
                                # 处理函数调用
                                elif hasattr(part, 'function_call') and part.function_call:
                                    try:
                                        chunk_delta.role = "assistant"
                                        chunk_delta.tool_calls = [{
                                            "id": f"call_{time.time()}",
                                            "type": "function",
                                            "function": {
                                                "name": part.function_call.name,
                                                "arguments": json.dumps(part.function_call.args)
                                            }
                                        }]
                                        chunk_choices.append(StreamingChoices(index=index, delta=chunk_delta, finish_reason="tool_calls"))
                                    except Exception as e:
                                        print("Error processing function call in stream:", str(e))
                        else:
                            print("Candidate content has no parts attribute or parts is None")
                    else:
                        print("Candidate has no content attribute")
                
                # 只有当有内容时才生成响应
                if chunk_choices:
                    chunk_response = ModelResponse(
                        id=f"gemini-{time.time()}",
                        choices=chunk_choices,
                        created=int(time.time()),
                        model=model,
                        stream=True
                    )
                    index += 1
                    yield chunk_response
        except Exception as e:
            # raise e
            print(f"Error in stream processing: {str(e)}")
            # raise GeminiError(status_code=500, message=f"Stream processing error: {str(e)}")

    def create_model_response_wrapper(self, response, model):
        choices = []
        content = ""
        tool_calls = []
                
        # 处理所有输出部分(文本、图片和函数调用)
        for part in response.candidates[0].content.parts:            
            if part.text is not None:
                content += part.text
            elif part.inline_data is not None:
                try:
                    # 将图片转换为base64字符串
                    image = Image.open(BytesIO(part.inline_data.data))                    
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # 添加markdown格式的图片
                    content += f"\n![generated_image](data:image/png;base64,{img_str})\n"
                except Exception as e:
                    print("Error processing image:", str(e))  # 打印任何图片处理错误
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
                except Exception as e:
                    print("Error processing function call:", str(e))  # 打印任何函数调用处理错误
        
        message = Message(
            content=content,
            role="assistant"
        )
        
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
        
        response = ModelResponse(
            id=f"gemini-{time.time()}",
            choices=choices,
            created=int(time.time()),
            model=model
        )
        return response

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

            if "multimodal" in kwargs:
                config = types.GenerateContentConfig(response_modalities = ['Text', 'Image'])
            else:
                config = types.GenerateContentConfig(response_modalities = ['Text'])

            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                # 获取最后一条消息内容
                last_message = messages[-1]["content"]
                
                # 如果有system instruction,添加到配置中
                if "system_instruction" in new_kwargs:
                    config.system_instruction = new_kwargs["system_instruction"]
                
                # 处理工具函数
                if "tools" in new_kwargs:
                    config.tool_config = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            allowed_function_names=[tool["function"]["name"] for tool in new_kwargs["tools"]],
                            mode="AUTO" if new_kwargs.get("tool_choice") == "auto" else "ANY"
                        )
                    )
                    config.tools = [
                        types.Tool(
                            function_declarations=[
                                types.FunctionDeclaration(
                                    name=tool["function"]["name"],
                                    description=tool["function"].get("description", ""),
                                    parameters=tool["function"].get("parameters", {})
                                ) for tool in new_kwargs["tools"]
                            ]
                        )
                    ]
                
                # 处理图片URL
                contents = []

                if isinstance(last_message, list):
                    # 处理OpenAI格式的多模态消息
                    text_parts = []
                    image_parts = []
                    video_url = None
                    
                    for content in last_message:
                        if content.get("type") == "text":
                            text_parts.append(content.get("text", ""))
                        elif content.get("type") == "image_url":
                            try:
                                image_url = content.get("image_url", {}).get("url", "")
                                if image_url:
                                    # 下载图片
                                    response = requests.get(image_url)
                                    image = Image.open(BytesIO(response.content))
                                    image_parts.append(image)

                            except Exception as e:
                                print(f"Error downloading image: {str(e)}")
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
                                    print(f"Error processing file data: {str(e)}")
                                    contents = [last_message]
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
                            raise e
                            print(f"Error processing video URL: {str(e)}")
                            # 如果视频处理失败，尝试回退到普通文本处理
                            if text_parts:
                                contents.append(" ".join(text_parts))
                    else:
                        # 添加文本部分
                        if text_parts:
                            contents.append(" ".join(text_parts))
                        
                        # 添加图片部分
                        contents.extend(image_parts)
                        
                        # 设置响应模态
                        if image_parts:
                            config.response_modalities = ['Text', 'Image']
                elif "image_url" in new_kwargs:
                    # 处理image_url参数
                    try:
                        image_url = new_kwargs["image_url"]
                        # 添加文本部分
                        contents.append(last_message)
                        
                        # 下载图片
                        response = requests.get(image_url)
                        image = Image.open(BytesIO(response.content))
                        contents.append(image)
                        
                        # 设置响应模态
                        config.response_modalities = ['Text', 'Image']
                    except Exception as e:
                        print(f"Error downloading image: {str(e)}")
                        contents = last_message
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
                        print(f"Error processing video URL: {str(e)}")
                        contents = last_message
                else:
                    # 普通文本消息
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
            