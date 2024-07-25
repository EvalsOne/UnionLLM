from .base_provider import BaseProvider
from unionllm.utils import ModelResponse, Message, Choices, Usage, Context, generate_unique_uid, Delta, StreamingChoices
from openai import OpenAI
import logging, json, time, requests, os


class CozeAIError(Exception):
    def __init__(
        self,
        status_code,
        message,
    ):
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)


class CozeAIProvider(BaseProvider):
    def __init__(self, **model_kwargs):
        # Get COZE_API_KEY from environment variables
        _env_api_key = os.environ.get("COZE_API_KEY")
        _env_bot_id = os.environ.get("COZE_BOT_ID")
        self.api_key = model_kwargs.get("api_key") if model_kwargs.get("api_key") else _env_api_key
        self.bot_id = model_kwargs.get("bot_id") if model_kwargs.get("bot_id") else _env_bot_id
        self.conversation_id = model_kwargs.get("conversation_id", None)
        if not self.api_key:
            raise CozeAIError(
                status_code=422, message=f"Missing API key"
            )
        if not self.bot_id:
            raise CozeAIError(
                status_code=422, message=f"Missing Bot ID"
            )
        
        self.base_url = "https://api.coze.com/v3"
        self.base_url_v2 = "https://api.coze.com/open_api/v2"
        self.endpoint_url = self.base_url + "/chat"
        self.endpoint_url_v2 = self.base_url_v2 + "/chat"
        if self.conversation_id:
            self.endpoint_url = self.endpoint_url + "?conversation_id=" + self.conversation_id

    def pre_processing(self, **kwargs):
        # 处理参数兼容性问题，不支持的参数全部舍弃
        supported_params = [
            "model", "messages", "max_tokens", "temperature", "top_p", "n",
            "logprobs", "stream", "stop", "presence_penalty", "frequency_penalty",
            "best_of", "logit_bias"
        ]
        for key in list(kwargs.keys()):
            if key not in supported_params:
                kwargs.pop(key)
        return kwargs

    def to_formatted_prompt(self, messages):
        last_message = messages[-1]
        # if last message role is not user, return error
        if last_message["role"] != "user":
            raise CozeAIError(
                status_code=422, message=f"Last message role should be user"
            )
        query = last_message["content"]
        # 从messages去掉最后一条消息
        history = messages[:-1]
        history_messages = []
        for message in history:
            if message["role"] == "assistant":
                # 在message中追加content_type字段和content字段
                message["type"] = "answer"
                message["content_type"] = "text"
                history_messages.append(message)
            elif message["role"] == "user":
                message["content_type"] = "text"
                history_messages.append(message)
        return history_messages, query

    def post_stream_processing_wrapper(self, model, messages, **new_kwargs):
        # 预处理消息内容并提取用户问题
        history, query = self.to_formatted_prompt(messages)

        # 接收user_id作为用户唯一标识传入参数
        if 'user_id' not in new_kwargs:
            user_id = generate_unique_uid()
        else:
            user_id = new_kwargs['user_id']
        payload = {"user_id": user_id,"bot_id": self.bot_id,"additional_messages":messages, "stream": True, "auto_save_history": True}
        if self.conversation_id:
            payload['conversation_id'] = self.conversation_id
        payload = json.dumps(payload)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint_url, headers=headers, data=payload)
        event_type = None
        index = 0
        for line in response.iter_lines():
            if line:
                if line.startswith(b"event:"):
                    event_type = line.decode("utf-8").replace("event:", "").strip()
                    continue
                if line.startswith(b"data:"):
                    new_line = line.decode("utf-8").replace("data:", "")
                    if new_line == "[DONE]":
                        break
                    data = json.loads(new_line)
                    chunk_context = []
                    chunk_choices = []
                    chunk_usage = Usage()

                    if isinstance(data, dict):
                        conversation_id = data['conversation_id']
                        chat_id = data['id']
    
                    if event_type=="conversation.message.delta":
                        if "type" in data:
                            message_type = data["type"]
                            if message_type == "answer":
                                chunk_choices = []
                                chunk_delta = Delta()
                                if "role" in data:
                                    chunk_delta.role = data["role"]
                                if "content" in data:
                                    chunk_delta.content = data["content"]
                                chunk_choices.append(StreamingChoices(index=str(index), delta=chunk_delta))
                    elif event_type=="conversation.message.completed":
                        if "type" in data:
                            message_type = data["type"]
                            if message_type == "answer":
                                chunk_choices = []
                                chunk_delta = Delta()
                                if "role" in data:
                                    chunk_delta.role = data["role"]
                                if "content" in data:
                                    if "content_type" in data and data["content_type"]=="image":
                                        contents = json.loads(data["content"])
                                        for content in contents:                                 
                                            image_url = content['image_ori']['url']
                                            image_markdown = f"![image]({image_url})"
                                            chunk_delta.content = image_markdown
                                            chunk_choices.append(StreamingChoices(index=str(index), delta=chunk_delta))                                                                                        
                    elif event_type=="conversation.chat.completed":
                        if "usage" in data and data["usage"]:
                            if "input_count" in data["usage"]:
                                chunk_usage.prompt_tokens = data["usage"]["input_count"]
                            if "output_count" in data["usage"]:
                                chunk_usage.completion_tokens = data["usage"]["output_count"]
                            if "token_count" in data["usage"]:
                                chunk_usage.total_tokens = data["usage"]["token_count"]   
                        
                    chunk_response = ModelResponse(
                        id=chat_id,
                        conversation_id=conversation_id,
                        choices=chunk_choices,
                        context=chunk_context,
                        created=int(time.time()),
                        model=model,
                        usage=chunk_usage if chunk_usage else None,
                        stream=True
                    )
                    index += 1
                    yield chunk_response

    def create_model_response_wrapper(self, result, model):
        choices = []
        context = []

        result_dict = result.json()
        messages = result_dict.get('messages', [])
        if not messages:
            code = result_dict.get('code', 500)
            message = result_dict.get('msg', 'Internal Server Error')
            raise CozeAIError(status_code=code, message=message)
        results = []
        cotext_pre = []
        
        for message in messages:
            # 根据消息类型处理数据
            if message['role'] == 'assistant' and message['type'] == 'verbose':
                # 解析知识型内容
                content = json.loads(message['content'])
                if "verbose_type" in content and content["verbose_type"] == "knowledge":
                    chunks = content["chunks"]
                    for chunk in chunks:
                        cotext_pre.append({"content":chunk["slice"],"score":chunk["score"]})

            elif message['role'] == 'assistant' and message['type'] == 'answer':
                # 直接添加回复内容
                results.append(message['content'])

        message = Message(
            content=results[0],
            role="assistant"
        )
        choices.append(
            Choices(
                message=message,
                index=0,
                finish_reason="stop",
            )
        )

        result_usage = result_dict.get('usage', {})
        usage = Usage(
            prompt_tokens=result_usage.get('prompt_tokens', 0),
            completion_tokens=result_usage.get('completion_tokens', 0),
            total_tokens=result_usage.get('completion_tokens', 0),
        )

        # Assume response_dict contains a field 'responseData' with quoteList to extract context
        if cotext_pre:
            i = 0
            for context_item in cotext_pre:
                i += 1
                context.append(Context(id=i, content=context_item['content'], score=context_item['score']))

        response = ModelResponse(
            id=self.conversation_id,
            conversation_id=self.conversation_id,
            choices=choices,
            context=context,
            created=int(time.time()),
            model=self.bot_id,
            usage=usage,
        )
        return response

    def completion(self, model: str, messages: list, **kwargs):
        try:
            if model is None or messages is None:
                raise CozeAIError(
                    status_code=422, message="Missing model or messages"
                )

            message_check_result = self.check_prompt("coze", model, messages)            
            if message_check_result['pass_check']:
                messages = message_check_result['messages']
            else:
                raise CozeAIError(
                    status_code=422, message=message_check_result['reason']
                )
                
            for message in messages:
                if 'content_type' not in message:
                    message['content_type'] = 'text'
            new_kwargs = self.pre_processing(**kwargs)
            stream = kwargs.get("stream", False)

            if stream:
                return self.post_stream_processing_wrapper(model, messages, **new_kwargs)
            else:
                history, query = self.to_formatted_prompt(messages)

                # 接受user_id作为用户唯一身份标识传入参数
                if 'user_id' not in kwargs:
                    user_id = generate_unique_uid()
                else:
                    user_id = kwargs['user_id']
                payload = {"query": query,"user": user_id,"bot_id": self.bot_id,"chat_history":history,"steam":False}
                if self.conversation_id:
                    payload['conversation_id'] = self.conversation_id
                payload = json.dumps(payload)

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                result = requests.post(self.endpoint_url_v2, headers=headers, data=payload)
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise CozeAIError(status_code=e.status_code, message=str(e))
            else:
                raise CozeAIError(status_code=500, message=str(e))