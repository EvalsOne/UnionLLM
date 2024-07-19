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
        
        self.base_url = "https://api.coze.com/open_api/v2"
        self.endpoint_url = self.base_url + "/chat"

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
        payload = {"query": query,"user": user_id,"bot_id": self.bot_id,"chat_history":history, "stream": True}
        if self.conversation_id:
            payload['conversation_id'] = self.conversation_id
        payload = json.dumps(payload)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint_url, headers=headers, data=payload)

        for line in response.iter_lines():
            if line:
                if line.startswith(b"data:"):
                    new_line = line.decode("utf-8").replace("data:", "")
                    if new_line == "[DONE]":
                        continue

                    data = json.loads(new_line)
                    chunk_context = []
                    chunk_choices = []
                    index = 0
                    if "message" in data:
                        message = data["message"]
                        if message['role'] == 'assistant':
                            if message['type'] == 'knowledge':
                                # 解析知识型内容
                                content = message['content']
                                # 假设knowledge类型内容是由"---\nrecall slice X:\n"分隔的
                                slices = content.split('---\n')
                                index = 0
                                for slice in slices:
                                    if slice.strip().startswith('recall slice'):
                                        # 去掉前缀找到JSON部分
                                        json_part = slice.strip().split('\n', 1)[-1].strip()
                                        # 如果不是以\"}结尾，则强制添加
                                        if not json_part.endswith('\"}'):
                                            json_part += '\"}'
                                        try:
                                            # 尝试解析JSON数据
                                            recall_data = json.loads(json_part)
                                            chunk_context.append({
                                                "id": index,
                                                "content": str(recall_data)    
                                            })
                                            index += 1
                                        except json.JSONDecodeError:
                                            raise CozeAIError(status_code=500, message=f"Error decoding JSON from slice: {json_part}")

                            elif message['type'] == 'answer':
                                chunk_choices = []
                                chunk_delta = Delta()
                                if "role" in message:
                                    chunk_delta.role = message["role"]
                                if "content" in message:
                                    chunk_delta.content = message["content"]
                                chunk_choices.append(StreamingChoices(index=str(index), delta=chunk_delta))
                        
                        chunk_usage = Usage()
                        # if data["usage"]:
                        #     if "prompt_tokens" in data["usage"]:
                        #         chunk_usage.prompt_tokens = data["usage"]["prompt_tokens"]
                        #     if "completion_tokens" in data["usage"]:
                        #         chunk_usage.completion_tokens = data["usage"]["completion_tokens"]
                        #     if "total_tokens" in data["usage"]:
                        #         chunk_usage.total_tokens = data["usage"]["total_tokens"]                        

                        chunk_response = ModelResponse(
                            id=self.conversation_id,
                            conversation_id=self.conversation_id,
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
                result = requests.post(self.endpoint_url, headers=headers, data=payload)
                # print("CozeAI response")
                # print(result.json())
                return self.create_model_response_wrapper(result, model=model)
        except Exception as e:
            if hasattr(e, "status_code"):
                raise CozeAIError(status_code=e.status_code, message=str(e))
            else:
                raise CozeAIError(status_code=500, message=str(e))