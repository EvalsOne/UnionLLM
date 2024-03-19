from .base_provider import BaseProvider
from cnlitellm.models import ResponseModel
from zhipuai import ZhipuAI
import tenacity, logging, json

class ZhipuAIProvider(BaseProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    @tenacity.retry(stop=tenacity.stop_after_attempt(1), reraise=True)
    def completion(self, model: str, messages: list, **kwargs) -> ResponseModel:
        # Check if api_key is set, if not, try to get it from kwargs
        if 'api_key' in kwargs:
            self.api_key = kwargs.get('api_key')
            kwargs.pop('api_key')

        self.client = ZhipuAI(api_key=self.api_key)

        # Check if stream is set, if not, try to get it from kwargs
        stream = kwargs.get('stream', False)

        if stream:
            def generate_stream():
                for chunk in self.client.chat.completions.create(model=model, messages=messages, **kwargs):
                    # Convert the chunk to the format expected by the outer function
                    delta = chunk.choices[0].delta
                    line = {
                        "choices": [{
                            "delta": {
                                "role": delta.role,
                                "content": delta.content
                            }
                        }]
                    }
                    if chunk.usage is not None:                    
                        line['usage'] = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        }
                    yield "data: " + json.dumps(line) + "\n\n"

                # yield "data: [DONE]"
            return generate_stream()

        else:
            result = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
            # convert result to dict

            print(f"ZhipuAIProvider.completion: result={result}")
            logging.info(f"ZhipuAIProvider.completion: result={result}")

            return result
            # result = result.dict()
            # usage = result['usage']

            # response =  ResponseModel(
            #     prompt_tokens=usage['prompt_tokens'],
            #     completion_tokens=usage['completion_tokens'],
            #     total_tokens=usage['total_tokens'],
            #     total_attempts=1,  # Since retries are handled by tenacity
            #     raw_response=result,
            #     usage=usage
            # )
            # return response.to_dict()
        
    
