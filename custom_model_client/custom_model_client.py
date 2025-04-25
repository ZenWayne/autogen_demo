# use autogen to create a custom model client
import autogen
from autogen import AssistantAgent, ModelClient

import numpy as np
np.float_ = np.float64 #https://github.com/facebook/prophet/issues/2595

import logging
import os
import time
import random
from openai import OpenAI
from autogen.oai.oai_models import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建配置并设置API密钥
api_key = os.getenv("OPENAI_API_KEY")
config_list = [
    {
        "model": "deepseek-chat",
        "api_key": api_key,
        "base_url": "https://api.deepseek.com/v1",
        "model_client_cls": "CustomModelClient",
    }
]

class CustomModelClient(ModelClient):
    def __init__(self, config, model_name, base_url, api_key):
        print(f"CustomModelClient config: {config} model_name: {model_name} base_url: {base_url} api_key: {api_key}")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name
        self._cost_per_output_token = 0.006/1000
        self._cost_per_input_token = 0.002/1000

    def create(self, params):
        """处理API请求并返回符合autogen要求的响应格式"""
        logger.info(f"CustomModelClient initiate request: {params}")
        
        # 初始化计数器和结果容器
        input_tokens, output_tokens = 0, 0
        choices = []
        
        # 在方法本地处理所有流式响应，避免持有对stream对象的引用
        try:
            # 创建流式请求
            stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=params["messages"],
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # 处理流式响应
            message = ChatCompletionMessage(
                role="assistant",
                content=""
            )
            finish_reason = ""
            for chunk in stream:
                # 处理内容部分
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    #print(f"chunk.choices: {chunk.choices}")

                    if delta.content:
                        message.content += delta.content  
                    finish_reason = chunk.choices[0].finish_reason
                        
                # 处理使用量统计
            if chunk.usage:
                output_tokens = chunk.usage.completion_tokens
                input_tokens = chunk.usage.prompt_tokens

            choices.append(
                Choice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason
                )
            )

                
        except Exception as e:
            logger.error(f"流式请求出错: {e}")
            full_message = ChatCompletionMessage(
                role="assistant",
                content=str(e)
            )
            choices.append(
                Choice(
                    index=0,
                    message=full_message,
                    finish_reason="stop"
                )
            )
        # 计算成本
        total_cost = (self._cost_per_output_token * output_tokens) + (self._cost_per_input_token * input_tokens)


        # 创建符合autogen期望的响应对象 - 使用全局函数
        response = ChatCompletion(
            id=f"chatcmpl-{random.randint(1000, 9999)}",
            model=self._model_name,
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            ),
            cost=total_cost
        )
        
        return response

    def message_retrieval(self, response) -> list: #will be assigned to response.message_retrieval_function
        """Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        return [choice.message for choice in response.choices]

    def cost(self, response) -> float:
        return response.cost

    @staticmethod
    def get_usage(response :ChatCompletion):
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "cost": response.cost,
            "model": response.model
        }

logger.info(f"create assistant with config: {config_list}")

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=config_list[0]
)

assistant.register_model_client(CustomModelClient, 
                        model_name=config_list[0]["model"],
                        api_key=config_list[0]["api_key"],
                        base_url=config_list[0]["base_url"]
)

# 创建用户代理
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

# 测试对话
if __name__ == "__main__":
    result = user_proxy.initiate_chat(
        assistant,
        message="hello, please introduce yourself add TERMINATE at the end of your response"
    )
    print(result.chat_history)