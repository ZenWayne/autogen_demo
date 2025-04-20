from turtle import mode
import autogen
from autogen import AssistantAgent, ModelClient

from autogen import oai
import numpy as np
from urllib3 import response
np.float_ = np.float64 #https://github.com/facebook/prophet/issues/2595

from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb
from llama_index.core.schema import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceWindowNodeParser

from chromadb.api.types import Documents, Embeddings
from chromadb.utils import embedding_functions
import logging
import aiohttp
import asyncio
import os
import time
import random
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import Stream, OpenAI
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel
from typing import Optional, Any
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartTextParam
from typing import Union

from autogen.oai.oai_models import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage


logging.basicConfig(
    level=logging.INFO, # Or logging.DEBUG to see DEBUG level messages too
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

markdown_node_parser = MarkdownNodeParser.from_defaults()
window_node_parser = SentenceWindowNodeParser()

def custom_text_split_function(text: str) -> list[str]:
    #logger.info(f"custom_text_split_function: {text}")
    docs=[Document(text=text)]
    nodes = markdown_node_parser.get_nodes_from_documents(documents=docs, show_progress=True)
    return [node.text for node in nodes]

def split_function_by_sentence_window(text: str) -> list[str]:
    docs=[Document(text=text)]
    nodes = markdown_node_parser.get_nodes_from_documents(documents=docs, show_progress=True)
    #nodes = window_node_parser.get_nodes_from_documents(documents=nodes, show_progress=True)
    #return [node.metadata[window_node_parser.window_metadata_key] for node in nodes]
    return [node.text for node in nodes]

class RemoteEmbeddingFunction(embedding_functions.EmbeddingFunction[Documents]):
    def __init__(
        self,
        url: str,
        api_key: str,
        model: str
    ):
        self.base_url = url
        self.api_key = api_key
        self.model = model
    
    async def _async_embed_text(self, text :str) -> list[float]:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "input": text,
                "encoding_format": "float"
            }
            #logger.info(f"send embedding request: {text}")
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                obj = await response.json()
                #logger.info(f"embedding result: success embedding {obj['data'][0]['embedding']}")
                return obj['data'][0]['embedding']

    async def async_embed_text(self, text :str, max_retries: int = 3) -> list[float]:
        for i in range(max_retries):
            try:
                await asyncio.sleep(random.randint(0, 3))
                return await self._async_embed_text(text)
            except Exception as e:
                if isinstance(e, aiohttp.ClientResponseError):
                    headers = e.headers
                    if e.status == 429:
                        if 'Retry-After' in e.headers:
                            retry_delay = int(e.headers['Retry-After'])
                        else:
                            retry_delay = retry_delay * 2 + random.randint(1, 3)
                    
                    logger.error(f"错误响应，状态码: {e.status}, 响应头: {headers} retry_delay: {retry_delay}")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"嵌入文本时出错: {e}")
                return []


    async def _gather_tasks(self, texts: list[str]):
        """异步任务聚合执行"""
        tasks = [self.async_embed_text(text) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def __call__(self, input: Documents) -> Embeddings:
        logger.info(f"embed input: {len(input)}")
        # replace newlines, which can negatively affect performance.
        #input = [t.replace("\n", " ") for t in input]

        # Create new event loop for synchronous call
        loop = asyncio.get_event_loop()
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)        
        
        # Batch process embeddings using async gather
        results = loop.run_until_complete(self._gather_tasks(input))
        
        # Filter out failed embeddings and log errors
        embeddings = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Error embedding text {i}")
            elif res is not None:
                embeddings.append(res)
        return embeddings

openai_ef = None
embedding_service_api_key = os.getenv("DASHSCOPE_API_KEY")

if embedding_service_api_key:
    openai_ef = RemoteEmbeddingFunction(
            api_key=embedding_service_api_key,
            model="text-embedding-v3",
            url="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    )
    # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    #             api_key=embedding_service_api_key,
    #             api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #             api_type="azure",
    #             api_version="YOUR_API_VERSION",
    #             model_name="text-embedding-3-small"
    #         )
else: 
    openai_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"
    )

api_key = os.getenv("DASHSCOPE_API_KEY")
# config_list = [
#     {
#         "model": "gemini-2.0-flash-lite",
#         "api_type": "google",
#         "api_key": os.getenv("GOOGLE_API_KEY"),
#     }
# ]

config_list = [
    {
        "model": "qwq-32b",
        "api_key": api_key,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_client_cls": "CustomModelClient",
        # "params": {
        #     "stream": True
        # }
    }
    # ,{
    #     "model": "qwq-32b",
    #     "api_key": api_key,
    #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # }
]

# 添加一个全局函数用于消息提取，确保可以序列化
def extract_messages_from_choices(response : ChatCompletion)->list[ChatCompletionMessage]:
    """从choices中提取消息内容"""
    print(f"extract_messages_from_choices response: {response}")
    return [response]

class CustomModelClient(ModelClient):
    def __init__(self, config, model_name, base_url, api_key):
        print(f"CustomModelClient config: {config} model_name: {model_name} base_url: {base_url} api_key: {api_key}")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name
        self._cost_per_output_token = 0.006/1000
        self._cost_per_input_token = 0.002/1000

    def create(self, params):
        """处理API请求并返回符合autogen要求的响应格式"""
        logger.info(f"CustomModelClient创建请求: {params}")
        
        # 初始化计数器和结果容器
        input_tokens, output_tokens = 0, 0
        response_text = ""
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
            for chunk in stream:
                # 处理内容部分
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        response_text += delta.content
                        
                # 处理使用量统计
                if chunk.usage:
                    if hasattr(chunk.usage, "completion_tokens"):
                        output_tokens = chunk.usage.completion_tokens
                    if hasattr(chunk.usage, "prompt_tokens"):
                        input_tokens = chunk.usage.prompt_tokens
            
            # 确保关闭流式响应
            del stream
            
            # 构建最终响应
            if response_text:
                # 创建完整的消息对象
                full_message = ChatCompletionMessage(
                    role="assistant",
                    content=response_text
                )
                
                # 添加到choices中
                choices.append(
                    Choice(
                        index=0,
                        message=full_message,
                        finish_reason="stop"
                    )
                )
                
        except Exception as e:
            logger.error(f"流式请求出错: {e}")
            # 在出错时仍然尝试提供部分结果
            if response_text:
                full_message = ChatCompletionMessage(
                    role="assistant",
                    content=response_text
                )
                choices.append(
                    Choice(
                        index=0,
                        message=full_message,
                        finish_reason="error"
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

        #print(f"response: {response.message_retrieval_function(response)}")
        
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

chroma_client = chromadb.PersistentClient(path="./chromadb")

collection_name = "ag2-docs"
try:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception as e:  # 捕获更通用的异常
    print(f"Collection '{collection_name}' not found, creating new one. Error: {e}")
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created successfully.")

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "collection_name": collection_name,  # 添加集合名称配置
        "docs_path": [
            "./rag/docs/1810.04805.md",
        ],
        "custom_text_types": ["mdx"],
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "embedding_function": openai_ef,
        "db_config": {
            "path": "./chromadb",
        },
        "get_or_create": True,
        "custom_text_split_function": custom_text_split_function,
    },
    code_execution_config=False,
)

code_problem = "What's BERT"
chat_result = ragproxyagent.initiate_chat(
    assistant, message=ragproxyagent.message_generator, problem=code_problem
)
print(chat_result.chat_history)