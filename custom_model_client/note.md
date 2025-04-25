## How to customize the model client for autogen
when using autogen, under some circumstances, we may need to customize the model client to fit our needs. For example, we may need to use stream mode of the model to get the response from the model with autogen, However, the default model client in autogen does not support stream mode. Or we may need to caculate the cost of the response from the model with autogen. And this article will provide a way to customize the model client for autogen.

## Customize the model client
first we inherit the autogen.ModelClient class and override the following methods:
- create: handle the API request and return the response that autogen expects
- message_retrieval: retrieve the message from the response, will be assigned to response.message_retrieval_function(in oai.client.OpenAIWrapper.create)
- cost: calculate the cost of the response
- get_usage: get the usage of the response

we use openai-compatible model for this example, so we need to import the openai and autogen.


```python
from openai import OpenAI
from autogen import ModelClient
class CustomModelClient(ModelClient):
    def __init__(self, config, model_name, base_url, api_key):
        print(f"CustomModelClient config: {config} model_name: {model_name} base_url: {base_url} api_key: {api_key}")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name
        self._cost_per_output_token = 0.006/1000 #cost per-thousand output token
        self._cost_per_input_token = 0.002/1000 #cost per-thousand input token

    def create(self, params):
        """handle the API request and return the response that autogen expects"""
        print(f"CustomModelClient initiate request: {params}")
        
        # initialize the counters and the result container
        input_tokens, output_tokens = 0, 0
        choices = []
        
        # handle the stream response locally to avoid holding the reference to the stream object
        try:
            # create the stream request
            stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=params["messages"],
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # handle the stream response
            message = ChatCompletionMessage(
                role="assistant",
                content=""
            )
            finish_reason = ""
            # concatenate the content of choices
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if delta.content:
                        message.content += delta.content  
                    finish_reason = chunk.choices[0].finish_reason
                        
            # handle the usage statistics
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
            print(f"stream request error: {e}")
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
        # calculate the cost
        total_cost = (self._cost_per_output_token * output_tokens) + (self._cost_per_input_token * input_tokens)


        # create the response object that autogen expects
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

    def message_retrieval(self, response) -> list: 
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
```

## Use the custom model client
first we need to config the model, so we get the api key for the model, you can get it from the model provider. 
Then we need to set model_client_cls to the CustomModelClient class.
```python
api_key = os.getenv("OPENAI_API_KEY") #your api key for the model, you can get it from the model provider
config_list = [
    {
        "model": "deepseek-chat", #your model name
        "api_key": api_key,
        "base_url": "https://api.deepseek.com/v1",
        "model_client_cls": "CustomModelClient",
    }
]
```

Then we create the assistant agent and register the custom model client.
```python
#create the assistant agent
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=config_list[0]
)
#register the custom model client
assistant.register_model_client(CustomModelClient, 
                        model_name=config_list[0]["model"],
                        api_key=config_list[0]["api_key"],
                        base_url=config_list[0]["base_url"]
)
```

Then we create the user proxy agent for human input.
```python
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

#test the conversation you should tell the model to add TERMINATE at the end of your response, otherwise it will not stop
if __name__ == "__main__":
    result = user_proxy.initiate_chat(
        assistant,
        message="hello, please introduce yourself add TERMINATE at the end of your response"
    )
    print(result.chat_history)
```

## Result
```
assistant (to user_proxy):

Hello! I'm an AI assistant here to help you with a wide range of tasks, from answering questions and providing information to assisting with creative writing, problem-solving, and more. I can also help with coding, learning new topics, and generating ideas. Let me know how I can assist you today!

TERMINATE
```

