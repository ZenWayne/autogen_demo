#package you need to install:
#pip install autogen-ext[mcp]
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from autogen_agentchat.ui import Console

import asyncio
from asyncio import coroutines
from autogen_core import CancellationToken
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily
from autogen_ext.tools.mcp import StdioMcpToolAdapter, StdioServerParams
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
import logging

from autogen_agentchat.teams import RoundRobinGroupChat
# import nest_asyncio
# nest_asyncio.apply()

# logging.basicConfig(level=logging.DEBUG)
# logging.Logger("asyncio").setLevel(logging.DEBUG)
logging.Logger(EVENT_LOGGER_NAME).setLevel(logging.DEBUG)


async def main() -> None:
    src_project_root = "/c/fs_test"
    project_root = "/app/project/"
    
    server_params = StdioServerParams(
            command = "docker",
            args = [
                "run",
                "-i",
                "--rm",
                "--mount", f"type=bind,src={src_project_root},dst={project_root}",
                "mcp/filesystem",
                project_root
                ],
            env = None
    )

    # Get the tools from the server
    write_file = await StdioMcpToolAdapter.from_server_params(server_params, "write_file")
    list_directory = await StdioMcpToolAdapter.from_server_params(server_params, "list_directory")
    create_directory = await StdioMcpToolAdapter.from_server_params(server_params, "create_directory")

    model_client = OpenAIChatCompletionClient(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("OPENAI_API_KEY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.R1,
            "structured_output": True,
        }
    )
    model_client.component_label="deepseek-r1 - Deepseek Official"
    print(f"model_config: {model_client.dump_component().model_dump_json()}")
    tailing_message = "when finished, respond with 'TERMINATE'"
    system_message = \
f"""You are a helpful file system assistant.
add, delete, modify, and list files in the file system.{tailing_message}
"""
    agent = AssistantAgent(
        name="file_system_assistant",
        model_client=model_client,
        tools=[write_file, list_directory, create_directory],
        system_message=system_message
    )

    max_msg_termination = MaxMessageTermination(max_messages=10)
    text_termination = TextMentionTermination("TERMINATE")
    combined_termination = max_msg_termination | text_termination

    round_robin_team = RoundRobinGroupChat(
        participants=[agent],
        termination_condition=combined_termination
        )
    round_robin_team.component_label = "file_system_group_chat"

    print(f"AssistantAgent config: {round_robin_team.dump_component().model_dump_json()}")

    await Console(
        round_robin_team.run_stream(task="list all files under the /app/project directory", cancellation_token=CancellationToken())
    )

async def simple_user_agent():
    agent = UserProxyAgent("user_proxy")
    response = await asyncio.create_task(
        agent.on_messages(
            [TextMessage(content="What is your name? ", source="user")],
            cancellation_token=CancellationToken(),
        )
    )
    assert isinstance(response.chat_message, TextMessage)
    print(f"Your name is {response.chat_message.content}")

if __name__ == "__main__":
    ## should not use asyncio.run() cos it will close loop which will be close after main function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

