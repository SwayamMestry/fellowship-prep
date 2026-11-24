from dotenv import load_dotenv
import anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import asyncio
import json
from contextlib import AsyncExitStack
from typing import List, TypedDict, Dict
import mcp.shared.exceptions as mcp_exceptions

_ = load_dotenv()

class ToolDefinition(TypedDict):
    name : str
    description : str
    input_schema : dict

class ArgDefinition(TypedDict):
    name : str
    description : str
    required : bool

class PromptDefinition(TypedDict):
    name : str
    description : str
    arguments : List[ArgDefinition]

class MCP_Chatbot:
    def __init__(self):
        self.session : List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.tools_to_session: Dict[str,ClientSession] = {}
        self.anthropic = anthropic.Anthropic()
        self.available_tools : List[ToolDefinition] = []
        self.prompts_to_session: Dict[str,ClientSession] = {}
        self.available_prompts : List[PromptDefinition] = []

    async def connect_to_server(self,server_name:str,server_config:dict):
        '''connect to a single MCP server'''
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport

            session = await self.exit_stack.enter_async_context(
                ClientSession(read,write)
            )
            await session.initialize()
            self.session.append(session)
            response = await session.list_tools()
            tools = response.tools

            for tool in tools:
                self.tools_to_session[tool.name] = session
                self.available_tools.append({
                    'name':tool.name,
                    'description':tool.description,
                    'input_schema':tool.inputSchema
                })

            try:
                response = await session.list_prompts()
            except mcp_exceptions.McpError as e:
                print(f"{server_name} does not offer prompts")
            else:
                prompts = response.prompts
                for prompt in prompts:
                    arguments = []
                    self.prompts_to_session[prompt.name] = session
                    if prompt.arguments is not None:
                        for argument in prompt.arguments:
                            arguments.append({
                                'name':argument.name,
                                'description':argument.description,
                                'required':argument.required
                            })
                    self.available_prompts.append({
                        'name':prompt.name,
                        'description':prompt.description,
                        'arguments':arguments
                    })
                print('connected with following prompts:',[prompt.name for prompt in prompts])
            print('connected with following tools:',[tool.name for tool in tools])
        except Exception as e:
            print(f'Failed to connect to server {server_name}: {e}')

    async def connect_to_servers(self):
        '''connect to all configured MCP servers'''
        try:
            with open('server_config.json','r') as file:
                data = json.load(file)
            servers = data.get('mcpServers',{})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name,server_config)

        except Exception as e:
            print(f'Error loading server config: {e}')
            raise

    async def chat_loop(self):
        '''run chat loop'''
        print('Type your query or "quit" to exit')
        while True:
            try:
                query = input('\nQuery: ').strip()
                if query.lower()=='quit':
                    break
                await self.process_query(query)
                print('\n')
            except Exception as e:
                print(f'Error: {str(e)}')
    
    async def process_query(self, query):
        messages = [{'role':'user','content':query}]
        response = self.anthropic.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=2000,
            tools=self.available_tools,
            messages=messages
        )
        
        process = True
        while process:
            tool_result = []
            ai_content = []
            tools_use = False
            for content in response.content:
                if content.type == 'text':
                    ai_content.append(content)
                    print(content.text)
                elif content.type == 'tool_use':
                    tools_use = True
                    ai_content.append(content)
                    tool_name = content.name
                    tool_id = content.id
                    tool_args = content.input

                    print('calling: ',tool_name, tool_args)

                    session = self.tools_to_session[tool_name]

                    result = await session.call_tool(tool_name,arguments=tool_args)
                    tool_result.append(
                        {
                            'type':'tool_result',
                            'tool_use_id':tool_id,
                            'content':result.content
                        }
                    )

            messages.append({'role':'assistant','content':ai_content})
            if tools_use:
                messages.append({'role':'user','content':tool_result})
                response = self.anthropic.messages.create(
                    model='claude-haiku-4-5-20251001',
                    max_tokens=2000,
                    tools=self.available_tools,
                    messages=messages
                )
            else:
                process = False

    async def cleanup(self):
        '''closes all resources'''
        await self.exit_stack.aclose()

async def main():
    chatbot = MCP_Chatbot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == '__main__':
    asyncio.run(main())