"""
应该使用 create_tool_calling_agent 而非 create_react_agent，
- 前者工具调用代理更适合企业场景，它明确区分工具使用和自然语言处理，
- 后者 React代理更适合研究场景，在企业场景中可能导致不可预测的行为

即使要使用 React模式，也要用 from langchain.agents import create_react_agent，以保证实时更新
"""
from typing import List, Dict, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
import gradio as gr
import logging
from mcp_config import Config
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential


# 初始化LLM，本地部署的
llm = ChatopenAI(
	temperature=0,
	model="qwen3-8b",
	api_key="EMPTY",
	api_base="http://localhost:6006/v1"
	# 启用深度思考模式
	extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 系统提示模板
SYSTEM_PROMPT = """你是一个智能助手，尽可能的调用工具回答用户的问题。
请遵循以下规则:
1. 确保回答准确、专业
2. 对于不确定的信息明确说明
3. 遵守企业数据安全政策
4. 避免提供敏感信息"""

prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name='chat_history', optional=True),
    ('human', '{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
])


"""
asynccontextmanager 实际是通过 MultiServerMCPClient 的异步上下文管理器接口隐式调用的，企业级代码中应避免直接操作底层异步原语  
asyncio 作为运行时基础依赖，应由框架层（如 `langchain_mcp_adapters`）统一管理，而非业务代码显式引入。
直接使用 `asyncio` 可能导致线程安全问题，而框架提供的客户端（如 `MultiServerMCPClient`）已实现线程安全的异步封装

仅在以下场景保留直接导入（其他情况应优先使用框架提供的异步抽象）：
1. **编写基础设施组件**（如自定义连接池）  
2. **性能关键型代码**需精细控制事件循环策略  
3. **兼容旧版Python**（<3.7需`@asyncio.coroutine`）  

# 在基础设施层集中管理（如 async_utils.py），若确实需要自定义异步逻辑，应采用以下模式：
from contextlib import asynccontextmanager
import asyncio

# 异步生命周期管理（app），异步上下文管理器在进入和退出上下文时可以执行异步操作。
@asynccontextmanager
async def managed_client(config: dict):
    """企业级封装的异步客户端"""
    async with MultiServerMCPClient(config) as client:
        try:
            yield client
        except asyncio.TimeoutError:
            logger.error("MCP client timeout")
            raise ServiceUnavailableError()
"""

# 在 agent 中连接 MCP_SERVER 时，必须是在异步环境下建立连接的
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def execute_graph(chat_bot: List[Dict]) -> List[Dict]:
    """执行工作流的函数，增加重试机制和错误处理"""
    try:
        user_input = chat_bot[-1]['content']
        if not user_input.strip():
            raise ValueError("Empty user input")
            
        inputs = {"input": user_input}
        
		"""
		async def 的作用是声明函数为协程函数，使其内部可以包含 await、async with 等异步操作。但函数本身的定义不会自动使其内部代码异步执行。
		- 若内部代码需要异步执行（如数据库连接建立、资源释放、网络会话等），则必须用 async with 来让异步上下文管理器管理其生命周期（普通with会阻塞事件循环）。
		- 若内部代码操作对象是同步的（如本地计算、非异步I/O），则无需加 async。
		"""
		# MultiServerMCPClient 可以接收多个 server 配置，即可以连接多个 MCP 服务器
		# with 自动释放资源
        async with MultiServerMCPClient(Config.MCP_SERVER_CONFIG) as client:
            tools = client.get_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            # agent = create_react_agent(llm, client.get_tools())
            # 使用工具调用代理而非React代理，更适合企业场景
			agent = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(
                agent=agent, 
                tools=tools,
                handle_parsing_errors=True,
                max_iterations=10  # 限制迭代次数防止无限循环
            )
            
            response = await executor.ainvoke(input=inputs)
            result = response["output"]
            
            # 记录交互历史
            logger.info(f"User: {user_input}\nAssistant: {result}")
            
            chat_bot.append({'role': 'assistant', 'content': result})
            return chat_bot
            
    except Exception as e:
        logger.error(f"Error in execute_graph: {str(e)}", exc_info=True)
        chat_bot.append({
            'role': 'assistant', 
            'content': "抱歉，处理您的请求时遇到问题。我们的技术团队已收到通知。"
        })
        return chat_bot
		
def do_graph(user_input: str, chat_bot: List[Dict]) -> tuple:
    """输入处理函数，增加输入验证"""
    if user_input and user_input.strip():
        # 简单的内容过滤
        if any(word in user_input.lower() for word in ["密码", "敏感", "机密"]):
            chat_bot.append({
                'role': 'assistant',
                'content': "抱歉，我无法处理包含敏感信息的请求。"
            })
            return '', chat_bot
            
        chat_bot.append({'role': 'user', 'content': user_input.strip()})
    return '', chat_bot


with gr.Blocks(title='调用MCP服务的Agent项目', css=Config.CSS) as instance:
    gr.Label('调用MCP服务的Agent项目', container=False)

    chatbot = gr.Chatbot(type='messages', height=450, label='AI客服')  # 聊天记录组件

    input_textbox = gr.Textbox(label='请输入你的问题📝', value='')  # 输入框组件

    input_textbox.submit(do_graph, [input_textbox, chatbot], [input_textbox, chatbot])
	.then(execute_graph, chatbot, chatbot)


if __name__ == '__main__':
    # 生产环境启动
    instance.launch(**{
        "auth": Config.GRADIO_AUTH,
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "debug": False
    })
	
	
	