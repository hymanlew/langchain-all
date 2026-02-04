"""
在生产环境中，使用 LangGraph 时，更推荐通过显式地自定义工作流（包括状态（State）管理、节点（Nodes）定义和条件边（Conditional Edges））
来构建代理（Agent），而非依赖于单一的预置创建函数（如 create_react_agent）

应该使用 from langchain.agents import create_tool_calling_agent 而非 create_react_agent（已经过时）
- 前者工具调用代理更适合企业场景，它明确区分工具使用和自然语言处理，适用于快速创建标准工具调用节点，但仍需将其放入自定义的图工作流中，并配置条件边来实现循环。

**LangGraph Prebuilt ReAct Agent (create_react_agent)**
https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.chat_agent_executor.create_react_agent
"""
import gradio as gr
import logging
from typing import List, Dict, Optional
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mcp_config import Config
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Annotated, Literal
from langchain_ollama import ChatOllama
import asyncio


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- State Definition ---
class State(TypedDict):
    messages: Annotated[list, add_messages]


# --- Tool Definition ---
search_tool = TavilySearch(max_results=2, name="web_search")
tools = [search_tool]
tool_node = ToolNode(tools)

def format_to_agent_scratchpad(messages: list[BaseMessage]) -> list[BaseMessage]:
    scratchpad = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            scratchpad.append(msg)
        elif isinstance(msg, ToolMessage):
            scratchpad.append(msg)
    return scratchpad

# 初始化LLM，本地部署的
llm = ChatOpenAI(
    temperature=0,
    model="qwen3-8b",
    api_key="EMPTY",
    api_base="http://localhost:6006/v1",
    # 启用深度思考模式
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

# 系统提示模板
SYSTEM_PROMPT = """你是一个智能助手，尽可能的调用工具回答用户的问题。
请遵循以下规则:
1. 确保回答准确、专业
2. 对于不确定的信息明确说明
3. 遵守企业数据安全政策
4. 避免提供敏感信息"""

prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages", optional=True),
    MessagesPlaceholder(variable_name='agent_scratchpad', optional=True),
])

"""
- 系统消息：设置代理的上下文，告诉它必须使用工具（web_search）来回答问题，不依赖内部知识。
- messages：包含用户和AI之间的对话历史。例如，用户之前的问题和AI的回答（可能是之前的对话轮次）。
- agent_scratchpad：在代理运行过程中记录中间步骤。例如，当代理思考要调用什么工具时，它会将工具调用的请求和工具的响应记录到agent_scratchpad中。
这样，代理可以根据之前的工具使用情况来决定下一步行动。
- agent_scratchpad 实际是临时存储了当前对话轮次中代理的思考过程（工具调用和结果）。这样当代理再次被调用时（在多步工具调用中），它可以看到之前
已经做了什么工具调用以及得到了什么结果，从而决定下一步行动。

带 agent_scratchpad 的工作流程，核心是 ReAct (Reasoning + Acting) 模式，它通过一个循环来工作：
- 启动：用户提出问题。
- 思考：LLM根据当前信息（包括已填充了历史步骤的agent_scratchpad）进行推理，决定下一步是调用工具还是给出最终答案。
- 行动：如果决定调用工具，则生成工具名称和输入参数。
- 观察：系统执行工具，并将结果记录为Observation。
- 记录与循环：将本次的Thought、Action和Observation添加到agent_scratchpad中，然后整个上下文再次送入LLM，循环步骤2-4。
- 结束：当LLM认为已经掌握足够信息时，会生成Final Answer。
在这个过程中，agent_scratchpad 起到了 “工作记忆”或“思维链” 的关键作用，它将整个推理和执行的轨迹记录下来，使得LLM能够基于完整的上下文进行下一步决策。

而 prompt 中使用不带 ️agent_scratchpad 的工作流程：
- 绑定工具：通过 llm.bind_tools(tools) 将工具的定义“告知”LLM，使其能生成符合规范的参数。
- 生成调用：用户提问后，LLM直接分析问题，并可能在一次响应中生成一个或多个结构化的工具调用请求（如 tool_calls）。
- 自动执行：ToolNode（在LangGraph中）或类似的执行器会接收这些请求，自动地、并行地调用相应的工具。
- 汇总结果：工具执行的结果会被收集起来，可以直接返回给用户，或者作为下一步LLM调用的输入。
这种方式下，LLM的思考过程是内化的，开发者看到的是直接的“输入-工具调用-输出”。

如何选择这两种方式：
选择带 agent_scratchpad 的传统Agent (ReAct) 方式，当你的任务是：
- 复杂且需要多步推理：例如，需要先搜索信息，再进行分析，最后进行计算。
- 需要高透明度和可控性：你想清晰地了解AI的每一步思考和决策过程，便于调试和优化。
= 任务步骤不确定性高：下一步要做什么，严重依赖于上一步的执行结果。

选择 llm.bind_tools + ToolNode 的方式，当你的场景是：
- 工具调用密集或可并行：例如，需要同时查询天气和搜索新闻。
- 追求更高的执行效率：希望减少与LLM的交互次数，降低延迟。
- 构建基于图的复杂工作流：在使用LangGraph等框架时，ToolNode可以作为一个高效的、专门化的工具执行节点。
"""


def agent_node(state: State):
    agent_scratchpad = format_to_agent_scratchpad(state["messages"])
    chain = prompt | llm.bind_tools(tools)
    response = chain.invoke({
        "messages": state["messages"],
        "agent_scratchpad": agent_scratchpad
    })
    return {"messages": [response]}

# --- Graph Definition ---
def should_continue(state: State) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()

# --- Main Interaction Loop (Asynchronous) ---
async def main():
    while True:
        user_input = await aioconsole.ainput("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        response = await app.ainvoke({"messages": [("human", user_input)]})
        print(f"AI: {response['messages'][-1].content}")


if __name__ == "__main__":
    print("Starting chatbot with local Ollama model. Make sure Ollama is running.")
    asyncio.run(main())


""" -------------------- MCP_SERVER 配置 ---------------------- """
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
    \"\"\"企业级封装的异步客户端\"\"\"
    async with MultiServerMCPClient(config) as client:
        try:
            yield client.get_tools()
        except asyncio.TimeoutError:
            logger.error("MCP client timeout")
            raise ServiceUnavailableError()
            
async with managed_client() as tools:
    llm_with_tool = prompt | llm.bind_tools(tools)
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
        
        # MultiServerMCPClient 可以接收多个 server 配置，即可以连接多个 MCP 服务器
        # with 自动释放资源
        """
        async def 的作用是声明函数为协程函数，使其内部可以包含 await、async with 等异步操作。但函数本身的定义不会自动使其内部代码异步执行。
        - 若内部代码需要异步执行（如数据库连接建立、资源释放、网络会话等），则必须用 async with 来让异步上下文管理器管理其生命周期（普通with会阻塞事件循环）。
        - 若内部代码操作对象是同步的（如本地计算、非异步I/O），则无需加 async。
        """
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
    input_textbox.submit(do_graph, [input_textbox, chatbot], [input_textbox, chatbot]).then(execute_graph, chatbot, chatbot)


if __name__ == '__main__':
    # 生产环境启动
    instance.launch(**{
        "auth": Config.GRADIO_AUTH,
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "debug": False
    })
