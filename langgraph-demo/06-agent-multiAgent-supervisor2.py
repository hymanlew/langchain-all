import os
from functools import partial
from typing import Annotated, Sequence, TypedDict, Literal
import yfinance as yf
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent, tools_condition
import functools
import operator


llm = ChatOpenAI(model="gpt-4o-mini")

# Route Response structure for supervisor's decision
class RouteResponseFin(BaseModel):
    next: Literal["Market_Data_Agent", "Analysis_Agent", "News_Agent", "FINISH"]

# Define agent members
members_fin = ["Market_Data_Agent", "Analysis_Agent", "News_Agent"]

# Supervisor prompt setup
system_prompt_fin = (
    "You are a Financial Services Supervisor managing the following agents: "
    f"{', '.join(members_fin)}. Select the next agent to handle the current query."
)

prompt_fin = ChatPromptTemplate.from_messages([
    ("system", system_prompt_fin),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Choose the next agent from: {options}."),
]).partial(options=str(members_fin))

# Supervisor Agent
def supervisor_agent_fin(state):
    supervisor_chain_fin = prompt_fin | llm.with_structured_output(RouteResponseFin)
    return supervisor_chain_fin.invoke(state)

# Define Tools and Agent Prompts
# 1. Market Data Tool and Agent Prompt
def fetch_stock_price(query):
    """Fetch the current stock price of a given stock symbol."""
    stock_symbol = query.split()[-1]
    stock = yf.Ticker(stock_symbol)
    try:
        current_price = stock.info.get("currentPrice")
        return f"The current stock price of {stock_symbol} is ${current_price}."
    except Exception as e:
        return f"Error retrieving stock data for {stock_symbol}: {str(e)}"

def agent_node(state, agent, name):
    result = agent.invoke(state)
    print(f"{name} Output: {result['messages'][-1].content}")
    return {
        "messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
    }

market_data_prompt = (
    "You are the Market Data Agent. Your role is to retrieve the latest stock prices or "
    "market information based on user queries. Ensure your response includes the current price "
    "and any relevant market details if available."
)
market_data_agent = create_react_agent(llm, tools=[fetch_stock_price], state_modifier=market_data_prompt)
market_data_node = functools.partial(agent_node, agent=market_data_agent, name="Market_Data_Agent")

# 2. Financial Analysis Tool and Agent Prompt
def perform_financial_analysis(query):
    """Perform financial analysis based on user query."""
    if "ROI" in query:
        initial_investment = 1000
        final_value = 1200
        roi = ((final_value - initial_investment) / initial_investment) * 100
        return f"For an initial investment of ${initial_investment} yielding ${final_value}, the ROI is {roi}%."
    return "No relevant financial analysis found."

analysis_prompt = (
    "You are the Financial Analysis Agent. Analyze the financial data provided in the query. "
    "Perform calculations like ROI, growth rates, or other financial metrics as required. "
    "Provide a clear and concise response."
    "Only use the following tools:"
    "perform_financial_analysis"

)

analysis_agent = create_react_agent(llm, tools=[perform_financial_analysis], state_modifier=analysis_prompt)
analysis_node = functools.partial(agent_node, agent=analysis_agent, name="Analysis_Agent")

# 3. Financial News Tool and Agent Prompt
financial_news_tool = TavilySearchResults(max_results=5)
news_prompt = (
    "You are the Financial News Agent. Retrieve the latest financial news articles relevant to the user's query. "
    "Use search tools to gather up-to-date news information and summarize key points."
    "Do not quote sources, just give a summary."
)
financial_news_agent = create_react_agent(llm, tools=[financial_news_tool], state_modifier=news_prompt)
financial_news_node = functools.partial(agent_node, agent=financial_news_agent, name="News_Agent")

# Define Workflow State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Define the workflow with the supervisor and agent nodes
workflow_fin = StateGraph(AgentState)
workflow_fin.add_node("Market_Data_Agent", market_data_node)
workflow_fin.add_node("Analysis_Agent", analysis_node)
workflow_fin.add_node("News_Agent", financial_news_node)
workflow_fin.add_node("supervisor", supervisor_agent_fin)

# Define edges for agents to return to the supervisor
for member in members_fin:
    workflow_fin.add_edge(member, "supervisor")

# Conditional map for routing based on supervisor's decision
conditional_map_fin = {
    "Market_Data_Agent": "Market_Data_Agent",
    "Analysis_Agent": "Analysis_Agent",
    "News_Agent": "News_Agent",
    "FINISH": END  # This will end the workflow when supervisor decides
}
workflow_fin.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map_fin)
workflow_fin.add_edge(START, "supervisor")

# Compile the workflow
graph_fin = workflow_fin.compile()

# Testing the workflow with an example input
inputs_fin = {"messages": [HumanMessage(content="What is the stock price of AAPL?")]}
for output in graph_fin.stream(inputs_fin):
    if "__end__" not in output:
        print(output)


"""
ToolNode 是一个运行在最后一条 AIMessage 中调用工具的节点。
它可以在 StateGraph 中与"messages"状态键一起使用(或通过 TooINode 的"mesages_key“传递自定义键)。
如果需要请求多个工具调用，它们将并行运行。输出是一个 ToolMessages 列表，每个工具调用对应一个。

功能              BasicToolNode(自定义)          ToolNode(官方)
工具调用逻辑      需手动解析 tool calls 并执行工具  自动解析 tool calls，支持同步/异步工具调用
状态管理         需手动封装 ToolMessage           自动将结果封装为 ToolMessage 并更新状态
错误处理         需手动捕获异常(如工具未找到)        内置工具名称校验和异常处理机制
LangGraph 集成    需手动配置节点和边               深度集成 StateGraph，支持 tools condition
性能优化         依赖手动实现的并发逻辑              内置并发调度和资源管理
"""
class BasicToolsNode:
    """
    异步工具节点，用于并发执行 AIMessage 中请求的工具调用
    功能:
    1.接收工具列表并建立名称索引
    2.并发执行消息中的工具调用请求
    3.自动处理同步/异步工具适配
    """
    def __init__(self, tools: list):
        """初始化工具节点"""
        self.tools_by_name = {tool.name: tool for tool in tools} #所有

    async def __call__(self, state: Dict[str, Any])-> Dict[str, List[ToolMessagel]]:
        """
        异步调用入口
        Args:
            state: 输入字典，需包含 "messages" 字段
        Returns:
            包含ToolMessage列表的字典
        Raises:
            ValueError: 当输入无效时抛出
        """
        # 1.输入验证
        if not (messages := state.get("messages")):
            raise ValueError("输入数据中未找到消息内容")  # 改进后的中文错误提示
        message:AIMessage = messages[-1] #取最新消息:AIMessage

        tool_name = message.tool-calls[0]['name'] if message.tool-calls else None
        if tool_name == 'webSearchStd' or tool_name == 'webSearchSogou':
            # Command
            response = interrupt(
                f"AI大模型尝试调用工具、{tool_name}'，\n"
                "请审核并选择:批准(y)或直接给我工具执行的答案。"
            )
            # response(字典):由人工输入的:批准(y),工具执行的答案或者拒绝执行工具的理由
            # 根据入工响应类型处理
            if response["answer"] == "y":
                pass  # 直接使用原参数继续执行
            else:
                return {"messages": [ToolMessage(
                    content=f"人工终止了该工具的调用，给出的理由或者答案是:{response['answer']}",
                    name=tool_name,
                    tool_callid = message.tool-calls[0]['id']
                )]}

        # 2.并发执行工具调用
        outputs = await self.execute-tool-calls(message.tool_calls)
        return {'messages": outputs}

    async def execute-tool-calls(self, tool_calls: List[Dict]) -> List[ToolMessage]:
        """执行实际工具调用
        Args:
            tool_calls:工具调用请求列表
        Returns:
            ToolMessage结果列式
        """
        async def invoke_tool(tool_call: Dict)-> ToolMessage:
            """执行单个工具调用
            Args:
                tool_call:工具调用请求字典，需包含name/args/id字段
            Returns:
                封装的ToolMessage
            Raises:
                KeyError:工具未注册时抛出
                RuntimeError:工具调用失败时抛出
            """
            try:
                # 异步调用工具
                tool = self.tools_by_name.get(tool_call["name"]) #验证工具是否在之前的工具集合中
                if not tool:
                    raise KeyError(f"未注册的工具:{tool_call['name']}")

                if hasattr(tool, 'ainvoke'): #优先使用异步方法
                    tool_result = await tool.ainvoke(tool_call["args"])
                else: #同步工具通过线程池转异步
                    loop = asyncio.get-running_loop()
                    tool-result = await loop.run-in-executor(
                        None，#使用默认线程池
                        tool.invoke，#同步调用方次
                        tool_call["args"] # 参数
                    )

                # 构造ToolMessage
                return ToolMessage(
                    content=json.dumps(tool_result,ensure_ascii=False),
                    name=tool-call["name"],
                    tool_call-id = tool_call["id"]
                )
            except Exception as e:
                raise RuntimeError(f"工具调用失败:{tool_call['name']}") from e

        try:
            # 并发执行所有工具调用
            # 并发执行:所有传入的协程会被同时调度到事件循环中，通过非阻塞I0实现并行处理。
            #结果收集:按输入顺序返回所有协程的结果(或异常)，与任务完成顺序无关。
            #异常处理:默认情况下，任一任务失败会立即取消其他任务并抛出异常;若设置 return_exceptions=True，则异常会作为结来
            return await asyncio.gather(*[invoke-tool(tool-call) for tool-call in tool-calls])
        except Exception as e:
            raise RuntimeError("并发执行工具时发生错误")from e

def route_tools_func(state: AgentState):
    """
    动态路由函数，如果从大模型输出后的AIMessage，包含有工具调用的请求(指令)，就进入到tools节点，否则则结束
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages",[]):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message,"tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

tools = []
# tool_node =  ToolNode(tools)
# workflow_fin.add_conditional_edges("supervisor", tools_condition)
tool_node =  BasicToolsNode(tools)
workflow_fin.add_node("tools", tool_node)
workflow_fin.add_conditional_edges("supervisor", route_tools_func,{"tools":"tOOLS",END: END})
