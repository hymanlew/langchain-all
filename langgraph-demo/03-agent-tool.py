from langchain_core.messages import (
	BaseMessage,
	HumanMessage,
	ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.utils.pydantic import TBaseModel
from langgraph.graph import END, StateGraph, START
#导入注解类型
from typing import Annotated, List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, InjectedToolCallId
from langchain_experimental.utilities import PythonREPL
#导入操作符和类型注解
import operator
from langgraph.prebuilt import ToolNode, InjectedState
from typing import Literal
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
import functools
from langchain_core.messages import AIMessage
import requests
from langgraph.types import Command
from pydantic import BaseModel, Field


#定义一个函数，用于创建代理
def create_agent(llm, tools, system_message: str):
	"""创建一个代理"""
	prompt = ChatPromptTemplate.from_messages(
		[
			("system",
			"你是一个有帮助的AI助手，与其他助手合作。使用提供的工具来推进问题的回答。"
			"如果你不能完全回答，没关系，另一个拥有不同工具的助手，它会接着你的位置继续帮助。执行你能做的以取得进展。"
			"如果你或其他助手有最终答案或交付物，在你的回答前加上 FINAL ANSWER，以便团队知道停止。"
			"你可以使用以下工具:{tool_names} \n{system_message}"
			),
			MessagesPlaceholder(variable_name="messages")
		]
	)
	# 传递系统消息参数，工具名称参数
	prompt = prompt.partial(system_message=system_message)
	prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
	# 绑定工具并返回提示模板
	return prompt | llm.bind_tools(tools)
	

#警告:这会在本地执行代码，未沙箱化时可能不安全
tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

#定义一个工具函数，用于执行Python代码
@tool("python_code_tool")
def python_repl(code: Annotated[str, "要执行以生成图表的Python代码。"]):
	try:
		result = repl.run(code)
	except BaseException as e:
		return f"执行失败。错误:{repr(e)}"
		
	result_str = f"成功执行:\n``python\n{code}\n```\nStdout: {result}"
	return (result_str + "\n\n如果你已完成所有任务，请回复 FINAL ANSWER。")

'''
code 值如下: 
import matplotlib.pyplot as plt
# Data
years = ["2018","2019","2020","2021"，"2022"]
market_size = [10.1，14.69，22.59，34.87，62.5]

# Create the plot
plt.figure(figsize=(10，6))
plt.plot(years, market_size, marker='o', linestyle='-', color='b')

# Adding titles and labels
plt.title("Global AI Software Market size(2018-2022)")
plt.xlabel("Year")
plt.ylabel("Market Size(in billion usD)")
plt.grid(True)

# Display the plot
plt.show()
'''

def lookup_stock_symbol(company_name: str, state: Annotated[AgentState, InjectedState]) -> str:
	"""
    将公司名称转换为股票代码使用金融API，并获取其财务数据。
    参数:
        company_name (str): 公司全名（例如 'Tesla'）。
    返回:
        str: 股票代码（例如 'TSLA'）或错误信息。
    """
	sender = state.sender
	print(sender)
	
	api_url = "https://www.alphavantage.co/query"
	params = {
		"function": "SYMBOL_SEARCH",
		"keywords": company_name,
		"apikey": "your_alphavantage_api_key"
	}
	response = requests.get(api_url, params=params)
	data = response.json()

	if "bestMatches" in data and data["bestMatches"]:
		return data["bestMatches"][0]["1. symbol"]
	else:
		return f"Symbol not found for {company_name}."


# Create tool bindings with additional attributes
lookup_stock = Tool.from_function(
    func=lookup_stock_symbol,
    name="lookup_stock_symbol",
    description="Converts a company name to its stock symbol using a financial API.",
    return_direct=False  # Return result to be processed by LLM
)

"""
InjectedToolCallId 自动注入工具ID
工具处理逻辑中，要更新 state 值，并不单单是返回工具结果
"""
@tool('cn_search')
def cn_search(
        query: str,
        too_call_id: Annotated[str, InjectedToolCallId],
        conf: RunnableConfig) -> Command:
	"""使用国内搜索引擎 API 进行实时搜索"""
	description = '使用国内搜索引擎 API 进行实时搜索'
	url = "https://api.example.cn/search/v1"  # 替换为实际 API 地址
	params = {
        "key": "your_search_api_key",
        "q": query,
        "num": 5  # 返回结果数量
    }
    response = requests.get(url, params=params)
    return Command(
		update={
			"sender": 'aa',
			"messages": [
				ToolMessage(
					content=response.json().get("results", []),
					tool_call_id=too_call_id
				)
			]
		}
	)


# 定义结构化输出 Schema
class FinalReport(BaseModel):
	"""智能体生成的最终报告，结构固定，便于下游系统消费。"""
	question: str = Field(description="用户提出的原始问题")
	summary: str = Field(description="对问题和所获信息的综合总结")
	answer: str = Field(description="给用户的直接、清晰的答案")
	data_sources: List[str] = Field(description="所使用的数据源或工具列表", default_factory=list)
	confidence: str = Field(description="答案的置信度", examples=["高", "中", "低"])


# 定义一个对象，用于在图的每个节点之间传递
# 为每个代理和工具创建不同的节点
class AgentState(BaseModel):
	# messages 字段用于存储消息的序列，并且通过 Annotated 和 operator.add 提供了额外的信息，解释如何处
	messages: Annotated[Sequence[BaseMessage],operator.add]
	# sender 用于存储当前消息的发送者。通过这个字段，系统可以知道当前消息是由哪个代理生成的。
	sender: str
	# 最终的结构化输出
	final_report: Optional[FinalReport]


#辅助函数，用于为给定的代理创建节点
def agent_node(state, agent, name):
	# 调用代理
	result = agent.invoke(state)
	if isinstance(result, ToolMessage):
		pass
	else:
		#将 tavily-result 转换为 AIMessage 类型，并且将 name 作为发送者的名称附加到消息中
		result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
	return {"messages": [result], "sender": name}


llm = ChatOpenAI(model="gpt-4o")
# 用于生成最终结构化答案的LLM
structured_llm = llm.with_structured_output(FinalReport)

#创建一个检索代理
research_agent = create_agent(llm, [tavily_tool], system_message="提供准确的数据供chart_generator使用。",)

#创建一个检索节点，并使用部分应用函数(partial function)
#其中 agent 固定为 research_agent, name 固定为 "Researcher"
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

#图表生成器
chart_agent = create_agent(llm, [python_repl], system_message="你展示的任何图表都将对用户可见。")
#创建图表生成节点
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

#创建工具节点
"""
使用 llm.bind_tools(tools) 创建的节点只是大模型推理，并生成含工具调用请求的 AIMessage（包含 tool_calls 字段），但不会自动调用工具，只负责生成工具调用请求。
而使用 ToolNode(tools) 创建的节点会自动调用工具，并生成 ToolMessage (包含工具执行结果)。
"""
tools = [tavily_tool, python_repl, lookup_stock]
tool_node = ToolNode(tools)

def generate_final_output(state: AgentState):
	"""最终输出节点：汇总所有信息，生成结构化报告。"""
	full_conversation = state["messages"]
	final_report = structured_llm.invoke(full_conversation)
	return {"final_report": final_report}

#任一代理都可以决定结束
def router(state) -> Literal["call_tool", "continue", "reporter"]:
	messages = state["messages"]
	last_message = messages[-1]
	
	#检査 last_message 是否包含工具调用(tool calls)
	if last_message.tool_calls:
		return "call_tool"
		
	#如果已经获取到最终答案，则返回结束节点
	if "FINAL ANSWER" in last_message.content:
		return "reporter"
	return "continue"
	

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)
workflow.add_node("reporter", RunnableLambda(generate_final_output))

workflow.add_edge(START, "Researcher")
workflow.add_conditional_edges("Researcher", router, {"continue":"chart_generator", "call_tool":"call_tool", "reporter": "reporter"})
workflow.add_conditional_edges("chart_generator", router, {"continue": "Researcher", "call_tool": "call_tool", "reporter": "reporter"})
workflow.add_conditional_edges(
	"call_tool",
	#lambda 函数作用是从状态中获取 sender 名称，以便在条件边的映射中使用。
	#如果 sender 是 "Researcher"，则工作流将转移到 “Researcher” 节点。
	#如果 sender 是 "chart_generator"，则工作流将转移到 "chart_generator” 节点
	lambda x: x["sender"],
	{"Researcher":"Researcher", "chart_generator":"chart_generator"}
)
workflow.add_edge("reporter", END)

graph = workflow.compile()
graph_png = graph.get_graph().draw_mermaid_png()
with open("collaboration.png","wb") as f:
	f.write(graph_png)

#事件流
input_data = AgentState(
	messages=[HumanMessage(content="获取过去5年AI软件市场规模，然后绘制一条折线图。一旦你编写好代码，就可以完成任务。")],
	final_report=None,
	sender="",
)
config = RunnableConfig(configurable={
        "configurable": {"thread_id": "1"},
		# 图中最多执行的步骤数
        "recursion_limit": 50
    })
events = graph.stream(
	input_data,
	config,
	stream_mode="updates"
)

#打印事件流中的每个状态
for s in events:
	print(s)
	print("----")

	