"""
LangGraph MultiAgent 架构
请注意，Agent 如果有文件系统访问权限，这在所有情况下都是不安全的，要在docker容器中运行。

协作多 Agent(multi-agent-collaboration)
参考: https://github.com/langchaln-al/langgraph/blob/main/docs/docs/tutorials/multil_agent/mult-agent-collaboration.ipynb
在这个例子中，不同的 Agent 在一个共享的消息暂存器上进行协作。这意味着他们所做的所有工作对对方都是可见的。
这样做的好处是其他 Agent 可以看到完成的所有单个步骤。但缺点是，有时传递所有这些信息过于冗长和不必要，有时只需要Agent的最终答案。
由于共享的性质，我们将这种协作称为暂存器。

控制状态转换的主要组件是路由器，它是一个基于规则的路由器，因此相当简单。基本上每次调用 LLM 后，它都会查看输出。
如果调用了一个工具，那么它会调用该工具。
如果没有调用任何工具，并且 LLM 响应“最终答案”，那么它会返回给用户。
否则（如果没有调用任何工具，并且 LLM 没有响应“最终答案”)，那么它会转到另一个 Agent。

但这种方式适应于简单逻辑的，或少量 AGENT 的任务协作，当有大量任务 agent 时，共享路由就很麻烦。这时就需要用到主管 AGENT（下一个文件代码）.
"""
from langchain_core.messages import (
	BaseMessage，
	HumanMessage.
	ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from angchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebui1t import ToolNode



# 是下面系统提示词的中文翻译
'''
"你是一个乐于助人的人工智能助手，与其他助手合作。”
”使用提供的工具来回答问题。”
"如果你不能完全回答，没关系，另一个助手用不同的工具, 这将有助于你职得进展。尽你所能取得进展。"
如果你或其他助理有最终答案或可交付成果，"
在你的回答前加上最终答案，这样团队就知道该停下来了。"
您可以访问以下工具:(tool_names)。\n(system_message}"，
'''

#创建Agent，指定大模型和 too1s, 以及自定义的系统提示词
def create_agent(llm, tools, system_message: str):
	"""Create an agent."""
	prompt = ChatPromptTemplate.from_messages(
		[
			(
				"system",
				"You are a helpful AI assistant, collaborating with other assistants."
				" Use the provided tools to progress towards answering the question."
				" If you are unable to fully answer, that's Ok, another assistant with different tools "
				" wi1l help where you left off, Execute what you can to make progress.""If you or any of the other assistants have the final answer or deliverable,"" prefix your response with FINAL ANSWER so the team knows to stop."
				" You have access to the following tools: {tool_names}.\n{system_message}",
			),
			Messagesp1aceholder(variable_name = "messages"),
		]
	)
	prompt = prompt.partial(system_message = system_message)
	prompt = prompt.partial(tool_names = ",".join([tool.name for tool in tools]))
	return prompt | llm.bind_too1s(tools)
	

@tool
def search():
	"""Search the web for information,"""
	return "test"
	
#tavily_tool = TavilySearchResults(max_results=5)
tavi1y_tool = search

# Warning: This executes code locally, which can be unsafe when not sandboxed
# PythonREPL 是 Python 的一个交互式解释器环境
repl = PythonREPL()

# 定义代码运行工具 python_repl
@tool
def python_repl(
	code: Annotated[str, "The python code to execute to generate your chart."],
):
	"""Use this to execute python code, If you want to see the output of a value,you should print it out with 'print(...)'. This is visible to the user."""
	try:
		result = repl.run(code)
	except BaseException as e:
		return f"Failed to execute. Error : {repr(e)}"
	result_str = f"successfully executed: \n```python\n{code}\n```\nstdout: {result}"
	return (
		result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
	)
	

#定义agent状态，保存agent执行过程中的状态数据，方便后面创建agentnode
import operator
from typing import Annotated, Sequence
from typing_extensions import Typedoict
from langchain openai import chatopenAI

# This defines the object that is passed between each node
# in the graph. we will create different nodes for each agent and tool
class AgentState(TypedDict):
	messages: Annotated[Sequence[BaseMessage], operator.add]
	sender: str
	

import functools
from langchain_core.messages import AIMessage

#agent从 llm, tool, system_msg创建
#Helper function to create a node for a given agent
def agent_node(state, agent, name):
	result = agent.invoke(state)
	# We convert the agent output into a format that is suitable to append to the global state
	if isinstance(result, ToolMessage):
		pass
	else:
		result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
	return {
		"messages": [result],
		# Since we have a strict workflow, we can track the sender so we know who to pass to next.
		"sender": name,
	}
	

llm= ChatopenAI(model="gpt-4o")

#Research agent and node
research_agent = create_agent(
	llm,
	[tavily_too1],
	# 系统提示词
	system_message = "You should provide accurate data for the chart_generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# chart_generator
chart_agent = create_agent(
	llm,
	[python_repl],
	system_message = "Any charts you display will be visible by the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")


#定义工具节点
tools = [tavi1y_tool, python_rep1]
tool_node = ToolNode(too1s)

#定义路由器router
# Either agent can decide to end
from typing import Literal

def router(state):
	# This is the router
	messages = state["messages"]
	last_message = messages[-1]
	if last_message.tool_ca1ls:
		# The previous agent is invoking a tool
		return "cal1_tool"
	if "FINAL ANSWER" in last_message.content:
		# Any agent decided the work is done
		return END
	return "continue"
	

#定义graph，由于可以循环，所以不再是 DAG（有向无环图）啦，添加节点，再添加边，定义初始节点，生成graph
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
	"Researcher",
	router,
	{"continue": "chart_generator", "call_tool": "call_tool", END: END},
)

workf1ow.add_conditional_edges(
	"chart_generator",
	router,
	{"continue": "Researcher", "call_tool": "call_tool", END: END],
)

# call_tool 节点的函数内部仅执行工具（如搜索或绘图），不修改状态中的 sender 字段。
# 当工具节点（call_tool）执行后，需要通过 sender 字段将控制权返回给调用该工具的代理（如 Researcher 或 chart_generator）。
# 若不保留原始 sender，工作流将无法确定下一步应路由到哪个代理节点，导致逻辑混乱。
workflow.add_conditional_edges(
	"call_tool",
	#Each agent node updates the 'sender' field
	#the tool calling node does not, meaning this edge will route back to the original agent
	#who invoked the tool
	1ambda x: x["sender"],
	{
		"Researcher": "Researcher",
		"chart_generator": "chart_generator",
	},
)

workflow.add_edge(START, "Researcher")
graph = workflow.compile()

#调用graph，得到结果
"""
任务：
“获取英国过去5年的国内生产总值，"
"然后画一个折线图。”
“一旦你把它编码好，就完成。"
"""
events = graph.stream(
	{
		"messages": [
			HumanMessage(
				content = "Fetch the Uk's Gop over the past s years ,"
				"then draw a line graph of it."
				"once you code it up, finish."
			)
		],
	},
	#Maximum number of steps to take in the graph
	{"recursion_limit": 150},
)

for s in events:
	print(s)
	print(”----”)


#可以把生成的 graph 结构显示出来
from IPython.display import Image, display
try:
	display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
	#This requires some extra dependencies and is optiona]
	pass

