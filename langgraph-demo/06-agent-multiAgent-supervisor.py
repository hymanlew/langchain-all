"""
多 Agen 协作的方式适应于简单逻辑的，或少量 AGENT 的任务协作，当有大量任务 agent 时，共享路由就很麻烦。这时就需要用到主管 AGENT.

主管多 Agent（Multi-agent supervisor）
参考: https://github,com/angchain-al/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb

在这个例子中，多个 Agen t连接在一起，但它们不共享一个暂存器。相反，它们有自己独立的暂存器，然后它们的最终响应被附加到全局暂存器中。
Agent 主管负责路由至各个 Agent，这样主管也可以被认为是一个 Agent，而其工具就是其他 Agent。
"""
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, Messagesplaceholder
from langchain_openai import ChatopenAI
from pydantic import BaseModel
from typing import Literal
import functools
import operator
from typing import Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

# 创建 too1, 各种 agent 架构都一样
@tool
def search():
	"""Search the web for information."""
	return "test"

#tavily_tool = TavilySearchResults(max results=5)
tavily_tool = search

# This executes code locally, which can be unsafe
python_rep1_tool = PythonREPLTool()

client = MultiServerMCPclient(
	{
		"mcp-tool": {
			"url": "xxx",
			"transport": "streamable_http"
		}
	}
)


#创建graph节点，agent 采用 ReAct
class AgentState(Typedpict):
	#有的节点返回 tuple 或自定义的序列类型。如果用 list，就会强制要求所有节点都返回 list，否则类型检查会报错。而 Sequence 能兼容任何序列类型，给开发者更多自由度。
	#Sequence 代表任何顺序的、可迭代的、可通过索引访问的容器，接受 list, tuple, ListWrapper 或任何实现了 __getitem__ 和 __len__ 的自定义序列。

	#指定序列中的元素类型必须是 BaseMessage 或其子类（如 AIMessage, HumanMessage, SystemMessage, tool, functionMessage 等），这是 LangChain 消息的标准类型。
	#AnyMessage 是 BaseMessage 的一个具体子类，特点是能够自动推断类型，优雅地处理来自不同格式的消息数据（如字典、字符串）。
	#AnyMessage 会尝试根据输入的内容自动推断并转换成正确的消息类型。
	#传入 {"role": "user", "content": "Hello"} -> 它会被实例化为一个 HumanMessage。
	#传入 {"role": "assistant", "content": "Hi"} -> 它会被实例化为一个 AIMessage。
	#传入一个字符串 "Hello" -> 默认情况下，它可能会被推断为 HumanMessage。

	#使用 Sequence[BaseMessage] (推荐)，明确消息类型，一目了然，易于维护和调试，避免了不必要的自动类型推断开销。
	#当需要处理大量来自外部、格式不一且不愿意手动预处理的原始消息数据，可能是字典、字符串、或者混合格式的数组。并希望 LangGraph 能自动帮你处理好这些转换时，使用 AnyMessage。
	messages: Annotated[Sequence[AnyMessage], operator.add]
	#The 'next' field indicates where to route to next
	next: str


'''
Define a helper function that we will use to create the nodes in the graph - it takes care of converting the agent response to ahuman message. 
This is important because that is how we will add it the global state of the graph
'''
#通过这个函数将 message 添加到全局
def agent_node(state, agent, next):
	if next == "joke":
		prompt = [
			SystemMessage(content="你是一个笑话大师"),
			HumanMessage(content=state["messages"][0].content),
		]
	if next == "travel":
		prompt = [
			SystemMessage(content="你是一个专业的旅行规划顾问师"),
			HumanMessage(content=state["messages"][0].content),
		]

	result = agent.invoke({"messages": prompt})
	writer = get_stream_writer()
	writer({"node": f"{next} {result}"})
	return {
		"messages": [result["messages"][-1].content], "next": "supervisor")]
	}

# Our team supervisor is an LLM node, It just picks the next agent to process and decides when the work is completed
members = ["research", "coder", "travel_node", "joke_node", "talk_node", "other_node"]
options = ["end"] + members

llm = ChatOpenAI(model="gpt-4o")

#创建主管 Agent，并使用函数调用选择下一个工作节点或完成处理
system_prompt = (
	"You are a supervisor tasked with managing a conversation between the following workers: {members}, Given the following user request, " +
	"respond with the worker to act next. Each worker will per form a task and respond with their results and status. when finished, respond with end." +
	"如果问题是笑话，返回 joke。如果问题是聊天，返回 talk。如果是其他，返回 other。"
)
prompt = ChatPromptTemplate.from_messages(
	[
		("system"，system_prompt),
		Messagesplaceholder(variable_name="messages")
		(
			"user",
			"{input}, Given the conversation above, who should act next?"
			"Or should we FINISH? Select one of: {options]",
		),
	]
).partial(options=str(options), members=",".join(members))

class routeResponse(BaseModel):
	next: Literal[*options]

def supervisor_node(state):
	writer = get_stream_writer()
	writer({"node": "==== supervisor_node"})

	#如果状态中已经有了 next 字段，表示之前已经有节点处理过了（会加上 node next 字段）
	if "next" in state:
		writer({"supervisor_node": f"已获得 {state['next']} 结果"})
		return {"next": "end"}
	else:
		supervisor_chain = prompt | llm.with_structured_output(routeResponse)
		result = supervisor_chain.invoke({"input": state['messages'][0]})
		writer({"supervisor_node": f"处理结果 {result}"})
		if result["next"] in members:
			return result
		else:
			return {"next": "other"}

def other_node(state):
	writer = get_stream_writer()
	writer({"node": "==== other_node"})
	return {
		"messages": [HumanMessage(content="我暂时无法回答这个问题", next="other")]
	}

def talk_node(state):
	writer = get_stream_writer()
	writer({"node": "==== talk_node"})

	#samples 是通过 state 的消息从向量库查询出来的结果集
	prompt_template = ChatPromptTemplate.from_messages([
		("system"，"""你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
			回答时，可以参考下面的参考对联。
			参考对联:
			{samples}
			请用中文回答问题"""),
		("user", "{text}")
	])
	prompt = prompt_template.invoke({"samples":samples, "text":state["messages"][0]})
	result = llm.invoke(prompt)
	writer({"node": result.content})
	return {
		"messages": [HumanMessage(content=result.content, next="supervisor")]
	}

research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, next="Researcher")

#NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, PROCEED WITH CAUTION
code_agent = create_react_agent(llm, tools=[python_rep1_tool])
code_node = functools.partial(agent_node, agent=code_agent, next="Coder")

joke_agent = create_react_agent(llm, tools=[])
joke_node = functools.partial(agent_node, agent=joke_agent, next="joke")

tools = asyncio.run(client.get_tools())
travel_agent = create_react_agent(llm, tools=tools)
travel_node = functools.partial(agent_node, agent=travel_agent, next="travel")

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("travel", travel_node)
workflow.add_node("joke", joke_node)
workflow.add_node("talk", talk_node)
workflow.add_node("other", other_node)
workflow.add_edge(START，"supervisor")

#条件映射：是一个路由表
conditional_map = {k: k for k in members}
conditional_map["end"] = END

#条件边设置中，第二个参数是一个条件函数（lambda表达式），参数 x 是 supervisor 节点执行后的完整状态。函数作用是从状态中提取一个决策值（routeResponse.next），LangGraph 会用这个值去 conditional_map 里查找下一个节点。
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
for member in members:
	#We want our workers to ALWAYS "report back" to the supervisor when done
	workflow.add_edge(member, "supervisor")


checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


#执行图
for s in graph.stream(
	{
		"messages": [
			#content=给我讲一个郭德纲的笑话
			#content=聊天对对联
			#content=今天天气怎么样
			HumanMessage(content="Code hello world and print it to the termina]")
		]
	},
	config={"configurable": {"thread_id": uuid.uuid()}},
	stream_mode="values"
):
	if "_end_" not in s:
		print(s)
		print('-----')

