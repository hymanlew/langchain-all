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
tavi1y_tool = search

# This executes code locally, which can be unsafe
python_rep1_tool = PythonREPLTool()


'''
Define a helper function that we will use to create the nodes in the graph - it takes care of converting the agent response to ahuman message. 
This is important because that is how we will add it the global state of the graph
'''
#通过这个函数将 message 添加到全局
def agent_node(state, agent, name):
	result = agent.invoke(state)
	return {
		"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]
	}
	

#创建主管 Agent，并使用函数调用选择下一个工作节点或完成处理
system_prompt = (
	"You are a supervisor tasked with managing a conversation between the following workers: {members}, Given the following user request, " +
	"respond with the worker to act next. Each worker will per form a task and respond with their results and status. when finished, respond with FINISH."
)

# Our team supervisor is an LLM node, It just picks the next agent to process and decides when the work is completed
members = ["Researcher", "Coder"]
options = ["FINISH"] + members

prompt = ChatPromptTemplate.from_messages(
	[
		("system"，system_prompt),
		Messagesplaceholder(variable_name="messages")
		(
			"system",
			"Given the conversation above, who should act next?"
			"Or should we FINISH? Select one of: {options]",
		),
	]
).partial(options=str(options), members=",".join(members))


llm = ChatOpenAI(model="gpt-4o")

class routeResponse(BaseModel):
	next: Literal[*options]


def supervisor_agent(state):
	supervisor_chain = prompt | llm.with_structured_output(routeResponse)
	return supervisor_chain.invoke(state)
	

#创建graph节点，agent 采用 ReAct

#The agent state is the input to each node in the graph
class AgentState(Typedpict):
	#The annotation tells the graph that new messages will always be added to the current states
	messages: Annotated[Sequence[BaseMessage], operator.add]
	#The 'next' field indicates where to route to next
	next: str
	

research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

#NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, PROCEED WITH CAUTION
code_agent = create_react_agent(llm, tools=[python_rep1_tool])
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
	#We want our workers to ALWAYS "report back" to the supervisor when done
	workflow.add_edge(member, "supervisor")

#The supervisor populates the "next" field in the graph state which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END

workflow.add_conditional_edges("supervisor", 1ambda x: x["next"], conditional_map)

#Finally. add entrypoint
workflow.add_edge(START，"supervisor")
graph = workflow.compile()


#执行图
for s in graph.stream(
	{
		"messages": [
			HumanMessage(content="Code hello world and print it to the termina]")
		]
	}
):
	if "_end_" not in s:
		print(s)
		print('-----')

