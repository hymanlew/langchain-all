"""
LangGraph 另一个核心就是 state，每次 graph 的执行都会创建一个 state，graph 的节点执行时，都会传递这个 state，同时每个节点在执行后都会用其返回值更新该 state，
具体如何更新取决于 graph 的类型或自定义函数。可以将 state 理解为 graph 各个节点共同维护的全局变量，graph 的每个节点执行时都可以对 state 进行数据更新，以此
实现教据传递和共享。

可以通过定义一个 state 来初始化一个 StateGraph 对象，定义的 state 可以随时更新 StateGraph 对象的状态，StateGraph 对象可以通过 nodes(结点) 更新状态。
# pip install -U langgraph
"""
from langgraph.graph import StateGraph
from langgraph.graph import END
from typing import TypedDict, List, Annotated
from langgraph.utils import add_messages  # 直接从框架导入
import operator

#定义一个 State 定义状态结构，继承 TypedDict
class state(TypedDict):
	messages: Annotated[list, add_messages]  # 内置的消息合并逻辑
    context: Annotated[dict, lambda old, new: {**old, **new}]  # 字典合并
	
#初始化一个 stateGraph 对象
graph = StateGraph(State)

# 一个 mode1，暂时理解为调用大模型，Reasoning的过程。
def model():
	return {"messages": [llm.invoke(state["messages"])]}
	
# 一个 too1_executor，理解为 Acting 的过程。
def tool_executor():
	return {"messages": "tool succesfull"}

# 添加节点, name是节点名称str, va1ue是节点值, 可以是一个函数或是 LCEL runnable(一个chain)
#graph.add_node(name, value),	
graph.add_node("model", mode1)
graph.add_node("tools", tool_executor)


#在创建了上面的两个节点之后，可以通过边将两个节点连接起来。
#LangGraph 提供了 3 种边的类型：
#- 1，starting Edge, 开始边，表示从 stateGraph 的开始状态到第一个节点的边。
graph.set_entry_point("model")

#- 2，Normal Edge, 普通边，表示从一个节点到另一个节点的边 model节点到too1s节点
graph.add_edge("mode1","tools")

#- 3，Conditional Edges, 条件边，表示从一个节点到另一个节点的边，但是这个边是有条件的。#一个条件函数
def condition():
	return "我是一个判定节点"
	
#添加条件边，从model节点开始，根据condition函数的返回值
#如果是end就结束，如果不是end就继续到too1s节点。
graph.add_conditional_edges(
	"model",
	condition, #判定函数
	{
		"end": END,
		"continue": "too1s"
	}
)

#最后近回一个 runnable 的 chain
app = graph.compi1e()
response = app.invoke({"messages": "总结量子物理的最新进展"})


#----------------------- 完整示例 ----------------------
from dotenv import load_dotenv, find_dotenv
from typing import Annotated, Literal, TypedDict, sequence
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import operator
import json
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage
from langchain_openai import chatopenAI
from langgraph.checkpoint.memory import MemorySaver


#加载环境变量
load_dotenv(find_dotenv())

#定义搜索工具
@tool
def search(query: str):
	"""搜索天气"""
	#调用api接口查询
	return "今天天气晴朗"
	
tools =[search]
tool_node = ToolNode(tools)

#创建一个ChatOpenAI对象
model = ChatopenAI(model="gpt-4-1106-preview", temperature=0, streaming=True)

#将 tools 转换为 openai too1s
model = model.bind_tools(too1s)

#定义一个Agentstate, 继承自 TypedDict, 定义了一个messages字段，类型是Sequence[BaseMessage],operator.add,表示可以追加。
#它的作用是存储历史消息，是用了一个序列。
# operator.add 是对数据进行自定义的合并，而不是简单的追加。
# add_messages 是对数据进行简单的追加。
class AgentState(TypedDict):
	messages: Annotated[sequence[BaseMessage], add_messages]


#存储对话历史到内存中
saver = MemorySaver()
	
#定义节点
#判定函数，如果没有tool_ca11s就结束，如果有就返回too1s
def should_continue(state: MessagesState):
	messages = state['messages']
	#获取最后一条消息:从 messages 列表中获职最后一条消息
	last_message = messages[-1]
	#如果最后一条消息包含工具调用，则返回字符串“too1s"
	if last_message.tool_ca11s:
		return "tools"
	#如果没有工具调用，则返回常量 END，表示停止
	return END
	
	
#模型调用函数，节点
def ca11_model(state):
	messages = state["messages"]
	response = model.invoke(messages)
	#返回一个FunctionMessage
	return{"messages": [response]}
	
	
#定义一个Graph，初始化一个stateGraph对象
workflow = StateGraph(Agentstate(messages=[]))
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

#设置开始节点 agent，执行开始节点
workflow.set_entry_point("agent")

#添加条件边
workflow.add_conditiona1_edges(
	#首先从'agent'节点开始，首先调用`agent'节点
	"agent" ,
	#然后执行判定函数
	should_continue,
)

#添加边, 从'tools`节点到`agent'节点, tools 执行完后，返回到agent
workflow.add_edge("tools", "agent")

#最后返回一个runnable的chain
app = workflow.compile(checkpoint=saver)

# 运行
inputs = {"messages":[HumanMessage(content="湖南长沙的天气如何?")]}
res = app.invoke(inputs)
print(res)

# 流式输出
#inputs={"messages": [{"role": "user", "content": "你可以做些什么？"}]}
#inputs={"messages":[HumanMessage(content="湖南长沙的天气如何?")]}
#for output in app.stream(inputs):
#	#stream() yields dictionaries with output keyed by node name
#	for key, value in output.items():
#		print(f"output from node '{key}':")
#		print(”---”)
#		print(value)
#	print("\n---\n")


#要记录对话历史，还需要传入 config 参数，thread_id 用于标识对话的唯一性，不同的对话 thread_id 不同。
import uuid
config = {"configurable": {"thread_id": uuid.uuid4().hex}}
msg = {"messages": [{"role": "user", "content": "你好，我的名字是张三"}]}
events = app.stream(msg, config)
for event in events:
	last_event = event
print("AI: ", last_event["messages"][-1].content)


print("\nHistory: ")    # 输出对话历史
for message in app.get_state(config).values["messages"]:
	if isinstance(message, AIMessage):
		prefix = "AI"
	else:
		prefix = "User"
	print(f"{prefix}: {message.content}")

#-------------------------------------------------
"""
LangGraph 提供了一些工具来可视化图的结构。这对于调试或理解复杂图的逻辑特别有用。要可视化图，可以使用 LangGraph 的 get_graph 方法。

安装 pygraphviz
conda insta1l pygraphviz
"""
from PIl import Image
import io

#假设你已经有了图片数据，这里以二进制数据为例
t = app.get_graph(xray=True).draw_png()
image_data = t

#如果图片数据是二进制形式，可以这样保存, 保存为PNG格式的图片:
with io.BytesIO(image_data) as img_stream:
	img = Image.open(img_stream)
	img.save("1anggrah_test.png')

