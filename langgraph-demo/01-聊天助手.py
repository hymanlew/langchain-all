"""
智能收集小助手：RactAgent 实现原理

在这个例子中，我们将创建一个帮助用户生成提示的聊天机器人。
它首先从用户那里收集需求，然后生成提示(并根据用户输入进行细化)。这些被分成两个独立的状态，LLM 决定何时在它们之间转换。
"""
from typing import List
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from langchain core.messages import AIMessage, HumanMessage, ToolMessage
from typing import Literal
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessageGraph
import uuid


'''
收集信息
首先，让我们定义图中用于收集用户需求的部分。这是一个带有特定系统消息的 LLM 调用。它将访问一个工具，当它准备好生成提示时可以调用该工具。
'''
# 定义一个模板，用于指导用户提供模板所需的信息
template = """Your job is to get information from a user about what type of prompt template they want to create.
You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NoT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
After you are able to discern all the information, call the relevant tool."""

"""
你的工作是从用户那里获取他们想要创建哪种类型的提示模板的信息。
您应该从他们那里获得以下信息:

- 提示的目的是什么
- 将向提示模板传递哪些变量
- 输出不应该做什么的任何限制
- 输出必须遵守的任何要求
如果你无法辨别这些信息，请他们澄清! 不要试图疯狂猜测。
在您能够辨别所有信息后，再调用相关工具。
"""
	
#定义一个数据模型，用于存储提示模板的指令信息
class PromptInstructions(BaseModel):
	"""Instructions on how to prompt the LLM"""
	#目标，变量（关注因素），约束，要求
	objective: str
	variables: List[str]
	constraints: List[str]
	requirements: List[str]


#定义一个函数，用于将系统消息和用户消息组合成一个消息列表
def get_messages_info(messages):
	return [SystemMessage(content=template)] + messages	
	
llm = ChatOpenAI(model="gpt-4o",temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])
chain = get_messages_info | llm_with_tool


#定义一个新的系统提示模板
prompt_system = """Based on the following requirements, write a good prompt template:
{regs}"""

#定义一个函数，用于获取生成提示模板所需的消息
#只获取工具调用之后的消息
def get_prompt_messages(messages: list):
	tool_call = None
	other_msgs = []
	for m in messages:
		if isinstance(m, AIMessage )and m.tool_calls:
			tool_call = m.tool_calls[0]["args"]
		elif isinstance(m, ToolMessage):
			continue
		elif tool_call is not None:
			other_msgs.append(m)
	return [SystemMessage(content=prompt_system.format(regs=tool_call))] + other_msgs
	
	
# 将消息处理链定义为 get_prompt_messages 函数和 LLM 实例
prompt_gen_chain = get_prompt_messages | llm

#定义一个函数，用于获取当前状态
def get_state(messages) -> Literal["add_tool_message", "info", "__end__"]:
	if isinstance(messages[-1], AIMessage) and messages[-1].tool calls:
		return "add_tool_message"
	elif not isinstance(messages[-1], HumanMessage):
		return END
	return "info"

#定义一个函数，用于添加工具消息
@workflow.add_node
def add_tool_message(state: list):
	return ToolMessage(
		content="Prompt generated!", tool_call id=state[-1].tool_calls[0]["id"]
	)


#创建聊天消息处理图，区别于状态传递图
workflow = MessageGraph()
workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_gen_chain)	

workflow.add_edge(START, "info")
workflow.add_conditional_edges("info", get_state)
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
graph_png = graph.get_graph().draw_mermaid_png()
with open("collect_chatbot.png","wb") as f:
	f.write(graph_png)
	

# 配置参数，生成一个唯一的线程 ID
config = {"configurable": {"thread _id": str(uuid.uuid4())}}

# 无限循环，直到用户输入“q”或“Q”退出
while True:
	user = input("User(q/Q to quit):")
	if user in {"q","Q"}:
		print("AI: Byebye")
		break
	
	output = None
	#处理用户输入的消息，并打印 AI 的响应
	for output in graph.stream(
		[HumanMessage(content=user)], config=config, stream_mode="updates"
	):
		last_message = next(iter(output.valves()))
		last_message.pretty_print()
		
	#如果输出包含“prompt"，打印“Done!"
	if output and "prompt" in output:
		print("Done!")

