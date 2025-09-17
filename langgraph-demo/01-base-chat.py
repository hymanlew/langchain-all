"""
智能收集小助手：RectAgent 实现原理

在这个例子中，我们将创建一个帮助用户生成提示的聊天机器人。
它首先从用户那里收集需求，然后生成提示(并根据用户输入进行细化)。这些被分成两个独立的状态，LLM 决定何时在它们之间转换。
"""
from typing import List
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field, BaseModel
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
	objective: str = Field(description="The objective of the prompt.")
	variables: List[str] = Field(description="The variables that will be passed into the prompt template.")
	constraints: List[str] = Field(description="The constraints for what the output should NOT do.")
	requirements: List[str] = Field(description="The requirements that the output MUST adhere to.")


#定义一个函数，用于将系统消息和用户消息组合成一个消息列表
def get_messages_info(messages):
	"""Change the agent's behavior depending on the query intent."""
	user_msg = messages[-1].content.lower()

	if "invest" in user_msg or "risks" in user_msg:
		prompt = "You are a financial advisor. Give clear analysis of risks and opportunities."
	elif "summarize" in user_msg:
		prompt = "You are a summarizer. Keep the answer short and clear."
	elif "explain" in user_msg:
		prompt = "You are a teacher. Explain concepts step by step in simple terms."
	else:
		prompt = template

	print(f"Selected prompt: {prompt}")
	# Filter out any existing system messages
	non_system_messages = [msg for msg in messages if msg.type != "system"]

	# Return the new system message + all non-system messages
	return [SystemMessage(content=prompt)] + non_system_messages

	
llm = ChatOpenAI(model="gpt-4o",temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])
chain = get_messages_info | llm_with_tool

#定义一个函数，用于获取生成提示模板所需的消息
#只获取工具调用之后的消息
def get_prompt_messages(messages):
	tool_call = None
	other_msgs = []
	for m in messages:
		if isinstance(m, AIMessage ) and m.tool_calls:
			tool_call = m.tool_calls[0]["args"]
		elif isinstance(m, ToolMessage):
			continue
		elif tool_call is not None:
			other_msgs.append(m)

	# 定义一个新的系统提示模板
	prompt_system = """Based on the following requirements, write a good prompt template: {regs}"""
	return [SystemMessage(content=prompt_system.format(regs=tool_call))] + other_msgs
	
	
# 将消息处理链定义为 get_prompt_messages 函数和 LLM 实例
prompt_create_chain = get_prompt_messages | llm

#定义一个函数，用于获取当前状态
def get_state(messages) -> Literal["prompt", "info", "__end__"]:
	if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
		return "prompt"
	elif not isinstance(messages[-1], HumanMessage):
		return END
	return "info"

#定义一个函数，用于添加工具消息
def add_tool_message(state):
	return ToolMessage(
		content="Prompt generated!",
		tool_call_id=state[-1].tool_calls[0]["id"],
	)


#创建聊天消息处理图，区别于状态传递图
workflow = MessageGraph()
workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_create_chain)
workflow.add_node("finish", add_tool_message)

workflow.add_edge(START, "info")
# 修正条件边逻辑：当 info 节点触发 tool_call 时，直接进入 prompt 节点
workflow.add_conditional_edges(
    "info", 
    get_state,
    {
        "prompt": "prompt",
        "info": "info",
        "__end__": END
    }
)
workflow.add_edge("prompt", "finish")
workflow.add_edge("finish", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
graph_png = graph.get_graph().draw_mermaid_png()
with open("collect_chatbot.png","wb") as f:
	f.write(graph_png)
	

# 配置参数，生成一个唯一的线程 ID
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

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
		last_message = next(iter(output.values()))
		last_message.pretty_print()
		
	#如果输出包含“prompt"，打印“Done!"
	if output and "prompt" in output:
		print("Done!")

