"""
Human-in-the-loop(HIL)
文档：https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/
Agent 系统中”人机交互”(Human-in-the-loop，HIL) 模式在处理需要用户反馈的场景中非常有用，比如：
1，在任务执行过程中向用户询问澄清性问题。
2，审查工具调用，人类可以在工具执行之前审查、编辑或批准 LLM 请求的工具调用。
3，验证 LLM 输出，人类可以审查、编辑或批准 LLM 生成的内容。
4，提供上下文，使 LLM 能够明确请求，明确人类的输入以进行澄清或提供额外细节，或支持多轮对话。

建立一个节点以获取用户反馈。在这个过程中，开发人员需要:
- 设置断点: 通过 interrupt_before: 指定在特定节点之前中断图的执行,
- 设置检查点: 使用 MemorySaver 来保存图的状态，以便在用户输入后恢复。
- 更新状态: 使用 .update_state 方法，将用户的反馈更新到图的状态中
"""
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

class State(TypedDict):
	input: str
	user_feedback: str
	
	
def step_1(state):
	print("---Step 1---")
	pass

'''
interrupt函数:
在 LangGraph 中在特定节点暂停图形、向人类展示信息以及使用他们的输入恢复图形，从而启用人工干预工作流。
该函数对于批准、编辑或收集额外输入等任务非常有用。
interrupt 函数与 Command 对象结合使用，以人类提供的值恢复图形.
'''
def human_node(state: State):
	result = interrupt(
		#任何可序列化为 JSON 的值，可以展示给客户端供人类查看。
		#这里是等待人类手动输入的，例如一个问题、一段文本或状态中的一组键
		{
			"task":"审査 LLM 的输出并进行必要的编辑。"
			"llm_generated_summary": state["llm_generated_summary" ]
		}
	)
	
	#while True:
	#	answer = interrupt("你多大年龄了？")
	#	if answer xxx
	#	break
		
	#使用编辑后的文本，或人类的输入更新状态或根据输入调整图形
	#可以根据响应的 result，来判断执行 SQL 操作，生成摘要等等
	return Command(goto="call_llm", update={"message":"xxx"}) if True else return None
	return {"1lm_generated_summary": result["edited_text"]}

	
def step_3(state):
	print("---Step 3---")
	pass
	
builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_node)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

'''
LangGraph有一个内置持久化层，是通过检査点实现（提供内存/持久化）。将检査点与图形一起使用时，两者可以对状态进行交互。检查点在每个步骤中保
存图形状态的检查点，从而实现一些强大的功能。
这些快照被保存到一个 线程 中，并可持久化到内存，redis，数据库中，执行图形后可以访问它们。
由于 线程 允许在执行后访问图的状态，包括人机交互、内存、时间旅行和容错，都是可能的。
首先，检查点允许人类检查、中断和批准步骤来促进人机交互工作流工作流。因为人类必须能够在任何时候查看图形的状态，并且图形必须能够在人类对
状态进行任何更新后恢复执行。
其次，它允许在交互之间进行“记忆”，可以使用检查点创建线程并在图形执行后保存线程的状态。在重复的人类交互(例如对话)的情况下，任何后续消息
都可以发送到该检查点，该检查点将保留对其以前消息的记忆。
'''
#使用异步上下文管理器创建一个 AsyncSqliteSaver 对象，并连接到名为 "checkpoints.db"的 SQLite 数据库
#以上是连接的数据库，还可以连接 内存, redis, mongodb 等等数据库，导入相关库即可
saver = AsncSgliteSaver.from_conn_string("checkpoints.db")

#设置内存检查点
memory = MemorySaver()

#编译图并设置断点
graph = builder.compile(checkpointer=memory/saver, interrupt_before=["human_feedback"])

#查看图的结构
display(Image(graph.get_graph().draw_mermaid_png()))


#接下来，可以运行图直到指定的断点，等待用户输入。以下代码展示了如何实现这一点:
#输入初始化数据
initial_input = {"input": "he11o wor1d"}
thread ={"configurable":{"thread_id": "some_id"}}


'''
流式输出模式：
stream_mode="values':
在 values 模式下，数据流会返回每个步骤的完整输出。即每一次迭代都会得到一个完整的结果对象。
适用场景: 适用于需要每一步的完整结果进行进一步处理的场景。
示例: 如果有一个复杂的计算过程，每一步都需要完整的上下文来进行下一步的计算，那么 values 模式是合适的选择。

stream_mode="updates"：
在 updates 模式下，数据流会返回每个步骤的增量更新。即每一次迭代只会得到自上一次迭代以来的变化部分。
适用场景: 适用于需要实时更新或增量更新的场景，比如实时显示处理进度或逐步输出的场景。
示例: 构建一个实时聊天应用程序，需要逐步显示对话内容，那么 updates 模式是合适的选择。
'''
#运行图，流式输出，直到第一个中断点
for event in graph.stream(initial_input, thread, stream_mode="values"):
	print(event)
	
	
#一旦程序到达断点，开发人员可以通过以下方式获取用户输入并更新图的状态
try:
	user_input = input("Tell me how you want to update the state: ")
except :
	user_input ="go to step 3!"
	
#更新状态
graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

#检查状态
print("--State after update-.")
print(graph.get_state(thread))

#检查下一个节点
print(graph.get_state(thread).next)

'''
中断非常强大且易于使用。在体验上类似于 Python 的 input() 函数，但要注意，它不会自动从中断点恢复执行。
相反，它们会重新运行使用中断的整个节点。
因此，中断通常最好放置在节点的开头或一个专用的节点中。
'''
#用人类的输入恢复图形
#传递值到图
graph.invoke(Command(resume=user_input), config=thread)
#更新图的状态
graph.invoke(Command(update={"foo":"bar"}, resume=user_input), config=thread)

