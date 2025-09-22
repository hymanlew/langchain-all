"""
Human-in-the-loop(HIL)
文档：https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/
Agent 系统中"人机交互"(Human-in-the-loop，HIL) 模式在处理需要用户反馈的场景中非常有用，比如：
1，在任务执行过程中向用户询问澄清性问题。
2，审查工具调用，人类可以在工具执行之前审查、编辑或批准 LLM 请求的工具调用。
3，验证 LLM 输出，人类可以审查、编辑或批准 LLM 生成的内容。
4，提供上下文，使 LLM 能够明确请求，明确人类的输入以进行澄清或提供额外细节，或支持多轮对话。

建立一个节点以获取用户反馈。在这个过程中，开发人员需要:
- 设置断点: 通过 interrupt_before: 指定在特定节点之前中断图的执行,
- 设置检查点: 使用 MemorySaver 来保存图的状态，以便在用户输入后恢复。
- 更新状态: 使用 .update_state 方法，将用户的反馈更新到图的状态中
"""
from typing import TypedDict
from typing_extensions import TypedDict as TypedDictExt
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

class State(TypedDict):
    input: str
    user_feedback: str
    llm_generated_summary: str
    human_reviewed_summary: str
    original_text: str
    
    
def step_1(state):
    print("---Step 1---")
    return {"llm_generated_summary": "这是LLM生成的摘要内容"}

'''
interrupt函数:
在 LangGraph 中在特定节点暂停图形、向人类展示信息以及使用他们的输入恢复图形，从而启用人工干预工作流。
该函数对于批准、编辑或收集额外输入等任务非常有用。
interrupt 函数与 Command 对象结合使用，以人类提供的值恢复图形.
'''
def human_node(state: State):
    result = interrupt(
        #调用 interrupt 发起中断，并传递任何可序列化为 JSON 的值，可以展示给客户端供人类查看。
        #这里是等待人类手动输入的，例如一个问题、一段文本或状态中的一组键
        {
            "task": "审查 LLM 的输出并进行必要的编辑。",
            "llm_generated_summary": state["llm_generated_summary"]
        }
    )

    #interrupt() 返回的是将用户输入通过 Command 封装后的响应（字符串或字典）
    #假设用户返回的是修改后的摘要字符串
    # result = state["user_feedback"]
    if isinstance(result, dict):
        reviewed_summary = result.get("reviewed_text")
    else:
        reviewed_summary = result

    #使用编辑后的文本，或人类的输入更新状态或根据输入调整图形
    #可以根据响应的 result，来判断执行 SQL 操作，生成摘要等等
    # return Command(goto="call_llm", update={"message":"xxx"}) if True else return None
    return {"human_reviewed_summary": reviewed_summary}


def step_3(state):
    print("---Step 3---")
    print(f"最终结果: {state['human_reviewed_summary']}")
    return {}

# 构建图
builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_node)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

'''
LangGraph有一个内置持久化层，是通过检査点实现（提供内存/持久化）。将检査点与图形一起使用时，两者可以对状态进行交互。检查点在每个步骤中保
存图形状态的检查点，从而实现一些强大的功能。
这些快照被保存到一个 线程 中，并可持久化到内存，redis，数据库中，执行图形后可以访问它们。
由于 线程 允许在执行后访问图的状态，包括人机交互、内存、时间旅行和容错，都是可能的。
首先，检查点允许人类检查、中断和批准步骤来促进人机交互工作流工作流。因为人类必须能够在任何时候查看图形的状态，并且图形必须能够在人类对
状态进行任何更新后恢复执行。
其次，它允许在交互之间进行"记忆"，可以使用检查点创建线程并在图形执行后保存线程的状态。在重复的人类交互(例如对话)的情况下，任何后续消息
都可以发送到该检查点，该检查点将保留对其以前消息的记忆。
'''
#使用异步上下文管理器创建一个 AsyncSqliteSaver 对象，并连接到名为 "checkpoints.db"的 SQLite 数据库
#以上是连接的数据库，还可以连接 内存-MemorySaver, SqliteSaver, RedisSaver, mongodb, PostgresSaver 等等数据库，导入相关库即可
saver = AsncSgliteSaver.from_conn_string("checkpoints.db")
# saver = PostgresSaver.from_conn_string("postgresql://user:password@localhost:5432/your_database")
# from psycopg_pool import ConnectionPool
# pool = ConnectionPool(conninfo="postgresql://user:password@localhost:5432/your_database", max_size=20) # 设置连接池大小
# saver = PostgresSaver(sync_connection=pool)
# saver.setup()  # 可选：自动创建所需的表

#设置内存检查点
memory = MemorySaver()

"""
检查点（Checkpoint）机制的核心作用是保存状态图执行过程中的状态快照，包含了当前所有状态通道（State Channels）的值、下一步要执行的节点信息以及相关的元数据。
在生产环境，优先使用官方推荐的 PostgresSaver。如果想使用 MySQL 作为持久化存储，官方当前不支持，只能自定义实现：实现 BaseSaver 接口
"""
#编译图并设置断点
graph = builder.compile(checkpointer=memory/saver)

# 查看图的结构 (仅在IPython环境中可用)
# display(Image(graph.get_graph().draw_mermaid_png()))


'''
流式输出模式：
stream_mode="values":
在 values 模式下，数据流会返回每个步骤的完整输出。即每一次迭代都会得到一个完整的结果对象。
适用场景: 适用于需要每一步的完整结果进行进一步处理的场景。
示例: 如果有一个复杂的计算过程，每一步都需要完整的上下文来进行下一步的计算，那么 values 模式是合适的选择。

stream_mode="updates"：
在 updates 模式下，数据流会返回每个步骤的增量更新。即每一次迭代只会得到自上一次迭代以来的变化部分。
适用场景: 适用于需要实时更新或增量更新的场景，比如实时显示处理进度或逐步输出的场景。
示例: 构建一个实时聊天应用程序，需要逐步显示对话内容，那么 updates 模式是合适的选择。

custom：从图节点内部流式传输自定义数据。通常用于调试。
可以自定义输出内容。在Node节点内或者Tod1s工具内，通过get_stream_writer()方法获取一个StreamWriter对象，然后使用write()方法将自定义数据写入流中。

messages：从任何调用大语言模型(LLM)的图节点中，流式传输二元组(LLM的Token，元数据)。

debug：在图的执行过程中尽可能多地传输信息。用得比较少。
'''
# 初始输入，例如用户的初始问题
user_input = None
initial_state = {"original_text": "这是一段很长的文本...", "llm_generated_summary": None, "human_reviewed_summary": None}

# 4. 主循环：运行图并处理中断
config = {"configurable": {"thread_id": "example_thread_1"}} # 线程ID，用于标识不同的执行会话
current_input = None # 初始输入，例如用户的初始问题
# 假设这是初始状态，或者从某个地方获取
initial_state = {"original_text": "这是一段很长的文本...", "llm_generated_summary": None, "human_reviewed_summary": None}

while True:
    interrupted = False
    # 使用 graph.stream 以流式方式执行，直到第一个中断点，便于捕获中断事件
    for event in graph.stream(initial_state if current_input is None else current_input, config, stream_mode="values"):
        # 1. 检查是否是中断事件
        if "__interrupt__" in event:
            interrupt_event = event["__interrupt__"][0] # 提取中断信息
            print(f"\n中断发生: {interrupt_event.value}") # 打印中断时传递的信息

            # 2. 模拟等待用户输入（在生产环境中，可能是从Web接口、队列等获取）
            try:
                user_response = input("请输入您的审查意见或修改后的摘要: ")
            except :
                user_response ="go to step 3!"

            # 3. 将用户的输入封装成 Command 对象，准备用于恢复执行
            # Command(resume=...) 中的值会被传递给 human_review_node 中的 interrupt() 调用
            current_input = Command(resume=user_response)
            interrupted = True

            #更新状态，并将用户输入保存为 user_feedback
            #注意，as_node 操作会直接替代 human_feedback 中断后的动作，即不会执行。而是直接执行下一节点
            #所以可以不添加此行代码，让节点内部读取反馈，然后继续执行
            # graph.update_state(config, {"user_feedback": user_response}, as_node="human_feedback")

            #检查状态
            print("--State after update--")
            print(graph.get_state(config))

            #检查下一个节点
            print(graph.get_state(config).next)
            break # 发生中断，跳出当前的事件循环

        else:
            # 如果是正常的状态更新，可以打印或处理
            print(f"状态更新: {event}")
            # 如果遇到结束信号或最终状态，也可以考虑跳出循环
            # if some_condition: break

    if not interrupted:
        # 如果没有发生中断，说明Graph执行完毕
        print("任务执行完毕！")
        break # 退出主循环


'''
中断非常强大且易于使用。在体验上类似于 Python 的 input() 函数，但要注意，它不会自动从中断点恢复执行。
相反，它们会重新运行使用中断的整个节点。
因此，中断通常最好放置在节点的开头或一个专用的节点中。

以上整个流程如下：
- 执行与中断：graph.stream 开始执行，运行到 human_review_node 中的 interrupt() 调用时，Graph 引擎会暂停在该节点，并抛出一个 __interrupt__ 事件。
- 捕获与等待：主循环（for event in graph.stream...）捕获到这个特殊事件，然后等待并获取用户输入（user_response = input(...)）。
- 准备恢复：将用户的输入包装成一个 Command(resume=...) 对象，赋值给 current_input。然后跳出当前事件循环（for 循环）
- 再次调用与恢复：由于外层的 while True 循环，代码会再次执行 graph.stream(current_input, config, ...)。这次，Graph 引擎会拿着 Command 对象回到上次中断的地方，interrupt() 调用此时会返回 user_response，然后 human_review_node 节点继续执行剩下的代码（即 return ...）。
- 继续执行：节点执行完毕后，Graph 会继续自动执行后面的节点，直到再次中断或结束。
'''

from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    writer = get_stream_writer()
    writer({"自定义key": "在节点内返回自定义信息"})
    return {"answer": "some data"}

graph = (
	StateGraph(State)
	.add_node(node)
	.add_edge(START, "node" )
	.compile()
)
inputs = {"query": "example"}

# custom 表示自定义的信息不会存入 state 中，并且图中每个节点执行的结果就是 writer 的信息并返回，而不是 state 执行结果。
# 这个设置主要用于调试图节点的执行过程，通过输出每一步的自定义信息来判断流程是否正常。
# 如果要正常输出节点的结果，即正常执行图流程，改为正常的 stream_mode 即可。
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)

