# Import necessary components
import os
from langgraph.errors import NodeInterrupt
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from display_graph import display_graph


@tool
def check_cpu_usage(tool_input: int):
    """Simulates checking the CPU usage of the server."""
    return "CPU Usage is 90%"


@tool
def check_disk_space(tool_input: int):
    """Simulates checking the available disk space on the server."""
    return "Disk space is below 15%"


@tool
def restart_server(tool_input: bool):
    """Simulates restarting the server."""
    return "Server restarted successfully"


# Define the human feedback tool (for confirming server restart)
class AskHuman(BaseModel):
    """Ask the human whether to restart the server."""
    question: str


# Set up the tools and tool node
model = ChatOpenAI(model="gpt-4o")
tools = [check_cpu_usage, check_disk_space, restart_server]
tool_node = ToolNode(tools)
model = model.bind_tools(tools + [AskHuman])


# Function to decide the next step based on the last message
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]

    # If no tool call, finish the process
    if not last_message.tool_calls:
        return "end"

    # If the tool call is AskHuman, return that node
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"

    # Otherwise, continue the workflow
    else:
        return "continue"

# Function to call the model and return the response
def call_model(state):
    input_length = len(state["input"])
    if input_length > 10:
        raise NodeInterrupt("Input length {input_length} exceeds threshold of 10.")

    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the human interaction node
def ask_human(state):
    pass  # No actual processing here, handled via breakpoint


# Create the state graph
workflow = StateGraph(MessagesState)
# Define the nodes for the workflow
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_node("ask_human", ask_human)

# Set the starting node
workflow.add_edge(START, "agent")

# Define conditional edges based on the agent's output
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",  # Proceed to the tool action
        "ask_human": "ask_human",  # Ask human for input
        "end": END,  # Finish the process
    }
)

# Add the edge from action back to agent for continued workflow
workflow.add_edge("action", "agent")
# Add the edge from ask_human back to agent after human feedback
workflow.add_edge("ask_human", "agent")

# Set up memory for checkpointing
memory = MemorySaver()

# Compile the graph with a breakpoint before ask_human
# interrupt_before/interrupt_after，旧版本，主要用于调用第三方定义的节点前
app = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])

# Visualize the workflow
display_graph(app, file_name=os.path.basename(__file__))

# Initial configuration and user message
config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(
    content="Check the CPU usage and disk space of the server, and restart it if necessary."
)

# Start the interaction with the agent
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Get the ID of the last tool call (AskHuman tool call)
tool_call_id = app.get_state(config).values["messages"][-1].tool_calls[0]["id"]

# Ask the user whether they want to approve the server restart
user_input = input("Do you want to restart the server? (yes/no): ")

# Create the tool response message based on actual user input
tool_message = [
    {"tool_call_id": tool_call_id, "type": "tool", "content": user_input}  # Use real user input
]

# Update the state as if the response came from the user
app.update_state(config, {"messages": tool_message}, as_node="ask_human")

for event in app.stream(None, config, stream_mode="values"):
    event["messages"][-1].pretty_print()



def get_answer(tool_message, user_answer):
    """让人工介入，并且给一个问题的答案"""
    tool_name = tool_message.tool_calls[0]['name']
    answer = f"人工强制终止了工具:{tool_name}的执行，拒绝的理由是:{user_answer}"

    # 创建一个消息
    new_message = [
        ToolMessage(content=answer, tool_call_id=tool_message.tool_calls[0]["id"]
        AIMessage(content=answer),
    ]

    # 把新人造的消息，添加到工作流的state中
    app.update_state(
        config=config,
        Values={'messages':new_message}
    )

async def execute_graph(user_input: str)-> str:
    """执行工作流的函数"""
    result = '' # AI助手的最后一条消息
    if user_input.strip().lower() != 'y': # 正常的用户提问 或 拒绝
        current_state = app.get_state(config)
        # 当工作流被中断时，它会停在某个节点之前或之后，此时 next 会包含接下来要执行的节点（即中断点之后的节点）。
        # 所以如果 next 不为空，说明工作流被中断了，还有节点等待执行。应该继续执行。
        if current_state.next:
            tools_script_message = current_state.values['messages'][-1]
            # 通过提供关于请求的更改/改变主意的指示 来满足工具调用
            get_answer(tools_script_message, user_input)
            message = app.get_state(config).values['messages'][-1]
            result = message.content
            return result
        else:
            # 如果没有下一步（即 next 为空），说明工作流已经结束或还没有开始（第一次调用时）。在这种情况下，你使用 app.astream 来启动一个新的流（或者继续一个没有中断的工作流？）。
            # 不是可等待对象(Awaitable)，因此不能直接用于await 表达式，必须通过 async for 继续执行
            async for chunk in app.astream({'messages':('user', user_input)}, config, stream_mode="values"):
                print(chunk)
            result = 'normal'

    else: #用户输入了y 想继续工具的调用
        # 在继续中断的工作流时，我们不需要输入新的消息，因为中断的工作流已经在等待某个节点的执行，而不是等待用户输入。所以，我们使用 None 作为输入，并传入之前的配置（config）来继续
        async for chunk in app.astream(None, config, stream_mode="values"):
            print(chunk)
        result = 'continue'

    current_state = app.get_state(config)
    if current_state.next:  # 出现了工作流的中断
        ai_message = current_state.values['messages'][-1]
        tool_name = ai_message.tool_calls[0]['name']
        # ai_message.tool_calls[]['args']
        result = f"AI助手马上根据你要求，执行{tool_name}工具。您是否批准继续执行?输)"

    return result

async def main():
    while True:
        user_input = input('用户:')
        res = await execute_graph(user_input)
        print(f'=== {res} ===')


# 执行工作流
if __name__ == "__main__":
    asyncio.run(main())
