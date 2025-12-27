import asyncio
from operator import add
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, MessagesState


class State(MessagesState):
    message: Annotated[list[str], add]

# 创建图
builder = StateGraph(State)

def node1(state):
    print("Node 1 executed")
    return {"message": ["from node1"]}

def node2(state):
    print("Node 2 executed")
    return {"message": ["from node2"]}

builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.set_entry_point("node1")
builder.add_edge("node1", "node2")
builder.add_edge("node2", END)

graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["node2"])

# 第一次执行，会在 node2 之前中断
config = {"configurable": {"thread_id": "thread1"}}

async def execute_graph(user_input: str)-> str:
    """执行工作流的函数"""
    result = '' # AI助手的最后一条消息
    if user_input.strip().lower() != 'y': # 正常的用户提问 或 拒绝
        current_state = graph.get_state(config)
        # 当工作流被中断时，它会停在某个节点之前或之后，此时 next 会包含接下来要执行的节点（即中断点之后的节点）。
        # 所以如果 next 不为空，说明工作流被中断了，还有节点等待执行。应该继续执行。
        if current_state.next:
            # 在继续中断的工作流时，我们不需要输入新的消息，因为中断的工作流已经在等待某个节点的执行，而不是等待用户输入。所以，我们使用 None 作为输入，并传入之前的配置（config）来继续。
            graph.update_state(config, {"message": ["skip node2"]}, as_node='node2')
            # graph.update_state(config, {"message": ["update node1"]})
            async for chunk in graph.astream(None, config, stream_mode="values"):
                print(chunk)
        else:
            # 如果没有下一步（即 next 为空），说明工作流已经结束或还没有开始（第一次调用时）。在这种情况下，你使用 app.astream 来启动一个新的流（或者继续一个没有中断的工作流？）。
            # 不是可等待对象(Awaitable)，因此不能直接用于await 表达式，必须通过 async for 继续执行
            async for chunk in graph.astream({'messages':('user', user_input)}, config, stream_mode="values"):
                print(chunk)

    else: #用户输入了y 想继续工具的调用
        async for chunk in graph.astream(None, config, stream_mode="values"):
            print(chunk)

    return result

async def main():
    while True:
        user_input = input('用户:')
        res = await execute_graph(user_input)
        # print(f'=== {res} ===')


# 执行工作流
if __name__ == "__main__":
    asyncio.run(main())

