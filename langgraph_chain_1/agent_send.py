"""
| 特性         | **`Send()` 指令**                        | **`Command` 指令**                                           |
| **核心作用** | 动态创建并分发并行任务 (Map-Reduce)          | 恢复被中断的工作流或跳转到指定节点                           |
| **触发方式** | 由条件边函数**自动返回**                     | 在调用 `graph.invoke()` 或 `graph.stream()` 时**手动传入**   |
| **工作机制** | 根据数据（如列表）动态生成多个任务分支，并行执行 | 与 `interrupt()` 函数配合，用于提供恢复所需的值或指定下一个节点 |
| **应用场景** | 批量处理文档、并行调用API、多主题内容生成      | 人工审核、编辑状态、多轮对话验证等“人在环路”场景             |
"""
import json
import operator
import os
from typing import TypedDict, Annotated

from langgraph.config import get_stream_writer
from langgraph.types import Send
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from chain_graph_1.display_graph import display_graph


# 定义状态
class State(TypedDict):
    item: str
    items: list[str]
    results: Annotated[list[str], operator.add]

# 条件边函数：为每个item创建并行任务
def map_router(state: State):
    writer = get_stream_writer()
    writer({"router": state["items"]})

    # 需要注意，Send() 不是条件边返回值，而是节点内部指令
    # Send 是用于在节点内部动态创建并行任务的指令（Map-Reduce模式），返回的是 Send 对象（而不是作为条件边的返回值），所以这里不是图的条件边。
    # 在图中，条件边函数应该返回下一个节点的名称（字符串）或包含节点名称的列表。如此才能展示并关联到图中
    # 为每个item创建并行任务
    return [Send("process_item", {"item": s}) for s in state['items']]

def process_item(state: State):
    # get_stream_writer()用于在节点执行过程中流式输出数据，但需要配合适当的流式调用方式才能看到输出
    writer = get_stream_writer()
    writer({"process": state["item"]})
    return {"results": [f"Processed {state['item']}"]}


# 构建图
# process_item 节点是通过 Send 动态调用的，但它不是静态图的一部分，所以不会在图中显示。
builder = StateGraph(State)
builder.add_node("map_router", map_router)
builder.add_node("process_item", process_item)

# 这里必须使用 add_conditional_edges，因为 map_router 是动态的创建并行任务。而 add_edge 是静态边。
# builder.add_edge(START, "map_router")
builder.add_conditional_edges(START, map_router)
# map_router的条件边：可以返回节点名或Send()列表
builder.add_conditional_edges("map_router", process_item)
# 处理节点后汇聚到END
builder.add_edge("process_item", END)

graph = builder.compile()
display_graph(graph)

# 执行：两个任务会并行处理
inputs = {"items": ["cat", "dog", "bird"]}
for step in graph.stream(inputs, config={"thread_id": 1}, stream_mode="values"):
    # node, output = list(step.items())[0]
    # print(f"[节点 {node}] → {output}")
    print(step)
# 结果: {'items': ['cat', 'dog'], 'results': ['Processed cat', 'Processed dog']}


def human_review_node(state):
    # 执行到此处，工作流会暂停并等待人工输入
    human_input = interrupt("请审核此内容")
    # 当通过Command恢复后，human_input会获得传入的值
    return {"reviewed_output": human_input}

# ... 构建图并启用检查点（必须）...
checkpointer = MemorySaver()

builder.add_node(human_review_node)
builder.add_edge("process_item", 'human_review_node')
builder.add_edge('human_review_node', END)
graph = builder.compile(checkpointer=checkpointer)

# 第一次调用，会在interrupt处暂停
# graph.invoke(...)

# 人工审核后，使用Command携带结果恢复执行
resume_command = Command(resume="审核通过，可以继续")
final_result = graph.invoke(resume_command, config={"thread_id": 1})
