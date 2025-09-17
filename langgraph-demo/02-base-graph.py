"""
在 LangGraph 的设计中，更常见和推荐的是使用 input_schema 和 output_schema 参数来明确指定输入和输出的状态模式（Schema）
"""
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage


# 1. 定义内部完整状态
class OverallState(TypedDict):
    foo: str           # 内部处理字段
    user_input: str    # 输入字段
    graph_output: str  # 输出字段

# 2. 定义输入状态模式 (可选)
class InputState(TypedDict):
    user_input: str

# 3. 定义输出状态模式 (可选)
class OutputState(TypedDict):
    graph_output: str

# 4. 创建状态图 - 推荐写法
builder = StateGraph(
    state_schema=OverallState,   # 内部完整状态
    input_schema=InputState,     # 输入限制模式
    output_schema=OutputState    # 输出限制模式
)
builder.compile()


# Define a tool to get the weather for a city
@tool
def get_weather(location: str):
    """Fetch the current weather for a specific location."""
    weather_data = {
        "San Francisco": "It's 60 degrees and foggy.",
        "New York": "It's 90 degrees and sunny.",
        "London": "It's 70 degrees and cloudy.",
        "Nairobi": "It's 27 degrees celsius and sunny."
    }
    return weather_data.get(location, "Weather information is unavailable for this location.")

message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "Nairobi"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)
tools = [get_weather]
tool_node = ToolNode(tools)
result = tool_node.invoke({"messages": [message_with_single_tool_call]})
print(result)

"""
在最新版本 langgraph 0.5 中，实际创建图节点时，推荐哪一种写法：

| 特性维度              | prompt + llm.bind_tools(tools) (推荐)                | 写法一: create_tool_calling_agent(llm, tools, prompt)        |
| :-------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **API 设计理念**      | **组合式，基于 LangChain Runnable 提供更大的灵活性和控制力。 | **声明式 (Declarative)** 属于高层预构建 API，为特定模式提供快速实现。 |
| **灵活性**            | **高** 可以轻松插入其他 Runnable 步骤（如自定义函数、输出解析器、条件逻辑等），构建复杂的工作流。 | **相对较低** 主要专注于“模型绑定工具并根据提示生成消息”这一特定模式，定制能力有限。 |
| **与 LangGraph 集成** | **更自然、更底层** Agent 本身就是一个 Runnable 对象，可以直接作为 LangGraph 的节点函数使用，或嵌入到更复杂的节点逻辑中。 | **需要额外包装** 通常需要将返回的 Agent 包装在特定的执行函数中（如 `agent_executor`），再作为节点。 |
| **控制力**            | **强** 可以清晰地看到数据流（prompt → model → tool binding），并完全控制每一步的输入输出。 | **较弱** 底层的一些实现细节被封装起来，调试和理解内部状态可能稍显不便。 |
| **适用场景**          | **需要精细控制的 LangGraph 节点** 尤其是需要自定义状态处理、错误处理或与其他 Runnable 组合时。 | **快速原型开发** 构建简单的、标准化的工具调用智能体，希望用最少的代码实现功能。 |
| **官方趋势**          | **LangChain 核心范式** 是官方积极推荐和主要演进的方向。      | **预构建方法** 仍然有效，但在追求高度定制和复杂集成的场景下，可能不是首选。 |

但需要注意：
使用 llm.bind_tools(tools) 创建的节点只是大模型推理，并生成含工具调用请求的 AIMessage（包含 tool_calls 字段），但不会自动调用工具，只负责生成工具调用请求。
而使用 ToolNode(tools) 创建的节点会自动调用工具，并生成 ToolMessage (包含工具执行结果)。
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import json
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from display_graph import display_graph
import os


class AgentState(TypedDict):
    """The state of the agent."""
    # `add_messages` is a reducer that manages the message sequence.
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def analyze_sentiment(feedback: str) -> str:
    """Analyze customer feedback sentiment with below custom logic"""
    from textblob import TextBlob
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0.5:
        return "positive"
    elif analysis.sentiment.polarity == 0.5:
        return "neutral"
    else:
        return "negative"


@tool
def respond_based_on_sentiment(sentiment: str) -> str:
    """Only respond as below to the customer based on the analyzed sentiment."""
    if sentiment == "positive":
        return "Thank you for your positive feedback!"
    elif sentiment == "neutral":
        return "We appreciate your feedback."
    else:
        return "We're sorry to hear that you're not satisfied. How can we help?"


tools = [analyze_sentiment, respond_based_on_sentiment]

llm = ChatOpenAI(model="gpt-4o-mini")
llm = llm.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def call_model(state: AgentState, config: RunnableConfig):
    system_prompt = SystemMessage(
        content="You are a helpful assistant tasked with responding to customer feedback."
    )
    response = llm.invoke([system_prompt] + state["messages"], config)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")
graph = workflow.compile()
display_graph(graph, file_name=os.path.basename(__file__))

# Initialize the agent's state with the user's feedback
initial_state = {"messages": [("user", "The product was great but the delivery was poor.")]}

# Helper function to print the conversation
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# Run the agent
print_stream(graph.stream(initial_state, stream_mode="values"))
