import asyncio
from pydantic import BaseModel

# Ragas 提供了修改或替换默认提示为自定义提示词，用于评估的方式。但一般使用默认即可
from ragas.prompt import PydanticPrompt

# 导入评估相关模块
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.dataset_schema import MultiTurnSample
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.metrics import ToolCallAccuracy, AgentGoalAccuracyWithReference
import ragas.messages as r

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
# from langchain.agents.tool_node import ToolNode
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from typing import Annotated


metal_price = {
    "gold": 88.1553,
    "silver": 1.0523,
    "platinum": 32.169,
}

@tool
def get_metal_price(metal_name: str) -> float:
    """Fetches the current per gram price of the specified metal."""
    try:
        metal_name = metal_name.lower().strip()
        if metal_name not in metal_price:
            raise KeyError(
                f"Metal '{metal_name}' not found. Available metals: {', '.join(metal_price['metals'].keys())}"
            )
        return metal_price[metal_name]
    except Exception as e:
        raise Exception(f"Error fetching metal price: {str(e)}")


class GraphState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]


def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Define the function that calls the model
def call_model(state: GraphState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Node
def assistant(state: GraphState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = [get_metal_price]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

builder = StateGraph(GraphState)
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", should_continue, ["tools", END])
builder.add_edge("tools", "assistant")

react_graph = builder.compile()
react_graph.get_graph(xray=True).draw_mermaid_png()


async def main():
    # 我们将使用一个查询运行Agent。Agent将使用metals.dev API获取铜的价格。
    messages = [HumanMessage(content="What is the price of copper?")]
    result = react_graph.invoke({"messages": messages})
    print(result)

    # 将LangChain消息列表（例如，HumanMessage、AIMessage和ToolMessage）转换为Ragas期望的格式，以便评估框架能够正确理解和处理它们
    ragas_trace = convert_to_ragas_messages(result["messages"])

    # 工具调用准确性：ToolCallAccuracy 指标用于评估LLM识别和调用所需工具以完成给定任务的性能。
    # 返回一个二元指标，1表示AI已实现目标，0表示AI未实现目标。
    sample = MultiTurnSample(
        user_input=ragas_trace,
        reference_tool_calls=[
            r.ToolCall(name="get_metal_price", args={"metal_name": "copper"})
        ],
    )
    tool_accuracy_scorer = ToolCallAccuracy()
    await tool_accuracy_scorer.multi_turn_ascore(sample)


    messages = [HumanMessage(content="What is the price of 10 grams of silver?")]
    result = react_graph.invoke({"messages": messages})
    ragas_trace = convert_to_ragas_messages(result["messages"])

    sample = MultiTurnSample(
        user_input=ragas_trace,
        reference="Price of 10 grams of silver",
    )
    # Agent目标准确性：Agent goal accuracy 指标用于评估LLM识别和实现用户目标的性能。
    # 返回一个二元指标，1表示AI已实现目标，0表示AI未实现目标。
    scorer = AgentGoalAccuracyWithReference()
    scorer.llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    await scorer.multi_turn_ascore(sample)


if __name__ == "__main__":
    asyncio.run(main())
