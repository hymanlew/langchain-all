'''
send 机制
默认情况下，Nodes 和 Edges 是提前定义的，并对同一个共享状态进行操作。但某些情况下，确切的边可能无法提前知道，或者可能希望同时存在 state 的不同版本。
一个常见的例子是 map-reduce 批处理模式，其中第一个节点可能会生成一个对象列表，并希望对所有这些对象应用另一个节点。而对象的数量可能事先未知(这意味着边的数量可能未知)，
并且输入 state 到下游 Node 应该是不同的(每个生成的对象对应一个 node)。
为了支持这种设计模式，LangGraph 支持从条件边返回 Send 对象。Send 接受两个参数：第一个是节点的名称，第二个是要传递到该节点的状态。

官方文档地址：https://langchain-aigithub.io/langgraph/concepts/low_level/#send
中文文档地址：https://www.aidoczh.com/langgraph/concepts/low level/#send

以下是:map-reduce 工作流程图
'''
# 导入operator模块，用于后续操作
import operator
from typing import Annotated, List
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import Send
from langgraph.graph import END, START
from display_graph import display_graph

#定义一个名为 OverallState 的 TypedDict 类
class OverallState(TypedDict):
	subjects: list[str]
	# jokes是一个带有 operator.add 注解的字符串列表
	jokes: Annotated[List[str], operator.add]
	
	
#定义一个函数 continue_to_jokes，接受一个 OverallState 类型的参数 state
def continue_to_jokes(state: OverallState):
	#返回一个 Send 对象列表，每个对象指定一个处理节点的名称，和对应主题的字典
	return [Send("generate_joke", {"subject": s}) for s in state['subjects']]
	
	
builder = StateGraph(OverallState)
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject' ]}"]})
builder.add_edge("generate_joke", END)

graph = builder.compile()
display_graph(graph)

# 调用graph对象，并传入包含两个主题的初始状态，结果是为每个主题生成一个笑话
result = graph.invoke({"subjects": ["cats","dogs"]})

# {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
print(result)

