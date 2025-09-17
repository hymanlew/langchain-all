'''
send 机制
默认情况下，Nodes 和 Edges 是提前定义的，并对同一个共享状态进行操作。但某些情况下，确切的边可能无法提前知道，或者希望同时路由到多个 node 节点
一个常见的例子是 map-reduce 批处理模式，其中第一个节点可能会生成一个对象列表，并希望对所有这些对象应用另一个节点。而对象的数量可能事先未知(这意味着边的数量可能未知)，
并且输入 state 到下游 Node 应该是不同的(每个生成的对象对应一个 node)。

为了支持这种设计模式，LangGraph 支持从条件边返回 Send 对象。Send 接受两个参数：第一个是节点的名称，第二个是要传递到该节点的状态。
Send 对象用于动态指定要调用的下游节点及其输入状态

官方文档地址：https://langchain-aigithub.io/langgraph/concepts/low_level/#send
中文文档地址：https://www.aidoczh.com/langgraph/concepts/low level/#send

第一个节点产生一个列表（Map），然后通过条件边函数为列表中的每个元素创建一个 Send 对象，动态调用下一个处理节点。这些节点并行处理各自分配到的数据，
最终的结果会被归约（Reduce）到整体状态中（例如代码中通过 Annotated[List[str], operator.add] 注解实现的 jokes 列表合并）。
'''
# 导入operator模块，用于后续操作
import operator
from typing import Annotated, List
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Send
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
	#条件边函数可以返回 Send 对象的列表，以实现向多个节点动态发送状态
	return [Send("generate_joke", {"subject": s}) for s in state['subjects']]


builder = StateGraph(OverallState)
builder.add_conditional_edges(START, continue_to_jokes)
builder.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subject']}"]})
builder.add_edge("generate_joke", END)

graph = builder.compile()
display_graph(graph)

# 调用graph对象，并传入包含两个主题的初始状态，结果是为每个主题生成一个笑话
result = graph.invoke({"subjects": ["cats","dogs"]})

# {'subjects': ['cats', 'dogs'], 'jokes': ['Joke about cats', 'Joke about dogs']}
print(result)


from langchain_core.runnables import Runnableconfig
from langgraph.constants impOrt START, END
from langgraph.graph import StateGraph
from langgraph.types import cachePolicy
from langgraph.cache.memory import InMemoryCache

#Node 是图中的一个处理数据的节点。也有以下几个需要注意的地方：
#在 LangCraph 中，Node 通常是一个Python的函数，它接受一个State对象作为输入，返回一个State对象作为输出。每个Node都有一个唯一的名称，通常是一个字符串。如果没有提供名称，LangGraph会自动生成一个和函数名一样的名称。
#在具体实现时，通常包含两个具体的参数，第一个是State，这个是必选的。第二个是一个可选的配置项config。这里面包含了一些节点运行的配置参数。
class State(TypedDict):
	number: int
	user id:str

class ConfigSchema(TypedDict):
	user_id: str

def node_1(state:State, config: Runnableconfig):
	time.sleep(3)
	user_id = config["configurable"]["user_id"]
	return {"number":state["number"]+ 1, "user_id":user_id}

builder = StateGraph(State, config_schema=ConfigSchema)

#LangGraph对每个Node提供了缓存机制。只要Node的传入参数相同，LangGraph就会优先从缓存当中获取Node的执行结果。从而提升Node的运行速度
graph = builder.compile(cache=InMemoryCache)

#对于Node，LangGraph除了提供缓存机制，还提供了重试机制可以针对单个节点指定，例如:
from langgraph.types import RetryPolicy
builder.add_node("node1", node1, retry=RetryPolicy(max_attempts=4))

#另外，也可以针对某一次任务调用指定，例如
print(graph.invoke(xxxxx, config={"recursion_limit":25}))
