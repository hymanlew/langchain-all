from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
import datetime
from typing import List, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph


"""
Reflexion 模式是通过自我反思（Self-Reflection）迭代改进输出（生成答案 → 评估质量 → 结合外部反馈调整策略 -> 修正错误），
模拟人类从错误中学习的能力，让 Agent 具备动态记忆和自我反思能力，以提高推理能力的框架。
Reflexion 的核心思想是让智能体在生成回答后进行自我批判和反思，然后根据反思结果进行信息检索，进而修订答案，如此迭代多次以提高答案质量。
"""
class State(TypedDict):
    messages: List[BaseMessage]


# 用于存储对当前答案的批评意见（缺失和多余的部分）
class Reflection(BaseModel):
    # 描述对当前回答中缺失部分的批评意见
    missing: str = Field(description="Critique of what is missing.")
    # 描述对当前回答中多余部分的批评意见
    superfluous: str = Field(description="critique of what is superfluous")


# 用于存储初始回答，包括答案文本、反思对象和搜索查询列表
class AnswerQuestion(BaseModel):
    """Answer the question. provide an answer, reflection, and then follow up with search queries to improve the answer."""
    # 回答问题。提供答案、反思，然后使用搜索查询来改进答案
    # 250 个字的回答
    answer: str = Field(description="~250 word detailed answer to the question.")
    # 对你初始答案的反思
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    # 1-3个搜索查询，用于研究改进方案 以解决 对你当前答案的批评意见
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

# 负责调用运行链（runnable）并验证输出
class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke({"messages": state["messages"]}, {"tags": [f"attempt: {attempt}"]})
            try:
                # invoke 方法是调用解析器来处理 LLM 的响应。检查响应匹配指定的 BaseModel 模型的模式
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                # 如果输出不符合指定的Pydantic模型（即验证失败），则会重试（最多3次，3次循环）
                # 每次重试时，它会将之前的错误信息作为工具消息，添加到消息历史中，以便模型能够根据错误调整输出
                state["messages"] = state["messages"] + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\n Pay close attention to the function schema.\n\n" +
                                self.validator.schema_json() +
                                " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return {"messages": response}


# 初始问题模板
"""
system, 你是一位专业的研究员。当前时间：{time}
    1. 第一条指令
    2. 反思并批判你的答案。要严苛以最大化改进效果。
    3. 推荐搜索查询以获取最新的研究信息并改进你的答案。
    
user, 反思用户的原始问题和迄今为止采取的行动。使用(function_name)函数（AnswerQuestion 结构）进行回应
"""
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher. Current time: {time}
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>Reflect on the user's original question and the"
            " actions taken thus far. Respond using the (function_name) function.</system>",
        ),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat(),)

llm = ChatOpenAI(model="gpt-4o")

# 初始请求链
"""
LangChain 机制：bind_tools() 方法的作用是让LLM 按照指定格式生成结构化输出。AnswerQuestion 类通过 Field 定义了输出模板，
LLM 会生成符合该结构的 JSON（类似 OpenAI 的 Function Calling）。
也可以在 bind_tools 方法中，添加参数 tool_choice="AnswerQuestion"  # 显式指定格式

即 bind_tools 的用途是双重的，具体行为取决于传入的参数类型：
- 传入 Tool 或 StructuredTool 的实例时，是绑定工具进行调用
- 传入 BaseModel 的子类（通过 Field 定义字段）时，绑定输出格式（Pydantic类）
"""
initial_answer_chain = actor_prompt_template.partial(
    first_instruction="provide a detailed ~250 word answer.",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])

# 初始验证器
# PydanticToolsParser 是 LangChain 中的一个工具，用于解析 LLM 的输出，使其符合 Pydantic 模型定义的格式。
# 它接收一个工具列表（通常是 Pydantic 模型类），并解析 LLM 的响应，确保响应匹配这些模型的模式。
validator = PydanticToolsParser(tools=[AnswerQuestion])

# 第一次请求结果
first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)

# Revision，反省模板，评价，修正
"""
使用新信息修改你之前的答案。
-你应该使用之前的评论为你的答案添加重要信息。
-你必须在修改后的答案中包含数字引用，以确保它可以被验证。
-在答案底部添加“参考文献”部分（不计入字数限制）。形式如下：
1.https://example.com
2.https://example.com
-你应该用之前的评论从你的答案中删除多余的信息，并确保它不超过250个单词
"""
revise_instructions = """Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# 根据新信息修订答案，并添加文献引用字段 references
class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,
cite your reflection with references, and finally add search queries to improve the answer."""
    references: list[str] = Field(description="citations motivating your updated answer.")

# 反省动作链
revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer])

revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

# 反省后处理的结果
revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)


# 构造搜索工具，并限制每次搜索返回5个结果
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

def run_queries(search_queries: List[str], **kwargs):
    """Run the generated queries."""
    # 每个搜索查询会返回一个结果列表（且每个查询最多返回 5 个结果）。每个结果是一个字典，包含标题、URL、内容等信息（取决于 Tavily API的响应）
    # 这些搜索结果会被 ToolNode 自动包装成 ToolMessage 对象，并添加到消息历史中。
    # ToolMessage 包含工具调用的结果，供后续节点（如 revise 节点）使用
    return tavily_tool.batch([{"query": query} for query in search_queries])

tool_node = ToolNode(
    [
        # 将搜索函数包装成结构化工具，并分别用 AnswerQuestion和ReviseAnswer 命名（实际上这两个工具的功能相同，都是执行搜索）
        # 并根据 初始回答或修订回答 的调用请求，调用对应的工具
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)

MAX_ITERATIONS = 5
builder = StateGraph(State)
builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

# 迭代次数计算函数
def get_num_iterations(messages: List[BaseMessage]) -> int:
    """通过工具消息数量判断迭代次数"""
    return sum(1 for msg in messages if isinstance(msg, ToolMessage))

# 更新事件循环逻辑
def event_loop(state: dict) -> str:
    # 在我们的例子中，我们将在 N 次计划后停止
    num_iterations = get_num_iterations(state["messages"])
    return END if num_iterations >= MAX_ITERATIONS else "execute_tools"

# 根据状态决定是否继续循环，revise -> execute-tools OR end
builder.add_conditional_edges(
    "revise",
    event_loop  # 传递完整状态字典
)

# 设置初始节点
builder.set_entry_point("draft")
graph = builder.compile()

"""
整个流程如下：
1，初始回答：智能体首先根据用户问题生成一个初始回答（约250字），同时进行自我反思，指出回答中缺失和多余的部分，并提出1-3个搜索查询用于改进答案。
2，执行搜索：根据生成的搜索查询，使用搜索工具获取新的信息。
3，修订答案：智能体根据搜索得到的新信息以及之前的反思，修订初始答案。修订要求包括：
  - 使用之前的反思来增加重要信息，删除多余信息。
  - 必须在修订的答案中加入数字引用（如[1]、[2]）以确保可验证性。
  - 在答案底部添加“参考文献”部分（不计入字数限制）。
4，迭代：重复步骤2和3，直到达到最大迭代次数（本例中为5次）或满足其他终止条件。


在这个示例中，我们首先创建了一个 MessageGraph 实例，然后添加了三个节点: draft、execute_tools、revise。
这些节点分别对应于 Refexion 流程中的生成、执行工具和修订步骤。
我们设置了初始节点为 draft，并定义了节点之间的边来指定执行顺序。最后，我们定义了一个循环逻辑，当迭代次数超过最大值时，流程将结束。

执行流程为：user request --> responder --> tools --> revise --> tools --> revise --> ... --> END
"""



"""
Time Travel 时间回溯：
由于大语言模型回答问题的不确定性，基于大语言模型构建的应用，也是充满不确定性的。就有必要进行更精确的检查，当某一个步骤出现问题时，才能及时发现问题并处理，再进行重演。
LangGraph提供了TimeTravel时间回溯功能，可以保存Graph的运行过程，并可以手动指定从Graph的某一个Node开始进行重演。

具体实现时，需要注意以下几点:
运行时，指定thread_id，checkpoint检查点，graph 将在每一个Node执行后，生成一个check_point_id.
重演时，指定thread_id和check_point_id 进行任务重演。
重演前，可以选择更新state，当然，如果没问题，也可以不指定。
"""
# 构建一个图。图中两个步骤:第一步让大模型推荐一个有名的作家，第二步，让大模型用推荐的作家的风格写一个100字以内的笑话。
# Time Travel 示例代码（注释掉以避免直接执行）
"""
llm = ChatTongyi(
    model="qwen-plus",
    api_key="YOUR_API_KEY"  # 需要替换为实际的API密钥
)

class State(TypedDict):
    author: NotRequired[str]
    joke: NotRequired[str]

def author_node(state: State):
    prompt = "帮我推荐一位受人们欢迎的作家。只需要给出作家的名字即可。"
    author = llm.invoke(prompt)
    return {"author": author.content}

def joke_node(state: State):
    prompt = f"用作家: {state['author']} 的风格，写一个100字以内的笑话"
    joke = llm.invoke(prompt)
    return {"joke": joke.content}

builder = StateGraph(State)
builder.add_node("author_node", author_node)
builder.add_node("joke_node", joke_node)
builder.add_edge(START, "author_node")
builder.add_edge("author_node", "joke_node")
builder.add_edge("joke_node", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
state = graph.invoke({}, config)
print(state["author"])
print(state["joke"])

# 查看所有checkpoint检查点，输出每一个Node执行后，生成的 check_point_id.
states = list(graph.get_state_history(config))
for state in states:
    print(state.next)
    print(state.config["configurable"]["checkpoint_id"])
    print()

# 选定某一个检查点。这里选择 author_node，让大模型重新推荐作家
selected_state = states[1]
print(selected_state.next)
print(selected_state.values)

# 为了后面的重演，需要更新state。可选步骤:
new_config = graph.update_state(selected_state.config, values={"author": "郭德纲"})
print(new_config)

# 接下来，指定thread_id和checkpoint_id，进行重演
graph.invoke(None, new_config)

for event in graph.stream(None, new_config):
    for v in event.values():
        print(v)
"""
