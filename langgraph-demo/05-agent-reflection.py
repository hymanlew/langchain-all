from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIwrapper
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, Messagesplaceholder
from pydantic import ValidationError
from pydantic import BaseModel, Field
from typing import List
import datetime

"""
Reflexion 模式是通过自我反思（Self-Reflection）迭代改进输出（生成答案 → 评估质量 → 结合外部反馈调整策略 -> 修正错误），
模拟人类从错误中学习的能力**，让 Agent 具备动态记忆和自我反思能力，以提高推理能力的框架。
"""

# 构造工具
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)


# 初始响应
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. provide an answer, reflection, and then follow up with search queries to improve the answer."""
    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke({"messages": state["messages"]}, {"tags": [f"attempt: {attempt}"]})
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                # state 在LangGraph中通常是 字典（如 {"messages": [...]}），而 + 操作符仅适用于列表
                state["messages"] = state["messages"] + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\n Pay close attention to the function schema.\n\n" +
                                self.validator.schema_json() +
                                " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return {"messages": response]



        # 初始问题模板
        actor_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are expert researcher.
                Current time: {time}
                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend search queries to research information and improve your answer .""",
                ),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "\n\n<system> Reflect on the user's original question and the"
                    " actions taken thus far. Respond using the (function_name) function.</reminder>",
                ),
            ]
        ).partial(
            time=lambda: datetime.datetime.now().isoformat(),
        )

        # 初始请求链
        """
    LangChain 机制：bind_tools() 方法的作用是让LLM 按照指定格式生成结构化输出。AnswerQuestion 类通过 Field 定义了输出模板，LLM会生成符合该结构的JSON（类似OpenAI的Function Calling）。
    也可以在 bind_tools 方法中，添加参数  tool_choice="AnswerQuestion"  # 显式指定格式
    
    即 bind_tools 的用途是双重的，具体行为取决于传入的参数类型：
    - 传入 Tool 或 StructuredTool 的实例时，是绑定工具进行调用
    - 传入 BaseModel 的子类（通过 Field 定义字段）时，绑定输出格式（Pydantic类）
    """
        initial_answer_chain = actor_prompt_template.partial(
            first_instruction="provide a detailed ~250 word answer .",
            function_name=AnswerQuestion.__name__,
        ) | llm.bind_tools(tools=[AnswerQuestion])

        # 初始验证器
        validator = PydanticToolsParser(tools=[AnswerQuestion])

        # 第一次请求结果
        first_responder = ResponderWithRetries(
            runnable=initial_answer_chain, validator=validator
        )

        # Revision，反省模板，评价，修正
        revise_instructions = """Revise your previous answer using the new information.
	- You should use the previous critique to add important information to your answer.
	- You MUST include numerical citations in your revised answer to ensure it can be verified.
	- Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
		- [1] https://example.com
		- [2] https://example.com
	- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words."""

        # Extend the initial answer schema to include references.
        # Forcing citation in the model encourages grounded responses
        class ReviseAnswer(AnswerQuestion):
            """Revise your original answer to your question. Provide an answer, reflection,
        cite your reflection with references, and finally add search queries to improve the answer ."""
            references: list[str] = Field(description="citations motivating your updated answer .")

        # 反省动作链
        revision_chain = actor_prompt_template.partial(
            first_instruction=revise_instructions,
            function_name=ReviseAnswer.__name__),
        ) | llm.bind_tools(tools=[ReviseAnswer])

        revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

        # 反省后处理的结果
        revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)

        # 创建节点
        from langchain_core.tools import StructuredTool
        from langgraph.prebuilt import ToolNode

        def run_queries(search_queries: List[str], **kwargs):
            """Run the generated queries."""
            return tavily_tool.batch([{"query": query} for query in search_queries])

        tool_node = ToolNode(
            [
                StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
                StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
            ]
        )

        # 构建 graph
        from langgraph.graph import END, MessageGraph

        MAX_ITERATIONS = 5
        builder = MessageGraph()

        builder.add_node("draft", first_responder.respond)
        builder.add_node("execute_tools", tool_node.run)
        builder.add_node("revise", revisor.respond)

        # draft -> execute tools
        builder.add_edge("draft", "execute_tools")
        # execute-tools -> revise
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

        # revise -> execute-tools OR end
        # 根据状态决定是否继续循环
        builder.add_conditional_edges(
            "revise",
            lambda state: event_loop(state)  # 传递完整状态字典
        )

        # 设置初始节点
        builder.set_entry_point("draft")
        graph = builder.compile()

        """
    在这个示例中，我们首先创建了一个 MessageGraph 实例，然后添加了三个节点: draft、execute_tools、revise。
    这些节点分别对应于 Refexion 流程中的生成、执行工具和修订步骡。
    我们设置了初始节点为 draft，并定义了节点之间的边来指定执行顺序。最后，我们定义了一个循环逻辑，当迭代次数超过最大值时，流程将结束。
    
    执行流程为：user request --> responder --> tools --> revise --> tools --> revise --> ... --> END
    """

        """
    Time Travel 时间回溯：
    由于大语言模型回答问题的不确定性，基于大语言模型构建的应用，也是充满不确定性的。就有必要进行更精确的检查，当某一个步骤出现问题时，才能及时发现问题并处理，再进行重演。LangGraph提供了TimeTravel时间回溯功能，可以保存Graph的运行过程，并可以手动指定从Graph的某一个Node开始进行重演。
    
    具体实现时，需要注意以下几点:
    运行时，指定thread_id，checkpoint检查点，graph 将在每一个Node执行后，生成一个check_point_id.
    重演时，指定thread_id和check_point_id 进行任务重演。
    重演前，可以选择更新state，当然，如果没问题，也可以不指定。
    """
        # 构建一个图。图中两个步骤:第一步让大模型推荐一个有名的作家，第二步，让大模型用推荐的作家的风格写一个100字以内的笑话。
        from typing import TypedDict
        from typing_extensions import NotRequired
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.constants import START, END
        from langgraph.graph import StateGraph
        from langchain_community.chat_models import chatTongyi
        import uuid

        llm = ChatTongyi(
            model="awen-plus",
            api_key="BAILIAN API KEY"

        class State(TypedDict):
            author: NotRequired[str]
            joke: NotRequired[str]

        def author_node(state: State):
            prompt = "帮我推荐一位受人们欢迎的作家。只需要给出作家的名字即可。"
            joke = llm.invoke(prompt)
            return {"author": author}

        def joke_node(state: State):
            prompt = f"用作家: {state['author']} 的风格，写一个100字以内的笑话"
            joke = llm.invoke(prompt)
            return {"joke": joke}

        builder = StateGraph(State)
        builder.add_node(author_node)
        builder.add_node(joke_node)
        builder.add_edge(START, "author_node")
        builder.add_edge("author_node", "joke_node")
        builder.add_edge("joke_node", END)

        checkpointer = InMemorySaver()
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread id": uuid.uuid4(), }}
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
        print(selected
        state.next)
        print(selected
        state.values)

        # 为了后面的重演，需要更新state。可选步骤:
        new_config = graph.update_state(selected_state.config, values={"author": "郭德纲"})
        print(new_config)

        # 接下来，指定thread_id和checkpoint_id，进行重演
        graph.invoke(None, new_config)
