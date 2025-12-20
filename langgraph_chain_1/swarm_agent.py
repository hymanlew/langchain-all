from typing import Annotated, Literal, Union, TypedDict
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
# from langgraph.checkpoint.sqlite import SqliteSaver
import json
import asyncio
from langgraph.types import Send, Command


"""
active_agent: 存储当前活动智能体的名称
add_active_agent_router 函数负责实现此动态路由
create_handoff_tool 函数创建移交工具

Send 先行通知：当主控（planner）调用 handoff_to_flight_expert 时，Send 指令会首先向 flight_expert 节点发送一个包含任务详情的 ToolMessage。这个通知会加入执行队列，但不会中断当前主控节点的继续执行。
Command 完成跳转：紧接着，Command(goto="flight_expert") 执行，它会更新状态（添加上下文），并将图的当前执行点强制跳转到 flight_expert 节点。
专家处理与返回：flight_expert 节点开始执行时，不仅能从共享状态中获取对话历史，还会处理之前通过 Send 收到的那个专属通知消息。专家完成任务后，通过自身的移交工具（handoff_to_planner）将控制权和结果再交还给主控。

简单跳转，移交上下文：直接使用 Command(goto=..., update=...)。这是最常见、最直接的Handoff方式。
复杂协作，预先通信：如果需要移交前“打招呼”、或向多个节点广播信息，可结合 Send。例如，在主控移交前，同时Send通知日志记录节点和下一个执行节点。
"""
# ========== 1. 定义Swarm状态 ==========
class SwarmState(TypedDict):
    """客户服务Swarm的全局状态"""
    # 核心路由状态
    active_agent: str  # 当前激活的智能体名称

    # 对话上下文
    messages: Annotated[list, add_messages]
    user_intent: Literal["general", "complaint", "technical", None]  # 识别的用户意图

    # 业务数据（随流程丰富）
    user_id: str
    problem_category: Union[str, None]

    # 各智能体执行结果摘要（用于最终报告）
    summary: dict


# ========== 2. 定义各智能体的“移交工具” ==========
def create_handoff_tool(agent_name: str, description: str = None):
    """
    创建移交工具，这是Swarm智能体间协作的关键
    每个智能体都配备一个或多个“移交工具”，当它判断任务应由其他智能体处理时，就调用对应工具，触发控制权的转移。

    agent_name: 目标智能体在图中的节点名称。
    description: 工具描述，用于指导LLM何时调用此工具。
    """
    tool_name = f"transfer_to_{agent_name}"
    description = description or f"将对话和控制权移交给 {agent_name} 智能体。"

    @tool(tool_name, description=description)
    def handoff_to_agent(
        final_reply_to_user: str,
        state: Annotated[SwarmState, ...],  # 注入当前图状态
        tool_call_id: Annotated[str, ...]  # 注入当前工具调用ID
    ):
        """当本专家无法完全处理时，将对话移交给另一个专家。

        Args:
            final_reply_to_user: 在移交控制权前，对用户说的最后一句话。
        """

        # 在实际企业应用中，这里可能会记录移交日志、更新工单状态等
        ToolMessage(
            content=f"主控已将任务移交给 {agent_name}, 具体要求: {final_reply_to_user}。",
            tool_call_id=tool_call_id,
            name=tool_name,
            next_agent=agent_name,
            handoff_message=f"移交说明：{final_reply_to_user}",
            internal_note=f"Control handed off to {agent_name}"
        )

        # 使用 Send 指令，异步向目标代理发送一个准备消息
        # 这不会立刻跳转，但目标代理会在后续收到此消息
        send_notification = Send(
            node=agent_name,
            # 构造一个工具消息，携带移交说明
            arg=[ToolMessage]
        )

        # 使用 Command 指令，准备正式跳转到目标代理
        # 状态更新 (update): 将当前信息（如对话历史、中间结果）写入图的共享状态，传递给下一个智能体。
        # 流程跳转 (goto): 指定图中下一个要执行的节点（即目标智能体），实现控制权的显式交接。
        command_transfer = Command(
            # 关键：指定下一个执行的节点（目标智能体）
            goto=agent_name,
            # 关键：更新共享状态，将移交消息加入历史
            update={"messages": state["messages"] + [ToolMessage]},
        )

        # 关键：返回一个包含两个指令的列表。LangGraph 会按顺序执行它们。
        return [send_notification, command_transfer]

    return handoff_to_agent

@tool
def book_general(from_city: str, to_city: str, date: str) -> str:
    """预订机票。"""
    # 此处应集成真实API
    return f"已成功处理 {date} 从{from_city}普通咨询。"

@tool
def book_technical(from_city: str, to_city: str, date: str) -> str:
    """预订机票。"""
    # 此处应集成真实API
    return f"已成功处理 {date} 从{from_city}技术问题。"

@tool
def book_complaint(city: str, check_in_date: str, nights: int) -> str:
    """预订酒店。"""
    # 此处应集成真实API
    return f"已成功处理 {city}成功预订酒店，入住日期{check_in_date}，投诉或纠纷。"

transfer_to_general = create_handoff_tool(agent_name="general_consultant", description="当用户需要普通咨询（产品信息、价格、政策等）时调用此工具。")
transfer_to_complaint = create_handoff_tool(agent_name="complaint_specialist", description="当用户需要投诉或纠纷（情绪负面，涉及退款、赔偿等）时调用此工具。")
transfer_to_technical = create_handoff_tool(agent_name="technical_specialist", description="当用户需要技术问题（无法登录、页面错误、功能故障等）时调用此工具。")


# ========== 3. 定义各专家智能体的行为函数 ==========
def general_consultant_node(state: SwarmState):
    """常规咨询专家：处理一般性查询，并充当初始路由判断器"""
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 获取最新的用户消息
    last_user_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    # 1. 意图识别
    intent_prompt = f"""
    用户最新消息: {last_user_message}
    历史上下文: {state['messages'][-5:] if len(state['messages']) > 5 else state['messages']}

    请分析用户意图，从以下选项中选择：
    - general: 普通咨询（产品信息、价格、政策等）
    - complaint: 投诉或纠纷（情绪负面，涉及退款、赔偿等）
    - technical: 技术问题（无法登录、页面错误、功能故障等）

    返回JSON格式：{{"intent": "general|complaint|technical", "confidence": 0.95}}
    """
    # intent_result = llm.invoke([SystemMessage(content="你是一个意图分类器"),
    #                             HumanMessage(content=intent_prompt)])
    # intent_data = json.loads(intent_result.content)

    intent_data = {"intent": "complaint", "confidence": 0.95}
    # 更新状态中的意图
    state['user_intent'] = intent_data['intent']

    # 2. 根据意图决定下一步行动
    if intent_data['intent'] == 'general' and intent_data['confidence'] > 0.7:
        # 高置信度的常规咨询，由本专家处理
        # response = llm.invoke([
        #     SystemMessage(content="你是专业的常规客服，友好、准确地回答用户关于产品、价格、政策的咨询。"),
        #     *state['messages'][-6:],  # 携带最近上下文
        # ])
        response = '处理完成了'
        state['messages'].append(AIMessage(content=response))
        # 处理完毕，不移交
        state['active_agent'] = "__end__"
    else:
        # 识别为投诉或技术问题，或低置信度，准备移交
        handoff_tool = create_handoff_tool(
            "complaint_specialist" if intent_data['intent'] == 'complaint' else "technical_specialist"
        )
        # 模拟调用移交工具（实际中由智能体的工具执行机制触发）
        handoff_result = handoff_tool(
            f"我已经了解您的问题属于{intent_data['intent']}类别，现在为您转接资深专家。"
        )
        state['active_agent'] = handoff_result['next_agent']
        state['messages'].append(AIMessage(content=handoff_result['handoff_message']))

    return state


def complaint_specialist_node(state: SwarmState):
    """纠纷投诉专家：处理负面情绪、退款、赔偿等复杂问题"""
    # 获取最新的用户消息和对话历史
    last_user_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break

    # 构建完整的对话上下文
    context_messages = [msg for msg in state['messages'] if not isinstance(msg, ToolMessage)]
    # tools = [book_complaint, transfer_to_general, transfer_to_technical]
    # llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

    # 模拟处理投诉并生成解决方案
    # 企业级实践中，这里可能会调用：情感分析API、工单系统API、赔付政策知识库等
    system_prompt = SystemMessage(content="""你是投诉处理专家，专业、同理心且严谨。
    你的任务：
    1. 安抚用户情绪，真诚道歉
    2. 详细了解投诉的具体细节和用户诉求
    3. 根据公司政策（可引用政策编号）提供解决方案
    4. 如需升级或跨部门协调，明确告知用户下一步

    注意：所有承诺必须符合公司规定，不越权。
    
    如果用户的需求涉及普通咨询或者技术问题，请务必调用 'transfer_to_general', 'transfer_to_technical' 工具，将服务移交给对应的专家。
    在移交时，请简要总结当前已完成的情况。
    """)

    # response = llm.invoke([
    #     system_prompt,
    #     *context_messages[-6:],  # 携带最近上下文（包括移交消息）
    # ])
    # state['messages'].append(AIMessage(content=response.content))

    # 实际响应生成（此处简化）
    response = "非常抱歉给您带来不好的体验。我是投诉处理专员，将全力为您解决。请问订单号和具体情况是？"
    state['messages'].append(AIMessage(content=response))

    # 假设本环节处理完毕，可以结束或移交回常规客服
    state['active_agent'] = "general_consultant"  # 或 handoff 回 general_consultant, __end__
    return state


def technical_specialist_node(state: SwarmState):
    """技术问题专家：处理故障、Bug、技术集成等"""
    # tools = [book_complaint, transfer_to_general, transfer_to_complaint]
    # llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)
    context_messages = [msg for msg in state['messages'] if not isinstance(msg, ToolMessage)]

    # 企业级实践中，这里可能会调用：错误日志查询系统、知识库、故障排查指南等
    system_prompt = SystemMessage(content="""你是技术支援专家，擅长排查系统故障。
    你的任务：
    1. 引导用户提供详细错误信息（截图、错误代码、操作步骤）
    2. 根据知识库提供初步排查步骤
    3. 判断是否需要创建技术工单并转交研发团队
    4. 告知预计处理时间

    注意：技术描述要准确，步骤要清晰可操作。
    
    如果用户的需求涉及普通咨询或者纠纷，请务必调用 'transfer_to_general', 'transfer_to_technical' 工具，将服务移交给对应的专家。
    在移交时，请简要总结当前已完成的情况。
    """)

    # response = llm.invoke([
    #     system_prompt,
    #     *context_messages[-6:],  # 携带最近上下文（包括移交消息）
    # ])
    # state['messages'].append(AIMessage(content=response.content))

    response = "收到您的技术问题反馈。我是技术专员，请提供错误截图或代码，并描述您的操作步骤，我将立即为您排查。"
    state['messages'].append(AIMessage(content=response))

    # 通常技术问题需要多轮交互，此处暂不移交
    state['active_agent'] = "transfer_to_complaint"
    return state


# ========== 4. 构建Swarm图 ==========
def create_customer_service_swarm():
    """创建并编译客户服务Swarm图"""
    workflow = StateGraph(SwarmState)

    # 添加节点（每个节点是一个专家智能体的入口函数）
    workflow.add_node("general_consultant", general_consultant_node)
    workflow.add_node("complaint_specialist", complaint_specialist_node)
    workflow.add_node("technical_specialist", technical_specialist_node)

    # 设置起始节点（所有对话都先由常规咨询专家处理）
    workflow.add_edge(START, "general_consultant")

    # 定义路由逻辑：根据 `active_agent` 状态决定下一个节点
    def route_by_active_agent(state: SwarmState):
        next_agent = state.get('active_agent', 'general_consultant')
        if next_agent == '__end__':
            return END
        return next_agent

    # 为所有专家节点配置路由
    for agent in ["general_consultant", "complaint_specialist", "technical_specialist"]:
        workflow.add_conditional_edges(
            agent,
            route_by_active_agent,
            {
                "general_consultant": "general_consultant",
                "complaint_specialist": "complaint_specialist",
                "technical_specialist": "technical_specialist",
                "__end__": END,
                END: END
            }
        )

    # 编译工作流，并配置持久化（企业级核心要求）
    # memory = SqliteSaver.from_conn_string("sqlite:///customer_service_swarm.db")  # 持久化到文件
    memory = InMemorySaver()
    compiled_workflow = workflow.compile(checkpointer=memory)

    return compiled_workflow


async def main():
    # 初始化Swarm
    swarm = create_customer_service_swarm()

    # 模拟用户输入
    test_cases = [
        "你们的产品价格是多少？",  # 常规咨询
        "我要投诉！昨天买的东西今天就坏了，必须给我退款！",  # 投诉
        "你们的API接口返回500错误，怎么解决？",  # 技术问题
    ]
    for i, user_input in enumerate(test_cases):
        print(f"\n{'=' * 50}")
        print(f"测试用例 {i + 1}: {user_input}")

        # 初始化状态
        initial_state = SwarmState(
            active_agent="general_consultant",
            messages=[HumanMessage(content=user_input)],
            user_intent=None,
            user_id="test_user_001",
            problem_category=None,
            summary={}
        )

        # 执行Swarm（带有异步支持）
        config: RunnableConfig = {"configurable": {"thread_id": f"test_thread_{i}"}}
        # config: dict[str, dict[str, str]] = {"configurable": {"thread_id": f"test_thread_{i}"}}
        try:
            # LangGraph 1.0+ 推荐使用 astream_events 进行流式事件处理
            final_state = initial_state
            async for event in swarm.astream_events(initial_state, config, version="v1"):
                # 企业级监控：可在此处捕获事件，用于日志、审计、实时看板
                kind = event.get("event")
                print(f"[状态] {event.get('name')} -- {kind}")
                if kind == "on_chain_end" and event.get("name") == "general_consultant":
                    print(f"[状态] 常规咨询专家处理完成")
                elif kind == "on_chain_end" and event.get("name") == "complaint_specialist":
                    print(f"[状态] 投诉专家处理完成")
                elif kind == "on_chain_end" and event.get("name") == "technical_specialist":
                    print(f"[状态] 技术专家处理完成")

                # 获取中间状态
                if "data" in event and "output" in event["data"]:
                    result = event["data"]["output"]
                    if isinstance(result, dict) and "__end__" == result['active_agent']:
                        final_state = event["data"]["output"]
                    # final_state = next(iter(final_state.values()))
                if kind == 'on_chat_model_stream':
                    # 可处理token流
                    pass
                elif kind == 'on_tool_start':
                    # 可记录工具调用
                    print(f"[监控] 工具调用: {event['name']}")
        except Exception as e:
            # 企业级错误处理
            print(f"[错误] 执行异常: {e}")
            continue

        # 获取最终状态
        print(f"[结果] 最终激活专家: {final_state['active_agent']}")
        print(f"[结果] 识别意图: {final_state['user_intent']}")
        if final_state['messages']:
            # 找到最后一条AI消息
            last_ai_msg = None
            for msg in reversed(final_state['messages']):
                if isinstance(msg, AIMessage):
                    last_ai_msg = msg
                    break

            if last_ai_msg:
                print(f"[结果] 最后回复: {last_ai_msg.content[:100]}...")

        # 打印完整的对话历史
        print(f"\n[完整对话历史]:")
        for idx, msg in enumerate(final_state['messages']):
            if isinstance(msg, HumanMessage):
                print(f"  用户[{idx}]: {msg.content[:50]}...")
            elif isinstance(msg, AIMessage):
                print(f"  AI助手[{idx}]: {msg.content[:50]}...")


# ========== 5. 企业级使用示例 ==========
if __name__ == "__main__":
    asyncio.run(main())

