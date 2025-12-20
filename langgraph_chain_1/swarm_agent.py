# Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.
from typing import Annotated, Literal, Union
# from typing import TypedDict
from typing_extensions import TypedDict
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

Swarm智能体协作客户服务系统
主要特点：
1. 使用Command指令进行智能体间控制权转移
2. 使用Send指令进行预先通知
3. 基于状态的路由机制
4. 企业级错误处理和监控

简单跳转，移交上下文：直接使用 Command(goto=..., update=...)。这是最常见、最直接的Handoff方式。
复杂协作，预先通信：如果需要移交前“打招呼”、或向多个节点广播信息，可结合 Send。例如，在主控移交前，同时Send通知日志记录节点和下一个执行节点。


# ========== 8. Send指令详解 ==========
Send指令的核心特点和使用场景：

1. 异步通知：Send会在当前节点执行过程中，异步发送消息到目标节点
2. 不中断当前执行：发送后，当前节点继续执行直到完成
3. 消息队列：发送的消息会进入目标节点的消息队列
4. 配合Command使用：通常Send + Command组合使用

使用场景：

场景1：预通知（Handoff前的打招呼）
  当前节点 -> Send(通知目标节点) -> 继续执行 -> Command(跳转到目标节点)
  目标节点执行时，会先收到Send的消息

场景2：广播通知
  Send(节点A) + Send(节点B) + Send(日志节点) -> Command(跳转)
  多个节点同时收到通知

场景3：并行处理
  当前节点 -> Send(启动子任务到工作节点) -> 继续处理主任务
  工作节点异步处理子任务

场景4：状态同步
  当多个节点需要共享信息时，可以用Send广播状态更新

场景5：日志和监控
  Send(日志节点)记录关键操作，不影响主流程

与Command的区别：
- Send: 只发送消息，不改变当前执行流程
- Command: 改变状态和路由，控制执行流程

实际例子：
  客服转接专家：
  1. Send(专家): "有客户需要技术支持，这是背景信息..."
  2. Command(goto=专家): 实际转接客户
  3. 专家收到Send的消息，了解背景

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
    # 用于存储移交相关信息
    handoff_info: dict


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
        tool_call_id: Annotated[str, ...] = None  # 注入当前工具调用ID
    ):
        """当本专家无法完全处理时，将对话移交给另一个专家。

        Args:
            final_reply_to_user: 在移交控制权前，对用户说的最后一句话。
            tool_call_id: 工具调用ID
            state: 当前状态
        """
        print(f"[工具执行] 执行中 {tool_name} ====")

        # 在实际企业应用中，这里可能会记录移交日志、更新工单状态等
        tool_message = ToolMessage(
            content=f"已收到移交请求，即将由{agent_name}处理。移交说明：{final_reply_to_user}",
            tool_call_id=tool_call_id,
            name=tool_name,
            next_agent=agent_name,
            handoff_message=f"移交说明：{final_reply_to_user}",
            internal_note=f"Control handed off to {agent_name}"
        )

        # 使用 Command 指令，准备正式跳转到目标代理
        # 状态更新 (update): 将当前信息（如对话历史、中间结果）写入图的共享状态，传递给下一个智能体。
        # 流程跳转 (goto): 指定图中下一个要执行的节点（即目标智能体），实现控制权的显式交接。
        command_transfer = Command(
            # 关键：指定下一个执行的节点（目标智能体）
            goto=agent_name,
            # 关键：更新共享状态，将移交消息加入历史
            update={
                "active_agent": agent_name,
                "handoff_info": {
                    "from_tool": tool_name,
                    "message": final_reply_to_user,
                    "time": "now"
                },
                "messages": [tool_message, AIMessage(content=final_reply_to_user)]},
        )

        # 关键：在LangGraph 1.0中，工具应该返回可执行的操作指令
        # 返回一个包含两个指令的列表。LangGraph 会按顺序执行它们。
        return {
            "command": command_transfer,
        }

    return handoff_to_agent


# ========== 3. 高级移交工具（结合Send和Command） ==========
def create_advanced_handoff_tool(source_agent: str, target_agent: str, description: str = None):
    """创建高级移交工具，结合Send和Command指令"""
    tool_name = f"advanced_transfer_{source_agent}_to_{target_agent}"
    description = description or f"从{source_agent}向{target_agent}移交控制权，并发送预通知。"

    @tool(tool_name, description=description)
    def advanced_handoff(
        final_reply_to_user: str,
        internal_notes: str = "",
        tool_call_id: str = None,
        state: SwarmState = None
    ) -> dict:
        """
        高级移交：先发送通知，再转移控制权

        创建Send指令：向目标代理发送预通知, 异步向目标代理发送一个准备消息
        Send指令使用场景详解：
        1. Send会在当前节点继续执行完之前，预先异步发送消息到目标节点
        2. 消息会被添加到目标节点的消息队列，但不会立即执行目标节点（而不立即转移控制权）
        3. 通常配合Command使用，Command的goto才会实际跳转执行
        4. 目标代理会在后续收到此消息

        它通常用于：
        # - 在转移控制权之前向目标节点发送一个通知或准备消息。
        # - 向多个节点广播消息，例如日志节点、监控节点等。

        典型工作流程：
        1. Send(通知目标节点) -> 消息进入队列
        2. Command(更新状态 + goto目标节点) -> 实际跳转
        3. 目标节点执行时，会收到Send发送的消息
        """
        print(f"[高级移交] 从 {source_agent} 到 {target_agent}")
        print(f"[内部说明] {internal_notes}")

        send_instruction = Send(
            node=target_agent,
            # 发送一个内部通知消息（不会显示给用户）
            arg=[ToolMessage(
                content=f"预通知：{source_agent}即将移交任务。内部说明：{internal_notes}",
                name=tool_name,
                tool_call_id=tool_call_id,
                internal_note=internal_notes if internal_notes else f"Control handed off to {target_agent}"
            )]
        )

        # 2. 创建Command指令：更新状态并跳转
        command_instruction = Command(
            update={
                "active_agent": target_agent,
                "handoff_info": {
                    "from_tool": tool_name,
                    "message": final_reply_to_user,
                    "internal_note": internal_notes,
                    "pre_notified": True
                },
                "messages": [
                    AIMessage(content=final_reply_to_user),
                    ToolMessage(
                        content=f"控制权已从{source_agent}移交至{target_agent}",
                        name=tool_name,
                        tool_call_id=tool_call_id,
                        internal_note=internal_notes if internal_notes else f"Control handed off to {target_agent}"
                    )
                ],
                "summary": {
                    **state.get("summary", {}),
                    f"{source_agent}_handoff": {
                        "time": "now",
                        "reason": internal_notes,
                        "target": target_agent
                    }
                }
            },
            goto=target_agent
        )

        # 返回指令序列
        return {
            "commands": [send_instruction, command_instruction]
        }

    return advanced_handoff


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

# 高级移交工具
advanced_general_to_complaint = create_advanced_handoff_tool(
    source_agent="general_consultant",
    target_agent="complaint_specialist",
    description="从常规咨询到投诉专家的高级移交"
)

advanced_complaint_to_technical = create_advanced_handoff_tool(
    source_agent="complaint_specialist",
    target_agent="technical_specialist",
    description="从投诉专家到技术专家的高级移交"
)


# ========== 4. 定义各专家智能体的行为函数 ==========
def general_consultant_node(state: SwarmState):
    """常规咨询专家：处理一般性查询，并充当初始路由判断器"""
    print("\n[常规咨询专家] 开始处理...")
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 获取最新的用户消息
    last_user_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    if not last_user_message:
        # 如果没有用户消息，保持当前状态
        return {"messages": [AIMessage(content="您好，请问有什么可以帮助您的？")]}

    # 1. 意图识别
    print(f"[意图识别] 用户消息: {last_user_message}")
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

    # 简单关键词匹配的意图识别（实际应用中可使用更复杂的NLP模型）
    if "投诉" in last_user_message or "退款" in last_user_message or "赔偿" in last_user_message:
        intent_data = {"intent": "complaint", "confidence": 0.95}
    elif "错误" in last_user_message or "故障" in last_user_message or "技术" in last_user_message:
        intent_data = {"intent": "technical", "confidence": 0.90}
    else:
        intent_data = {"intent": "general", "confidence": 0.85}

    # 更新状态中的意图
    print(f"[意图识别] 结果: {intent_data}")
    state['user_intent'] = intent_data['intent']

    # 2. 根据意图决定下一步行动
    if intent_data['intent'] == 'general' and intent_data['confidence'] > 0.7:
        # 高置信度的常规咨询，由本专家处理
        print("[常规咨询专家] 处理常规咨询...")

        # response = llm.invoke([
        #     SystemMessage(content="你是专业的常规客服，友好、准确地回答用户关于产品、价格、政策的咨询。"),
        #     *state['messages'][-6:],  # 携带最近上下文
        # ])
        # 模拟LLM处理
        response = "感谢您的咨询。关于产品价格，我们的标准套餐是每月99元，包含所有基础功能。您需要了解更多详细信息吗？"
        state['messages'].append(AIMessage(content=response))

        # 处理完毕，准备结束或等待下一轮用户输入
        state['active_agent'] = "__end__"
        print("[常规咨询专家] 处理完成，准备结束。")
    else:
        # 识别为投诉或技术问题，或低置信度，准备移交
        print(f"[常规咨询专家] 识别为{intent_data['intent']}问题，准备移交...")

        # 创建移交响应，根据意图选择目标代理
        transfer_message = f"我已经了解您的问题属于{intent_data['intent']}类别，现在为您转接资深专家。"
        target_agent = "complaint_specialist" if intent_data['intent'] == 'complaint' else "technical_specialist"

        # 创建移交消息
        handoff_msg = AIMessage(
            content=transfer_message,
            tool_calls=[{
                "name": f"transfer_to_{target_agent}",
                "args": {"final_reply_to_user": transfer_message},
                "id": "handoff_tool_call"
            }]
        )

        # 更新状态
        state['messages'].append(handoff_msg)
        state['active_agent'] = target_agent
        print(f"[常规咨询专家] 已移交至{target_agent}")

    return state


def complaint_specialist_node(state: SwarmState):
    """纠纷投诉专家：处理负面情绪、退款、赔偿等复杂问题"""
    print("\n[投诉专家] 开始处理投诉...")

    # 构建完整的对话上下文
    context_messages = [msg for msg in state['messages'] if not isinstance(msg, ToolMessage)]

    # 获取最新的用户消息和对话历史
    last_user_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break

    if not last_user_message:
        return {"messages": [AIMessage(content="您好，请问有什么可以帮助您的？")]}

    # 检查是否有移交工具调用
    tool_calls_pending = False
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_pending = True
            break

    if tool_calls_pending:
        # 如果有待处理的工具调用，模拟工具执行
        print("[投诉专家] 处理移交请求...")

        # 模拟处理投诉并生成解决方案
        # tools = [book_complaint, transfer_to_general, transfer_to_technical]
        # llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

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
        response = "非常抱歉给您带来不好的体验。我是投诉处理专员，将全力为您解决。为了快速处理您的问题，请提供订单号和具体情况描述。"

        # state['messages'].append(AIMessage(content=response.content))
        state['messages'].append(AIMessage(content=response))
        state['active_agent'] = "__end__"  # 或设置为"general_consultant"进行进一步处理
    else:
        # 直接处理用户消息
        print("[投诉专家] 处理用户投诉...")

        # 模拟处理逻辑
        if "订单" in last_user_message or "编号" in last_user_message:
            response = "感谢提供订单信息。我已记录您的问题，我们的处理专员将在24小时内联系您，并提供解决方案。同时，为表歉意，我们为您申请了一张50元优惠券。"
            handoff_msg = AIMessage(
                content=response,
            )
        else:
            response = "理解您的不满情绪。为了更好协助您，请提供订单号、问题发生时间以及您的联系方式。"
            handoff_msg = AIMessage(
                content=response,
                tool_calls=[{
                    "name": f"transfer_to_general",
                    "args": {"final_reply_to_user": response},
                    "id": "handoff_tool_call"
                }]
            )
        state['messages'].append(handoff_msg)

        # 根据情况决定下一步
        if "解决" in response or "优惠券" in response:
            state['active_agent'] = "__end__"
        elif "请提供" in response or "<UNK>" in response:
            state['active_agent'] = "general_consultant"  # 继续处理
        else:
            state['active_agent'] = "complaint_specialist"

    print("[投诉专家] 处理完成")
    return state


def technical_specialist_node(state: SwarmState):
    """技术问题专家：处理故障、Bug、技术集成等"""
    print("\n[技术专家] 开始处理技术问题...")

    last_user_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    if not last_user_message:
        return {"messages": [AIMessage(content="您好，请问有什么可以帮助您的？")]}

        # 检查是否有移交工具调用
    tool_calls_pending = False
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_pending = True
            break

    if tool_calls_pending:
        # 处理移交请求
        print("[技术专家] 处理移交请求...")

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

        response = "收到您的技术问题反馈。我是技术专员，请提供以下信息以帮助排查：1) 错误截图或代码 2) 操作步骤 3) 使用的设备和浏览器版本。"
        state['messages'].append(AIMessage(content=response))
        state['active_agent'] = "technical_specialist"  # 继续处理
    else:
        # 直接处理技术问题
        print("[技术专家] 处理用户技术问题...")

        # 模拟技术问题处理逻辑
        if "API" in last_user_message or "接口" in last_user_message:
            response = "关于API接口500错误，这通常是由于服务器端问题。建议：1) 检查API密钥是否正确 2) 确认请求参数格式 3) 查看服务状态页。如果问题持续，请提供更多错误详情。"
        elif "登录" in last_user_message or "无法访问" in last_user_message:
            response = "登录问题可能由多种原因引起：1) 清除浏览器缓存 2) 检查网络连接 3) 重置密码。如果仍无法解决，请提供具体错误信息。"
        else:
            response = "请详细描述您遇到的技术问题，包括错误信息、操作步骤和期望结果，以便我们准确排查。"

        state['messages'].append(AIMessage(content=response))

        # 根据问题复杂度决定下一步
        if "提供" in response or "检查" in response:
            state['active_agent'] = "technical_specialist"  # 需要更多信息
        else:
            state['active_agent'] = "__end__"

    print("[技术专家] 处理完成")
    return state


async def execute_tools(state: SwarmState):
    """
    专门用于执行工具调用的节点

    工作流程：
    1. 检查最新的AI消息是否有tool_calls
    2. 如果有，执行对应的工具
    3. 工具返回Command/Send指令
    4. LangGraph执行这些指令
    """
    # 查找最新的AI消息
    last_ai_msg = None
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break

    if not last_ai_msg or not hasattr(last_ai_msg, 'tool_calls') or not last_ai_msg.tool_calls:
        # 没有工具调用，直接返回
        return state

    print(f"[工具执行节点] 发现工具调用: {last_ai_msg.tool_calls}")

    # 执行每个工具调用
    for tool_call in last_ai_msg.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_call_id = tool_call['id']

        # 根据工具名称选择对应的工具
        if tool_name == "transfer_to_complaint_specialist":
            tool_func = transfer_to_complaint
        elif tool_name == "transfer_to_technical_specialist":
            tool_func = transfer_to_technical
        elif tool_name == "transfer_to_general_consultant":
            tool_func = transfer_to_general
        elif tool_name == "advanced_transfer_general_consultant_to_complaint_specialist":
            tool_func = advanced_general_to_complaint
        elif tool_name == "advanced_transfer_complaint_specialist_to_technical_specialist":
            tool_func = advanced_complaint_to_technical
        else:
            print(f"[警告] 未知工具: {tool_name}")
            continue

        try:
            # 执行工具
            print(f"[工具执行] 执行 {tool_name}，参数: {tool_args}")
            result = tool_func.invoke({
                **tool_args,
                "tool_call_id": tool_call_id
            })

            # 添加工具响应消息
            tool_response = ToolMessage(
                content=f"工具 {tool_name} 执行完成",
                tool_call_id=tool_call_id,
                name=tool_name
            )
            state['messages'].append(tool_response)

            # 工具返回的指令会被LangGraph自动处理
            # 这里我们只需返回包含指令的状态
            if 'command' in result:
                state['__command__'] = result['command']
            elif 'commands' in result:
                state['__commands__'] = result['commands']

        except Exception as e:
            print(f"[工具执行错误] {e}")
            error_msg = ToolMessage(
                content=f"工具执行失败: {str(e)}",
                tool_call_id=tool_call_id,
                name=tool_name
            )
            state['messages'].append(error_msg)

    return state


# ========== 5. 构建Swarm图 ==========
def create_customer_service_swarm():
    """创建并编译客户服务Swarm图"""
    workflow = StateGraph(SwarmState)

    # 添加节点（每个节点是一个专家智能体的入口函数）
    workflow.add_node("general_consultant", general_consultant_node)
    workflow.add_node("complaint_specialist", complaint_specialist_node)
    workflow.add_node("technical_specialist", technical_specialist_node)
    workflow.add_node("execute_tools", execute_tools)

    # 设置起始节点（所有对话都先由常规咨询专家处理）
    workflow.add_edge(START, "general_consultant")

    # 定义路由逻辑：根据 `active_agent` 状态决定下一个节点
    def route_by_active_agent(state: SwarmState):
        # 优先处理工具执行
        last_ai_msg = None
        for msg in reversed(state['messages']):
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                break

        if last_ai_msg and hasattr(last_ai_msg, 'tool_calls') and last_ai_msg.tool_calls:
            print(f"[路由] 检测到工具调用，转到工具执行节点")
            return "execute_tools"

        next_agent = state.get('active_agent', 'general_consultant')
        print(f"[路由决策] 当前agent: {state.get('active_agent')}, 下一节点: {next_agent}")

        if next_agent == '__end__':
            return END
        elif next_agent in ["general_consultant", "complaint_specialist", "technical_specialist"]:
            return next_agent
        else:
            # 默认返回常规咨询专家
            return "general_consultant"

    # 为所有专家节点配置条件边
    for agent in ["general_consultant", "complaint_specialist", "technical_specialist", "execute_tools"]:
        workflow.add_conditional_edges(
            agent,
            route_by_active_agent,
            {
                "general_consultant": "general_consultant",
                "complaint_specialist": "complaint_specialist",
                "technical_specialist": "technical_specialist",
                "execute_tools": "execute_tools",
                END: END
            }
        )

    # 编译工作流，并配置持久化（企业级核心要求）
    # memory = SqliteSaver.from_conn_string("sqlite:///customer_service_swarm.db")  # 持久化到文件
    memory = InMemorySaver()
    compiled_workflow = workflow.compile(
        checkpointer=memory,
        # 启用调试
        debug=True
    )
    return compiled_workflow


# ========== 6. 企业级使用示例 ==========
async def run_conversation(swarm, user_input: str, thread_id: str = "test_thread"):
    """运行单轮对话"""
    print(f"\n{'=' * 60}")
    print(f"用户输入: {user_input}")
    print(f"{'=' * 60}")

    # 初始化状态
    initial_state = SwarmState(
        active_agent="general_consultant",
        messages=[HumanMessage(content=user_input)],
        user_intent=None,
        user_id="test_user_001",
        problem_category=None,
        summary={}
    )

    # 执行Swarm，当达到递归限制时，会抛出RecursionError异常
    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "user_id": "user_vip_123",
        },
        recursion_limit=3
    )
    try:
        # 使用astream进行流式执行
        final_state = initial_state
        async for event in swarm.astream(initial_state, config, stream_mode="values"):
            # 获取最新状态
            latest_state = event
            final_state = latest_state

            # 监控事件
            print(f"[状态更新] active_agent: {latest_state.get('active_agent', 'unknown')}")
            print(f"[状态更新] user_intent: {latest_state.get('user_intent', 'unknown')}")

            # 打印最新消息
            if latest_state['messages']:
                last_msg = latest_state['messages'][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"[AI回复] {last_msg.content[:100]}...")
                elif isinstance(last_msg, ToolMessage):
                    print(f"[工具消息] {last_msg.content}")

        print(f"\n{'=' * 60}")
        print("对话完成!")
        print(f"最终状态: {final_state['active_agent']}")
        print(f"识别意图: {final_state['user_intent']}")
        print(f"消息数量: {len(final_state['messages'])}")

        # 显示完整对话
        print(f"\n完整对话记录:")
        for idx, msg in enumerate(final_state['messages']):
            if isinstance(msg, HumanMessage):
                print(f"  用户[{idx}]: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"  助手[{idx}]: {msg.content}")
            elif isinstance(msg, ToolMessage):
                print(f"  系统[{idx}]: {msg.content}")

        return final_state

    except Exception as e:
        print(f"[错误] 执行异常: {e}")
        return None


async def main():
    """主函数"""
    print("初始化客户服务Swarm系统...")

    # 初始化Swarm
    swarm = create_customer_service_swarm()

    # 测试用例
    test_cases = [
        ("你们的产品价格是多少？", "test_case_1"),  # 常规咨询
        ("我要投诉！昨天买的东西今天就坏了，必须给我退款！", "test_case_2"),  # 投诉
        ("你们的API接口返回500错误，怎么解决？", "test_case_3"),  # 技术问题
        ("我想了解下你们的服务政策", "test_case_4"),  # 常规咨询
    ]

    # 运行测试
    for user_input, thread_id in test_cases:
        await run_conversation(swarm, user_input, thread_id)

    print("\n所有测试完成!")


# ========== 7. 企业级监控和错误处理 ==========
class SwarmMonitor:
    """Swarm系统监控器"""

    @staticmethod
    async def monitor_execution(swarm, state: SwarmState, config: dict):
        """监控执行过程"""
        try:
            async for event in swarm.astream_events(state, config, version="v1"):
                event_type = event.get("event")
                name = event.get("name")

                # 记录不同类型的事件
                if event_type == "on_chain_start":
                    print(f"[监控] 节点 {name} 开始执行")
                elif event_type == "on_chain_end":
                    print(f"[监控] 节点 {name} 执行完成")
                elif event_type == "on_tool_start":
                    print(f"[监控] 工具 {name} 开始调用")
                elif event_type == "on_tool_end":
                    print(f"[监控] 工具 {name} 调用完成")
                elif event_type == "on_chat_model_stream":
                    # LLM流式输出
                    pass

                # 企业级：这里可以添加日志记录、性能监控、异常报警等

        except Exception as e:
            print(f"[监控异常] {e}")
            # 企业级：发送警报、记录错误日志等



# ========== 5. 企业级使用示例 ==========
if __name__ == "__main__":
    asyncio.run(main())

