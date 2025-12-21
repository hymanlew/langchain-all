import asyncio
from typing import Any, Callable, List, Optional, Dict
from datetime import datetime
from langchain_core.runnables import Runnable, RunnableLambda, RunnableWithFallbacks, RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest, ModelResponse, AgentState,
    before_model, after_model, wrap_tool_call, dynamic_prompt, AgentMiddleware
)
from langchain.agents.middleware import PIIMiddleware, ToolCallLimitMiddleware, ModelCallLimitMiddleware
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from functools import wraps


# ---------- 1. 定义企业级工具 ----------
@tool
def query_customer_database(customer_id: str, query_type: str) -> str:
    """
    模拟查询内部客户数据库。这是一个高风险操作，需要审计。
    """
    # 模拟数据库查询
    print(f"[AUDIT] 查询客户 {customer_id} 的 {query_type} 信息")
    return f"客户 {customer_id} 的 {query_type} 信息：余额为 $1,234.56，状态为活跃。"


# ---------- 2. 企业级自定义 Middleware (装饰器模式) ----------
# 通过 @before_model， @after_model， @wrap_tool_call 等装饰器定义，这些装饰器会自动应用于其作用域内的所有相应操作。
# @before_model 等装饰器，会自动应用于 node_plan_with_llm 函数中的 llm.ainvoke(state["messages"]) 调用。这是LangChain 1.0运行时的一部分。
# 但前提是已经将对应的中间件注册到了 chain/agent 中

@wrap_tool_call
async def validate_tool_execution(request: Any, handler: Callable) -> Any:
    """
    工具调用护栏：在执行高风险工具前进行参数验证和权限检查。
    这是对内置PIIMiddleware、ToolCallLimitMiddleware的补充。
    """
    tool_call = request.tool_call
    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
    args = tool_call.get("args", {})

    # 1. 高风险工具拦截示例
    high_risk_tools = ["format_hard_drive", "delete_database"]
    if tool_name in high_risk_tools:
        raise PermissionError(f"拒绝执行高风险工具 '{tool_name}'，此操作被企业安全策略禁止。")

    # 2. 参数校验示例：确保订单数量为正数
    if tool_name == "place_order":
        args = tool_call.get("args", {})
        if args.get("quantity", 0) <= 0:
            raise ValueError("订单数量必须为正整数。")

    # 3. 调用原始处理器执行工具
    result = await handler(request)
    return result

@dynamic_prompt
def inject_context_based_on_session(request: ModelRequest) -> str:
    """
    动态上下文工程：根据会话状态和用户身份动态注入系统提示词。
    这是Context Engineering的核心实践。
    """
    base_prompt = "你是一个专业的客户服务AI助手。"
    state = request.state
    runtime = request.runtime

    # 从运行时配置获取用户上下文（例如从JWT令牌解析）
    user_tier = runtime.config.get("user_tier", "standard")  # 'vip', 'standard'
    conversation_length = len(state.get("messages", []))

    # 根据上下文动态构建提示词
    if user_tier == "vip":
        base_prompt += " 当前用户是我们的VIP客户，请提供优先和详尽的帮助。"
    if conversation_length > 10:
        base_prompt += " 当前对话轮次较多，请注意保持回答简洁。"

    # 注入当前时间和会话ID以增强模型上下文感知
    base_prompt += f"\n\n当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | 会话ID：{runtime.config.get('session_id', 'N/A')}"

    return base_prompt

# ---------- 3. 组装企业级 Agent ----------
def create_demo_agent():
    """
    创建集成了多层防护和上下文管理能力的企业级Agent。
    """
    # 初始化模型
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)

    # 组合中间件：顺序很重要，按注册顺序形成执行管道
    # 列表的顺序决定了中间件从外到内的包装顺序。m1 最先执行 before，最后执行 after
    middleware_stack = [
        # 第一层：官方内置安全护栏
        PIIMiddleware(),  # 自动检测并处理邮箱、电话等敏感信息
        ModelCallLimitMiddleware(max_calls=10),  # 防止无限循环
        ToolCallLimitMiddleware(max_calls_per_tool=5),  # 限制单个工具调用次数

        # 第二层：企业自定义审计与护栏
        BusinessContextMiddleware(),
        AuditLogMiddleware(service_name="CustomerServiceAI"),
        validate_tool_execution,  # 工具调用参数校验与拦截

        # 第三层：动态上下文工程，在模型调用、工具执行前后插入审计、验证和状态管理逻辑
        inject_context_based_on_session,
    ]

    # 创建Agent
    # 在LangChain中，AgentMiddleware确实是为agent设计的中间件，但通常是与AgentExecutor结合使用，而不是直接用于LCEL链。
    agent = create_agent(
        model=llm,
        tools=[query_customer_database],
        middleware=middleware_stack,
        system_prompt="基础系统提示词将被 dynamic_prompt 中间件动态覆盖。"  # 将被inject_context_based_on_session覆盖
    )
    return agent


# ---------- 2. 企业级自定义 Middleware (类模式) ----------
class AuditLogMiddleware(AgentMiddleware):
    """
    企业审计中间件：记录所有模型调用和工具调用的审计日志。
    生产环境中应将日志发送至ELK/Splunk等系统。
    """
    def __init__(self, service_name: str):
        self.service_name = service_name

    async def before_model(self, state: AgentState, runtime: Runtime) -> None:
        """在模型调用前记录审计日志"""
        user_input = state.get("messages", [])[-1].content if state.get("messages") else "N/A"
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event": "MODEL_CALL_START",
            "session_id": runtime.config.get("session_id", "unknown"),
            "user_input": user_input[:500]  # 截断长文本
        }
        # 此处应替换为实际的日志服务调用，例如：
        # audit_logger.info(log_entry)
        print(f"[AUDIT_LOG] {log_entry}")

    async def after_tool(self, state: AgentState, runtime: Runtime, tool_name: str, tool_input: dict, tool_output: str, error: Exception = None) -> None:
        """在工具执行后记录审计日志（包含成功/失败状态）"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "event": "TOOL_CALL_COMPLETE",
            "session_id": runtime.config.get("session_id", "unknown"),
            "tool": tool_name,
            "input": tool_input,
            "success": error is None,
            "error": str(error) if error else None
        }
        # 此处应替换为实际的日志服务调用
        print(f"[AUDIT_LOG] {log_entry}")

    async def after_model(self, response: ModelResponse) -> ModelResponse:
        """记录模型的所有输出，特别是工具调用决策。"""
        ai_msg = response.state.get("ai_message")
        session_id = response.runtime.config.get("configurable", {}).get("session_id", "unknown")

        if ai_msg and hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "event": "MODEL_TOOL_DECISION",
                "tools_called": [tc['name'] for tc in ai_msg.tool_calls],
                "raw_response": ai_msg.model_dump_json()  # 记录原始响应供审计
            }
            # 生产环境应写入外部日志系统 (如Logstash, CloudWatch)
            print(f"[AuditLogMiddleware] 审计日志: {json.dumps(log_entry, indent=2, ensure_ascii=False)}")
        return response


class BusinessContextMiddleware(AgentMiddleware):
    """业务上下文中间件：动态注入用户和会话信息。"""

    async def before_model(self, request: ModelRequest) -> ModelRequest:
        """在模型调用前注入系统级业务上下文。"""
        config = request.runtime.config.get("configurable", {})
        user_tier = config.get("user_tier", "standard")

        context_msg = SystemMessage(content=(
            f"[业务上下文] 用户等级: {user_tier} | "
            f"会话: {config.get('session_id')} | "
            f"时间: {datetime.now().isoformat()}"
        ))

        # 将上下文消息插入到现有消息列表的头部
        existing_msgs = request.state.get("messages", [])
        request.state["messages"] = [context_msg] + existing_msgs

        print(f"[BusinessContextMiddleware] 已为 {user_tier} 用户注入上下文。")
        return request


async def node_plan_with_llm(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """节点1: 让LLM分析需求并规划工具调用。"""
    tools = [query_customer_database]
    llm = ChatOpenAI(model="gpt-4", temperature=0).bind_tools(tools)

    # 关键：在此函数内，上述通过装饰器定义的Middleware会自动生效（使用 creat_agent）。
    # - 中间件是按照 before_model、after_model、wrap_tool_call 等不同类型执行的（严格按照此装饰器的顺序）
    # - 如果是有使用同一类装饰器，则是以它们的从上往下的定义顺序来组织的，形成一个“洋葱模型”式的执行管道。
    # 当调用 `llm.ainvoke` 时，流程会经过：
    # 1. before_model -> 2. validate_and_sanitize_input -> 3. inject_business_context → 模型调用。
    # 然后执行真正的LLM调用
    # 4. audit_model_output
    ai_msg: AIMessage = await llm.ainvoke(state["messages"])

    return {
        **state,
        "ai_message": ai_msg,
        "tool_calls": ai_msg.tool_calls if hasattr(ai_msg, 'tool_calls') else []
    }

async def node_execute_tools(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """节点2: 执行工具调用。"""
    tool_calls = state.get("tool_calls", [])
    tool_messages = []

    for tc in tool_calls:
        tool_name = tc.get("name")
        tool_args = tc.get("args", {})

        try:
            # 模拟通过中间件处理器调用（使用 creat_agent）
            # 关键：这里的工具调用会触发 `validate_tool_execution` middleware
            # 因为它被 `@wrap_tool_call` 装饰器包裹
            # 这里直接调用，但逻辑上等同于通过了validate_tool_execution校验
            result = await query_customer_database.ainvoke(tool_args)
            tool_msg = ToolMessage(content=str(result), tool_call_id=tc.get("id"))
            tool_messages.append(tool_msg)
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            tool_msg = ToolMessage(content=error_msg, tool_call_id=tc.get("id"))
            tool_messages.append(tool_msg)

    all_messages = state.get("messages", []) + [state.get("ai_message")] + tool_messages
    return {**state, "messages": all_messages}

async def node_generate_response(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """节点3: 生成最终回复。"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    final_msg: AIMessage = await llm.ainvoke(state["messages"])

    # 最终状态更新
    return {
        **state,
        "final_response": final_msg.content,
        "messages": state["messages"] + [final_msg]
    }

def create_business_agent() -> Runnable:
    # 将节点组合成链
    middleware = AuditLogMiddleware(service_name='a')

    # 请注意：在LangChain 1.0中，AgentMiddleware 是专为 create_agent 设计的抽象，无法直接用于通用LCEL链（无内置中间件概念）。
    # 但我们可以通过自定义Runnable来模拟中间件的行为。由于LCEL链是由多个Runnable组成的，我们可以将中间件逻辑嵌入到链的构建中。
    #
    # 如果我们已经有一个继承自AgentMiddleware的类，我们可能需要将其转换为LCEL兼容的形式。
    # - 将AgentMiddleware中的逻辑提取出来，然后包装成RunnableLambda，或者创建一个自定义的Runnable来调用中间件的方法。
    # - 将before方法放在链的开始处，作为一个RunnableLambda。
    # - 将after方法放在链的结束处，作为一个RunnableLambda。
    # - 使用下面声明的自定义装饰器实现

    # 将中间件的方法转换为RunnableLambda
    before_runnable = RunnableLambda(middleware.before_model())
    after_runnable = RunnableLambda(middleware.after_model())

    # 构建链
    chain = (
            before_runnable
            | RunnableLambda(node_plan_with_llm)
            | after_runnable
            | RunnableLambda(node_execute_tools)
            | RunnableLambda(node_parse_and_plan)
            | RunnableLambda(node_generate_response)
    )
    fallback_chain = RunnableWithFallbacks(
        primary=chain,
        fallbacks=[
            # 示例：如果主链失败，返回一个友好的错误消息
            RunnableLambda(lambda x: {
                **x,
                "final_response": "系统暂时繁忙，请稍后重试。",
                "error": "primary_chain_failed"
            })
        ]
    )
    return fallback_chain


# ---------- 自定义实现“护栏”和“上下文”装饰器 ----------
def audit_log_decorator(func):
    """审计日志护栏：记录函数调用和结果。"""

    @wraps(func)
    async def wrapper(state: Dict[str, Any], config: Optional[RunnableConfig] = None):
        session_id = config.get("configurable", {}).get("session_id", "unknown") if config else "unknown"
        func_name = func.__name__

        # 1. 调用前审计
        print(f"[AUDIT][{session_id}] 开始执行: {func_name}, 输入: {state.get('input')}")

        try:
            # 2. 执行原始函数
            result = await func(state, config)

            # 3. 成功审计
            print(f"[AUDIT][{session_id}] 执行成功: {func_name}, 输出状态键: {list(result.keys())}")
            return result
        except Exception as e:
            # 4. 失败审计
            print(f"[AUDIT][{session_id}][ERROR] 执行失败: {func_name}, 错误: {e}")
            raise

    return wrapper

def inject_context_decorator(func):
    """上下文工程装饰器：动态注入业务上下文到提示词。"""

    @wraps(func)
    async def wrapper(state: Dict[str, Any], config: Optional[RunnableConfig] = None):
        # 从配置或状态中获取业务上下文
        user_context = config.get("configurable", {}) if config else {}
        user_id = user_context.get("user_id", "unknown")

        # 动态构建系统上下文
        dynamic_context = {
            "current_time": datetime.now().isoformat(),
            "user_id": user_id,
            "service_region": "CN"  # 可从配置读取
        }

        # 将上下文注入到状态中，供后续节点使用
        state_with_context = {**state, "injected_context": dynamic_context}

        # 增强用户输入（可选）
        if "messages" in state and state["messages"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                enhanced_content = f"[系统上下文: 尊贵的客户] {last_msg.content}"
                state_with_context["messages"][-1] = HumanMessage(content=enhanced_content)

        return await func(state_with_context, config)

    return wrapper

# ---------- 手动构建 Chain 节点 ----------
@inject_context_decorator
@audit_log_decorator
async def node_parse_and_plan(state: Dict, config: RunnableConfig) -> Dict:
    """节点1: 解析用户输入并规划工具调用。"""
    llm = ChatOpenAI(model="gpt-4", temperature=0).bind_tools(ENTERPRISE_TOOLS)

    # 使用注入的上下文
    context = state.get("injected_context", {})
    system_prompt = f"""你是一个订单助手。当前上下文：
    - 客户级别: {context.get('customer_tier')}
    - 时间: {context.get('current_time')}
    - 用户ID: {context.get('user_id')}
    请根据用户请求，判断是否需要调用工具以及调用哪些工具。"""

    messages = [{"role": "system", "content": system_prompt}] + state.get("messages", [])
    ai_msg: AIMessage = await llm.ainvoke(messages)

    return {
        **state,
        "ai_message": ai_msg,
        "tool_calls": ai_msg.tool_calls if hasattr(ai_msg, 'tool_calls') else []
    }


# ---------- 4. 执行企业级对话 ----------
async def main():
    print("=== DEMO 级 AI Agent，集成Middleware护栏与上下文工程 ===\n")
    # agent = create_enterprise_agent()

    print("=== 启动企业级AI Agent，集成Middleware护栏与上下文工程 ===\n")
    agent = create_business_agent()

    # 模拟带有用户上下文的运行时配置（通常来自Web请求）
    runtime_config = RunnableConfig(
        configurable={
            "thread_id": "thread_enterprise_001",
            "session_id": "sess_enterprise_20250415_001",
            "user_id": "user_vip_123",
            "audit_logger": AuditLogMiddleware(service_name='config'), # 注入审计中间件
            "input_checker": BusinessContextMiddleware(), # 注入中间件
        },
        recursion_limit=20  # 防止无限循环的重要防护
    )
    """
    在LangGraph中，recursion_limit 用于控制图中节点之间或整个图的递归调用次数，防止无限递归。
    当达到递归限制时，会抛出RecursionError异常。
    在生产环境中不能用它，因为需要更优雅地处理这种情况，比如给出用户友好的提示，或者将对话转给人工客服。实现方案：
    - 自己在状态中维护一个计数器，并在每个节点中检查它，并在达到阈值时主动终止或转移。
    - 在每个节点中检查状态中的执行次数，如果超过限制，则返回一个特定的状态，然后通过路由函数跳转到结束节点，或者跳转到一个处理超限的节点
    """

    # 第一轮对话：触发动态提示词和工具调用
    print("用户: 我的客户ID是CUST-1001，我想查一下我的余额，然后订购3个产品PROD-xyz。")
    try:
        response1 = await agent.ainvoke(
            input={"messages": [{"role": "user", "content": "我的客户ID是CUST-1001，我想查一下我的余额，然后订购3个产品PROD-xyz。"}]},
            config=runtime_config
        )
        print(f"\nAI回复: {response1['messages'][-1].content}\n")
        print("-" * 50)
    except Exception as e:
        print(f"\n[企业护栏生效] 对话被拦截: {type(e).__name__}: {e}\n")

    # 第二轮对话：测试敏感信息拦截（内置PIIMiddleware）
    print("用户: 我的邮箱是private@company.com，帮我重置密码。")
    try:
        response2 = await agent.ainvoke(
            input={"messages": [{"role": "user", "content": "我的邮箱是private@company.com，帮我重置密码。"}]},
            config=runtime_config
        )
        # PIIMiddleware可能已替换邮箱为占位符，输出将不会包含真实邮箱
        print(f"\nAI回复 (PII已处理): {response2['messages'][-1].content}\n")
    except Exception as e:
        print(f"\n[安全拦截] 请求被阻止: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())


