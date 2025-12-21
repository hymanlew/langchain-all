from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END, add_messages
import operator
from langgraph.types import Send


# # 1. 定义状态（使用通用的 StateGraph，需自定义结构）
# class OrderState(TypedDict):
#     order_id: str
#     items: list
#     # 使用 Annotated 实现自动合并
#     logs: Annotated[list[str], operator.add]
#     inventory_checked: bool
#     logistics_prepared: bool
#
# # 2. 订单创建节点
# def create_order(state: OrderState):
#     """创建订单，并更新状态"""
#     order_id = state.get("order_id", "ORD-001")
#     return {
#         "order_id": order_id,
#         "logs": [f"[订单] 订单 {order_id} 创建成功"]
#     }
#
# # 3. 库存检查节点
# def check_inventory(state: OrderState):
#     """处理库存检查"""
#     return {
#         "inventory_checked": True,
#         "logs": [f"[仓储] 订单 {state['order_id']} 库存检查完成"]
#     }
#
# # 4. 物流准备节点
# def prepare_logistics(state: OrderState):
#     """处理物流准备"""
#     return {
#         "logistics_prepared": True,
#         "logs": [f"[物流] 订单 {state['order_id']} 运力调度完成"]
#     }
#
# # 5. 关键：分发函数（返回 Send 指令列表）
# def after_order_created(state: OrderState):
#     """订单创建后，广播预通知到库存和物流节点"""
#     # 返回两个 Send 指令，实现并行广播
#     return [
#         Send("check_inventory", {"order_id": state["order_id"]}),
#         Send("prepare_logistics", {"order_id": state["order_id"]})
#     ]
#
# # 6. 构建 StateGraph
# workflow = StateGraph(OrderState)
# workflow.add_node("create_order", create_order)
# workflow.add_node("check_inventory", check_inventory)
# workflow.add_node("prepare_logistics", prepare_logistics)
#
# # 设置入口
# workflow.set_entry_point("create_order")
# # 关键：使用条件边，将分发函数与 create_order 连接
# workflow.add_conditional_edges("create_order", after_order_created)
# # 库存和物流节点执行后结束
# workflow.add_edge("check_inventory", END)
# workflow.add_edge("prepare_logistics", END)
#
# # 编译并执行
# app = workflow.compile()
# for result in app.stream({
#     "order_id": "ORD-2025-001",
#     "items": [{"product": "A", "qty": 2}],
#     "logs": [],
#     "inventory_checked": False,
#     "logistics_prepared": False
# }):
#     print("最终状态:", result)


"""
金融交易风控
"""
# 1. 状态定义
class TransactionState(TypedDict):
    messages: Annotated[list, add_messages]
    tx_id: str
    amount: float
    from_account: str
    to_account: str
    risk_level: Literal["low", "medium", "high", "unknown"]
    audit_logged: bool
    user_notified: bool
    tx_status: Literal["initiated", "processing", "completed", "rejected"]


# 2. 交易执行节点（入口，使用send广播）
def execute_transaction_node(state: TransactionState):
    """执行交易，并广播通知相关系统"""
    print(f"[交易执行] 开始处理交易 {state['tx_id']}...")

    # 1. 执行核心交易逻辑（模拟）
    import random
    success = random.random() > 0.05  # 95%成功率

    if not success:
        new_state = {
            "tx_status": "rejected",
            "messages": add_messages(
                state.get("messages", []),
                f"[交易执行] 交易 {state['tx_id']} 执行失败"
            )
        }
        return new_state

    # 2. 交易成功，更新状态
    new_state = {
        "tx_status": "processing",
        "messages": add_messages(
            state.get("messages", []),
            f"[交易执行] 交易 {state['tx_id']} 执行成功，开始广播通知..."
        )
    }

    # 3. 关键：使用send指令广播到多个监控系统
    # 根据交易金额决定广播范围
    broadcast_targets = []

    # 总是通知审计系统
    broadcast_targets.append({
        "node": "compliance_audit",
        "data": {
            "tx_id": state["tx_id"],
            "amount": state["amount"],
            "messages": add_messages(
                state.get("messages", []),
                f"[广播通知] 交易 {state['tx_id']} 已发送至审计系统"
            )
        }
    })

    # 大额交易额外通知风控系统
    if state["amount"] > 10000:
        broadcast_targets.append({
            "node": "risk_monitoring",
            "data": {
                "tx_id": state["tx_id"],
                "amount": state["amount"],
                "from_account": state["from_account"],
                "to_account": state["to_account"],
                "messages": add_messages(
                    state.get("messages", []),
                    f"[广播通知] 大额交易 {state['tx_id']} 已发送至风控系统"
                )
            }
        })

    # 总是通知用户
    broadcast_targets.append({
        "node": "notify_user",
        "data": {
            "tx_id": state["tx_id"],
            "amount": state["amount"],
            "from_account": state["from_account"],
            "messages": add_messages(
                state.get("messages", []),
                f"[广播通知] 交易 {state['tx_id']} 已发送至用户通知系统"
            )
        }
    })

    return new_state, Send(*broadcast_targets)


# 3. 风控监控节点
def risk_monitoring_node(state: TransactionState):
    """接收广播通知，进行风险监控"""
    print(f"[风控监控] 收到交易 {state['tx_id']} 通知，开始风险分析...")

    # 风险分析逻辑
    risk_score = min(state["amount"] / 50000, 1.0)  # 金额越大风险越高

    if risk_score > 0.8:
        risk_level = "high"
        message = f"交易 {state['tx_id']} 风险较高，建议人工审核"
    elif risk_score > 0.5:
        risk_level = "medium"
        message = f"交易 {state['tx_id']} 中等风险，已记录"
    else:
        risk_level = "low"
        message = f"交易 {state['tx_id']} 风险较低"

    return {
        "risk_level": risk_level,
        "messages": add_messages(
            state.get("messages", []),
            f"[风控监控] {message}"
        )
    }


# 4. 合规审计节点
def compliance_audit_node(state: TransactionState):
    """接收广播通知，进行合规审计"""
    print(f"[合规审计] 收到交易 {state['tx_id']} 通知，开始审计记录...")

    # 模拟审计记录逻辑
    import datetime
    audit_timestamp = datetime.datetime.now().isoformat()

    return {
        "audit_logged": True,
        "messages": add_messages(
            state.get("messages", []),
            f"[合规审计] 交易 {state['tx_id']} 审计记录已保存，时间: {audit_timestamp}"
        )
    }


# 5. 用户通知节点
def notify_user_node(state: TransactionState):
    """接收广播通知，通知用户"""
    print(f"[用户通知] 收到交易 {state['tx_id']} 通知，开始发送用户提醒...")

    # 模拟通知逻辑
    notification_method = "SMS" if state["amount"] < 5000 else "Email"

    return {
        "user_notified": True,
        "messages": add_messages(
            state.get("messages", []),
            f"[用户通知] 已通过{notification_method}通知用户 {state['from_account']}"
        )
    }


# 6. 聚合节点（可选，用于收集所有广播结果）
def aggregate_results_node(state: TransactionState):
    """收集所有广播节点的结果，进行最终处理"""
    print(f"[结果聚合] 收集交易 {state['tx_id']} 的所有处理结果...")

    # 检查所有系统是否都已完成处理
    all_completed = (
            state.get("audit_logged", False) and
            state.get("user_notified", False) and
            state.get("risk_level", "unknown") != "unknown"
    )

    if all_completed:
        final_status = "completed"
        message = "所有系统处理完成，交易流程结束"
    else:
        final_status = state.get("tx_status", "processing")
        message = "部分系统仍在处理中"

    return {
        "tx_status": final_status,
        "messages": add_messages(
            state.get("messages", []),
            f"[结果聚合] {message}"
        )
    }


# 7. 构建图
workflow = StateGraph(TransactionState)

# 添加节点
workflow.add_node("execute_transaction", execute_transaction_node)
workflow.add_node("risk_monitoring", risk_monitoring_node)
workflow.add_node("compliance_audit", compliance_audit_node)
workflow.add_node("notify_user", notify_user_node)
workflow.add_node("aggregate_results", aggregate_results_node)

# 设置入口
workflow.set_entry_point("execute_transaction")

# 注意：这里不定义从execute_transaction出去的边！
# 广播连接由send指令在运行时动态创建

# 编译
app = workflow.compile()

# 8. 执行示例
if __name__ == "__main__":
    # 测试不同金额的交易
    test_transactions = [
        {
            "tx_id": "TX-SMALL-001",
            "amount": 1500.0,
            "from_account": "ACC-001",
            "to_account": "ACC-002",
            "risk_level": "unknown",
            "audit_logged": False,
            "user_notified": False,
            "tx_status": "initiated",
            "messages": []
        },
        {
            "tx_id": "TX-LARGE-002",
            "amount": 25000.0,
            "from_account": "ACC-003",
            "to_account": "ACC-004",
            "risk_level": "unknown",
            "audit_logged": False,
            "user_notified": False,
            "tx_status": "initiated",
            "messages": []
        }
    ]

    for i, tx in enumerate(test_transactions):
        print(f"\n{'=' * 60}")
        print(f"执行测试交易 {i + 1}: {tx['tx_id']} (金额: ${tx['amount']:,.2f})")
        print('=' * 60)

        for result in app.stream(tx):
            print(f"【执行】{result}")

        print(f"\n交易 {tx['tx_id']} 处理完成!")
