"""
ReAct 是当接收到某些用户输入，Agent 就思考并决定使用哪个工具，以及该工具的输入应该是什么，随后调用该工具并记录观察结果。之后，Agent 将工具、工具输入和观察历史传回，
以决定接下来采取哪个步骤，一直重复此过程直到Agent确定不再需要使用工具，然后直接回应用户。
基于这个框架的 Agent 在大多数情况下运行良好，但当用户目标变得更加复杂，请求复杂时，面对提高可靠性以及越来越复杂的需求，大模型往往不堪重负，此时就有了 Plan-and-Solve
 认知框架和 Zero/Few-shot-CoT 认知框架。

**Zero/Few-shot-CoT 认知框架**
它是将目标问题的定义与 “Let's think step by step” 连接起来作为输入提示。这种方法结合了零样本学习和思维链推理旨在提升模型处理未见过任务的能力，即在没有直接训练样本的情况下解决问题。
- 零样本学习是一种让机器学习模型能够识别和处理它在训练阶段从未见过的数据或任务的方法。这种方法依赖于模型的泛化能力，即利用已有的知识和理解来推断新的概念或任务。
- 思维链推理是一种模拟人类解决问题过程的方法，通过生成一系列中间步骤和解释来得到最终答案。这种方法帮助模型在解决复杂问题时能够展示其推理过程，从而提高解决问题的准确性和可解释性。

Plan-and-Solve 认知框架
要求大模型分两步走：首先制订一个解决问题的计划，这个计划会生成一个逐步行动的方案，然后实施这个方案来找到答案。这种认知框架首先规划解决方案的每个步骤，然后按照计划执行这些步骤。它先设计一个计划，将整个任务划分为较小的子任务，然后根据计划执行子任务。**核心就是把复杂任务的解决过程分解为两个阶段：**
- **计划阶段涉及理解问题、分析任务结构，并制定一个详细的解决方案。**
- **执行阶段则是根据计划的步骤来实际解决问题。**
- **总结成一句话：计划和执行的解耦。**
"""
import operator
import os
import platform
import subprocess
import psutil
from langchain.agents import create_tool_calling_agent
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# Define diagnostic and action tools
@tool
def check_cpu_usage():
    """Checks the actual CPU usage."""
    cpu_usage = psutil.cpu_percent(interval=1)
    return f"CPU Usage is {cpu_usage}%."


@tool
def check_disk_space():
    """Checks actual disk space."""
    disk_usage = psutil.disk_usage('/').percent
    return f"Disk space usage is at {disk_usage}%."


@tool
def check_network():
    """Checks network connectivity by pinging a reliable server."""
    response = subprocess.run(["ping", "-c", "1", "8.8.8.8"], stdout=subprocess.PIPE)
    if response.returncode == 0:
        return "Network connectivity is stable."
    else:
        return "Network connectivity issue detected."


@tool
def restart_server():
    """Restarts the server with an OS-independent approach."""
    current_os = platform.system()

    try:
        if current_os == "Windows":
            os.system("shutdown /r /t 0")  # Windows restart command
        elif current_os == "Linux" or current_os == "Darwin":  # Darwin is macOS
            os.system("sudo shutdown -r now")  # Linux/macOS restart command
        else:
            return "Unsupported operating system for server restart."
        return "Server restart initiated successfully."
    except Exception as e:
        return f"Failed to restart server: {e}"


# Tools setup
tools = [check_cpu_usage, check_disk_space, check_network, restart_server]

# 您是一名IT诊断代理。遵循以下指南：
# 1.按顺序检查指标：CPU->磁盘->网络
# 2.分析阈值：
# -CPU使用率>80%：严重
# -磁盘空间<15%：严重
# -网络：必须稳定
# 3.采取行动：
# -如果任何指标至关重要，建议重新启动服务器
# -如果所有指标正常，报告健康状态
# 4.除非明确需要，否则切勿重复检查
# 5.服务器重启后，执行最后一次检查以验证改进情况
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an IT diagnostics agent. Follow these guidelines:
    1. Check metrics in order: CPU -> Disk -> Network
    2. Analysis thresholds:
       - CPU Usage > 80%: Critical
       - Disk Space < 15%: Critical
       - Network: Must be stable
    3. Take action:
       - If any metric is critical, recommend server restart
       - If all metrics normal, report healthy status
    4. Never repeat checks unless explicitly needed
    5. After server restart, perform one final check to verify improvement"""),
    ("placeholder", "{messages}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_executor = create_tool_calling_agent(llm, tools, prompt)


# Modified state structure to track check history and results
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    checks_complete: bool
    restart_performed: bool
    final_check: bool
    messages: Annotated[List[str], operator.add]  # Add messages to track each step and result


# 检查和解决服务器问题的任务 计划类
class Plan(BaseModel):
    steps: List[str] = Field(description="Tasks to check and resolve server issues")


class Response(BaseModel):
    response: str


class Act(BaseModel):
    action: Union[Response, Plan] = Field(description="Action to perform")


# 制定一个重点诊断计划：
# 1.仅包括必要的检查
# 2.跟踪已检查的内容
# 3.如果超过阈值，则包括重新启动
# 4.重启后的最后一次验证
# 可用工具：check_cpu_usage、check_disk_space、check_network、restart_server
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """Create a focused diagnostic plan:
    1. Only include necessary checks
    2. Track what's already been checked
    3. Include restart if thresholds exceeded
    4. One final verification after restart
    Available tools: check_cpu_usage, check_disk_space, check_network, restart_server"""),
    ("placeholder", "{messages}"),
])

planner = planner_prompt | llm.bind_tools(tools).with_structured_output(Plan)

# 分析当前形势并确定下一步行动：
# 任务：｛input｝
# 已完成的步骤：｛past_steps｝
# 检查完成：{Checks_complete}
# 已执行重新启动：｛Restart_performed｝
# 最终检查：{Final_check}
#
# 规则：
# 1.除非重新启动后进行验证，否则不要重复检查
# 2.如果CPU>80%或磁盘<15%，请继续重新启动
# 3.重新启动后，进行最后一次检查
# 4.最终验证后结束流程
#
# 可用工具：
# -check_cpu_usage
# -check_disk_space
# -检查网络
# -restart_server
replanner_prompt = ChatPromptTemplate.from_template("""
Analyze the current situation and determine next steps:

Task: {input}
Completed steps: {past_steps}
Checks complete: {checks_complete}
Restart performed: {restart_performed}
Final check: {final_check}

Rules:
1. Don't repeat checks unless verifying after restart
2. If CPU > 80% or Disk < 15%, proceed to restart
3. After restart, do one final check
4. End process after final verification

Available tools:
- check_cpu_usage
- check_disk_space
- check_network
- restart_server
""")

replanner = replanner_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Act)


# Enhanced execution step with state tracking
async def execute_step(state: PlanExecute):
    if not state.get("checks_complete"):
        state["checks_complete"] = False
    if not state.get("restart_performed"):
        state["restart_performed"] = False
    if not state.get("final_check"):
        state["final_check"] = False

    plan = state["plan"]
    if not plan:
        return state

    task = plan[0]
    tool_map = {
        "check_cpu_usage": check_cpu_usage,
        "check_disk_space": check_disk_space,
        "check_network": check_network,
        "restart_server": restart_server
    }

    if task in tool_map:
        result = tool_map[task].invoke({})
        state["past_steps"].append((task, result))
        state["messages"].append(f"Executed {task}: {result}")  # Log the message here
        state["plan"] = state["plan"][1:]

        # Update state flags based on actions
        if task == "restart_server":
            state["restart_performed"] = True
        elif state["restart_performed"] and not state["final_check"]:
            state["final_check"] = True

        # Check if all initial checks are complete
        if len(state["past_steps"]) >= 3 and not state["checks_complete"]:
            state["checks_complete"] = True

    return state


# Initial planning step
async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    state["plan"] = plan.steps
    state["messages"].append(f"Planned {plan}: {plan.steps}")  # Log the message here
    state["checks_complete"] = False
    state["restart_performed"] = False
    state["final_check"] = False
    return state


# Enhanced replanning with better decision making
async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)

    if isinstance(output.action, Response):
        return {"response": output.action.response}

    # Avoid repeating checks unless doing final verification
    if state["restart_performed"] and not state["final_check"]:
        state["plan"] = ["check_cpu_usage", "check_disk_space", "check_network"]
    else:
        state["plan"] = [step for step in output.action.steps
                         if step not in [s[0] for s in state["past_steps"]] or
                         (state["restart_performed"] and not state["final_check"])]

    return state


# Enhanced end condition check
def should_end(state: PlanExecute):
    # End conditions:
    # 1. All checks complete and no issues found
    # 2. Restart performed and final check complete
    # 3. Maximum steps reached (safety check)
    if (state["checks_complete"] and not state["plan"]) or \
            (state["restart_performed"] and state["final_check"]) or \
            len(state["past_steps"]) > 15:  # Safety limit
        return END
    return "agent"


# Build the workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])

app = workflow.compile()


# Example usage
async def run_plan_and_execute():
    inputs = {
        "input": "Diagnose the server issue and restart if necessary.",
        "past_steps": [],
        "checks_complete": False,
        "restart_performed": False,
        "final_check": False,
        "messages": []  # Initialize an empty list for messages
    }
    config = {"recursion_limit": 15}

    async for event in app.astream(inputs, config=config):
        print(event)
        print("\n\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_plan_and_execute())
