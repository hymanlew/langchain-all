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
#导入相关的工具
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
#设置OpenA网站和SerpApi 网站提供的 AP密钥
from langchain.chat_models import ChatOpenAl
from dotenv import load_dotenv
from langchain.tools import tool

#用于加载环境变量, 加载.env文件中的环境变量
load_dotenv()

#库存查询
@tool
def check_inventory(flower_type: str)-> int:
    """
    查询特定类型花的库存数量。
    参数:
    - flower_type:花的类型
    返回:
    - 库存数量(暂时返回一个固定的数字)
    DM MM 如
    """
    # 实际应用中这里应该是数据库查询或其他形式的库存检查
    # 假设每种花都有100个单位
    return 100

#定价函数
@tool
def calculate_price(base_price: float, markup: float) -> float:
    """
    根据基础价格和加价百分比计算最终价格。
    参数:
    - base_price: 基础价格
    - markup: 加价百分比
    返回:
    - 最终价格
    """
    return base_price * (1 + markup)

# 调度函数
@tool
def schedule_delivery(order_id: int, delivery_date: str):
    """
    安排订单的配送
    参数:
    - order_id:订单编号
    - delivery_date: 配送日期
    返回:
    - 配送状态或确认信息
    """
    # 在实际应用中这里应该是对接配送系统的过程
    return f"订单{order_id}已安排在{delivery_date}配送"

tools = [check_inventory, calculate_price]

"""
以上这个任务其实是一个“不可能完成的任务”，因为任务需求根本不清晰。我们来看看 Agent是会坦诚交代自己的能力不足以完成任务，还是会“自信地胡说八道”。
"""
# 设置大模型
model = ChatOpenAl(temperature=0)

#设置计划者和执行者
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
#初始化 Plan-and-Execute Agent
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
# 运行 Agent 解决问题
agent.run("查查玫瑰的库存然后给出出货方案!")

"""
它会先输出第一部分：计划阶段 plan，给出具体操作流程和思路。
然后会一步步输出分行阶段 Execute 的第一步骤。
由于上面给出的问题无法完成，所以它不会有最终的答案，只需要给出有答案的问题，它才会正常分析出答案。
"""
