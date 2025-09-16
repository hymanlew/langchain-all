"""
AutoGen 是由微软开发的一个多 Agent 框架，它是一种协作 Agent，且这些 Agent 可以通过对话交互来完成任务。AutoGen 允许人类无缝参与
（也就是说，人类可以在 Agent 执行任务的过程中提供反馈）。
凭借可定制和可对话的 Agent，可以利用 AutoGen 构建涉及对话自主性、Agent 数量和 Agent 对话拓扑结构等方面的广泛对话模式。

pip install autogen-agentchat~=0.2 #支持0.2版

流程如下：
1，基于大模型创建 Agent 助手角色
2，创建用户代理，用于人工介入或 Agent 自动处理
3，使用 Agent 发起对话，指定消息，及消息的发送者（用户代理），接收者（Agent 助手），日志，对话摘要等
4，但是它也只是适合少量或几个 Agent 代理的编排，如果是大量的代理编排及复杂任务流程编排，还是要使用 langGraph。
"""
import os
import autogen
from autogen import AssistantAgent, UserProxyAgent


# 配置大模型
llm_config = {
    "config_list": [{"model": "gpt-4", "api_key": os.envirn["OPENAI_API_KEY"]}],
}

# 定义一个与花语秘境运营相关的任务
inventory_tasks = [
    """查看当前库存中各种鲜花的数量，并报告哪些鲜花库存不足。""",
    """根据过去一个月的销售数据，预测接下来一个月哪些鲜花的需求量会增加。""",
]
market_research_tasks = ["""分析市场趋势，找出当前最受欢迎的鲜花种类及其可能的原因。"""]
content_creation_tasks = ["""利用提供的信息，撰写一篇吸引人的博客文章，介绍最受欢迎的鲜花及选购技巧。"""]

'''
ConversableAgent：
- 是用于管理每个角色的行为。
- 参数 system_message 指定其用途，
- 参数 description 告知其他 Agent 什么时候该调用它
- Agent 的子类 ConversableAgent 是会话的基类，一般不直接使用，而是作为其他类的父类。它能保持对话状态、历史记录，并调用其他工具。

ConversableAgent 的子类 UserProxyAgent:
- 用户角色，用于模拟用户输入和执行代码等。
- 能够安排任务发起任务，user_proxy 充当的角色之一是老板，而其他 agent 是员工
- 无论其他 agent 做什么，至少需要一个user_proxy提出目标问题

ConversableAgent 的子类 AssistantAgent:
- AI角色，用于执行任务处理、调用 API和逻辑推理等相关代码.

对话至少包含一个 AssistantAgent 和一个 UserProxyAgent。
- 对话开始后，在两个gent之间展开，A.send0)·>B.recv0, B.send0)·>A.recv0.,如此往复直到结束(找到答案、超过对话轮数或满足退出条件)
- 每次 AssistantAgent 调用 LLM 时，会参考初始问题和当前的多轮对话内容做出回应
- UserProxyAgent 可以让用户参与、执行代码或调用工具来完成具体任务
'''
# 创建 Agent 助手角色，负责处理业务逻辑
inventory_assistant = autogen.AssistantAgent(
    name="库存管理助理",
    llm_config=llm_config,
)
market_research_assistant = autogen.AssistantAgent(
    name="市场研究助理",
    llm_config=llm_config,
)
content_creator = autogen.AssistantAgent(
    name="内容创作助理",
    llm_config=llm_config,
    system_message="""
        你是一名专业的撰稿人，以洞察力强和文章引人人胜著称。
        你能将复杂的概念转化为引人入胜的叙述。当一切完成后，请回复"结束".
    """
)

# 创建用户代理，负责模拟用户行为（如输入消息、执行代码）
user_proxy_auto = autogen.UserProxyAgent(
    name="用户代理_自动",
    # 全自动执行，无需人工干预。默认值是最后一步与人交互
    human_input_mode="NEVER",
    # 结合 max_turns 和 is_termination_msg 函数，在超时或任务失败时触发备用流程
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith(" 结束"),
    # 代码执行配置
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
		
		#在 docker 容器中执行代码，安全
		"executor": DockerCommandLinecodeExecutor()
		#Loca1commandLineCodeExecutor 是本地执行
		"executor": autogen.coding.LocalcommandLineCodeExecutor(work_dir="coding")
    },
)

user_proxy = autogen.UserProxyAgent(
    name="用户代理_手动",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith(" 结束"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },
)

'''
GroupChat 用于管理多个 Agent 的协作。是通过多个 agent 以一定规则结合实现一个大功能，具体规则可以由聊天组来定义。
所有Agent都参与到一个对话线程中并共享相同的上下文。这对于需要多个Agent之间协作的任务非常重要.如果有两个以上Agent，就不确定的下
一句话该传给谁，就需要使用 ChatGroup 管理，并利用 transitions 设置Agent之间的衔接,

GroupChat:管理多个Agent之间的对话，相当于管理一个群聊房间。可通过字典图限定各个 Agent 可与谁交互。
GroupChatManager:管理多个 GroupChat，相当于多个群聊房间的管理者，负责统筹多个对话组的协调和资源分配。
'''
# 发起对话，顺序对话模式，包含了三个阶段：
chat_results = autogen.initiate_chats(
    [
        {
            # 1. 消息发送者配置
            "sender": user_proxy_auto,
            # 2. 消息接收者配置
            "recipient": inventory_assistant,
            # 3. 发送的具体消息内容
            "message": inventory_tasks[0],
            # 4. 是否清空历史对话记录
            "clear_history": True,
            # 5. 是否静默执行（不打印日志）
            "silent": False,
            # 6. 对话摘要生成方式，摘要会存储在返回的 chat_results 中
            # - "last_msg" 表示直接取最后一条消息作为摘要
            # - "reflection_with_llm"：用LLM生成总结
            # - "accumulate"：合并所有消息
            "summary_method": "last_msg",
        },
        {
            "sender": user_proxy_auto,
            "recipient": market_research_assistant,
            "message": market_research_tasks[0],
            # 定义单次对话的最大轮数（即消息交换次数），达到该数值后对话自动终止（无论任务是否完成）
            # 作用是防止无限循环：当代理无法达成共识或任务超时时强制终止对话
            "max_turns": 2,
            # "summary_args": {"summary_prompt": "以{'趋势':'', '原因':''}格式返回"}
            "summary_method": "reflection_with_llm",
            # "carryover": chat_results[0].summary  # 若需将库存结果传递给市场分析，显式传递库存摘要
        },
        {
            "sender": user_proxy,
            "recipient": content_creator,
            "message": content_creation_tasks[0],
            # 将前序对话的上下文信息传递到后续对话，覆盖默认行为
            "carryover": "我希望在博客文章中包含一张数据表格或图。",
        },
    ]
)

"""
carryover 是 AutoGen 顺序对话模式中的上下文传递机制，用于将前序对话的结构化摘要传递给后续对话，实现跨代理的任务接力。其核心特点包括：
1. 信息继承：避免重复输入，确保任务连续性（如库存分析→市场预测→内容创作）。
2. 动态组合：与当前对话的 `message` 自动合并，形成完整的输入上下文。
3. 摘要依赖：依赖 `summary_method`（如 `reflection_with_llm`）生成高质量摘要。

在连续对话模式（如 initiate_chats 的多个对话序列）中，carryover 默认会将前序对话的上下文信息传递到后续对话。
它可以是字符串、字典或自定义对象，用于保留关键状态或总结。使用场景：
- 任务分阶段执行：例如先由技术支持代理解决问题，再将对话摘要传递给满意度调查代理。
- 跨对话记忆：避免重复输入相同信息。
            
carryover 常依赖 summary_method 生成的前序对话摘要，确保传递的信息是结构化且相关的。
当 human_input_mode="TERMINATE" 时，max_turns 达到后仍需人工确认终止，否则仅自动停止。
即它的默认行为：是将前序对话的 summary 自动传递到后续阶段。如果手动显示设置 "carryover": chat_results[0].summary，
则可以覆盖默认行为，直接注入自定义上下文。

上下文传递的完整流程（以上面代码为例）
1：库存管理对话，传递关系：user_proxy_auto -> inventory_assistant → last_msg 摘要 -> chat_results[0].summary`  
- 用户代理自动发送库存查询指令
- 库存助理生成报告（如"玫瑰库存不足"）
- 取最后一条消息作为摘要（如"百合剩余200支"） 

2：市场研究对话，实际输入就是指定的 分析市场趋势，输出摘要由 llm 生成的趋势总结
例如 `{"热门趋势": "玫瑰需求上涨30%", "原因": "情人节临近"}`。

3：内容创作对话（自定义 `carryover`）
实际输入是 撰写博客文章，并自定义 carryover 会覆盖前序摘要，直接传递人工指令（人工干预）。
否则自动继承前一个 chat_results 的摘要。

传递路径：  
A[库存对话] -> carryover=chat_results[0].summary -> B[市场对话]
B -> carryover=chat_results[1].summary -> C[内容对话]

如需进一步优化，可参考 AutoGen 官方文档中的 [顺序对话案例](https://cloud.tencent.com/developer/article/2505824)。
"""



