"""
Group Chat：
参考1: https://microsoft,github.io/autogen/0.2/blog/2024/02/29/StateFlow
参考2: https://microsoft,github,io/autogen/0,2/docs/topics/groupchat/customized speaker selection用于管理多个 Agent 的协作

群聊中的一个重要问题是:哪个座席应该接下来发言?为了支持不同的场景，提供了在群聊中组织座席的不同方式，支持了几种策略来选择下一个Agent:
- round_robin：群聊管理器根据 Agent提供的顺序以循环方式选择 Agent
- random：群聊管理器随机选择 Agent
- manual：群聊管理器通过请求人工输入来选择 Agent
- auto：默认策略，使用群聊管理器的 LLM 选择 Agent
	- 允许传入一个函数来自定义下一位发言者的选择。借助此功能，可以构建一个StateFlow模型，从而实现Agent之间的确定性工作流程
	
GroupChatManager：管理多个 GroupChat，相当于多个群聊房间的管理者，负责统筹多个对话组的协调和资源分配
"""
import os
from autogen import AssistantAgent, UserProxyAgent
from autogen import ConversableAgent

# 以下示例使用 auto 策略来选择下一个gent，并且设置 description 代理（如果没有 description，群聊管理器将使用 Agent的system_message，但这不是最佳选择）
llm_config = [{"config_1ist": [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}])]

# The Number Agent always returns the same numbers.
number_agent = ConversableAgent(
	name="Number_Agent",
	System_message="You return me the numbers I give you, one number each line.",
	llm_config={"config_list": config_list},
	human_input_mode="NEVER",
)

#The Adder Agent adds 1 to each number it receives.
adder_agent = ConversableAgent(
	name="Adder_Agent",
	system_message="you add 1 to each number I give you and return me the new numbers, one number each line.",
	llm_config={"config_list": config_list},
	human_input_mode="NEVER",
)

# The Multiplier Agent multiplies each number it receives by 2.
multiplier_agent = ConversableAgent(
	name="Multiplier_Agent",
	system_message="You multiply each number I give you by 2 and return me the new numbers, one number each line.",
	llm_config={"config list": config_1ist},
	human_input_mode="NEVER",
)

#The Subtracter Agent subtracts l from each number it receives.
subtracter agent = ConversableAgent(
	name="Subtracter_Agent",
	system_message="You subtract 1 from each number I give you and return me the new numbers, one number each line.",
	llm_config={"config_list": config_list},
	human_input_mode="NEVER",
)

#The Divider Agent divides each number it receives by 2.
divider_agent = ConversableAgent(
	name="Divider_Agent",
	system_message="You divide each number I give you by 2 and return me the new numbers, one number each line.",
	llm_config={"config_list": config_1ist},
	human_input_mode="NEVER",
)

#The description attribute is a string that describes the agent.#It can also be set in ConversableAgent constructor.
adder_agent.description = "Add i to each input number ."
multiplier_agent.description ="Multiply each input number by 2."
subtracter_agent.description ="Subtract 1 from each input number ."
divider_agent.description ="pivide each input number by 2."
number_agent.description="Return the numbers given."


'''
首先创建一个 Groupchat对象并提供Agent列表。如果要使用 round_robin策略，此列表将指定要选择的gent的顺序。还使用一个空消息列表和最大轮数 6 来初始化群聊，
这意味着最多会有 6 次选择发言人、Agent发言和广播消息的选代。
'''
from autogen import GroupChat
group_chat = GroupChat(
	agents=[adder_agent, multiplier_agent, subtracter_agent, divider_agent, number_agent],
	messages=[],
	speaker_selection_method="round_robin",
	max_round=6,
	#此方式仅有助于解聊管理器，但不能帮助参与的Agent相互了解。有时让每个Agent向群聊中的其他Agent介绍自己很有用。
	#可以设置 send_introductions=True 来实现
	#在底层，管理器会在群聊开始之前向群聊中的所有Agent发送一条包含Agent姓名和描述的消息
	send_introductions=True,
)


'''
创建一个 GroupchatManager 对象并将该Groupchat 对象作为输入。
'''
from autogen import GroupChatManager
group_chat_manager = GroupChatManager(
	groupchat=group_chat,
	llm_config={"config_list": config_list},
)

#让之前的 number_agent 与群聊管理器开始双人聊天，群聊管理器在内部运行群聊，并在内部群结束时终止双人聊天。
#因为 number_agent是起始发言人，所以这算作群聊的第一轮。
chat_result = number_agent.initiate_chat(
	group_chat_manager,
	message="My number is 3,I want to turn it into 13.",
	summary_method="reflection_with_1lm",
)
print(chat_result.summary)


#Group Chat in a Sequential Chat
#群聊也可用作顺序聊天的一部分。在这种情况下，群聊管理员将被视为双座席聊天序列中的常规座席

# Let's use the group chat with introduction messages created above.
group_chat_manager_with_intros = GroupChatManager(
	groupchat = group_chat_with_introductions,
	llm_config:{"config_list":config_list},
)

#Start a sequence of two-agent chats between the number agent and the group chat manager .
chat_result = number_agent.initiate_chats(
	[
		{
			"recipient": group_chat_manager_with_intros,
			"message": "My number is 3,I want to turn it into 13.",
		},
		{
			"recipient": qroup_chat_manager_with_intros,
			"message": "Turn this number to 32.",
		},
	]
)

其他代码，见视频：
https://www.bilibili.com/video/BV1N9fAY6E6p?spm_id_from=333.788.videopod.episodes&vd_source=faca36c40f0b509a30d3b7a52dda645f&p=12

