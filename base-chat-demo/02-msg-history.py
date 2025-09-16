# pip install langchain_community -i https://pypi.org/simple
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

# 聊天机器人案例，创建模型
model = ChatOpenAI(
    model='glm-4-0520',
    temperature='0.6',
    api_key='0884a4262379e6b9e98d08be606f2192.TOaCwXTLNYo1GlRM',
    base_url='https://open.bigmodel.cn/api/paas/v4/'
)

# 定义提示模板，其中一定要配置历史记录的占位符
"""
MessagesPlaceholder 用于在提示模板中插入动态生成的消息列表，即对话历史。
否则每次请求都是独立的，无法进行连贯的多轮对话。
"""
# 因为每次发送消息时，要把历史记录也发送，否则大模型不能关联上下文
prompt_template = ChatPromptTemplate.from_messages([
    ('system', '你是一个乐于助人的助手。用{language}尽你所能回答所有问题。'),
    MessagesPlaceholder(variable_name='history'),
	("human", "{my_msg}"),
])

# 生成任务调用链
chain = prompt_template | model

# 保存聊天的历史记录
# 所有用户的聊天记录都保存到 store 字典。一个用户对应一个 sessionId
# key: sessionId, value: 历史聊天记录对象
store = {}

# 此函数接收一个 session_id，并返回一个消息历史记录对象。
# 如果没有会话时，就新建一个会话记录对象（是一个消息列表，用于保存历史聊天记录），存储不同会话ID的消息历史
def get_session_history(user_id: str, session_id: str) -> BaseChatMessageHistory:
    if (user_id, session_id) not in store:
        store[(user_id, session_id)] = ChatMessageHistory()
    return store[(user_id, session_id)]

# 创建会话聊天对象，来执行对话相关的操作（包括自动追加聊天记录到 ChatMessageHistory 中）
do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg',  # 每次聊天时要发送的 msg 的 key
	history_message_key='history',
	history_factory_config=[
		ConfigurableFieldSpec(
			id="user_id",
			annotation=str,
			name="User ID",
			description="用户的唯一标识符",
			default="",
			is_shared=True,
		),
		ConfigurableFieldSpec(
			id="session_id",
			annotation=str,
			name="Session ID",
			description="对话的唯一标识符",
			default="",
			is_shared=True,
		),
	],
)

# 手动发送聊天请求，给当前会话定义一个sessionId，第一个用户
config = {'configurable': {'user_id':'111', 'session_id': 'user1'}}
# 第一轮
resp1 = do_message.invoke(
    {
        'my_msg': '你好啊！ 我是 HYMAN',
        'language': '中文'
    },
    config=config
)
print(resp1.content)

# 第二轮
resp2 = do_message.invoke(
    {
        'my_msg': '请问：我的名字是什么？',
        'language': '中文'
    },
    config=config
)
print(resp2.content)

# 给当前会话定义一个sessionId，第二个用户
# 第3轮：返回的数据是流式的（一个 token 一个 token 的返回）
config = {'configurable': {'user_id':'222', 'session_id': 'user2'}}
for resp in do_message.stream(
	{
        'my_msg': '请给我讲一个笑话？',
        'language': 'English'
     },
    config=config
):
    # 每一次resp都是一个token
    print(resp.content, end='-', flush=True)


#运行异步流处理
async def async_stream():
    async for chunk in do_message.astream({"input":"鹦鹉"}):
        print(chunk, end="", flush=True)

asyncio.run(async_stream())