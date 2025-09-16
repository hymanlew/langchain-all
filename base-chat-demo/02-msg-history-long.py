# pip install langchain_community -i https://pypi.org/simple
import os
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAI
import asyncio

llm = ChatOpenAI(model="gpt-4")
'''
- ChatMessageHistory：维护聊天上下文所有记录
- ConversationBufferMemory：是对 ChatMessageHistory 的包装器，字节缓存，无窗口
- ConversationBuferWindowMemory：用于跟踪会话中的来回消息，然后使用大小为 k 的窗口将最近的 k 条来回消息提取出来存储到内存。字节缓存 - 历史记录（次数）遗忘机制
- ConversationTokenBufferMemory：字节缓存 - 历史记录（基于字符长度）遗忘机制
- ConversationSummaryMemory：可将当前对话上下文进行总结并注入到提示中，对较长的对话进行总结时非常有用，能节省 token。字节缓存 - 对过往历史记录做摘要存储，并在新对话中提前读取这些历史上下文，无窗口
- ConversationSummaryBufferMemory：有窗口的会话摘要
- ConversationKGMemory：使用知识图谱来重建记忆
- ConversationEntityMemory：使用语言横型(LLMS)提取实体相关的信息，并随着时间的推移逐渐积累对该实体的知识
- VectorStoreRetrieverMemory：将记忆存储在向量数据库中，并在每次调用时査询最“显著”的前 K 个文档
'''

# --- 获取全量上下文对话信息
history = ChatMessageHistory()
history.add_ai_message('xxx')
history.add_user_message('xxx')


# --- Agent 获取全量上下文对话信息
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory
memory = ConversationBufferMemory(
    return_message=True, memory_key='history', output_key='output'
)
print(memory.load_memory_variables({"input": "查询的消息"}))


# --- 滑动窗口获取最近部分对话内容
from langchain.memory import ConversationBufferWindowMemory
# 只保留最后1次互动的记忆
memory = ConversationBufferWindowMemory(k=1)


# --- 获取历史对话中实体信息，让 AI 记住对话中的关键实体和实体关系细节，而非全量
memory = ConversationEntityMemory(llm=llm)
memory.save_context({"input": "你好"}, {"output": "怎么了"})
variables = memory.load_memory_variables({})


# --- 利用知识图谱获取历史对话中的实体及其联系
from langchain.memory import ConversationKGMemory
llm = OpenAI(temperature=0)
memory = ConversationKGMemory(llm=llm)
memory.save_context({"input": "小李是程序员"}, {"output": "知道了，小李是程序员"})
variables = memory.load_memory_variables({"input": "告诉我关于小李的信息"})
print(variables)
# 输出 {'history': 'On 小李: 小李 is 程序员. 小李 的笔名 莫尔索.'}


# --- 基于 Redis 存储对话信息
chain = None
def get_session_history(user_id: str, session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(user_id, session_id)

do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg',  # 每次聊天时要发送的 msg 的 key
    history_message_key='history',
)


# --- 基于向量存储并检索对话信息
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)
memory.save_context({"input": "我喜欢吃火锅"}, {"output": "听起来很好吃"})
memory.save_context({"input": "我不喜欢看摔跤比赛"}, {"output": "我也是"})

PROMPT_TEMPLATE = """以下是人类和 AI 之间的友好对话。AI 话语多且提供了许多来自其上
下文的具体细节。如果 AI 不知道问题的答案，它会诚实地说不知道。
以前对话的相关片段：
{history}
（如果不相关，你不需要使用这些信息）
当前对话：
人类：{input}
AI：
"""
prompt = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True # 打印输出详细日志
)
print(conversation_with_summary.predict(input="你好，我是莫尔索，你叫什么"))
print(conversation_with_summary.predict(input="我喜欢的食物是什么？"))
print(conversation_with_summary.predict(input="我提到了哪些运动？"))

#异步流处理, 主要用于日志打印，排查线上问题
async def async_stream():
    events = []
    async for event in llm.astream_events("hello", version="v2"):
        events.append(event)
    print(events)

#运行异步流处理
asyncio.run(async_stream())
