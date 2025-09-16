# 爬虫使用：requests 请求，from bs4 import BeautifulSoup 响应解析
import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

model = ChatOpenAI(model='gpt-4-turbo')

"""
RAG 检索增强生成，实现思路:
1.加载：首先加载数据。通过 DocumentLoaders 完成的。
2.分割：Text splitters 将大型文档分割成小块。这对于索引数据和将其传递给模型很有用，因为大块数据更难搜索，并且不适合模型的有限上下文窗口。
3.存储：需要存储和索引我们的分割，以便以后搜索。通常使用 VectorStore 和 Embeddings 模型完成。
4.检索：给定用户输入，使用检索器从存储中检索相关分割。
5.生成：ChatModel/LLM 使用包括问题和检索到的数据的提示生成答案
"""
# 1、加载数据: 网上的一篇博客内容，类似于爬虫爬数据
loader = WebBaseLoader(
    web_paths=['https://lilianweng.github.io/posts/2023-06-23-agent/'],
    # 使用 BeautifulSoup - bs4 解析响应的 html 数据
    # 这里只解析 parse_only 指定的 class 标签
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)
docs = loader.load()
# print(len(docs))
# print(docs)

# 2、大文本的切割
"""
chunk_size 是每个文本块的最大长度(字符数)
chunk_overlap 是块之间的重叠量。重叠可以帮助保持上下文，避免在分割时切断重要信息，比如句子中间断开。重叠200个字符可以在相邻块之间保留部分内容，确保语义连贯。
RecursiveCharacterTextSplitter 类的工作方式：
递归分割意味着它首先尝试用较大的分隔符（比如双换行）分割，如果块太大，再逐步使用较小的分隔符（比如单换行、句号、空格）。这种方法尽可能保持段落或句子的完整性，同时确保块大小符合要求。
"""
text = "hello world, how about you? thanks, I am fine.  the machine learning class. So what I wanna do today is just spend a little time going over the logistics of the class, and then we'll start to talk a bit about machine learning"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
res = splitter.split_text(text)
for s in res:
    print(s)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# 2、存储，并向量化
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# 3、检索器
retriever = vectorstore.as_retriever()

# 整合
# 创建一个问题的模板
system_prompt = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""
# 提问和回答的 历史记录  模板
prompt_templa = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),  #
        ("human", "{input}"),
    ]
)

# 创建一个多文本处理的 chain 链
doc_chain = create_stuff_documents_chain(model, prompt_templa)
# 创建一个检索的 chain 链，执行的顺序 = 参数的逆顺序
# retrie_chain = create_retrieval_chain(retriever, doc_chain)
# resp = retrie_chain.invoke({'input': "What is Task Decomposition?"})
# print(resp['answer'])

'''
注意：
一般情况下，我们构建的链（chain）直接使用输入问答记录来关联上下文。但在此案例中，
查询检索器也需要对话上下文才能被理解。

解决办法：
添加一个子链(chain)，它采用最新用户问题和聊天历史，并在它引用历史信息中的任何信息
时重新表述问题。这可以被简单地认为是构建一个新的“历史感知”检索器。
这个子链的目的：就是让检索过程融入了对话的上下文。
'''

# 子链的提示模板
son_qa_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

retriever_history_templa = ChatPromptTemplate.from_messages(
    [
        ('system', son_qa_prompt),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)

# 创建一个子链
history_retrie_chain = create_history_aware_retriever(model, retriever, retriever_history_templa)

# 保持问答的历史记录
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建父链chain: 把前两个链整合
chain = create_retrieval_chain(history_retrie_chain, doc_chain)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# 第一轮对话
resp1 = result_chain.invoke(
    {'input': 'What is Task Decomposition?'},
    config={'configurable': {'session_id': 'user1'}}
)
print(resp1['answer'])

# 第二轮对话
resp2 = result_chain.invoke(
    {'input': 'What are common ways of doing it?'},
    config={'configurable': {'session_id': 'user1'}}
)
print(resp2['answer'])
