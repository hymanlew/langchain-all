# 导入 langchain 的内置向量数据库 chroma
# pip install langchain_chroma -i https://pypi.org/simple
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGSMITH_API_KEY"] = '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'

# 聊天机器人案例, 创建模型
# model = ChatOpenAI(model='gpt-4-turbo')
model = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your_aliyun_api_key",
    model="qwen-turbo",
    temperature=0.6
)
model = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key="your_aliyun_api_key",
    model="glm-4-0520",
    temperature=0.6
)

# 真实文档内容读取 @see pdf_search_ai.py
# 准备测试数据 ，假设我们提供的文档数据如下：
documents = [
    Document(
        # 表示文档的内容，可以存储真实文档内的所有内容
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",
        # 文档的元数据，可以是摘要，作者，来源等等，可以自定义
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

# 实例化一个向量数空间
# vector_store = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
# 使用国内魔塔社区的嵌入模型，且要注意，一定只能使用 langchain 支持的模型
# https://www.modelscope.cn/models/iic/nlp_gte_sentence-embedding_chinese-large/summary
# 推荐中文环境下使用，个人资源有限用 base 或 small，在公司商业中用 large
from langchain_community.embeddings import ModelScopeEmbeddings
embedding = ModelScopeEmbeddings(model_id='iic/nlp_gte_sentence-embedding_chinese-base')
vector_store = Chroma.from_documents(documents, embedding=embedding, persist_directory="./db")

# 相似度的查询: 返回相似的分数（是指搜索到的内容与指定搜索的词，的距离长度），
# 分数越低相似度越高，即距离越近则相似度越高
print(vector_store.similarity_search_with_score('咖啡猫'))

# 检索器（是让 chain 调用执行的）: bind(k=1) 返回相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)
# print(retriever.batch(['咖啡猫', '鲨鱼']))

# 提示模板
message = """
使用提供的上下文仅回答这个问题:
{question}
上下文:
{context}
"""
prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

# 用 chain 链连接的对象，必须是 runnable 对象，否则是无效不执行的。且会报错
# RunnablePassthrough 是代表接收用户的问题，然后再传递给 prompt 和 model。
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model | StrOutputParseer()
chain = {'question': RunnablePassthrough()} | RunnablePassthrough.assign(context=itemgetter("question") | retriever) | prompt_temp | model

resp = chain.invoke('请介绍一下猫？')
print(resp.content)
